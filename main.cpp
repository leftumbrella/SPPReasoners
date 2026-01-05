#include <opencv2/opencv.hpp>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// ======================= 你原来的结构体（保持不变） =======================
struct Border {
    float fPossibility;
    unsigned short usX1;
    unsigned short usY1;
    unsigned short usX2;
    unsigned short usY2;
};

// ======================= CUDA 错误检查 =======================
#define CHECK_CUDA(call)                                                                 \
    do {                                                                                 \
        cudaError_t _e = (call);                                                         \
        if (_e != cudaSuccess) {                                                         \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_e)); \
        }                                                                                \
    } while (0)

// ======================= TensorRT Logger =======================
class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
};

// ======================= TensorRT 对象释放 =======================
template <typename T>
struct TrtDeleter {
    void operator()(T* p) const noexcept {
        delete p;
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;

// ======================= 读 plan 文件 =======================
static std::vector<std::uint8_t> readBinaryFile(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open engine file: " + path);
    }
    ifs.seekg(0, std::ios::end);
    const std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    std::vector<std::uint8_t> data(static_cast<size_t>(size));
    if (!ifs.read(reinterpret_cast<char*>(data.data()), size)) {
        throw std::runtime_error("Failed to read engine file: " + path);
    }
    return data;
}

// ======================= FP16(half) <-> float（纯 CPU 位运算，避免依赖 cuda_fp16.h） =======================
static inline std::uint16_t floatToHalfBits(float f) {
    std::uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(x));

    const std::uint32_t sign = (x >> 16) & 0x8000u;
    const std::uint32_t mantissa = x & 0x007FFFFFu;
    const std::int32_t exp = static_cast<std::int32_t>((x >> 23) & 0xFFu);

    if (exp == 255) { // Inf/NaN
        if (mantissa != 0) {
            return static_cast<std::uint16_t>(sign | 0x7E00u); // qNaN
        }
        return static_cast<std::uint16_t>(sign | 0x7C00u);     // Inf
    }

    // 127 是 float bias，15 是 half bias
    std::int32_t halfExp = exp - 127 + 15;

    if (halfExp >= 31) { // overflow -> Inf
        return static_cast<std::uint16_t>(sign | 0x7C00u);
    }
    if (halfExp <= 0) { // subnormal / underflow
        if (halfExp < -10) { // too small -> 0
            return static_cast<std::uint16_t>(sign);
        }
        // subnormal: 1.mantissa * 2^(exp-127) 变成 half 的 0.mantissa
        std::uint32_t m = mantissa | 0x00800000u; // 补上隐含 1
        const std::int32_t shift = 14 - halfExp;  // 24(含隐含1) -> 10
        std::uint32_t halfMant = (m >> shift);

        // round to nearest
        if ((m >> (shift - 1)) & 1u) {
            halfMant += 1u;
        }
        return static_cast<std::uint16_t>(sign | (halfMant & 0x03FFu));
    }

    // normal
    std::uint32_t halfMant = mantissa >> 13;
    // round to nearest
    if (mantissa & 0x00001000u) {
        halfMant += 1u;
        if (halfMant == 0x0400u) { // mantissa overflow
            halfMant = 0;
            halfExp += 1;
            if (halfExp >= 31) {
                return static_cast<std::uint16_t>(sign | 0x7C00u);
            }
        }
    }

    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(halfExp) << 10) | (halfMant & 0x03FFu));
}

static inline float halfBitsToFloat(std::uint16_t h) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16;
    const std::uint32_t exp  = (h >> 10) & 0x1Fu;
    const std::uint32_t mant = h & 0x03FFu;

    std::uint32_t out = 0;
    if (exp == 0) {
        if (mant == 0) {
            out = sign; // 0
        } else {
            // subnormal -> normal float
            std::uint32_t m = mant;
            std::int32_t e = -1;
            do {
                e++;
                m <<= 1;
            } while ((m & 0x0400u) == 0);

            m &= 0x03FFu;
            const std::uint32_t floatExp = static_cast<std::uint32_t>(127 - 15 - e);
            const std::uint32_t floatMant = m << 13;
            out = sign | (floatExp << 23) | floatMant;
        }
    } else if (exp == 31) {
        // Inf/NaN
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        // normal
        const std::uint32_t floatExp = exp + (127 - 15);
        const std::uint32_t floatMant = mant << 13;
        out = sign | (floatExp << 23) | floatMant;
    }

    float f = 0.0f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

// ======================= 体积计算 =======================
static inline std::int64_t volumeDims(const nvinfer1::Dims& d) {
    std::int64_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        v *= static_cast<std::int64_t>(d.d[i]);
    }
    return v;
}

static inline bool hasDynamicDim(const nvinfer1::Dims& d) {
    for (int i = 0; i < d.nbDims; ++i) {
        if (d.d[i] < 0) return true;
    }
    return false;
}

// ======================= TensorRT 推理封装 =======================
class YoloTrtRunner {
public:
    explicit YoloTrtRunner(const std::string& enginePath)
        : mEnginePath(enginePath) {
        loadEngine();
        allocBuffers();
    }

    ~YoloTrtRunner() {
        if (mStream) {
            cudaStreamDestroy(mStream);
            mStream = nullptr;
        }
        if (mDeviceInput) {
            cudaFree(mDeviceInput);
            mDeviceInput = nullptr;
        }
        if (mDeviceOutput) {
            cudaFree(mDeviceOutput);
            mDeviceOutput = nullptr;
        }
    }

    // 输入：CHW float（与你原代码一致）
    // 输出：返回 float（若 engine 输出是 FP16，会转回 float）
    std::vector<float> infer(const std::vector<float>& inputCHW) {
        if (static_cast<std::int64_t>(inputCHW.size()) != mInputElems) {
            throw std::runtime_error("Input size mismatch. expected=" + std::to_string(mInputElems) +
                                     " got=" + std::to_string(inputCHW.size()));
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        // 1) H2D: input
        if (mInputType == nvinfer1::DataType::kHALF) {
            mHostInputHalf.resize(static_cast<size_t>(mInputElems));
            for (size_t i = 0; i < mHostInputHalf.size(); ++i) {
                mHostInputHalf[i] = floatToHalfBits(inputCHW[i]);
            }
            CHECK_CUDA(cudaMemcpyAsync(mDeviceInput, mHostInputHalf.data(),
                                       mInputBytes, cudaMemcpyHostToDevice, mStream));
        } else if (mInputType == nvinfer1::DataType::kFLOAT) {
            CHECK_CUDA(cudaMemcpyAsync(mDeviceInput, inputCHW.data(),
                                       mInputBytes, cudaMemcpyHostToDevice, mStream));
        } else {
            throw std::runtime_error("Unsupported input datatype (only FP32/FP16 handled)");
        }

        // 2) enqueue
        // TRT10：对每个 IO tensor 设地址，然后 enqueueV3（NVIDIA 官方迁移建议）:contentReference[oaicite:5]{index=5}
        if (!mContext->enqueueV3(mStream)) {
            throw std::runtime_error("enqueueV3 failed");
        }

        // 3) D2H: output
        if (mOutputType == nvinfer1::DataType::kHALF) {
            mHostOutputHalf.resize(static_cast<size_t>(mOutputElems));
            CHECK_CUDA(cudaMemcpyAsync(mHostOutputHalf.data(), mDeviceOutput,
                                       mOutputBytes, cudaMemcpyDeviceToHost, mStream));
        } else if (mOutputType == nvinfer1::DataType::kFLOAT) {
            mHostOutputFloat.resize(static_cast<size_t>(mOutputElems));
            CHECK_CUDA(cudaMemcpyAsync(mHostOutputFloat.data(), mDeviceOutput,
                                       mOutputBytes, cudaMemcpyDeviceToHost, mStream));
        } else {
            throw std::runtime_error("Unsupported output datatype (only FP32/FP16 handled)");
        }

        CHECK_CUDA(cudaStreamSynchronize(mStream));

        auto t1 = std::chrono::high_resolution_clock::now();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::cout << "Time:" << ms << " ms" << std::endl;

        // 4) 输出统一转 float 给你原来的后处理用
        std::vector<float> out(static_cast<size_t>(mOutputElems));
        if (mOutputType == nvinfer1::DataType::kHALF) {
            for (size_t i = 0; i < out.size(); ++i) {
                out[i] = halfBitsToFloat(mHostOutputHalf[i]);
            }
        } else {
            out = mHostOutputFloat;
        }
        return out;
    }

    nvinfer1::Dims getOutputDims() const { return mOutputDims; }

private:
    void loadEngine() {
        static TrtLogger logger;

        const auto engineData = readBinaryFile(mEnginePath);
        mRuntime.reset(nvinfer1::createInferRuntime(logger));
        if (!mRuntime) {
            throw std::runtime_error("createInferRuntime failed");
        }

        // TRT10：deserializeCudaEngine 接口仍然存在（返回 ICudaEngine*）
        mEngine.reset(mRuntime->deserializeCudaEngine(engineData.data(), engineData.size()));
        if (!mEngine) {
            throw std::runtime_error("deserializeCudaEngine failed");
        }

        mContext.reset(mEngine->createExecutionContext());
        if (!mContext) {
            throw std::runtime_error("createExecutionContext failed");
        }

        CHECK_CUDA(cudaStreamCreate(&mStream));

        // TRT10：用 IO tensor API 找输入输出名/shape/type :contentReference[oaicite:6]{index=6}
        const int32_t nIO = mEngine->getNbIOTensors();
        if (nIO < 2) {
            throw std::runtime_error("Engine IO tensors < 2");
        }

        for (int32_t i = 0; i < nIO; ++i) {
            const char* name = mEngine->getIOTensorName(i);
            const nvinfer1::TensorIOMode mode = mEngine->getTensorIOMode(name);
            const nvinfer1::Dims shape = mEngine->getTensorShape(name);
            const nvinfer1::DataType dtype = mEngine->getTensorDataType(name);

            std::cout << "[IO] " << name << " mode=" << (mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT")
                      << " dtype=" << static_cast<int>(dtype) << " shape=[";
            for (int d = 0; d < shape.nbDims; ++d) {
                std::cout << shape.d[d] << (d + 1 == shape.nbDims ? "" : ",");
            }
            std::cout << "]" << std::endl;

            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                mInputName = name;
                mInputType = dtype;
                mInputDims = shape;
            } else {
                mOutputName = name;
                mOutputType = dtype;
                mOutputDims = shape;
            }
        }

        // 如果是动态 shape，需要先 setInputShape，输出 shape 才稳定（官方文档也这么要求）:contentReference[oaicite:7]{index=7}
        if (hasDynamicDim(mInputDims)) {
            // 这里按你原代码固定 1x3x640x640
            nvinfer1::Dims4 fixed{1, 3, 640, 640};
            if (!mContext->setInputShape(mInputName.c_str(), fixed)) {
                throw std::runtime_error("setInputShape failed for input: " + mInputName);
            }
            mInputDims = fixed;
            // 重新从 context 取一次输出 shape（更可靠）
            mOutputDims = mContext->getTensorShape(mOutputName.c_str());
        }

        // 绑定地址（一次设置，后续反复 infer 就不用再设）
        // NVIDIA 官方迁移示例：遍历所有 IO tensor，context->setTensorAddress(name, ptr) :contentReference[oaicite:8]{index=8}
        // 这里先占位，真正的 ptr 会在 allocBuffers() 里分配后再 set
    }

    void allocBuffers() {
        // 计算元素数量
        mInputElems = volumeDims(mInputDims);
        if (mInputElems <= 0) {
            throw std::runtime_error("Invalid input dims volume");
        }

        // 输出 dims 可能是 [1,84,8400] 或 [84,8400]（取决于导出/插件）
        mOutputElems = volumeDims(mOutputDims);
        if (mOutputElems <= 0) {
            throw std::runtime_error("Invalid output dims volume");
        }

        const auto typeSize = [](nvinfer1::DataType t) -> size_t {
            switch (t) {
                case nvinfer1::DataType::kFLOAT: return 4;
                case nvinfer1::DataType::kHALF:  return 2;
                default: return 0;
            }
        };

        const size_t inTs = typeSize(mInputType);
        const size_t outTs = typeSize(mOutputType);
        if (inTs == 0 || outTs == 0) {
            throw std::runtime_error("Only FP32/FP16 buffers are supported in this sample");
        }

        mInputBytes = static_cast<size_t>(mInputElems) * inTs;
        mOutputBytes = static_cast<size_t>(mOutputElems) * outTs;

        CHECK_CUDA(cudaMalloc(&mDeviceInput, mInputBytes));
        CHECK_CUDA(cudaMalloc(&mDeviceOutput, mOutputBytes));

        // TRT10：分配完 device buffer 后，设置 tensor 地址
        const int32_t nIO = mEngine->getNbIOTensors();
        for (int32_t i = 0; i < nIO; ++i) {
            const char* name = mEngine->getIOTensorName(i);
            const nvinfer1::TensorIOMode mode = mEngine->getTensorIOMode(name);
            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                if (!mContext->setTensorAddress(name, mDeviceInput)) {
                    throw std::runtime_error(std::string("setTensorAddress failed for input: ") + name);
                }
            } else {
                if (!mContext->setTensorAddress(name, mDeviceOutput)) {
                    throw std::runtime_error(std::string("setTensorAddress failed for output: ") + name);
                }
            }
        }
    }

private:
    std::string mEnginePath;

    TrtUniquePtr<nvinfer1::IRuntime> mRuntime;
    TrtUniquePtr<nvinfer1::ICudaEngine> mEngine;
    TrtUniquePtr<nvinfer1::IExecutionContext> mContext;

    cudaStream_t mStream{nullptr};

    // TRT8/9
    int mInputIndex{0};
    int mOutputIndex{1};

    // TRT10
    std::string mInputName;
    std::string mOutputName;

    nvinfer1::Dims mInputDims{};
    nvinfer1::Dims mOutputDims{};
    nvinfer1::DataType mInputType{nvinfer1::DataType::kFLOAT};
    nvinfer1::DataType mOutputType{nvinfer1::DataType::kFLOAT};

    std::int64_t mInputElems{0};
    std::int64_t mOutputElems{0};
    size_t mInputBytes{0};
    size_t mOutputBytes{0};

    void* mDeviceInput{nullptr};
    void* mDeviceOutput{nullptr};

    // host 临时缓冲
    std::vector<std::uint16_t> mHostInputHalf;
    std::vector<std::uint16_t> mHostOutputHalf;
    std::vector<float> mHostOutputFloat;
};

// ======================= main：保持你原来逻辑，替换推理实现 =======================
int main() {
    try {
        // 1) 读图（与你原代码一致）:contentReference[oaicite:9]{index=9}
        cv::Mat imgSrc = cv::imread("images/times.jpg");
        if (imgSrc.empty()) {
            std::cerr << "Failed to load image" << std::endl;
            return -1;
        }

        // 2) 预处理：resize 640、BGR->RGB、float、/255、CHW（与你原代码一致）:contentReference[oaicite:10]{index=10}
        cv::Mat imgResized640;
        cv::resize(imgSrc, imgResized640, cv::Size(640, 640));

        cv::Mat imgRGB;
        cv::cvtColor(imgResized640, imgRGB, cv::COLOR_BGR2RGB);
        imgRGB.convertTo(imgRGB, CV_32F, 1.0 / 255.0);

        std::vector<cv::Mat> vecRGBChannelData(3);
        cv::split(imgRGB, vecRGBChannelData);

        std::vector<float> inputCHW;
        inputCHW.reserve(1 * 3 * 640 * 640);
        for (int c = 0; c < 3; ++c) {
            inputCHW.insert(
                inputCHW.end(),
                (float*)(vecRGBChannelData[c].datastart),
                (float*)(vecRGBChannelData[c].dataend)
            );
        }

        // 3) TensorRT：加载 plan 并推理（替换掉你原来的 Ort::Session + Run）:contentReference[oaicite:11]{index=11}
        YoloTrtRunner runner("./models/yolov8x_fp16.plan");
        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<float> out = runner.infer(inputCHW);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::cout << "Time:\t" << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) << std::endl;

        // 4) 后处理：尽量保持你原来的写法（按 (4+80, 8400) 取值）:contentReference[oaicite:12]{index=12}
        const nvinfer1::Dims outDims = runner.getOutputDims();

        // 兼容 outDims = [1,84,8400] 或 [84,8400]
        std::int64_t boxes = 0;
        std::int64_t channels = 0;
        if (outDims.nbDims == 3) {
            channels = static_cast<std::int64_t>(outDims.d[1]);
            boxes = static_cast<std::int64_t>(outDims.d[2]);
        } else if (outDims.nbDims == 2) {
            channels = static_cast<std::int64_t>(outDims.d[0]);
            boxes = static_cast<std::int64_t>(outDims.d[1]);
        } else {
            throw std::runtime_error("Unexpected output dims nbDims=" + std::to_string(outDims.nbDims));
        }

        if (channels < 5) {
            throw std::runtime_error("Output channels too small");
        }

        const std::int64_t nClasses = channels - 4;
        float fThreshold = 0.5f;

        std::map<unsigned int, std::pair<unsigned short, Border>> mapBeastType;

        for (std::int64_t i = 0; i < boxes; ++i) {
            std::pair<int, float> pairBestType(0, -1.0f);
            std::array<float, 4> arrXYWH{};

            for (std::int64_t j = 0; j < nClasses; ++j) {
                const float fThis = out[static_cast<size_t>((4 + j) * boxes + i)];
                const float fPossibility = fThis; // 你原代码也是直接用 fThis :contentReference[oaicite:13]{index=13}

                if (fPossibility > pairBestType.second) {
                    pairBestType.second = fPossibility;
                    pairBestType.first = static_cast<int>(j);
                    arrXYWH[0] = out[static_cast<size_t>(0 * boxes + i)];
                    arrXYWH[1] = out[static_cast<size_t>(1 * boxes + i)];
                    arrXYWH[2] = out[static_cast<size_t>(2 * boxes + i)];
                    arrXYWH[3] = out[static_cast<size_t>(3 * boxes + i)];
                }
            }

            if (pairBestType.second > 0.0f && pairBestType.second > fThreshold) {
                const auto usX1 = static_cast<unsigned short>(arrXYWH[0] - (arrXYWH[2] / 2.0f));
                const auto usY1 = static_cast<unsigned short>(arrXYWH[1] - (arrXYWH[3] / 2.0f));
                const auto usX2 = static_cast<unsigned short>(arrXYWH[0] + (arrXYWH[2] / 2.0f));
                const auto usY2 = static_cast<unsigned short>(arrXYWH[1] + (arrXYWH[3] / 2.0f));

                mapBeastType[static_cast<unsigned int>(i)].first = static_cast<unsigned short>(pairBestType.first);
                mapBeastType[static_cast<unsigned int>(i)].second.fPossibility = pairBestType.second;
                mapBeastType[static_cast<unsigned int>(i)].second.usX1 = usX1;
                mapBeastType[static_cast<unsigned int>(i)].second.usY1 = usY1;
                mapBeastType[static_cast<unsigned int>(i)].second.usX2 = usX2;
                mapBeastType[static_cast<unsigned int>(i)].second.usY2 = usY2;
            }
        }

        // 5) 画框（与你原代码一致）:contentReference[oaicite:14]{index=14}
        for (auto& elem : mapBeastType) {
            cv::rectangle(
                imgResized640,
                cv::Point(elem.second.second.usX1, elem.second.second.usY1),
                cv::Point(elem.second.second.usX2, elem.second.second.usY2),
                cv::Scalar(0, 255, 0),
                1
            );
            std::cout << elem.first << "\t" << elem.second.first << "\t"
                      << elem.second.second.fPossibility << std::endl;
        }

        cv::imshow("box", imgResized640);
        cv::waitKey(0);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }
}
