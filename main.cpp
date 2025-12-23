#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <vector>
#include <chrono>
#include <iostream>
struct Border {
    float fPossibility;
    unsigned short usX1;
    unsigned short usY1;
    unsigned short usX2;
    unsigned short usY2;
};

void print_model_inputs(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;

    size_t input_count = session.GetInputCount();
    std::cout << "Input count: " << input_count << std::endl;

    for (size_t i = 0; i < input_count; ++i) {
        Ort::AllocatedStringPtr input_name =
            session.GetInputNameAllocated(i, allocator);

        std::cout << "Input " << i
                  << " name: " << input_name.get() << std::endl;

        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        auto shape = tensor_info.GetShape();
        std::cout << "  Shape: ";
        for (auto d : shape) std::cout << d << " ";
        std::cout << std::endl;
    }
}

int main() {
    cv::Mat imgSrc = cv::imread("images/times.jpg");
    if (imgSrc.empty()) {
        return -1;
    }

    cv::Mat imgResized640;
    cv::resize(imgSrc, imgResized640, cv::Size(640, 640));

    cv::Mat imgRGB;
    cv::cvtColor(imgResized640, imgRGB, cv::COLOR_BGR2RGB);
    imgRGB.convertTo(imgRGB, CV_32F, 1.0/255.0);

    std::vector<cv::Mat> vecRGBChannelData(3);
    cv::split(imgRGB, vecRGBChannelData);

    std::vector<float> input_tensor;
    input_tensor.reserve(1 * 3 * 640 * 640);
    for (int c = 0; c < 3; ++c) {
        input_tensor.insert(
            input_tensor.end(),
            (float*)(vecRGBChannelData[c]).datastart,
            (float*)(vecRGBChannelData[c]).dataend
        );
    }

    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "onnx");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    Ort::Session session(env, "./models/yolov8x.onnx", session_options);
    size_t input_count = session.GetInputCount();
    std::cout << "Input count: " << input_count << std::endl;

    print_model_inputs(session);

    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
    const char* input_name = input_name_alloc.get();
    auto input_shape = session.GetInputTypeInfo(0)
                           .GetTensorTypeAndShapeInfo()
                           .GetShape();

    std::vector<int64_t> input_dims = {1, 3, 640, 640};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
          OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
        mem_info,
        input_tensor.data(),
        input_tensor.size(),
        input_dims.data(),
        input_dims.size()
    );

    const char* output_names[] = {"output0"};

    auto start_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        &input_name,
        &input_tensor_ort,
        1,
        output_names,
        1
    );
    auto stop_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
    std::cout << "Time:" << stop_time - start_time << std::endl;

    auto out_shape = output_tensors[0]
                         .GetTensorTypeAndShapeInfo()
                         .GetShape();

    float* out = output_tensors[0].GetTensorMutableData<float>();
    float fThreshold = 0.5;

    std::map<unsigned int, std::pair<unsigned short,Border>> mapBeastType;
    for (auto i=0;i<8400;++i) {
        // 找出80类中，本检测单元概率最大的类别
        std::pair<int,float> pairBestType(0,-1);
        std::array<float,4> arrXYWH{};
        for (auto j=0;j<80;++j) {
            float fThis = out[(4 + j) * 8400 + i];
            float fPossibility = fThis;//1.0f / (1.0f + std::exp(-fThis));
            if (fPossibility>pairBestType.second) {
                pairBestType.second = fPossibility;
                pairBestType.first = j;
                arrXYWH[0] = out[0 * 8400 + i];
                arrXYWH[1] = out[1 * 8400 + i];
                arrXYWH[2] = out[2 * 8400 + i];
                arrXYWH[3] = out[3 * 8400 + i];
            }
        }
        if (pairBestType.second > 0 && pairBestType.second>fThreshold) {
            auto usX1 = static_cast<unsigned short>(arrXYWH[0] - (arrXYWH[2] / 2.0));
            auto usY1 = static_cast<unsigned short>(arrXYWH[1] - (arrXYWH[3] / 2.0));
            auto usX2 = static_cast<unsigned short>(arrXYWH[0] + (arrXYWH[2] / 2.0));
            auto usY2 = static_cast<unsigned short>(arrXYWH[1] + (arrXYWH[3] / 2.0));
            mapBeastType[i].first = pairBestType.first;
            mapBeastType[i].second.fPossibility = pairBestType.second;
            mapBeastType[i].second.usX1 = usX1;
            mapBeastType[i].second.usY1 = usY1;
            mapBeastType[i].second.usX2 = usX2;
            mapBeastType[i].second.usY2 = usY2;
        }
    }

    for (auto& elem:mapBeastType) {
        cv::rectangle(imgResized640,cv::Point(elem.second.second.usX1,elem.second.second.usY1),cv::Point(elem.second.second.usX2,elem.second.second.usY2), cv::Scalar(0, 255, 0),1);
        std::cout << elem.first << "\t" << elem.second.first<< "\t" << elem.second.second.fPossibility << std::endl;
    }
    cv::imshow("box", imgResized640);
    cv::waitKey(0);
    return 0;
}
