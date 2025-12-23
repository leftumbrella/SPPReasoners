#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>
struct Border {
    float fPossibility;
    unsigned short usX1;
    unsigned short usY1;
    unsigned short usX2;
    unsigned short usY2;
};
int main() {
    cv::Mat img = cv::imread("images/times.jpg");
    if (img.empty()) {
        std::cerr << "Failed to load image\n";
        return -1;
    }

    cv::Mat img_resized;
    cv::resize(img, img_resized, cv::Size(640, 640));

    cv::Mat img_rgb;
    cv::cvtColor(img_resized, img_rgb, cv::COLOR_BGR2RGB);
    img_rgb.convertTo(img_rgb, CV_32F, 1.0/255.0);

    std::vector<cv::Mat> chw(3);
    cv::split(img_rgb, chw);

    std::vector<float> input_tensor;
    input_tensor.reserve(1 * 3 * 640 * 640);
    for (int c = 0; c < 3; ++c) {
        input_tensor.insert(
            input_tensor.end(),
            (float*)(chw[c]).datastart,
            (float*)(chw[c]).dataend
        );
    }

    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "onnx");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    Ort::Session session(env, "./models/yolov8n.onnx", session_options);

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

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        &input_name,
        &input_tensor_ort,
        1,
        output_names,
        1
    );

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
        cv::rectangle(img_resized,cv::Point(elem.second.second.usX1,elem.second.second.usY1),cv::Point(elem.second.second.usX2,elem.second.second.usY2), cv::Scalar(0, 255, 0),3);
        std::cout << elem.first << "\t" << elem.second.first<< "\t" << elem.second.second.fPossibility << std::endl;
    }
    cv::imshow("box", img_resized);
    cv::waitKey(0);
    return 0;
}
