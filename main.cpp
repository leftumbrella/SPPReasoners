#pragma warning( disable : 4819 )
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>

int main() {
    cv::Mat img = cv::imread("images/bus.jpg");
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

    std::cout << "Preprocess done.\n";
    std::cout << "Input tensor size: " << input_tensor.size() << std::endl;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    Ort::Session session(env, L"./models/yolov8n.onnx", session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
    const char* input_name = input_name_alloc.get();
    auto input_shape = session.GetInputTypeInfo(0)
                           .GetTensorTypeAndShapeInfo()
                           .GetShape();

    std::cout << "Input name: " << input_name << "\n";
    std::cout << "Input shape: ";
    for (auto s : input_shape) std::cout << s << " ";
    std::cout << "\n";

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

    std::cout << "Output shape: ";
    for (auto s : out_shape) std::cout << s << " ";
    std::cout << "\n";

    return 0;
}
