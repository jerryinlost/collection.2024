// sample.cpp
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

int main() {
    cv::Mat img = cv::imread("test.jpg");
    if (img.empty()) {
        std::cerr << "Could not read the image" << std::endl;
        return 1;
    }

    cv::dnn::Net net = cv::dnn::readNetFromONNX("model.onnx");
    cv::Mat blob = cv::dnn::blobFromImage(img, 1.0, cv::Size(224, 224), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    cv::Mat prob = net.forward();
    
    std::cout << "Inference completed" << std::endl;
    return 0;
}