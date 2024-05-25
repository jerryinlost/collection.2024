#pragma once
/*
// g++ -o yolo_detect yolo_detect.cpp -I<path_to_opencv_include> -L<path_to_opencv_lib> -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_dnn
//g++ -o yolo_detect yolo_detect.cpp -I"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\include"  -ID:\\opencv\\opencv\\build\\include -LD:\\opencv\\opencv\\build\\x64\\vc16\\lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_dnn
//g++ -std=c++11 -o yolo_detect yolo_detect.cpp -I"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\include" -I"D:\opencv\opencv\build\include" -D"F:\!sent\Programs\opencv\build\lib\Release"  -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_dnn

// -I"C:\cygnus\cygwin-b20\include\g++"
// -I"C:\\Program Files (x86)\\Microsoft Visual Studio\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include"
// -I"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\include"
// -I"C:\\w64devkit\\lib\\gcc\\x86_64-w64-mingw32\\14.1.0\\include\\c++"
// -IC:\w64devkit\lib\gcc\x86_64-w64-mingw32\14.1.0\include\c++

cl /EHsc yolo_detect.cpp /I"D:\opencv\opencv\build\include" /I"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\include" /link /LIBPATH:"F:\!sent\Programs\opencv\build\lib\Release" opencv_core490.lib opencv_imgcodecs490.lib opencv_highgui490.lib opencv_dnn490.lib
'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64\'cl.exe /EHsc yolo_detect.cpp /I"D:\opencv\opencv\build\include" /I"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\include" /link /LIBPATH:"F:\!sent\Programs\opencv\build\lib\Release" opencv_core490.lib opencv_imgcodecs490.lib opencv_highgui490.lib opencv_dnn490.lib
ols\\MSVC\\14.37.32822\\include" /link /LIBPATH:"F:\\!sent\\Programs\\opencv\\build\\lib\\Release" opencv_core490.lib opencv_imgcodecs490.lib opencv_highgui490.lib opencv_dnn490.lib
*/
// failed to build using command line on windows
// failed to build on linux
// succeed to build on windows after building opencv dnn with protobuf using cmake
// but failed to execute with the following error
/* 
[ERROR:0@0.093] global onnx_importer.cpp:1034 cv::dnn::dnn4_v20231225::ONNXImporter::handleNode DNN/ONNX: ERROR during processing node with 1 inputs and 3 outputs: [Split]:(onnx_node!/model.22/Split) from domain='ai.onnx'
OpenCV: terminate handler is called! The last OpenCV error is:
OpenCV(4.9.0) Error: Unspecified error (> Node [Split@ai.onnx]:(onnx_node!/model.22/Split) parse error: OpenCV(4.9.0) F:\!sent\Programs\opencv\opencv-4.9.0\modules\dnn\src\layers\slice_layer.cpp:243: error: (-215:Assertion failed) inputs.size() == 1 in function 'cv::dnn::SliceLayerImpl::getMemoryShapes'
> ) in cv::dnn::dnn4_v20231225::ONNXImporter::handleNode, file F:\!sent\Programs\opencv\opencv-4.9.0\modules\dnn\src\onnx\onnx_importer.cpp, line 1053
*/
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

std::vector<std::string> loadClassNames(const std::string& classFile) {
    std::vector<std::string> classNames;
    std::ifstream ifs(classFile.c_str());
    std::string line;
    while (getline(ifs, line)) {
        classNames.push_back(line);
    }
    return classNames;
}

void detectAndSave(const std::string& imageFile, const std::string& modelPath, const std::string& classFile) {
    std::vector<std::string> classNames = loadClassNames(classFile);

    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    cv::Mat image = cv::imread(imageFile);
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1/255.0, cv::Size(640, 640), cv::Scalar(0,0,0), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());

    float confThreshold = 0.5;
    float nmsThreshold = 0.4;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (const auto& out : outs) {
        float* data = (float*)out.data;
        for (int i = 0; i < out.rows; ++i, data += out.cols) {
            cv::Mat scores = out.row(i).colRange(5, out.cols);
            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold) {
                int centerX = (int)(data[0] * image.cols);
                int centerY = (int)(data[1] * image.rows);
                int width = (int)(data[2] * image.cols);
                int height = (int)(data[3] * image.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    std::string outputDir = "output";
    std::string command = "mkdir " + outputDir;
    system(command.c_str());

    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        cv::Mat croppedImage = image(box);

        std::string label = classNames[classIds[idx]];
        std::ostringstream filename;
        filename << outputDir << "/" << label << "_" << i << ".png";
        cv::imwrite(filename.str(), croppedImage);
    }
}

int main() {
    std::string imageFile = "screenshot.png";
    std::string modelPath = "yolov8n_blood_cell_detection.onnx";
    std::string classFile = "coco.names";

    detectAndSave(imageFile, modelPath, classFile);

    std::cout << "Processing complete. Check the 'output' directory for results." << std::endl;
    return 0;
}