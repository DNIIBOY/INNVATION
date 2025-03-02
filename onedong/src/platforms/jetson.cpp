#include "detector.h"
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace cv::dnn;
using namespace std;

class JetsonDetector : public GenericDetector {
private:
    Net net;

public:
    JetsonDetector(const string& modelPath, const vector<string>& targetClasses_)
        : GenericDetector(modelPath, targetClasses_) {
        initialize(modelPath);
    }

protected:
    void initialize(const string& modelPath) override {
#ifdef DEBUG
        cout << "Initializing JetsonDetector with modelPath: " << modelPath << endl;
#endif
        string cfg = modelPath + "/yolov7-tiny.cfg";
        string weights = modelPath + "/yolov7-tiny.weights";
#ifdef DEBUG
        cout << "Loading model: cfg=" << cfg << ", weights=" << weights << endl;
#endif
        net = readNet(cfg, weights);
#ifdef USE_CUDA
        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA);
#ifdef DEBUG
        cout << "Using CUDA backend and target for Jetson." << endl;
#endif
#else
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
#ifdef DEBUG
        cout << "Using OpenCV backend and CPU target." << endl;
#endif
#endif
        if (net.empty()) {
#ifdef DEBUG
            cerr << "Error: Neural network is empty after loading." << endl;
#endif
            throw runtime_error("Failed to load YOLOv7-tiny model");
        }
#ifdef DEBUG
        cout << "Model loaded successfully." << endl;
#endif

        width = 416;  // Hardcoded for simplicity, adjust as needed
        height = 416;
        channel = 3;
        initialized = true;
#ifdef DEBUG
        cout << "JetsonDetector initialized. Dimensions: " << width << "x" << height << "x" << channel << endl;
#endif
    }

    DetectionOutput runInference(const Mat& input) override {
#ifdef DEBUG
        cout << "Running inference on input: " << input.cols << "x" << input.rows << endl;
#endif
        Mat blob = blobFromImage(input, 1.0 / 255.0, Size(width, height), Scalar(0, 0, 0), true, false);
#ifdef DEBUG
        cout << "Blob created: " << blob.cols << "x" << blob.rows << "x" << blob.channels() << endl;
#endif
        net.setInput(blob);
#ifdef DEBUG
        cout << "Input set to network." << endl;
#endif

        vector<Mat> outs;
        vector<String> outNames = net.getUnconnectedOutLayersNames();
#ifdef DEBUG
        cout << "Output layer names: ";
        for (const auto& name : outNames) cout << name << " ";
        cout << endl;
#endif
        net.forward(outs, outNames);
#ifdef DEBUG
        cout << "Inference completed. Number of outputs: " << outs.size() << endl;
        for (size_t i = 0; i < outs.size(); i++) {
            cout << "Output " << i << ": " << outs[i].cols << "x" << outs[i].rows << endl;
        }
#endif

        DetectionOutput output;
        output.buffers.resize(outs.size());
        output.num_outputs = outs.size();
        for (size_t i = 0; i < outs.size(); i++) {
            output.buffers[i] = outs[i].data;  // Note: This assumes float output
#ifdef DEBUG
            cout << "Output " << i << " buffer assigned, size=" << outs[i].total() * outs[i].elemSize() << " bytes" << endl;
#endif
        }
        output.scales = vector<float>(outs.size(), 1.0);  // No quantization
        output.zps = vector<int32_t>(outs.size(), 0);     // No quantization
#ifdef DEBUG
        cout << "DetectionOutput prepared: " << output.num_outputs << " outputs, scales and zps set." << endl;
#endif

        return output;
    }

    // Destructor (optional, added for completeness)
    ~JetsonDetector() override {
#ifdef DEBUG
        cout << "Destroying JetsonDetector..." << endl;
#endif
        // Net is automatically cleaned up by OpenCV's destructor
    }
};

#ifdef USE_CUDA
Detector* createDetector(const string& modelPath, const vector<string>& targetClasses) {
    try {
#ifdef DEBUG
        cout << "Creating JetsonDetector..." << endl;
#endif
        return new JetsonDetector(modelPath, targetClasses);
    } catch (const exception& e) {
        cerr << "Error creating Jetson detector: " << e.what() << endl;
        return nullptr;
    }
}
#else
Detector* createDetector(const string&, const vector<string>&) { return nullptr; }
#endif