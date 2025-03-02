#include "detector.h"
#include "postprocess.h"
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace cv::dnn;
using namespace std;

class CPUDetector : public GenericDetector {
private:
    Net net;

public:
    CPUDetector(const string& modelPath, const vector<string>& targetClasses_)
        : GenericDetector(modelPath, targetClasses_) {
        initialize(modelPath);
    }

    void detect(Mat& frame) override {
        if (!initialized) {
            cerr << "Error: Detector not properly initialized" << endl;
            return;
        }
#ifdef DEBUG
        cout << "Detecting on frame: " << frame.cols << "x" << frame.rows << endl;
#endif

        Mat img;
        cvtColor(frame, img, COLOR_BGR2RGB);
        int img_width = img.cols;
        int img_height = img.rows;
#ifdef DEBUG
        cout << "Converted to RGB: " << img_width << "x" << img_height << endl;
#endif

        Mat resized_img(height, width, CV_8UC3, Scalar(114, 114, 114));
        float scale = min(static_cast<float>(width) / img_width, static_cast<float>(height) / img_height);
        int new_width = static_cast<int>(img_width * scale);
        int new_height = static_cast<int>(img_height * scale);
        int dx = (width - new_width) / 2;
        int dy = (height - new_height) / 2;
#ifdef DEBUG
        cout << "Resizing: scale=" << scale << ", new_size=" << new_width << "x" << new_height 
             << ", offsets=" << dx << "," << dy << endl;
#endif

        Mat resized_part;
        resize(img, resized_part, Size(new_width, new_height));
        resized_part.copyTo(resized_img(Rect(dx, dy, new_width, new_height)));
#ifdef DEBUG
        cout << "Image resized and letterboxed: " << resized_img.cols << "x" << resized_img.rows << endl;
#endif

        DetectionOutput output = runInference(resized_img);
#ifdef DEBUG
        cout << "Inference completed. Outputs: " << output.num_outputs << endl;
        for (int i = 0; i < output.num_outputs; i++) {
            if (!output.buffers[i]) {
                cerr << "Error: Output buffer " << i << " is null!" << endl;
            }
        }
#endif

        detect_result_group_t detect_result_group;
        // Explicitly call the float version of post_process
        post_process(
            static_cast<float*>(output.buffers[0]),
            static_cast<float*>(output.buffers[1]),
            static_cast<float*>(output.buffers[2]),
            height, width,
            BOX_THRESH, NMS_THRESH,
            scale, scale,
            output.zps, output.scales,
            &detect_result_group,
            false  // is_quantized = false for CPU
        );

#ifdef DEBUG
        cout << "Post-processing done. Found " << detect_result_group.count << " detections" << endl;
        for (int i = 0; i < detect_result_group.count; i++) {
            detect_result_t* det = &detect_result_group.results[i];
            cout << "Detection " << i << ": " << det->name << " (" 
                 << det->box.left << "," << det->box.top << ")-(" 
                 << det->box.right << "," << det->box.bottom << "), conf=" 
                 << det->prop << endl;
        }
#endif

        for (int i = 0; i < detect_result_group.count; i++) {
            detect_result_t* det_result = &detect_result_group.results[i];

            if (!targetClasses.empty()) {
                bool isTarget = false;
                for (const auto& target : targetClasses) {
                    if (strcmp(det_result->name, target.c_str()) == 0) {
                        isTarget = true;
                        break;
                    }
                }
                if (!isTarget) continue;
            }

            int x1 = static_cast<int>((det_result->box.left - dx) / scale);
            int y1 = static_cast<int>((det_result->box.top - dy) / scale);
            int x2 = static_cast<int>((det_result->box.right - dx) / scale);
            int y2 = static_cast<int>((det_result->box.bottom - dy) / scale);

            x1 = max(0, min(x1, img_width - 1));
            y1 = max(0, min(y1, img_height - 1));
            x2 = max(0, min(x2, img_width - 1));
            y2 = max(0, min(y2, img_height - 1));
#ifdef DEBUG
            cout << "Drawing box: (" << x1 << "," << y1 << ")-(" << x2 << "," << y2 << ")" << endl;
#endif

            rectangle(frame, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2);

            char text[256];
            sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
            int baseLine = 0;
            Size label_size = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            int x = x1;
            int y = y1 - label_size.height - baseLine;
            if (y < 0) y = y1 + label_size.height;
            if (x + label_size.width > frame.cols) x = frame.cols - label_size.width;

            rectangle(frame, Rect(Point(x, y), Size(label_size.width, label_size.height + baseLine)),
                      Scalar(255, 255, 255), -1);
            putText(frame, text, Point(x, y + label_size.height), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
        }
#ifdef DEBUG
        cout << "Frame processing completed." << endl;
#endif
    }

protected:
    void initialize(const string& modelPath) override {
#ifdef DEBUG
        cout << "Initializing CPUDetector with modelPath: " << modelPath << endl;
#endif
        string cfg = modelPath + "/yolov7-tiny.cfg";
        string weights = modelPath + "/yolov7-tiny.weights";
#ifdef DEBUG
        cout << "Loading model: cfg=" << cfg << ", weights=" << weights << endl;
#endif
        net = readNet(cfg, weights);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
#ifdef DEBUG
        cout << "Using OpenCV backend and CPU target." << endl;
#endif
        if (net.empty()) {
#ifdef DEBUG
            cerr << "Error: Neural network is empty after loading." << endl;
#endif
            throw runtime_error("Failed to load YOLOv7-tiny model for CPU");
        }
#ifdef DEBUG
        cout << "Model loaded successfully." << endl;
#endif

        width = 416;  // Hardcoded for simplicity, adjust as needed
        height = 416;
        channel = 3;
        initialized = true;
#ifdef DEBUG
        cout << "CPUDetector initialized. Dimensions: " << width << "x" << height << "x" << channel << endl;
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
            output.buffers[i] = outs[i].data;  // Float output assumed
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

    ~CPUDetector() override {
#ifdef DEBUG
        cout << "Destroying CPUDetector..." << endl;
#endif
    }
};

Detector* createDetector(const string& modelPath, const vector<string>& targetClasses) {
    try {
#ifdef DEBUG
        cout << "Creating CPUDetector..." << endl;
#endif
        return new CPUDetector(modelPath, targetClasses);
    } catch (const exception& e) {
        cerr << "Error creating CPU detector: " << e.what() << endl;
        return nullptr;
    }
}