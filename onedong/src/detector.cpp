#include "detector.h"
#include "postprocess.h"
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;

GenericDetector::GenericDetector(const string& modelPath, const vector<string>& targetClasses_) {
    targetClasses = targetClasses_;
    initialized = false;
#ifdef DEBUG
    cout << "GenericDetector constructed with modelPath: " << modelPath << endl;
#endif
}

void GenericDetector::detect(Mat& frame) {
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
    post_process(
        static_cast<int8_t*>(output.buffers[0]),
        static_cast<int8_t*>(output.buffers[1]),
        static_cast<int8_t*>(output.buffers[2]),
        height, width,
        BOX_THRESH, NMS_THRESH,
        scale, scale,
        output.zps, output.scales,
        &detect_result_group
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

    releaseOutputs(output);  // Call platform-specific cleanup
}