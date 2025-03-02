#include "detector.h"
#include "tracker.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

vector<string> loadClassNames(const string& filename) {
    vector<string> classNames;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open class names file: " << filename << endl;
        return classNames;
    }
    
    string line;
    while (getline(file, line)) {
        if (!line.empty()) classNames.push_back(line);
    }
    file.close();
#ifdef DEBUG
    cout << "Loaded " << classNames.size() << " class names from " << filename << endl;
#endif
    return classNames;
}

int main(int argc, char** argv) {
    string classFile = "../models/coco.names";
    string modelPath = "../models";
   
#ifdef DEBUG
    cout << "Starting object detection program..." << endl;
    cout << "Command line arguments: " << argc << endl;
    for (int i = 0; i < argc; i++) {
        cout << "argv[" << i << "]: " << argv[i] << endl;
    }
#endif

    vector<string> allClassNames = loadClassNames(classFile);
    if (allClassNames.empty()) {
        cerr << "Failed to load class names. Exiting." << endl;
        return -1;
    }
    
    vector<string> targetClasses = {"person", "dog"};
#ifdef DEBUG
    cout << "Target classes: ";
    for (const auto& cls : targetClasses) cout << cls << " ";
    cout << endl;
#endif
    
    unique_ptr<Detector> detector(createDetector(modelPath, targetClasses));
    if (!detector) {
        cerr << "Error: Failed to initialize detector." << endl;
        return -1;
    }
#ifdef DEBUG
    cout << "Detector initialized successfully." << endl;
#endif
    
    Mat frame;
    bool useImageMode = (argc > 1 && (string(argv[1]) == "--image" || string(argv[1]) == "-image"));
    if (useImageMode) {
        cout << "Running in image mode..." << endl;
        frame = imread("../sample.jpg");
        if (frame.empty()) {
            cerr << "Error: Could not load sample.jpg." << endl;
            return -1;
        }
#ifdef DEBUG
        cout << "Image loaded. Resolution: " << frame.cols << "x" << frame.rows << endl;
#endif
        
        detector->detect(frame);
        imshow("Object Detection", frame);
        waitKey(0);
        destroyAllWindows();
        return 0;
    } else {
        cout << "Running in camera mode..." << endl;
        VideoCapture cap(0);
        if (!cap.isOpened()) {
            cerr << "Error: Could not open video capture device 0." << endl;
            cerr << "Check permissions (e.g., 'sudo chmod 666 /dev/video0')." << endl;
            return -1;
        }
        
        cap >> frame;
        if (frame.empty()) {
            cerr << "Error: Initial frame capture failed." << endl;
            cap.release();
            return -1;
        }
        
        cout << "Camera opened successfully. Resolution: " 
             << frame.cols << "x" << frame.rows << endl;

        const int fpsBufferSize = 16;
        float fpsBuffer[fpsBufferSize] = {0.0};
        int frameCount = 0;
        chrono::steady_clock::time_point startTime;
        
        while (true) {
            startTime = chrono::steady_clock::now();
            
            cap >> frame;
            if (frame.empty()) {
                cerr << "Error: Frame capture failed during loop." << endl;
                break;
            }
#ifdef DEBUG
            cout << "Frame captured: " << frame.cols << "x" << frame.rows << endl;
#endif
            
            detector->detect(frame);
            
            auto endTime = chrono::steady_clock::now();
            float frameTimeMs = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
            float fps = (frameTimeMs > 0) ? 1000.0f / frameTimeMs : 0.0f;
            
            fpsBuffer[frameCount % fpsBufferSize] = fps;
            frameCount++;
            
            float avgFps = 0.0;
            for (int i = 0; i < min(frameCount, fpsBufferSize); i++) {
                avgFps += fpsBuffer[i];
            }
            avgFps /= min(frameCount, fpsBufferSize);
#ifdef DEBUG
            cout << "FPS: " << fps << ", Avg FPS: " << avgFps << endl;
#endif
            
            string fpsText = format("FPS: %.2f", avgFps);
            putText(frame, fpsText, Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
            
            imshow("Object Detection", frame);
            if (waitKey(1) == 'q') break;
        }
        
        cap.release();
        destroyAllWindows();
        return 0;
    }
}