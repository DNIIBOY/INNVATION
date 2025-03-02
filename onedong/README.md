# Object Detection on Orange Pi 5 and Jetson Nano

This project implements object detection using YOLO models on Orange Pi 5 (RK3588 NPU) and NVIDIA Jetson Nano (CUDA), with a portable codebase.

## Prerequisites
- **Orange Pi 5**: Install RKNN Toolkit and place `yolov5s-640-640.rknn` in `models/`.
- **Jetson Nano**: Install OpenCV with CUDA support and place `yolov7-tiny.cfg` and `yolov7-tiny.weights` in `models/`.
- **x86_64**: Install OpenCV (`sudo apt-get install libopencv-dev`).
- All platforms: Provide `coco.names` in `models/` (download from YOLO repository).

### models
**Cooc.Names**
```sh
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```
**RKNN models(NPU)**
pretrained models can be found at https://github.com/airockchip/rknn_model_zoo
```sh
wget https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov7/yolov7-tiny.onnx
```
**Nvidia jetson models(CUDA)**
```sh
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov7-tiny.cfg
```
## Build
```bash
chmod +x build-linux.sh
./build-linux.sh