# YOLOv9 视频人脸打码工具

基于 [YOLOv9](https://arxiv.org/abs/2402.13616) 的自动视频人脸打码工具，支持像素化和图片覆盖两种打码方式。

## 功能特点

- 🎯 准确的人脸检测 - 使用YOLOv9模型进行实时人脸检测
- 🎨 双重打码模式:
  - 像素化模式 - 自动将检测到的人脸进行像素化处理
  - 图片覆盖模式 - 用自定义图片覆盖检测到的人脸
- 🚀 多线程处理 - 支持多线程并行处理以提升性能
- 🔊 音频保留 - 自动保留原视频的音轨
- 🎮 GPU加速 - 支持CUDA加速(如果可用)

## 安装

1. 克隆仓库:
```bash
git clone https://github.com/wangyuanchuan2022/video_masking.git
cd yolov9-face-blur
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

## 使用方法

基本用法:

```bash
python detect.py -i input.mp4 -o output.mp4 [options]
```

### 参数说明

- `-i, --input`: 输入视频路径
- `-o, --output`: 输出视频路径
- `-m, --mode`: 打码模式
  - `0`: 像素化模式(默认)
  - `path/to/image.png`: 图片覆盖模式，使用指定图片
- `-t, --threads`: 处理线程数(默认24)

### 使用示例

1. 像素化打码:
```bash
python detect.py -i input.mp4 -o output.mp4 -m 0
```

2. 使用图片覆盖:
```bash
python detect.py -i input.mp4 -o output.mp4 -m mask.png
```

3. 指定线程数:
```bash
python detect.py -i input.mp4 -o output.mp4 -m 0 -t 16
```

## 性能优化

- 使用GPU加速(如果可用)
- 多线程并行处理
- 队列缓冲机制
- 保持视频帧顺序输出
- 自动音频处理

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🔗 Reference

- [YOLOv9](https://github.com/WongKinYiu/yolov9) - YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information
- [YOLOv9-Face-Detection](https://github.com/spacewalk01/yolov9-face-detection) - Training YOLOv9 for face detection on the WIDER Face dataset
