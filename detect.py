import argparse
import os
import platform
import sys
from pathlib import Path
import threading
from queue import Queue
import time
from typing import Union
import subprocess

import numpy as np
import torch
from tqdm import tqdm
from moviepy import VideoFileClip, AudioFileClip

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


def detect(im0: np.ndarray, model: DetectMultiBackend):
    res = []
    assert im0 is not None, f'Image Not Found'
    im = letterbox(im0)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    with Profile():
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
    pred = model(im)
    pred = non_max_suppression(pred)
    seen = 0

    for i, det in enumerate(pred):  # per image
        seen += 1

        gn = torch.tensor(im.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy()

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = xyxy
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                res.append((x1, y1, x2, y2))

    return res


def resize_image(image, width, height):
    return cv2.resize(image, (width, height))


def overlay_image(base_img, overlay_img, position, overlay_width=None, overlay_height=None):
    # Check if images are loaded
    if base_img is None:
        raise FileNotFoundError(f"Base image not found")
    if overlay_img is None:
        raise FileNotFoundError(f"Overlay image not found")

    # Resize overlay image if dimensions are provided
    if overlay_width and overlay_height:
        overlay_img = resize_image(overlay_img, overlay_width, overlay_height)

    # Get image dimensions
    base_h, base_w = base_img.shape[:2]
    overlay_h, overlay_w = overlay_img.shape[:2]

    # Get position
    x, y = position

    # 计算有效的覆盖区域（处理边界情况）
    x_start = max(0, x)
    y_start = max(0, y)
    x_end = min(base_w, x + overlay_w)
    y_end = min(base_h, y + overlay_h)

    # 计算overlay_img需要裁剪的区域
    overlay_x_start = max(0, -x)
    overlay_y_start = max(0, -y)
    overlay_x_end = overlay_w - max(0, (x + overlay_w) - base_w)
    overlay_y_end = overlay_h - max(0, (y + overlay_h) - base_h)

    # 确保有效区域
    if x_end <= x_start or y_end <= y_start:
        return base_img

    # 裁剪overlay_img到有效区域
    overlay_roi = overlay_img[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end]

    # 使用GPU如果可用
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        # Upload images to GPU
        base_img_gpu = cv2.cuda_GpuMat()
        overlay_roi_gpu = cv2.cuda_GpuMat()
        base_img_gpu.upload(base_img)
        overlay_roi_gpu.upload(overlay_roi)

        # Create GPU masks
        gray_overlay_gpu = cv2.cuda.cvtColor(overlay_roi_gpu, cv2.COLOR_BGR2GRAY)
        _, mask_gpu = cv2.cuda.threshold(gray_overlay_gpu, 10, 255, cv2.THRESH_BINARY)
        mask_inv_gpu = cv2.cuda.bitwise_not(mask_gpu)

        # Black-out the area of overlay in ROI on GPU
        roi_gpu = cv2.cuda_GpuMat(base_img_gpu, cv2.Rect(x_start, y_start, x_end-x_start, y_end-y_start))
        base_img_bg_gpu = cv2.cuda.bitwise_and(roi_gpu, roi_gpu, mask=mask_inv_gpu)

        # Take only region of overlay from overlay image on GPU
        overlay_img_fg_gpu = cv2.cuda.bitwise_and(overlay_roi_gpu, overlay_roi_gpu, mask=mask_gpu)

        # Add the two images on GPU
        combined_gpu = cv2.cuda.add(base_img_bg_gpu, overlay_img_fg_gpu)

        # Download the result back to CPU
        combined = combined_gpu.download()
        base_img[y_start:y_end, x_start:x_end] = combined
    else:
        # Fallback to CPU if no GPU available
        gray_overlay = cv2.cvtColor(overlay_roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        roi = base_img[y_start:y_end, x_start:x_end]
        base_img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        overlay_img_fg = cv2.bitwise_and(overlay_roi, overlay_roi, mask=mask)
        combined = cv2.add(base_img_bg, overlay_img_fg)
        base_img[y_start:y_end, x_start:x_end] = combined

    return base_img


def process_video(model, input_path, output_path, num_threads=24, mode=0):
    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if isinstance(mode, str):
        overlay_img = cv2.imread(mode)

    print(f"Video Info: {width}x{height} @ {fps}fps, {total_frames} frames")

    # 修改输出路径，先输出到临时文件
    temp_output = output_path.replace('.mp4', '_temp.mp4')

    # 原来的处理过程保存到temp_output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    if not out.isOpened():
        print("Failed to initialize video writer")
        return

    # 创建进度条
    pbar = tqdm(total=total_frames, desc="Processing")

    # 创建队列
    frame_queue = Queue(maxsize=128)  # 待处理帧队列
    result_queue = Queue(maxsize=128)  # 处理完成帧队列

    # 处理线程函数
    def process_frame_worker():
        while True:
            frame_data = frame_queue.get()
            if frame_data is None:
                frame_queue.task_done()
                break

            frame, frame_id = frame_data
            processed_frame = frame.copy()
            faces = detect(processed_frame, model)

            for (x1, y1, x2, y2) in faces:
                if mode == 0:
                    face_region = processed_frame[y1:y2, x1:x2]
                    ratio = min((x2 - x1)/16, (y2 - y1)/16)
                    x0, y0 = int((x2 - x1) * ratio), int((y2 - y1) * ratio)
                    if x0 == 0:
                        x0 = 1
                    if y0 == 0:
                        y0 = 1
                    face_region = cv2.resize(face_region, (x0, y0), interpolation=cv2.INTER_LINEAR)
                    face_region = cv2.resize(face_region, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                    processed_frame[y1:y2, x1:x2] = face_region
                else:
                    # 使用overlay_image覆盖人脸
                    assert isinstance(mode, str)
                    face_width = x2 - x1
                    face_height = y2 - y1
                    x1 -= int(face_width * 0.15)
                    y1 -= int(face_height * 0.45)
                    face_width *= 1.2
                    face_height *= 1.5
                    
                    # 保持长宽比计算新尺寸
                    overlay_h, overlay_w = overlay_img.shape[:2]
                    ratio = max(face_width/overlay_w, face_height/overlay_h)
                    new_width = int(overlay_w * ratio)
                    new_height = int(overlay_h * ratio)
                    
                    processed_frame = overlay_image(
                        processed_frame, 
                        overlay_img, 
                        (x1, y1),
                        new_width,
                        new_height
                    )
                
            result_queue.put((processed_frame, frame_id))
            frame_queue.task_done()

    # 创建处理线程
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=process_frame_worker)
        t.start()
        threads.append(t)

    # 读取帧线程
    def read_frames():
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_queue.put((frame, frame_id))
            frame_id += 1

        # 发送结束信号
        for _ in range(num_threads):
            frame_queue.put(None)

    # 启动读取帧线程
    read_thread = threading.Thread(target=read_frames)
    read_thread.start()

    # 写入结果
    frame_id = 0
    pending_frames = {}

    while True:
        # 检查是否所有帧都处理完成
        if not read_thread.is_alive() and frame_queue.empty() and result_queue.empty() and not pending_frames:
            break

        # 获取处理后的帧
        try:
            processed_frame, curr_frame_id = result_queue.get(timeout=0.1)
            pending_frames[curr_frame_id] = processed_frame
            result_queue.task_done()

            # 按顺序写入帧
            while frame_id in pending_frames:
                out.write(pending_frames[frame_id])
                del pending_frames[frame_id]
                frame_id += 1
                if frame_id == 3000:
                    break
                pbar.update(1)  # 更新进度条

        except:
            continue

    pbar.close()
    # 等待所有线程完成
    read_thread.join()
    for t in threads:
        t.join()

    cap.release()
    out.release()

    # 合并音频
    try:
        # 读取原始视频的音轨
        video = VideoFileClip(input_path)
        audio = video.audio

        if audio is not None:
            # 读取处理后的视频
            processed_video = VideoFileClip(temp_output)
            # 添加音轨并保存
            final_video = processed_video.set_audio(audio)
            final_video.write_videofile(
                output_path, 
                codec='libx264',
                audio_codec='aac',  # 指定音频编码器
                temp_audiofile="temp-audio.m4a",  # 临时音频文件
                remove_temp=True  # 处理完成后删除临时文件
            )

            # 清理资源
            video.close()
            audio.close()
            processed_video.close()
            final_video.close()

            # 删除临时视频文件
            if os.path.exists(temp_output):
                os.remove(temp_output)

    except Exception as e:
        print(f"Error processing audio: {e}")
        # 如果音频处理失败，至少保留处理好的视频
        if os.path.exists(temp_output):
            os.rename(temp_output, output_path)

@smart_inference_mode()
def res(
        weights='best.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=r'D:\yolov10-main\yolov9\assets\worlds-largest-selfie.jpg',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=True,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video with face detection')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input video path')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output video path')
    parser.add_argument('-m', '--mode', type=str, default='0', help='Processing mode: 0 for pixelation, path to image for overlay')
    parser.add_argument('-t', '--threads', type=int, default=24, help='Number of processing threads')
    args = parser.parse_args()

    # 转换mode参数
    if args.mode == '0':
        mode = 0
    else:
        mode = args.mode  # 图片路径

    model = DetectMultiBackend("best.pt", device=torch.device("cuda"))
    process_video(model, args.input, args.output, num_threads=args.threads, mode=mode)
