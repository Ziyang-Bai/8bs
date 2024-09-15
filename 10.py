import cv2
import numpy as np
import time
from multiprocessing import Pool, cpu_count

def quantize_colors(image):
    small = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    shifted = cv2.pyrMeanShiftFiltering(small, 21, 51)
    quantized = cv2.resize(shifted, image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    return quantized

def pixelate(image, scale=3):
    small = cv2.resize(image, None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_NEAREST)
    large = cv2.resize(small, image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    return large

def process_frame(frame):
    # 对每一帧进行像素化和颜色量化
    frame = pixelate(frame, scale=5)
    frame = quantize_colors(frame)
    return frame

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    start_time = time.time()
    
    # 使用多核并行处理帧
    pool = Pool(cpu_count())  # 创建与CPU核心数量相同的进程池
    
    print(f"Processing video with {total_frames} frames at {fps} FPS.")

    def frame_generator():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame

    # 用进程池并行处理帧
    for processed_frame in pool.imap(process_frame, frame_generator()):
        out.write(processed_frame)
        frame_count += 1

        # 打印进度和估计剩余时间
        if frame_count % 100 == 0:  # 每100帧打印一次进度
            elapsed_time = time.time() - start_time
            remaining_time = (elapsed_time / frame_count) * (total_frames - frame_count)
            m, s = divmod(remaining_time, 60)
            h, m = divmod(m, 60)
            print(f"Processing frame {frame_count}/{total_frames}, Estimated remaining time: {int(h)}h:{int(m)}m:{int(s)}s")

    cap.release()
    out.release()
    pool.close()
    pool.join()

    print("Video processing complete.")

video_path = 'input.mp4'
output_path = 'output_video_8bit.mp4'
process_video(video_path, output_path)
