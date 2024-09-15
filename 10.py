import cv2
import numpy as np
import time

def quantize_colors(image, colors=64):
    # 减少处理尺寸
    small = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    
    # 使用颜色均值迁移进行快速颜色量化
    shifted = cv2.pyrMeanShiftFiltering(small, 21, 51)
    
    # 放大图像回到原始大小
    quantized = cv2.resize(shifted, image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    
    return quantized

def pixelate(image, scale=10):
    small = cv2.resize(image, None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_NEAREST)
    large = cv2.resize(small, image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    return large

video_path = 'input.mp4'
output_path = 'output_video_8bit.mp4'

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
start_time = time.time()

print(f"Processing video with {total_frames} frames at {fps} FPS.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # 处理每一帧
    frame = pixelate(frame, scale=5)
    frame = quantize_colors(frame)
    out.write(frame)
    
    # 计算剩余时间
    elapsed_time = time.time() - start_time
    remaining_time = (elapsed_time / frame_count) * (total_frames - frame_count)
    m, s = divmod(remaining_time, 60)
    h, m = divmod(m, 60)
    
    if frame_count % 100 == 0:  # 每100帧打印一次进度
        print(f"Processing frame {frame_count}/{total_frames}, Estimated remaining time: {int(h)}h:{int(m)}m:{int(s)}s")

cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing complete.")
