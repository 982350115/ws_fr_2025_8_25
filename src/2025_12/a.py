import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def cv2_add_chinese_text(img, text, position, text_color=(0, 255, 0), text_size=40):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    draw = ImageDraw.Draw(img)
    # 字体路径 - 依然使用通用路径
    font_path = "/usr/share/fonts/truetype/arphic/uming.ttc" 
    try:
        font = ImageFont.truetype(font_path, text_size)
    except IOError:
        font = ImageFont.load_default()
        
    draw.text(position, text, fill=text_color[::-1], font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def process_video(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"错误：找不到文件 {input_path}")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("错误：无法打开视频流")
        return

    # --- 1. 获取并修正 FPS ---
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 针对你的 1000 FPS 异常情况进行修正
    if original_fps > 60 or original_fps <= 0:
        print(f"检测到异常 FPS ({original_fps})，强制修正为 30 FPS。")
        use_fps = 30.0
    else:
        use_fps = original_fps

    # --- 2. 确定剪辑范围 ---
    # 剪掉原视频前 1 秒 -> 从第 1*fps 帧开始
    start_frame_idx = int(1.0 * use_fps)
    # 原视频第 9 秒结束 -> 到第 9*fps 帧停止
    end_frame_idx = int(9.0 * use_fps)

    # --- 3. 初始化写入器 (AVI + MJPG) ---
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
    
    # 强制修改后缀为 .avi
    base_name = os.path.splitext(output_path)[0]
    output_path = base_name + ".avi"
    
    out = cv2.VideoWriter(output_path, fourcc, use_fps, (width, height))
    
    if not out.isOpened():
        print("错误：视频写入器无法初始化")
        return

    # 设置读取位置，跳过原视频的第一秒
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
    
    print(f"开始处理... 字幕规则：0-1s='我', 1-3s='王老师'")
    
    current_frame_idx = start_frame_idx
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or current_frame_idx > end_frame_idx:
            break
            
        # --- 时间轴计算 (新视频的时间) ---
        # 刚开始循环时，current == start，结果为 0.0秒
        time_in_new_video = (current_frame_idx - start_frame_idx) / use_fps
        
        # --- 字幕逻辑 (修改部分) ---
        # 0秒 到 1秒 -> 显示“我”
        if 0.0 <= time_in_new_video < 1.0:
            frame = cv2_add_chinese_text(frame, "我", (50, 50), text_color=(0, 0, 255), text_size=80)
            
        # 1秒 到 3秒 -> 显示“王老师”
        elif 1.0 <= time_in_new_video < 3.0:
            frame = cv2_add_chinese_text(frame, "王老师", (50, 50), text_color=(0, 255, 255), text_size=80)
            
        out.write(frame)
        current_frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"完成！视频已保存至: {output_path}")

if __name__ == "__main__":
    input_video = "jun.webm"
    output_video = "output_final.avi" 
    process_video(input_video, output_video)