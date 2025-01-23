from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import threading

app = Flask(__name__)

# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 假设摄像头的已知参考比例因子
PIXEL_TO_CM_RATIO = 0.16  # 示例值

# 全局变量用于共享身体维度数据
shared_data = {"dimensions": {}}
lock = threading.Lock()  # 确保线程安全

def calculate_body_dimensions(landmarks, frame_shape):
    """计算身体维度（转换为厘米）"""
    if not landmarks:
        return {}
    
    height, width, _ = frame_shape
    points = [
        (int(landmark.x * width), int(landmark.y * height)) for landmark in landmarks
    ]

    body_dimensions = {
        "Left Leg Length": np.linalg.norm(np.array(points[23]) - np.array(points[27])),
        "Right Leg Length": np.linalg.norm(np.array(points[24]) - np.array(points[28])),
        "Shoulder Width": np.linalg.norm(np.array(points[11]) - np.array(points[12])),
        "Torso Height": np.linalg.norm(np.array(points[11]) - np.array(points[23]))
    }
    
    body_dimensions = {k: f"{v * PIXEL_TO_CM_RATIO:.2f} cm" for k, v in body_dimensions.items()}
    return body_dimensions

def generate_frames():
    """生成实时视频流帧"""
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            body_dimensions = {}
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )
                body_dimensions = calculate_body_dimensions(results.pose_landmarks.landmark, frame.shape)
                
                # 更新共享数据
                with lock:
                    shared_data["dimensions"] = body_dimensions

            # 显示身体维度
            y_offset = 20
            for key, value in body_dimensions.items():
                cv2.putText(
                    frame, f"{key}: {value}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                y_offset += 30

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    """实时视频流路由"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_body_dimensions')
def get_body_dimensions():
    """返回实时的身体维度数据（JSON 格式）"""
    with lock:
        return jsonify(shared_data["dimensions"])

@app.route('/')
def home():
    """主页"""
    return render_template('home.html')

@app.route('/scan')
def scan():
    """扫描页面"""
    return render_template('scanning.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)
