import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 손 랜드마크 인덱스
LEFT_HAND_LANDMARKS = [15, 17, 19, 21]
RIGHT_HAND_LANDMARKS = [16, 18, 20, 22]

def draw_hand_landmarks(image, landmarks, hand_landmarks_indices):
    for index in hand_landmarks_indices:
        if index < len(landmarks):  # 인덱스가 범위를 벗어나지 않도록 확인
            cv2.circle(image, (int(landmarks[index].x * image.shape[1]), int(landmarks[index].y * image.shape[0])), 5, (0, 255, 0), -1)

def detect_hand_pose(image_path):
    print(f"Loading image from {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Failed to load image from {image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        draw_hand_landmarks(image, landmarks, LEFT_HAND_LANDMARKS)
        draw_hand_landmarks(image, landmarks, RIGHT_HAND_LANDMARKS)
        print("Hand landmarks detected and drawn")
    else:
        print("No pose landmarks detected")
        return None
    
    return image
