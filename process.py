import mediapipe as mp

def detect_keypoints(image):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Process image
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        return results.pose_landmarks
    return None
