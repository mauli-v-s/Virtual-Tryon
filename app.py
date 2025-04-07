import cv2
import mediapipe as mp
import numpy as np
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# =============================================================================
# Video Transformer that implements the Virtual Mirror functionality.
# =============================================================================
class VirtualMirrorTransformer(VideoTransformerBase):
    def __init__(self):
  
        self.shirts_list = ['tshirt4.jpg', 'top4.jpg', 'tshirt3.png', 'tshirt2.png', 'tshirt1.png']
       
        self.active_shirt_index = 0

        self.last_scroll_time = 0
        self.cooldown = 1.0 

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils

     
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

   
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")
 
        img = cv2.flip(img, 1)
        img_h, img_w = img.shape[:2]


        if "shirt_index" in st.session_state:
            self.active_shirt_index = st.session_state["shirt_index"]

        # Read gesture control flag from sidebar (enabled by default)
        gesture_control = st.session_state.get("gesture_control", True)


        preview_width = 200
        preview_height = 150
        margin = 20 
        preview_x = img_w - preview_width - margin
        preview_y = margin


        if gesture_control:
            # Convert to RGB for mediapipe processing.
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results_hands = self.hands.process(img_rgb)
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
             
                    self.mp_drawing.draw_landmarks(
                        img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                
                    index_tip = hand_landmarks.landmark[8]
                    finger_x = int(index_tip.x * img_w)
                    finger_y = int(index_tip.y * img_h)

                    if preview_x <= finger_x <= preview_x + preview_width and preview_y <= finger_y <= preview_y + preview_height:
                        rel_x = finger_x - preview_x 
                        left_threshold = 40
                        right_threshold = preview_width - 40
                        if rel_x < left_threshold:
                            if current_time - self.last_scroll_time > self.cooldown:
                                self.active_shirt_index = (self.active_shirt_index - 1) % len(self.shirts_list)
                                st.session_state["shirt_index"] = self.active_shirt_index
                                self.last_scroll_time = current_time
                        elif rel_x > right_threshold:
                            if current_time - self.last_scroll_time > self.cooldown:
                                self.active_shirt_index = (self.active_shirt_index + 1) % len(self.shirts_list)
                                st.session_state["shirt_index"] = self.active_shirt_index
                                self.last_scroll_time = current_time

        # ---------------------------------------------------------------------
        # Draw the preview area background (filled white rectangle)
        cv2.rectangle(img, (preview_x, preview_y), (preview_x + preview_width, preview_y + preview_height), (255, 255, 255), -1)


        shirt_path = self.shirts_list[self.active_shirt_index]
        shirt_img = cv2.imread(shirt_path)
        if shirt_img is not None:
         
            shirt_img_rgb = cv2.cvtColor(shirt_img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = shirt_img_rgb.shape[:2]
 
            arrow_margin = 20
            arrow_size = 10
            margin_space = arrow_margin + arrow_size + 5
            available_width = preview_width - 2 * margin_space
            available_height = preview_height - 2 * margin_space
            scale = min(available_width / orig_w, available_height / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            resized_shirt = cv2.resize(shirt_img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
   
            center_x = preview_x + preview_width // 2
            center_y = preview_y + preview_height // 2
            top_left_x = center_x - new_w // 2
            top_left_y = center_y - new_h // 2

            resized_shirt_bgr = cv2.cvtColor(resized_shirt, cv2.COLOR_RGB2BGR)
            img[top_left_y:top_left_y + new_h, top_left_x:top_left_x + new_w] = resized_shirt_bgr


        left_arrow = np.array([
            [preview_x + arrow_margin, preview_y + preview_height // 2],
            [preview_x + arrow_margin + arrow_size, preview_y + preview_height // 2 - arrow_size],
            [preview_x + arrow_margin + arrow_size, preview_y + preview_height // 2 + arrow_size]
        ], np.int32)
        cv2.fillPoly(img, [left_arrow], (0, 0, 0))

    
        right_arrow = np.array([
            [preview_x + preview_width - arrow_margin, preview_y + preview_height // 2],
            [preview_x + preview_width - arrow_margin - arrow_size, preview_y + preview_height // 2 - arrow_size],
            [preview_x + preview_width - arrow_margin - arrow_size, preview_y + preview_height // 2 + arrow_size]
        ], np.int32)
        cv2.fillPoly(img, [right_arrow], (0, 0, 0))


        cv2.putText(
            img,
            "Swipe hand left/right",
            (preview_x + 10, preview_y + preview_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA
        )

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results_pose = self.pose.process(img_rgb)
        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark
            try:
                left_shoulder = landmarks[11]
                right_shoulder = landmarks[12]
                left_hip = landmarks[23]
                right_hip = landmarks[24]
            except IndexError:
                return img 
            ls_x, ls_y = int(left_shoulder.x * img_w), int(left_shoulder.y * img_h)
            rs_x, rs_y = int(right_shoulder.x * img_w), int(right_shoulder.y * img_h)
            lh_x, lh_y = int(left_hip.x * img_w), int(left_hip.y * img_h)
            rh_x, rh_y = int(right_hip.x * img_w), int(right_hip.y * img_h)
            shoulder_mid_x = (ls_x + rs_x) // 2
            shoulder_mid_y = (ls_y + rs_y) // 2
            hip_mid_y = (lh_y + rh_y) // 2

            shirt_width = int(2.0 * abs(rs_x - ls_x))
            shirt_height = int(1.5 * (hip_mid_y - shoulder_mid_y))
            uplift = int(0.15 * shirt_height)
            shirt_x1 = shoulder_mid_x - shirt_width // 2
            shirt_y1 = max(0, shoulder_mid_y - uplift)
            shirt_x2 = shirt_x1 + shirt_width
            shirt_y2 = shirt_y1 + shirt_height

            shirt_x1 = max(0, shirt_x1)
            shirt_y1 = max(0, shirt_y1)
            shirt_x2 = min(img_w, shirt_x2)
            shirt_y2 = min(img_h, shirt_y2)

            active_shirt_path = self.shirts_list[self.active_shirt_index]
            imgshirt = cv2.imread(active_shirt_path)
            if imgshirt is not None:
                musgray = cv2.cvtColor(imgshirt, cv2.COLOR_BGR2GRAY)
                threshold = [200, 254] 
                ret_val, orig_mask = cv2.threshold(
                    musgray,
                    threshold[self.active_shirt_index % len(threshold)],
                    255,
                    cv2.THRESH_BINARY
                )
                orig_mask_inv = cv2.bitwise_not(orig_mask)
                region_width = shirt_x2 - shirt_x1
                region_height = shirt_y2 - shirt_y1
                if region_width > 0 and region_height > 0:
                    shirt_resized = cv2.resize(imgshirt, (region_width, region_height), interpolation=cv2.INTER_AREA)
                    mask_resized = cv2.resize(orig_mask, (region_width, region_height), interpolation=cv2.INTER_AREA)
                    mask_inv_resized = cv2.resize(orig_mask_inv, (region_width, region_height), interpolation=cv2.INTER_AREA)
                    roi = img[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
                    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_resized)
                    roi_fg = cv2.bitwise_and(shirt_resized, shirt_resized, mask=mask_inv_resized)
                    combined = cv2.add(roi_bg, roi_fg)
                    img[shirt_y1:shirt_y2, shirt_x1:shirt_x2] = combined
        else:
            # -----------------------------------------------------------------
            # Fallback: Use face detection if pose landmarks are not found.
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            active_shirt_path = self.shirts_list[self.active_shirt_index]
            imgshirt = cv2.imread(active_shirt_path)
            if imgshirt is not None:
                musgray = cv2.cvtColor(imgshirt, cv2.COLOR_BGR2GRAY)
                threshold = [200, 254]
                ret_val, orig_mask = cv2.threshold(
                    musgray,
                    threshold[self.active_shirt_index % len(threshold)],
                    255,
                    cv2.THRESH_BINARY
                )
                orig_mask_inv = cv2.bitwise_not(orig_mask)
                for (x, y, w, h) in faces:
                    shirtWidth = int(2.9 * w + 10)
                    shirtHeight = int((shirtWidth * imgshirt.shape[0] / imgshirt.shape[1]) + 10 / 3)
                    shirt_x1 = x + w - int(w / 2) - int(shirtWidth / 2)
                    shirt_x2 = shirt_x1 + shirtWidth
                    shirt_y1 = y + h + 5
                    shirt_y2 = shirt_y1 + shirtHeight

                    shirt_x1 = max(0, shirt_x1)
                    shirt_y1 = max(0, shirt_y1)
                    shirt_x2 = min(img_w, shirt_x2)
                    shirt_y2 = min(img_h, shirt_y2)
                    region_width = shirt_x2 - shirt_x1
                    region_height = shirt_y2 - shirt_y1
                    if region_width > 0 and region_height > 0:
                        shirt_resized = cv2.resize(imgshirt, (region_width, region_height), interpolation=cv2.INTER_AREA)
                        mask_resized = cv2.resize(orig_mask, (region_width, region_height), interpolation=cv2.INTER_AREA)
                        mask_inv_resized = cv2.resize(orig_mask_inv, (region_width, region_height), interpolation=cv2.INTER_AREA)
                        roi = img[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
                        roi_bg = cv2.bitwise_and(roi, roi, mask=mask_resized)
                        roi_fg = cv2.bitwise_and(shirt_resized, shirt_resized, mask=mask_inv_resized)
                        combined = cv2.add(roi_bg, roi_fg)
                        img[shirt_y1:shirt_y2, shirt_x1:shirt_x2] = combined
        return img


# =============================================================================
# Streamlit UI
# =============================================================================
st.title("Virtual Mirror with Shirt Overlay")

RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
webrtc_streamer(
    key="virtual-mirror",
    rtc_configuration=RTC_CONFIGURATION,
    video_transformer_factory=VirtualMirrorTransformer,
)
