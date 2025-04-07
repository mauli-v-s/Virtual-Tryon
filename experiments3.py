import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
import mediapipe as mp
import time

# Global variable for the active shirt.
ID = 0                    
cooldown = 1.0            
last_shirt_change = 0     
offset = 10               

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # Get screen dimensions.
        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()

        # Enlarge preview area.
        self.preview_width = 500
        self.preview_height = 300
        # Position the preview at the right top.
        self.preview_x = self.screen_width - self.preview_width - 20  
        self.preview_y = 20  
        # List of available shirts.
        self.shirts_list = ['tshirt4.jpg', 'top4.jpg', 'tshirt3.png', 'tshirt2.png', 'tshirt1.png']

        # Pass UI parameters to VideoCapture.
        self.cap = VideoCapture(self.screen_width, self.screen_height,
                                self.preview_x, self.preview_y,
                                self.preview_width, self.preview_height,
                                self.shirts_list)

        self.window.attributes('-fullscreen', True)
        self.window.bind('<Escape>', self.exit_fullscreen)
        
        # Main canvas for the live camera feed.
        self.canvas = tk.Canvas(self.window, width=self.screen_width,
                                height=self.screen_height)
        self.canvas.pack(fill=tk.BOTH, expand=True)
       
        self.preview_canvas = tk.Canvas(self.window, width=self.preview_width,
                                        height=self.preview_height, highlightthickness=0, bg='white')
        self.preview_canvas.place(x=self.preview_x, y=self.preview_y)
        
       
        self.preview_photo = None

        self.delay = 5
        self.update()
        self.window.mainloop()

    def exit_fullscreen(self, event=None):
        self.window.attributes('-fullscreen', False)

    def update(self):
        ret, frame = self.cap.get_frame()
        if ret:
            # Update the live camera feed.
            frame = cv2.resize(frame, (self.screen_width, self.screen_height))
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            
      
            self.preview_canvas.delete("all")
           
            self.preview_canvas.create_rectangle(0, 0, self.preview_width, self.preview_height, 
                                                fill="white", outline="")

            # Get the current preview index.
            preview_index = self.cap.preview_index

            # Load the corresponding shirt image.
            shirt_file = self.shirts_list[preview_index]
            img = cv2.imread(shirt_file)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
             
                orig_h, orig_w = img_rgb.shape[:2]
               
                arrow_margin = 40
                arrow_size = 20
                margin_space = arrow_margin + arrow_size + 5  
                available_width = self.preview_width - 2 * margin_space
                available_height = self.preview_height - 2 * margin_space
                scale = min(available_width / orig_w, available_height / orig_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                preview_img = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                self.preview_photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(preview_img))
                # Display the shirt image at the center.
                self.preview_canvas.create_image(self.preview_width // 2, self.preview_height // 2, 
                                                image=self.preview_photo, anchor=tk.CENTER)
            
          
            arrow_margin = 40
            arrow_size = 20
            # Left arrow (points left)
            left_arrow = [arrow_margin, self.preview_height // 2,
                        arrow_margin + arrow_size, self.preview_height // 2 - arrow_size,
                        arrow_margin + arrow_size, self.preview_height // 2 + arrow_size]
            # Right arrow (points right)
            right_arrow = [self.preview_width - arrow_margin, self.preview_height // 2,
                        self.preview_width - arrow_margin - arrow_size, self.preview_height // 2 - arrow_size,
                        self.preview_width - arrow_margin - arrow_size, self.preview_height // 2 + arrow_size]
            self.preview_canvas.create_polygon(left_arrow, fill="black")
            self.preview_canvas.create_polygon(right_arrow, fill="black")
            
            # Instruction text.
            self.preview_canvas.create_text(self.preview_width // 2, self.preview_height - 20,
                                            text="Swipe hand left/right in preview area",
                                            fill="blue", font=("Helvetica", 14))
        self.window.after(self.delay, self.update)


class VideoCapture:
    def __init__(self, screen_width, screen_height, preview_x, preview_y, preview_width, preview_height, shirts_list):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("Unable to open Camera.")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.preview_x = preview_x
        self.preview_y = preview_y
        self.preview_width = preview_width
        self.preview_height = preview_height
        
        self.shirts_list = shirts_list

        # For virtual scrolling & selection, we store a preview index.
        self.preview_index = 0  
        self.last_scroll_time = 0
        self.hover_start = None  
        
        # Define left/right thresholds (in pixels) within the preview area.
        self.left_threshold = 100
        self.right_threshold = self.preview_width - 100
        
        # Initialize Mediapipe Hands.
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize Mediapipe Face Detection (fallback).
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.7)
        self.face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
        
        # Initialize Mediapipe Pose for body detection.
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
        self.pose.close()

    def get_frame(self):
        global ID, last_shirt_change, cooldown, offset
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                return ret, None

            # Mirror the frame.
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]

            # --- Virtual Mouse for Preview Control using Hand Gestures ---
            results_hands = self.hands.process(frame_rgb)
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    index_tip = hand_landmarks.landmark[8]
                    finger_x = int(index_tip.x * img_w)
                    finger_y = int(index_tip.y * img_h)
                    scale_x = self.screen_width / img_w
                    scale_y = self.screen_height / img_h
                    finger_screen_x = finger_x * scale_x
                    finger_screen_y = finger_y * scale_y

                    # Check if the finger is within the preview area.
                    if (self.preview_x <= finger_screen_x <= self.preview_x + self.preview_width) and \
                       (self.preview_y <= finger_screen_y <= self.preview_y + self.preview_height):
                        rel_x = finger_screen_x - self.preview_x
                        current_time = time.time()
                        if rel_x < self.left_threshold:
                            if current_time - self.last_scroll_time > 0.5:
                                self.preview_index = (self.preview_index - 1) % len(self.shirts_list)
                                self.last_scroll_time = current_time
                                self.hover_start = None
                        elif rel_x > self.right_threshold:
                            if current_time - self.last_scroll_time > 0.5:
                                self.preview_index = (self.preview_index + 1) % len(self.shirts_list)
                                self.last_scroll_time = current_time
                                self.hover_start = None
                        else:
                            if self.hover_start is None:
                                self.hover_start = current_time
                            elif current_time - self.hover_start > cooldown:
                                if ID != self.preview_index:
                                    ID = self.preview_index
                                    last_shirt_change = current_time
                                self.hover_start = current_time

            # --- Pose Detection: Overlay the Active Shirt ---
            results_pose = self.pose.process(frame_rgb)
            if results_pose.pose_landmarks:
                landmarks = results_pose.pose_landmarks.landmark
                try:
                    left_shoulder = landmarks[11]
                    right_shoulder = landmarks[12]
                    left_hip = landmarks[23]
                    right_hip = landmarks[24]
                except IndexError:
                    return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

                shirts_type = self.shirts_list
                imgshirt = cv2.imread(shirts_type[ID])
                threshold = [200, 254]
                musgray = cv2.cvtColor(imgshirt, cv2.COLOR_BGR2GRAY)
                ret_val, orig_mask = cv2.threshold(musgray, threshold[ID % len(threshold)], 255, cv2.THRESH_BINARY)
                orig_mask_inv = cv2.bitwise_not(orig_mask)

                region_width = shirt_x2 - shirt_x1
                region_height = shirt_y2 - shirt_y1
                if region_width > 0 and region_height > 0:
                    shirt_resized = cv2.resize(imgshirt, (region_width, region_height), interpolation=cv2.INTER_AREA)
                    mask_resized = cv2.resize(orig_mask, (region_width, region_height), interpolation=cv2.INTER_AREA)
                    mask_inv_resized = cv2.resize(orig_mask_inv, (region_width, region_height), interpolation=cv2.INTER_AREA)

                    roi = frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
                    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_resized)
                    roi_fg = cv2.bitwise_and(shirt_resized, shirt_resized, mask=mask_inv_resized)
                    combined = cv2.add(roi_bg, roi_fg)
                    frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2] = combined
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                shirts_type = self.shirts_list
                imgshirt = cv2.imread(shirts_type[ID])
                threshold = [200, 254]
                musgray = cv2.cvtColor(imgshirt, cv2.COLOR_BGR2GRAY)
                ret_val, orig_mask = cv2.threshold(musgray, threshold[ID % len(threshold)], 255, cv2.THRESH_BINARY)
                orig_mask_inv = cv2.bitwise_not(orig_mask)
                for (x, y, w, h) in faces:
                    face_w = w
                    face_h = h
                    face_x1 = x
                    face_x2 = x + w
                    face_y1 = y
                    face_y2 = y + h

                    shirtWidth = int(2.9 * face_w + offset)
                    shirtHeight = int((shirtWidth * imgshirt.shape[0] / imgshirt.shape[1]) + offset / 3)

                    shirt_x1 = face_x2 - int(face_w / 2) - int(shirtWidth / 2)
                    shirt_x2 = shirt_x1 + shirtWidth
                    shirt_y1 = face_y2 + 5
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

                        roi = frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
                        roi_bg = cv2.bitwise_and(roi, roi, mask=mask_resized)
                        roi_fg = cv2.bitwise_and(shirt_resized, shirt_resized, mask=mask_inv_resized)
                        combined = cv2.add(roi_bg, roi_fg)
                        frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2] = combined

            return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None, None

    @staticmethod
    def count_raised_fingers(hand_landmarks):
       
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        thumb_up = hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x
        fingers_up = [thumb_up]
        for tip, pip in zip(finger_tips, finger_pips):
            fingers_up.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y)
        return sum(fingers_up)

App(tk.Tk(), "Virtual Mirror")
