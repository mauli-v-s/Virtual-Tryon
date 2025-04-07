import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
import mediapipe as mp
import time

ID = 0
cooldown = 2  
last_shirt_change = 0  
offset = 10

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.cap = VideoCapture()
       
        self.window.attributes('-fullscreen', True)  
        self.window.bind('<Escape>', self.exit_fullscreen) 
        
    
        self.canvas = tk.Canvas(self.window, width=self.window.winfo_screenwidth(), 
                                height=self.window.winfo_screenheight())
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.delay = 5
        self.update()

        self.window.mainloop()

    def exit_fullscreen(self, event=None):
        self.window.attributes('-fullscreen', False)

    def update(self):
        ret, frame = self.cap.get_frame()
        if ret:
     
            frame = cv2.resize(frame, (self.window.winfo_screenwidth(), self.window.winfo_screenheight()))
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(self.delay, self.update)

class VideoCapture:
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("Unable to open Camera.")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

   
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils

   
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.7)
        self.face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
        
     
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
        self.pose.close()

    def get_frame(self):
        global ID, last_shirt_change, cooldown
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                return ret, None

            frame = cv2.flip(frame, 1) 
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]

         
            results_hands = self.hands.process(frame_rgb)
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    fingers = self.count_raised_fingers(hand_landmarks)
                    if fingers == 5 and time.time() - last_shirt_change > cooldown:
                        ID = (ID + 1) % 2 
                        last_shirt_change = time.time()

           
            results_pose = self.pose.process(frame_rgb)
            if results_pose.pose_landmarks:
          
                landmarks = results_pose.pose_landmarks.landmark
                try:
                    left_shoulder  = landmarks[11]
                    right_shoulder = landmarks[12]
                    left_hip       = landmarks[23]
                    right_hip      = landmarks[24]
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

            
                shirts_type = ['tshirt4.jpg', 'top4.jpg']
                imgshirt = cv2.imread(shirts_type[ID])
             
                threshold = [200, 254]
                musgray = cv2.cvtColor(imgshirt, cv2.COLOR_BGR2GRAY)
                ret_val, orig_mask = cv2.threshold(musgray, threshold[ID], 255, cv2.THRESH_BINARY)
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
                shirts_type = ['tshirt4.jpg', 'top4.jpg']
                imgshirt = cv2.imread(shirts_type[ID])
                threshold = [200, 254]
                musgray = cv2.cvtColor(imgshirt, cv2.COLOR_BGR2GRAY)
                ret_val, orig_mask = cv2.threshold(musgray, threshold[ID], 255, cv2.THRESH_BINARY)
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
