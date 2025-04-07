import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
import mediapipe as mp
import time

root = tk.Tk()
frame_width = root.winfo_screenwidth()
frame_height = root.winfo_screenheight()
root.destroy()  
ID = 0
cooldown = 2  # Cooldown time in seconds
last_shirt_change = 0  # Timestamp of last shirt change
offset =10

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        global offset
        self.cap = VideoCapture()

        self.frame = tk.Frame(window)
        self.frame.place(relx=0.44, rely=0.05, anchor='n')

        self.canvas = tk.Canvas(self.frame, width=self.cap.width, height=self.cap.height)
        self.canvas.pack()

        self.delay = 5
        self.update()

        self.window.mainloop()

    def update(self):
        _, frame = self.cap.get_frame()
        if _:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(self.delay, self.update)


class VideoCapture:
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("Unable to open Camera.")

        self.width = frame_width
        self.height = frame_height

        # Initialize Mediapipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize Mediapipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.7)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

    def get_frame(self):
        global ID, last_shirt_change, cooldown
        if self.cap.isOpened():
            _, frame = self.cap.read()
            if not _:
                return _, None

            frame = cv2.flip(frame, 1)  # Flip for mirror view
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Hand Detection
            results = self.hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )

                    # Check for raised fingers
                    fingers = self.count_raised_fingers(hand_landmarks)
                    if fingers == 5 and time.time() - last_shirt_change > cooldown:
                        ID = (ID + 1) % 2  # Toggle shirt
                        last_shirt_change = time.time()  # Update timestamp

            # Face Detection
            face_results = self.face_detection.process(frame_rgb)
            if face_results.detections:
                for detection in face_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                        int(bboxC.width * w), int(bboxC.height * h)

                    # Render Shirt
                    shirts_type = ['tshirt4.jpg', 'top4.jpg']
                    imgshirt = cv2.imread(shirts_type[ID])
                    if imgshirt is not None:
                        self.overlay_shirt(frame, imgshirt, bbox)

            return _, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None, None

    @staticmethod
    def count_raised_fingers(hand_landmarks):
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]

        thumb_up = hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x

        fingers_up = [thumb_up]
        for tip, pip in zip(finger_tips, finger_pips):
            fingers_up.append(
                hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y
            )
        return sum(fingers_up)

    def overlay_shirt(self, frame, imgshirt, bbox):
        x, y, w, h = bbox
        offset=10
        shirt_width = int(2.9 * w + offset)
        shirt_height = int((shirt_width * imgshirt.shape[0] / imgshirt.shape[1]) + offset / 3)

        shirt_x1 = x + w // 2 - shirt_width // 2
        shirt_x2 = shirt_x1 + shirt_width
        shirt_y1 = y + h + 5
        shirt_y2 = shirt_y1 + shirt_height

        shirt_x1 = max(0, shirt_x1)
        shirt_y1 = max(0, shirt_y1)
        shirt_x2 = min(frame.shape[1], shirt_x2)
        shirt_y2 = min(frame.shape[0], shirt_y2)

        if shirt_x2 <= shirt_x1 or shirt_y2 <= shirt_y1:
            return

        shirt = cv2.resize(imgshirt, (shirt_x2 - shirt_x1, shirt_y2 - shirt_y1))
        mask = cv2.cvtColor(shirt, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        roi = frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
        roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        roi_fg = cv2.bitwise_and(shirt, shirt, mask=mask)

        frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2] = cv2.add(roi_bg, roi_fg)

App(tk.Tk(), "Virtual Mirror")
