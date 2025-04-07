import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
import mediapipe as mp
import time


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
       
        self.window.attributes('-fullscreen', True)  
        self.window.bind('<Escape>', self.exit_fullscreen) 
        self.frame = tk.Frame(window)
        self.frame.place(relx=0.44, rely=0.05, anchor='n')

        # self.canvas = tk.Canvas(self.frame, width=self.cap.width, height=self.cap.height)
        # self.canvas.pack()

        self.canvas = tk.Canvas(self.window, width=self.window.winfo_screenwidth(), height=self.window.winfo_screenheight())
        self.canvas.pack(fill=tk.BOTH, expand=True)  # Make the canvas fill the entire window


        self.delay = 5
        self.update()

        self.window.mainloop()

    def exit_fullscreen(self, event=None):
        self.window.attributes('-fullscreen', False)

    def update(self):
        ret, frame = self.cap.get_frame()
        if ret:
            # Resize the frame to fit the canvas dimensions
            frame = cv2.resize(frame, (self.window.winfo_screenwidth(), self.window.winfo_screenheight()))
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(self.delay, self.update)


    # def update(self):
    #     _, frame = self.cap.get_frame()
    #     if _:
    #         self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
    #         self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
    #     self.window.after(self.delay, self.update)

class VideoCapture:
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("Unable to open Camera.")

        # self.width = frame_width
        # self.height = frame_height

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


        # Initialize Mediapipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize Mediapipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.7)

        # Load face cascade
        self.face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

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
            img_h, img_w = frame.shape[:2]
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

            # Face Detection using Haar Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            shirts_type = ['tshirt4.jpg', 'top4.jpg']
            imgshirt = cv2.imread(shirts_type[ID])

            threshold = [200, 254]
            shirt_id = ID
            imgshirt = cv2.imread(shirts_type[shirt_id])
            musgray = cv2.cvtColor(imgshirt, cv2.COLOR_BGR2GRAY)  # Grayscale conversion
            ret, orig_mask = cv2.threshold(musgray, threshold[ID], 255, cv2.THRESH_BINARY)
            orig_mask_inv = cv2.bitwise_not(orig_mask)
            origshirtHeight, origshirtWidth = imgshirt.shape[:2]

            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            # origshirtHeight, origshirtWidth = imgshirt.shape[:2]
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                face_w = w
                face_h = h
                face_x1 = x
                face_x2 = face_x1 + face_w
                face_y1 = y
                face_y2 = face_y1 + face_h

                shirtWidth = int(2.9 * face_w+ offset)
                shirtHeight = int((shirtWidth * origshirtHeight / origshirtWidth)+offset/3)
                # cv2.putText(frame,(str(shirtWidth)+" "+str(shirtHeight)),(x+w,y+h),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)

                shirt_x1 = face_x2 - int(face_w / 2) - int(shirtWidth / 2)                             # setting shirt centered wrt recognized face
                shirt_x2 = shirt_x1 + shirtWidth
                shirt_y1 = face_y2 + 5                                                       # some padding between face and upper shirt. Depends on the shirt img
                shirt_y2 = shirt_y1 + shirtHeight

                if shirt_x1 < 0:
                    shirt_x1 = 0
                if shirt_y1 < 0:
                    shirt_y1 = 0
                if shirt_x2 > img_w:
                    shirt_x2 = img_w
                if shirt_y2 > img_h:
                    shirt_y2 = img_h

                shirtWidth = shirt_x2 - shirt_x1
                shirtHeight = shirt_y2 - shirt_y1
                if shirtWidth < 0 or shirtHeight < 0:
                    continue

            # Load the shirt image
                
                # # Set the shirt size in relation to the detected face
                # shirtWidth = int(2.9 * w)  # Adjust shirt width based on face width
                # shirtHeight = int((shirtWidth * origshirtHeight / origshirtWidth))

                # shirt_x1 = x + w // 2 - shirtWidth // 2  # Center the shirt horizontally with face
                # shirt_x2 = shirt_x1 + shirtWidth
                # shirt_y1 = y + h + 5  # Position the shirt slightly below the face
                # shirt_y2 = shirt_y1 + shirtHeight

                # # Avoid going out of frame bounds
                # shirt_x1 = max(0, shirt_x1)
                # shirt_y1 = max(0, shirt_y1)
                # shirt_x2 = min(frame.shape[1], shirt_x2)
                # shirt_y2 = min(frame.shape[0], shirt_y2)

                # shirtWidth = shirt_x2 - shirt_x1
                # shirtHeight = shirt_y2 - shirt_y1
                # if shirtWidth < 0 or shirtHeight < 0:
                #     continue

                # if imgshirt is not None:
                    # Overlay shirt on frame
                # self.overlay_shirt(frame, imgshirt, (shirt_x1, shirt_y1, shirtWidth, shirtHeight), orig_mask, orig_mask_inv)
                shirt = cv2.resize(imgshirt, (shirt_x2 - shirt_x1, shirt_y2 - shirt_y1), interpolation=cv2.INTER_AREA)
                mask = cv2.resize(orig_mask, (shirt_x2 - shirt_x1, shirt_y2 - shirt_y1), interpolation=cv2.INTER_AREA)
                mask_inv = cv2.resize(orig_mask_inv, (shirt_x2 - shirt_x1, shirt_y2 - shirt_y1), interpolation=cv2.INTER_AREA)

                # Define region of interest (ROI) on the frame
                roi = frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2]

                # Overlay shirt
                roi_bg = cv2.bitwise_and(roi, roi, mask=mask)  # Background
                roi_fg = cv2.bitwise_and(shirt, shirt, mask=mask_inv)  # Foreground (shirt)
                combined = cv2.add(roi_bg, roi_fg)

                # Place the combined image back onto the frame
                frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2] = combined

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

    def overlay_shirt(self, frame, imgshirt, bbox, orig_mask, orig_mask_inv):
        x, y, w, h = bbox

        # Adjust scaling factors for better fit
        shirt_width = int(3.0 * w)  # Increase scaling factor for shirt width
        shirt_height = int(shirt_width * imgshirt.shape[0] / imgshirt.shape[1])

        # Calculate shirt position
        shirt_x1 = x + w // 2 - shirt_width // 2
        shirt_x2 = shirt_x1 + shirt_width
        shirt_y1 = y + h + 10  # Slightly increase offset below the face
        shirt_y2 = shirt_y1 + shirt_height

        # Ensure bounding box stays within frame
        shirt_x1 = max(0, shirt_x1)
        shirt_y1 = max(0, shirt_y1)
        shirt_x2 = min(frame.shape[1], shirt_x2)
        shirt_y2 = min(frame.shape[0], shirt_y2)

        # Avoid invalid dimensions
        if shirt_x2 <= shirt_x1 or shirt_y2 <= shirt_y1:
            return

        # Resize shirt image and masks to fit the calculated region
        shirt = cv2.resize(imgshirt, (shirt_x2 - shirt_x1, shirt_y2 - shirt_y1), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(orig_mask, (shirt_x2 - shirt_x1, shirt_y2 - shirt_y1), interpolation=cv2.INTER_AREA)
        mask_inv = cv2.resize(orig_mask_inv, (shirt_x2 - shirt_x1, shirt_y2 - shirt_y1), interpolation=cv2.INTER_AREA)

        # Define region of interest (ROI) on the frame
        roi = frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2]

        # Overlay shirt
        roi_bg = cv2.bitwise_and(roi, roi, mask=mask)  # Background
        roi_fg = cv2.bitwise_and(shirt, shirt, mask=mask_inv)  # Foreground (shirt)
        combined = cv2.add(roi_bg, roi_fg)

        # Place the combined image back onto the frame
        frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2] = combined

App(tk.Tk(),"Virtual Mirror")
