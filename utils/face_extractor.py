import cv2
import numpy as np
import os

class MediaExtractor:
    def __init__(self, margin=0.20):
        """
        margin: Extra padding around the face bounding box (0.20 = 20% extra padding)
        to capture edge-blending artifacts created by generative AI models.
        """
        # Uses OpenCV's built-in face detector (100% reliable, zero external dependencies or API deprecations)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.margin = margin

    def extract_face(self, image):
        """Takes a BGR image array, detects the primary face, and returns a 256x256 cropped face."""
        h, w, _ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return None  # No face detected

        # If multiple faces exist, get the largest detected face (primary subject)
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        fx, fy, fw, fh = faces[0]

        # Add padding margin around the detected bounding box
        x = int(fx - fw * self.margin / 2)
        y = int(fy - fh * self.margin / 2)
        box_w = int(fw * (1 + self.margin))
        box_h = int(fh * (1 + self.margin))

        # Ensure coordinates stay within image boundaries
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + box_w)
        y2 = min(h, y + box_h)

        face_crop = image[y1:y2, x1:x2]
        
        # Standardize resolution for consistent mathematical Transforms
        if face_crop.size > 0:
            return cv2.resize(face_crop, (256, 256))
        return None

    def process_video_frames(self, video_path, max_frames=5):
        """Extracts exactly 1 frame per second from a video file, up to max_frames."""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 24  # Fallback FPS if video metadata is corrupted

        extracted_faces = []
        frame_count = 0
        
        while cap.isOpened() and len(extracted_faces) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Sample exactly one frame every second
            if frame_count % fps == 0:
                face = self.extract_face(frame)
                if face is not None:
                    extracted_faces.append(face)
            frame_count += 1
            
        cap.release()
        return extracted_faces