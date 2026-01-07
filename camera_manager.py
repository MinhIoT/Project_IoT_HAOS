import cv2
import threading
import time

class CameraManager:
    def __init__(self, rtsp_url):
        self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame = None
        self.running = True

        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
            else:
                time.sleep(0.5)

    def get_frame(self):
        return self.frame

    def release(self):
        self.running = False
        self.cap.release()
