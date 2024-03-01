import cv2

class MotionSensor:
    def __init__(self, motion_threshold=100, roi=None):
        self.motion_threshold = motion_threshold
        self.motion_detected = False
        self.prev_frame = None
        self.roi = roi

    def update(self, frame):
        gray_roi = self._extract_roi(frame)
        self._draw_rectangle(frame)

        if self.prev_frame is not None:
            self._detect_motion(gray_roi)
        else:
            self.prev_frame = gray_roi.copy()

    def _extract_roi(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        if self.roi is None:
            self.roi = (0, 0, width, height)

        x, y, w, h = self.roi
        return gray[y:y+h, x:x+w]

    def _detect_motion(self, gray_roi):
        frame_diff = cv2.absdiff(gray_roi, self.prev_frame)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > self.motion_threshold:
                self.motion_detected = True
                self.prev_frame = gray_roi.copy()
                return
        self.motion_detected = False
        self.prev_frame = gray_roi.copy()

    def _draw_rectangle(self, frame):
        x, y, w, h = self.roi
        color = (0, 0, 255) if self.motion_detected else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
