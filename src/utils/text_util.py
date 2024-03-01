import cv2
import numpy as np


class ScreenText:
    def __init__(self, frame):
        self.frame = frame

    def add_text(self, text, first_line):
        alpha = 0.25
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        x = self.frame.shape[1] - text_size[0] - 20
        y = first_line - 15
        w = text_size[0] + 20
        h = text_size[1] + 10
        region = self.frame[y:y+h, x:x+w]
        self.frame[y:y+h, x:x+w] = self._set_region_background(region, alpha)
        text_x = self.frame.shape[1] - text_size[0] - 10
        text_y = first_line
        cv2.putText(self.frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def _set_region_background(self, region, alpha):
        black_rect = np.zeros(region.shape, dtype=np.uint8)
        return cv2.addWeighted(region, alpha, black_rect, 0.5, 1.0)