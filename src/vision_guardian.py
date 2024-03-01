import os
import time
import cv2

from src.utils.video_utils import VideoCapture, VideoWriter
from src.utils.text_util import ScreenText


class VisionGuardian:
    def __init__(
        self,
        test_number='Test Default',
        capture_device=0,
        idle_timeout=5,
        full_screen_motion_sensor=None,
        area_motion_sensor=None,
        object_detector=None,
        root_dir=os.path.dirname(os.path.abspath(__file__)),
    ):
        self.test_number = test_number
        self.capture_device = capture_device
        self.idle_timeout = idle_timeout
        self.full_screen_motion_sensor = full_screen_motion_sensor
        self.area_motion_sensor = area_motion_sensor
        self.object_detector = object_detector
        self.root_dir = root_dir

        self.last_detection_time = None
        self.last_id_update_time = time.time()
        self.record = False
        self.motion_list = []

    def _update_motion_status(self, frame):
        if self.full_screen_motion_sensor.motion_detected:
            self.record = True
            self.last_detection_time = time.time()
            message = 'Motion Detected'
        elif self.last_detection_time and (time.time() - self.last_detection_time) < self.idle_timeout - 1:
            message = f'Motion Idle: {int(self.idle_timeout - (time.time() - self.last_detection_time))}'
        else:
            self.record = False
            message = 'No Motion Detected'

        cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (114, 45, 0), 1)

        return frame

    def _update_time(self, frame):
        time_running = time.strftime("%Y%m%d_%H%M%S")
        text_adder = ScreenText(frame)
        text_adder.add_text(time_running, first_line=30)
        return frame

    def _update_motion_list(self, frame):
        if self.full_screen_motion_sensor.motion_detected and not self.record:
            self.motion_list.append(time.strftime("%Y%m%d_%H%M%S"))
        for idx, text in enumerate(self.motion_list):
            text_adder = ScreenText(frame)
            text_adder.add_text(text, first_line=200 + 20 * idx)
        return frame

    def _check_key_interrupts(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            return True
        return False

    def _get_output_file(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.test_number}_{timestamp}.mp4"
        return os.path.join(
            self.root_dir,
            'data',
            'recordings',
            filename
        )

    def detect_motion(self):
        with VideoCapture(self.capture_device) as cap:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            _, init_frame = cap.read()
            with VideoWriter(self._get_output_file(), (init_frame.shape[1], init_frame.shape[0])) as writer:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Failed to capture frame.")
                        break
                    # Check for motion and update frame info
                    self.full_screen_motion_sensor.update(frame)
                    self.area_motion_sensor.update(frame)
                    frame = self._update_motion_list(frame)
                    frame = self._update_motion_status(frame)
                    frame = self._update_time(frame)
                    # Object detection
                    if self.object_detector is not None:
                        objects = self.object_detector.detect_objects(frame)
                        frame = self.object_detector.draw_detected_objects(frame, objects)
                    # Frame display and recording    
                    cv2.imshow('Vision Guardian', frame)
                    if self.record:
                        writer.write(frame)
                    # Key interrupts
                    if self._check_key_interrupts():
                        break


if __name__ == '__main__':
    detector = VisionGuardian(
        capture_device=0,
        idle_timeout=5,
        file_prefix='Test 00015',
        ai=False
    )
    detector.detect_motion()