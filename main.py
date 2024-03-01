import os

from src.utils.motion_sensor import MotionSensor
from src.utils.object_detector import ObjectDetector
from src.vision_guardian import VisionGuardian


def main(root_dir=os.path.dirname(os.path.abspath(__file__))):
    # Set up object detector
    ai = True
    object_detector = ObjectDetector(root_dir) if ai else None

    # Set up motion regions. If not region is defined, the entire frame is used.
    full_screen_motion_sensor = MotionSensor(
        motion_threshold=100
    )
    area_motion_sensor = MotionSensor(
        motion_threshold=175,
        roi=(450, 250, 550, 50)
    )

    detector = VisionGuardian(
        test_number='Test',
        capture_device=0,
        idle_timeout=5,
        full_screen_motion_sensor=full_screen_motion_sensor,
        area_motion_sensor=area_motion_sensor,
        object_detector=object_detector,
        root_dir=root_dir,
    )

    detector.detect_motion()


if __name__ == '__main__':
    main()