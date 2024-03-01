import cv2

class VideoCapture:
    def __init__(self, capture_device=0):
        self.cap = cv2.VideoCapture(capture_device)

    def __enter__(self):
        return self.cap

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()

    def read(self):
        return self.cap.read()


class VideoWriter:
    def __init__(self, output_file, frame_shape):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_file, fourcc, 30.0, (frame_shape[0], frame_shape[1]))
        if not self.writer.isOpened():
            print("Error: Could not open video writer.")

    def __enter__(self):
        return self.writer

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.release()
        if exc_type is not None:
            print("Error during video writing:", exc_val)
