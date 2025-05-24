import cv2

class VideoProcessor:
    def __init__(self, output_path, frame_width, frame_height, fps=30):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
