# # import cv2
# # from yolo_detector import YoloDetector
# # from tracker import Tracker
# # from global_id_manager import GlobalIDManager
# # from video_processor import VideoProcessor

# # MODEL_PATH = "models\\yolo12m.pt"
# # VIDEO_PATH_1 = "assets\\football copy.mp4"
# # VIDEO_PATH_2 = "assets\\football.mp4"
# # OUTPUT_PATH_1 = "output\\video1_output.mp4"
# # OUTPUT_PATH_2 = "output\\video2_output.mp4"

# # def process_frame(frame, tracker, detector, global_id_manager):
# #     detections = detector.detect(frame)
# #     tracking_ids, boxes = tracker.track(detections, frame)

# #     global_ids = []
# #     for box in boxes:
# #         x1, y1, x2, y2 = map(int, box)
# #         crop = frame[y1:y2, x1:x2]
# #         if crop.size == 0:
# #             global_ids.append(-1)
# #             continue
# #         gid = global_id_manager.assign_global_id(crop)
# #         global_ids.append(gid)

# #     return tracking_ids, boxes, global_ids

# # def draw_annotations(frame, tracking_ids, boxes, global_ids):
# #     for tid, box, gid in zip(tracking_ids, boxes, global_ids):
# #         x1, y1, x2, y2 = map(int, box)
# #         color = (gid * 37 % 256, gid * 17 % 256, gid * 97 % 256)
# #         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
# #         cv2.putText(frame, f"ID: {gid}", (x1, y1 - 10),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
# #     return frame

# # def main():
# #     cap1 = cv2.VideoCapture(VIDEO_PATH_1)
# #     cap2 = cv2.VideoCapture(VIDEO_PATH_2)

# #     detector = YoloDetector(model_path=MODEL_PATH, confidence=0.3)
# #     tracker1 = Tracker()
# #     tracker2 = Tracker()
# #     global_id_manager = GlobalIDManager()

# #     # Read first frames to get size info
# #     ret1, frame1 = cap1.read()
# #     ret2, frame2 = cap2.read()

# #     if not ret1 or not ret2:
# #         print("❌ Error reading one or both video files.")
# #         return

# #     # Get frame sizes for each video separately
# #     h1, w1 = frame1.shape[:2]
# #     h2, w2 = frame2.shape[:2]

# #     # Create two separate video writers for saving outputs
# #     video_writer1 = VideoProcessor(OUTPUT_PATH_1, w1, h1, fps=30)
# #     video_writer2 = VideoProcessor(OUTPUT_PATH_2, w2, h2, fps=30)

# #     # Reset video to first frame
# #     cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
# #     cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

# #     while True:
# #         ret1, frame1 = cap1.read()
# #         ret2, frame2 = cap2.read()

# #         if not ret1 and not ret2:
# #             break

# #         if ret1:
# #             tracking_ids, boxes, global_ids = process_frame(frame1, tracker1, detector, global_id_manager)
# #             frame1 = draw_annotations(frame1, tracking_ids, boxes, global_ids)
# #             cv2.imshow("Video 1", frame1)
# #             video_writer1.write(frame1)

# #         if ret2:
# #             tracking_ids, boxes, global_ids = process_frame(frame2, tracker2, detector, global_id_manager)
# #             frame2 = draw_annotations(frame2, tracking_ids, boxes, global_ids)
# #             cv2.imshow("Video 2", frame2)
# #             video_writer2.write(frame2)

# #         key = cv2.waitKey(1)
# #         if key == ord("q") or key == 27:
# #             break

# #     cap1.release()
# #     cap2.release()
# #     video_writer1.release()
# #     video_writer2.release()
# #     cv2.destroyAllWindows()

# # if __name__ == "__main__":
# #     main()

# import cv2
# from yolo_detector import YoloDetector
# from tracker import Tracker
# from global_id_manager import GlobalIDManager
# from video_processor import VideoProcessor

# MODEL_PATH = "models\\yolo12m.pt"
# VIDEO_PATH_1 = "assets\\football copy.mp4"
# VIDEO_PATH_2 = "assets\\football.mp4"
# OUTPUT_PATH_1 = "output\\video1_output.mp4"
# OUTPUT_PATH_2 = "output\\video2_output.mp4"

# def process_frame(frame, tracker, detector, global_id_manager, camera_id):
#     detections = detector.detect(frame)
#     tracking_ids, boxes = tracker.track(detections, frame)

#     global_ids = []
#     for box in boxes:
#         x1, y1, x2, y2 = map(int, box)
#         crop = frame[y1:y2, x1:x2]
#         if crop.size == 0:
#             global_ids.append(-1)
#             continue
#         # Pass camera_id here to keep track of source camera/video
#         gid = global_id_manager.assign_global_id(crop, camera_id=camera_id)
#         global_ids.append(gid)

#     return tracking_ids, boxes, global_ids

# def draw_annotations(frame, tracking_ids, boxes, global_ids):
#     for tid, box, gid in zip(tracking_ids, boxes, global_ids):
#         x1, y1, x2, y2 = map(int, box)
#         color = (gid * 37 % 256, gid * 17 % 256, gid * 97 % 256)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(frame, f"ID: {gid}", (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#     return frame

# def main():
#     cap1 = cv2.VideoCapture(VIDEO_PATH_1)
#     cap2 = cv2.VideoCapture(VIDEO_PATH_2)

#     detector = YoloDetector(model_path=MODEL_PATH, confidence=0.3)
#     tracker1 = Tracker()
#     tracker2 = Tracker()
#     global_id_manager = GlobalIDManager()

#     # Read first frames to get size info
#     ret1, frame1 = cap1.read()
#     ret2, frame2 = cap2.read()

#     if not ret1 or not ret2:
#         print("❌ Error reading one or both video files.")
#         return

#     # Get frame sizes for each video separately
#     h1, w1 = frame1.shape[:2]
#     h2, w2 = frame2.shape[:2]

#     # Create two separate video writers for saving outputs
#     video_writer1 = VideoProcessor(OUTPUT_PATH_1, w1, h1, fps=30)
#     video_writer2 = VideoProcessor(OUTPUT_PATH_2, w2, h2, fps=30)

#     # Reset video to first frame
#     cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
#     cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

#     while True:
#         ret1, frame1 = cap1.read()
#         ret2, frame2 = cap2.read()

#         if not ret1 and not ret2:
#             break

#         if ret1:
#             # Pass camera_id "cam1"
#             tracking_ids, boxes, global_ids = process_frame(frame1, tracker1, detector, global_id_manager, camera_id="cam1")
#             frame1 = draw_annotations(frame1, tracking_ids, boxes, global_ids)
#             cv2.imshow("Video 1", frame1)
#             video_writer1.write(frame1)

#         if ret2:
#             # Pass camera_id "cam2"
#             tracking_ids, boxes, global_ids = process_frame(frame2, tracker2, detector, global_id_manager, camera_id="cam2")
#             frame2 = draw_annotations(frame2, tracking_ids, boxes, global_ids)
#             cv2.imshow("Video 2", frame2)
#             video_writer2.write(frame2)

#         key = cv2.waitKey(1)
#         if key == ord("q") or key == 27:
#             break

#     cap1.release()
#     cap2.release()
#     video_writer1.release()
#     video_writer2.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

import cv2
from yolo_detector import YoloDetector
from tracker import Tracker
from global_id_manager import GlobalIDManager
from video_processor import VideoProcessor
import os

MODEL_PATH = "models\\yolo12m.pt"
INPUT_VIDEOS = [
    "assets\\Double1.mp4",
    "assets\\football.mp4"
]
OUTPUT_DIR = "output\\"

def process_frame(frame, tracker, detector, global_id_manager, camera_id):
    detections = detector.detect(frame)
    tracking_ids, boxes = tracker.track(detections, frame)

    global_ids = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            global_ids.append(-1)
            continue
        gid = global_id_manager.assign_global_id(crop, camera_id=camera_id)
        global_ids.append(gid)

    return tracking_ids, boxes, global_ids

def draw_annotations(frame, tracking_ids, boxes, global_ids):
    for tid, box, gid in zip(tracking_ids, boxes, global_ids):
        x1, y1, x2, y2 = map(int, box)
        color = (gid * 37 % 256, gid * 17 % 256, gid * 97 % 256)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID: {gid}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

def main():
    detector = YoloDetector(model_path=MODEL_PATH, confidence=0.3)
    global_id_manager = GlobalIDManager()

    caps = []
    trackers = []
    writers = []
    names = []

    # Initialize everything dynamically
    for idx, video_path in enumerate(INPUT_VIDEOS):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()

        if not ret:
            print(f"❌ Error reading video: {video_path}")
            continue

        h, w = frame.shape[:2]
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_output.mp4")

        caps.append(cap)
        trackers.append(Tracker())
        writers.append(VideoProcessor(output_path, w, h, fps=30))
        names.append(f"cam{idx+1}")

        # Reset to first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        all_done = True
        for i in range(len(caps)):
            ret, frame = caps[i].read()
            if not ret:
                continue

            all_done = False
            tracking_ids, boxes, global_ids = process_frame(frame, trackers[i], detector, global_id_manager, camera_id=names[i])
            annotated = draw_annotations(frame, tracking_ids, boxes, global_ids)
            cv2.imshow(f"Video {i+1}", annotated)
            writers[i].write(annotated)

        if all_done:
            break

        key = cv2.waitKey(1)
        if key == ord("q") or key == 27:
            break

    # Release everything
    for cap in caps:
        cap.release()
    for writer in writers:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
