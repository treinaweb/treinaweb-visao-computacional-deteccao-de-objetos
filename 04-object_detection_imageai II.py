from imageai.Detection import VideoObjectDetection
import os
import cv2

path_execucao = os.getcwd()

camera = cv2.VideoCapture(0)

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(path_execucao, "model/yolo.h5"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(
    camera_input=camera,
    output_file_path=os.path.join(path_execucao, "Img/camera_detected"),
    frames_per_second=20,
    log_progress=True,
    minimum_percentage_probability=30
)

print(video_path)