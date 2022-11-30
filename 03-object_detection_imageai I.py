from imageai.Detection import ObjectDetection
import os

path_execucao = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(path_execucao, "model/yolo.h5"))
detector.loadModel()

deteccoes = detector.detectObjectsFromImage(
    input_image=os.path.join(path_execucao, "Img/image4.jpg"),
    output_image_path=os.path.join(path_execucao, "Img/image4_detect.jpg"),
    minimum_percentage_probability=40
)

print(deteccoes)