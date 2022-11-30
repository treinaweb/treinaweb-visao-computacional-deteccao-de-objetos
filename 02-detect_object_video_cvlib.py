import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2

webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Webcam não funciona")
    exit()

while webcam.isOpened():
    status, frame = webcam.read()

    if not status:
        print("Frame não foi lido")
        exit()

    bbox, label, conf= cv.detect_common_objects(frame, enable_gpu=True)

    print(bbox, label, conf)

    out = draw_bbox(frame, bbox, label, conf)

    cv2.imshow("Real Time Object Detection", out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()