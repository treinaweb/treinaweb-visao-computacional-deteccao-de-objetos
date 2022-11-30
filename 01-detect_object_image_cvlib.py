import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2

imagem = cv2.imread("Img/image4.jpg")

bbox, label, conf = cv.detect_common_objects(imagem)

print(bbox, label, conf)

out = draw_bbox(imagem, bbox, label, conf)

cv2.imshow("Object Detection", out)
cv2.waitKey()

cv2.imwrite("Img/image4_detect.jpg", out)

cv2.destroyAllWindows()