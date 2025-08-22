
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255  for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)

img1=cv2.imread("/content/drive/MyDrive/yolov7-main/yolov7-main/inference/images/image3.jpg")
original = Image.open("/content/drive/MyDrive/yolov7-main/yolov7-main/inference/images/image3.jpg")

gamma_list=[0.7,1,2.5]
plt.figure(figsize=(15,20))
for i,gamma in enumerate(gamma_list):
  adjusted = adjust_gamma(img1, gamma=gamma)
  adjusted=cv2.cvtColor(adjusted,cv2.COLOR_BGR2RGB)
  cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 4)
  plt.subplot(1,3,i+1)
  plt.axis("off")
  plt.imshow(adjusted)
