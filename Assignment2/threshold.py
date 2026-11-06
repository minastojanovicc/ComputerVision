import cv2
import matplotlib.pyplot as plt
import numpy as np

def nothing(x): pass

def plot_image(img, color_map=None):
    plt.imshow(img, cmap=color_map)
    plt.show()

img = cv2.imread("./coins/coins.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plot_image(img)

img = cv2.GaussianBlur(img, (5, 5), 10)

grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.namedWindow("Threshold")
cv2.createTrackbar("T", "Threshold", 0, 255, nothing)

while True:
    t = cv2.getTrackbarPos("T", "Threshold")
    _, tmp = cv2.threshold(grayscale_img, t, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Threshold", tmp)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()