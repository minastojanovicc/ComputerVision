import cv2
import matplotlib.pyplot as plt

def nothing(x):
   pass

# Učitavanje slike
img = cv2.imread("./coins/coins.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# HSV konverzija
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
s_channel = hsv[:, :, 1]

# OpenCV prozor
cv2.namedWindow("S channel threshold")

# Trackbar za prag
cv2.createTrackbar("S threshold", "S channel threshold", 0, 255, nothing)

while True:
   thresh = cv2.getTrackbarPos("S threshold", "S channel threshold")

   # Threshold nad S kanalom
   _, mask = cv2.threshold(s_channel, thresh, 255, cv2.THRESH_BINARY)

   # Prikaz
   cv2.imshow("S channel", s_channel)
   cv2.imshow("S channel threshold", mask)

   key = cv2.waitKey(1) & 0xFF
   if key == 27:  # ESC za izlaz
       print(f"Izabrani prag za Saturation kanal: {thresh}")
       break

cv2.destroyAllWindows()