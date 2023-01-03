import numpy as np
import cv2

center = (200, 200)  # x,y
axes = (100, 75)  # first, second
angle = 0.  # clockwise, first axis, starts horizontal
for i in range(360):
    image = np.zeros((400, 400, 3))  # creates a black image
    image = cv2.ellipse(image, center, axes, angle, 0., 360, (0, 0, 255))
    image = cv2.ellipse(image, center, axes, angle, 0., i, (0, 255, 0))
    cv2.imshow("image", image)
    cv2.waitKey(5)

cv2.waitKey(0)
cv2.destroyAllWindows()
