import cv2
import numpy as np

cam = cv2.VideoCapture(1)

# lower_orange = (100, 200, 200)
# upper_orange = (140, 255, 255)
#
# lower_green = (30, 80, 80)
# upper_green = (70, 255, 255)
#
# lower_blue = (0, 180, 55)
# upper_blue = (20, 255, 200)


def check_color(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    counter = 1
    sum_hue = 0
    for i in range(50, width-50, 20):
        for j in range(50, height-50, 20):
            print(hsv[j, i, 0])
            print(hls[j, i, 0])
            sum_hue = sum_hue + (hls[j, i, 0]*2)
            sum_hue = sum_hue + (hsv[j, i, 0]*2)
            counter += 1
    hue = sum_hue / counter


while True:
    _, image = cam.read()

    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(img_hsv)
    print(h, s, v)
    # for hue in h:
    #     print(hue)

    h = cv2.inRange(h, 8, 20)
    orange = cv2.bitwise_and(img_hsv, img_hsv, mask=h)
    orange = cv2.cvtColor(orange, cv2.COLOR_HSV2BGR)

    # img_mask = cv2.inRange(img_hsv, lower_orange, upper_orange)
    # # img_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
    # img_result = cv2.bitwise_and(image, image, mask=img_mask)

    print(img_hsv)
    cv2.imshow('image', image)

    if cv2.waitKey(1) >= 0:
        break
