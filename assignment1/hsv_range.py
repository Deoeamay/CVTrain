import cv2
import numpy as np

'''
本模块可以轻松复用，用于观察所需分割颜色在HSV空间中的范围，从而确定阈值
'''

image = cv2.imread('1_1.png')
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

cv2.namedWindow('Trackbars')
# 定义crewatTrackbar回调函数
def nothing(x):
    pass

cv2.createTrackbar("H_MIN", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("H_MAX", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("S_MIN", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("S_MAX", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("V_MIN", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("V_MAX", "Trackbars", 255, 255, nothing)

while True:
    h_min = cv2.getTrackbarPos("H_MIN", "Trackbars")
    h_max = cv2.getTrackbarPos("H_MAX", "Trackbars")
    s_min = cv2.getTrackbarPos("S_MIN", "Trackbars")
    s_max = cv2.getTrackbarPos("S_MAX", "Trackbars")
    v_min = cv2.getTrackbarPos("V_MIN", "Trackbars")
    v_max = cv2.getTrackbarPos("V_MAX", "Trackbars")

    h_min = min(h_max, h_min)
    s_min = min(s_max, s_min)
    v_min = min(v_max, v_min)

    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv_image, lower_bound, upper_bound) # 构建掩膜，筛选出指定颜色范围内的部分
    result = cv2.bitwise_and(image, image, mask=mask) # 按位操作，传入同一张图片，实现了掩膜下的图像显示原色彩而非白色

    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)

    # 按q退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break