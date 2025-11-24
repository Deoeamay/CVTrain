import cv2
import numpy as np

image = cv2.imread('1_1.png')

# ------ 确定边缘 ------

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # canny检测最好用灰度图
blurred_image = cv2.GaussianBlur(gray_image, (11, 11), 0)
edges = cv2.Canny(blurred_image, 75, 160)

contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] # [0]实现了只获取轮廓列表
balls = {"centers": [], "contours": [], "radius": []}
for c in contours:
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        ((_, _), radius) = cv2.minEnclosingCircle(c)
        radius = int(radius)
        balls["centers"].append((cX, cY))
        balls["contours"].append(c)
        balls["radius"].append(radius)

# ------确定颜色 ------

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

color_ranges = {
    'Green': ([26, 48, 0], [96, 255, 255]),
    'Blue': ([89, 72, 0], [179, 255, 255]),
    'Yellow': ([13, 72, 0], [24, 255, 255]),
    'Pink': ([0, 23, 127], [82, 92, 255]),
    'Black': ([38, 0, 0], [127, 60, 115]),
    'Brown': ([0, 48, 0], [13, 107, 128]) ,
}

# ------ 绘图 ------

gray_mask = np.zeros_like(gray_image) # 生成与原来灰度图尺寸一样大的纯黑掩膜

for i in range(len(balls['centers'])):
    cX, cY = balls['centers'][i]
    c = balls['contours'][i]

    gray_mask.fill(0)  # 每次循环前清空掩膜
    cv2.drawContours(gray_mask, [c], -1, 255, -1)

    masked_hsv = cv2.bitwise_and(hsv_image, hsv_image, mask=gray_mask) # 只分析轮廓内的像素

    best_color = 'Unknown'
    max_count = 15

    for color_name, (lower, upper) in color_ranges.items():
            lower_bound = np.array(lower, dtype=np.uint8)
            upper_bound = np.array(upper, dtype=np.uint8)

            color_mask = cv2.inRange(masked_hsv, lower_bound, upper_bound)
            
            pixel_count = np.count_nonzero(color_mask)
            
            if pixel_count > max_count:
                max_count = pixel_count
                best_color = color_name

    cv2.circle(image, (cX, cY), balls["radius"][i], (0, 255, 0), 2) # 绘制轮廓
    #cv2.drawContours(image, [c], -1, (0, 255, 0), 2) # 绘制轮廓
    cv2.putText(image, best_color, (cX - 20, cY - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) # 标注颜色
    cv2.circle(image, (cX, cY), 3, (0, 0, 255), -1) # 标注中心

cv2.imshow('Answer', image)
cv2.waitKey(0)
cv2.destroyAllWindows()