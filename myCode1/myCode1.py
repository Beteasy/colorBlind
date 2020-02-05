import cv2
import numpy as np
import time

# 高斯滤波核大小
blur_ksize = 5

#Canny标远检测高低阈值
canny_lth = 50
canny_hth = 150

def process_an_image(img):

    # Lower = np.array([0, 43, 46])
    # Upper = np.array([25, 255, 255])
    Lower = np.array([0, 43, 63])
    Upper = np.array([20, 255, 255])
    changeColor(img, Lower, Upper)
    # LowerGreen = np.array([35,43,46])
    # UpperGreen = np.array([77,255,255])
    # changeColor(img,LowerGreen, UpperGreen)
    return

def changeColor(img, Lower, Upper):
    # img = cv2.GaussianBlur(img, (11, 11), 0)
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # mask是把HSV图片中在颜色范围内的区域变成白色其他区域变成黑色
    mask = cv2.inRange(HSV, Lower, Upper)
    # cv2.imshow("mask", mask)
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
    # # 进行OTUS二值化
    # ret, th = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow("th", th)
    # 进行边缘检测
    # edges = cv2.Canny(th, canny_lth, canny_hth)
    # cv2.imshow("edges", edges)
    erosion = cv2.erode(mask, (5, 5), iterations=1)
    erosion = cv2.erode(erosion, (5, 5), iterations=1)
    dilation = cv2.dilate(erosion, (5, 5), iterations=1)
    dilation = cv2.dilate(dilation, (5, 5), iterations=1)
    # cv2.imshow("dilation", dilation)
    target = cv2.bitwise_and(img, img, mask=dilation)
    cv2.imshow("target", target)
    # 轮廓检测
    # contours, hierarchy = cv2.findContours(dilation, cv2.RETR_CCOMP, 2)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # 找到面积最大的轮廓
        c = max(contours, key=cv2.contourArea)
        # 使用最小外接圆圈出面积最大的轮廓
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        # 计算轮廓的矩
        # M = cv2.moments(c)
        # 计算轮廓的重心
        # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # 只处理尺寸足够大的轮廓
        if radius > 5:
            # 画出最小外接圆
            cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            # 画出重心
            # cv2.circle(img, center, 5, (0, 0, 255), -1)
            # cv2.drawContours(img, contours, c, (255, 0, 0), -1)
    cv2.drawContours(img, contours, -1, (255, 0, 0), -1)
    cv2.imshow("img", img)
    end = time.clock()
    print(end - start)
    return

if __name__ == "__main__":
    start = time.clock()
    # img = cv2.imread("rabbit_red.jpg")
    # img = cv2.imread(("img3.png"))

    cap = cv2.VideoCapture(0)
    while(1):
        ret, frame = cap.read()
        process_an_image(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break