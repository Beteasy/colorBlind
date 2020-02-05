import cv2
import numpy as np
import time

# cap = cv2.VideoCapture('C:\\Users\\hp\\Desktop\\pythonfile\\课本原件.mp4')
cap = cv2.VideoCapture(0)
numFrame = 0
j = 0
while True:
    if cap.grab():
        numFrame += 1

        flag, frame = cap.retrieve()
        if not flag:
            continue
        else:
            #cv2.imshow('video', frame)
            newPath = 'C:\\Users\\hp\\Desktop\\pythonfile\\new5' + "\\" + str(int(numFrame/1)) + ".jpg"
            cv2.imencode('.jpg', frame)[1].tofile(newPath)

            if __name__ == '__main__':
                Img = cv2.imread('C:\\Users\\hp\\Desktop\\pythonfile\\new5' + "\\" + str(int(numFrame/1)) + ".jpg")  # 读入一幅图像
                cv2.imshow('Img', Img)
                kernel_2 = np.ones((2, 2), np.uint8)  # 2x2的卷积核
                kernel_3 = np.ones((3, 3), np.uint8)  # 3x3的卷积核
                kernel_4 = np.ones((4, 4), np.uint8)  # 4x4的卷积核
                if Img is not None:  # 判断图片是否读入
                    HSV = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)  # 把BGR图像转换为HSV格式
                    '''
                    HSV模型中颜色的参数分别是：色调（H），饱和度（S），明度（V）
                    下面两个值（Lower & Upper）是要识别的颜色范围
                    '''
                    Lower = np.array([50, 18, 80])  # 要识别颜色的下限
                    Upper = np.array([150, 150, 235])  # 要识别的颜色的上限
                    # mask是把HSV图片中在颜色范围内的区域变成白色其他区域变成黑色
                    mask = cv2.inRange(HSV, Lower, Upper)
                    # 下面四行是用卷积进行滤波
                    erosion = cv2.erode(mask, kernel_4, iterations=1)
                    erosion = cv2.erode(erosion, kernel_4, iterations=1)
                    dilation = cv2.dilate(erosion, kernel_4, iterations=1)
                    dilation = cv2.dilate(dilation, kernel_4, iterations=1)
                    # target是把原图中的非目标颜色区域去掉剩下的图像
                    target = cv2.bitwise_and(Img, Img, mask=dilation)
                    #  将滤波后的图像变成二值图像放在binary中
                    ret, binary = cv2.threshold(dilation, 127, 255, cv2.THRESH_BINARY)
                    # 在binary中发现轮廓，轮廓按照面积从小到大排列
                    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    '''Area = 0
                    s = 0
                    for i in contours:  # 遍历所有的轮廓
                        Area = max(Area, cv2.contourArea(i))  # 获取最大形状面积

                    p = 0
                    for i in contours:  # 遍历所有的轮廓
                        if (Area == cv2.contourArea(i)):  # 将小于最大面积1/10的图像过滤掉
                            # cv2.drawContours(Img, i, -1, (0, 255,), 10)
                            cv2.drawContours(Img, i, -1, (0, 255,), 20)
                    cv2.imshow('Img', Img)
                    '''
                    s = 0
                    for i in contours:                                                      # 遍历所有的轮廓
                        x, y, w, h = cv2.boundingRect(i)                                    # 将轮廓分解为识别对象的左上角坐标和宽、高
                        s = max(s, w * h)                                                   # 获取最大形状面积

                    p = 0
                    for i in contours:                                                      # 遍历所有的轮廓
                        x, y, w, h = cv2.boundingRect(i)                                    # 将轮廓分解为识别对象的左上角坐标和宽、高
                        if( w * h == s ):                                                # 将小于最大面积1/10的图像过滤掉
                            cv2.drawContours(Img, i, -1, (0, 255,), 20)
            cv2.imshow('Img', Img)
        #检测到按下Esc时，break（和imshow配合使用）
    if cv2.waitKey(10) == 1500:
        break
    j += 1