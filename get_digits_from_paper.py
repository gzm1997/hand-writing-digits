import cv2
from PIL import Image


#读取图片为rgb数组，用作第一次轮廓检测
im = cv2.imread('input.jpg')
#转为灰度矩阵，用以二值化
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#进行二值化
ret, thresh = cv2.threshold(imgray, 169, 255, 0)

#展示二值化后的图像
Image.fromarray(thresh).show()

#使用进行了二值化的矩阵进行轮廓检测
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#在原图中画出轮廓，画迹宽度为3
cv2.drawContours(im, contours, -1, (0,255,0), 3)
#对画出了轮廓了的图像矩阵进行第二次轮廓检测(这样检测出来的轮廓才是连续的)
#转为灰度矩阵
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#第二次二值化
ret, thresh = cv2.threshold(imgray, 169, 255, 0)
#第二次轮廓检测，此时获得的轮廓才是连续的，第一次获得的是断断续续的
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#去掉第一个轮廓，那时整张A4纸的边框
contours = contours[1:]

#新打开一次原图用作圈出轮廓
img = cv2.imread("input.jpg")
#对每个轮廓
index = 0
for c in contours:
	#获取每个轮廓被包围的矩形的横纵坐标和宽高
    x, y, w, h = cv2.boundingRect(c)
    if h < 45 or h > 160:
    	continue
    #画出包围每个轮廓的矩形
    #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #展示每个被圈住的轮廓
    Image.fromarray(thresh[y:y+h,x:x+w]).save("digits/" + str(index) + ".png")
    index += 1
 
#展示画出包围轮廓的矩形的图像
cv2.imshow("Contours", img)

# 等待键盘输入
cv2.waitKey(0)
#关闭展示窗口
cv2.destroyAllWindows()


