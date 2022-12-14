import cv2 as cv
from skimage import io
import matplotlib.pyplot as plt
from numpy import *
import numpy as np

def Hoff(path):
    print('start to Hoff')
    try:
        image = io.imread(path)
        print(image.shape[0])

        whole_area=image.shape[0]*image.shape[1]
        print(f"whole area is {whole_area}")

        area11 = []
        mask = np.zeros(image.shape[0:2], np.uint8)
        mask1 = np.zeros(image.shape[0:2], np.uint8)

        result = cv.medianBlur(image, 3)
        # io.imshow(result)
        # io.show()

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)
        # io.imshow(binary)
        # io.show()

        edge = cv.Canny(image, 180, 255)
        # io.imshow(edge)
        # io.show()
        lines = cv.HoughLinesP(edge, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(mask, (x1, y1), (x2, y2), (255, 255, 255), 1)
        # io.imshow(mask)
        # io.show()

        kernel = np.ones((18, 18), np.uint8)
        kernel_1 = np.ones((76, 76), np.uint8)
        closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel_1)

        # io.imshow(closing)
        # io.show()
        # io.imshow(opening)
        # io.show()

        opening_x = opening.shape[0]
        opening_y = opening.shape[1]
        opening[:, 0] = 0
        opening[:, opening_y - 1] = 0
        opening[0, :] = 0
        opening[opening_x - 1, :] = 0


        # io.imshow(opening)
        # io.show()

        contours, h = cv.findContours(opening, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for i, cont in enumerate(contours):
            cv.drawContours(mask1, contours, i, (255, 255, 67), 3)
            area = cv.contourArea(cont)
            area11.append(area)

        area11.sort()
        print('start to sort')
        print(area11)
        # if len(area)==0:
        #     return 0.1


        # io.imshow(mask1)
        # io.show()
        print('calculate percentage')
        if len(area11)==0:
            return 0.05
        percentage=area11[-1]/whole_area

        return percentage
    except:
        return 0.01
    

    # cv.waitKey(0)
    # cv.destroyAllWindows()

if __name__=='__main__':
    Hoff('/home/surf/jlb/2322.jpg')