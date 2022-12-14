# coding: utf-8
import cv2

img = cv2.imread("test.png", cv2.IMREAD_COLOR)
w = img.shape[1]
h = img.shape[0]
print(f"宽度：{w}, 高度：{h}")
img = cv2.resize(img, (960, 540))


# print img.shape
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        print(x, y)
        print(img[y, x])
        # cv2.putText(img, img[x, y], (x, y), cv2.FONT_HERSHEY_PLAIN,
        #             1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)


cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img)

while (True):
    try:
        cv2.waitKey(100)
    except Exception:
        cv2.destroyAllWindows()
        break

cv2.waitKey(0)
cv2.destroyAllWindows()