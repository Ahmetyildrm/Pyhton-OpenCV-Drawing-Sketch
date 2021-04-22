import cv2
import numpy as np

def distance(P1, P2):

    return ((P1[0] - P2[0])**2 + (P1[1] - P2[1])**2) ** 0.5

def optimized_path(coords, start=None):
    if start is None:
        start = coords[0]
    pass_by = coords
    path = [start]
    pass_by.remove(start)
    while pass_by:
        nearest = min(pass_by, key=lambda x: distance(path[-1], x))
        path.append(nearest)
        pass_by.remove(nearest)
    return path  #

def stackImages(imgArray, scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    return ver

img = cv2.imread("Images/berkay.jpeg")
# 1. YOL
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 3)
edges = cv2.Canny(blur, 100, 200)

# 2. YOL
thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
diff = cv2.absdiff(dilate, thresh)

ret, labels = cv2.connectedComponents(edges) # diff

canvas = np.zeros(img.shape, np.uint8)
cnt = 0

imageArray = ([img, gray, thresh], [dilate, diff, edges])
stackedImage = stackImages(imageArray, 0.3)
cv2.imshow('Stacked Images', stackedImage)
cv2.waitKey(0)
while True:
    all_coords = []
    for i in range(1, ret):
        print("%", round((100*i/ret), 1))
        pts = np.where(labels == i)
        coords = list(zip(*pts))
        start = (coords[0][0], coords[0][1])
        coords = optimized_path(coords, start)
        all_coords.append(coords)
    print("Hesaplama Bitti")
    inpt = input("Bekliyorum")
    for i in all_coords:
        cnt += 1
        print("%", round((100 * cnt / len(all_coords)), 1))
        for c in i:
            cv2.circle(canvas, (c[1], c[0]), 1, (255, 255, 255))
            imageArray = ([canvas])
            stackedImage = stackImages(imageArray, 0.7)
            cv2.putText(canvas, "Ahmet Yildirim", (canvas.shape[1] - 120, canvas.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 165, 255), 2)
            cv2.imshow('Stacked Images', stackedImage)
            cv2.waitKey(1)
    print("Çizim Bitti")



