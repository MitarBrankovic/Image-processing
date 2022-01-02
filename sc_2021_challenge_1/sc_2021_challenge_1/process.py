# import libraries here

from PIL.Image import merge
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

def count_cars(image_path):
    """
    Procedura prima putanju do fotografije i vraca broj prebrojanih automobila. Koristiti ovu putanju koja vec dolazi
    kroz argument procedure i ne hardkodirati nove putanje u kodu.

    Ova procedura se poziva automatski iz main procedure i taj deo koda nije potrebno menjati niti implementirati.

    :param image_path: <String> Putanja do ulazne fotografije.
    :return: <int>  Broj prebrojanih automobila
    """
    car_count = 0
    # TODO - Prebrojati auta i vratiti njihov broj kao povratnu vrednost ove procedure

    car_count = countCars(image_path)

    return car_count





def countCars(image_path):
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


    image = cv2.imread(image_path)
    original = image.copy()
    height, width, channel = image.shape
    if width > 2500 and height > 2500:
        image = decreaseImage(image)

    if width < 350 and height < 350:
        image = increaseImage(image)
        #return funkcijaZaMaleSlike(image)

    image_copy = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image)
    
    ret, img_bin = cv2.threshold(s, 0, 255, cv2.THRESH_OTSU)
    #ret, img_bin = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_bin = cv2.dilate(img_bin, kernel_cross, iterations=5)
    img_bin = cv2.erode(img_bin, kernel_cross, iterations=3)
    img_bin = cv2.dilate(img_bin, kernel_ellipse, iterations=2)
    #img_bin = cv2.erode(img_bin, kernel_cross, iterations=3)

    img_bin_opening = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel_cross)
    img_bin_eroded= cv2.dilate(img_bin_opening, kernel_cross, iterations=2)
    
     
    #https://python.hotexamples.com/examples/cv2/-/floodFill/python-floodfill-function-examples.html
    height, width = img_bin_eroded.shape[:2]
    img_for_floodfill = img_bin_eroded.copy()
    mask = np.zeros((height+2, width+2), np.uint8)
    cv2.floodFill(img_for_floodfill, mask, (0,0), 255)
    floodfill_inverted = cv2.bitwise_not(img_for_floodfill)
    img_floodfilled = img_bin_eroded | floodfill_inverted
    ###############


    #https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Watershed_Algorithm_Marker_Based_Segmentation_2.php
    dist_transform = cv2.distanceTransform(img_floodfilled, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    ###############

    img_cont, contours, hierarchy = cv2.findContours(sure_fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
    cv2.drawContours(img_cont, contours, -1, (255, 0, 0), 1)
    img_final, new_contours, area = removeInsideContours(image_copy, contours, hierarchy)
    
    broj = len(new_contours)


    plt.imshow(original)
    #plt.show()
    plt.imshow(image_copy)
    #plt.show()
    return broj


#def smallImg(image):
    




def funcSmallImages(image):
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    image = increaseImage(image)
    image_copy = image.copy()
    img_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_t = 0 - img_gs
    ret, img_bin = cv2.threshold(img_t, 0, 255, cv2.THRESH_OTSU)
    img_bin = 255 - img_bin

    ret, img_bin2 = cv2.threshold(img_bin, 0, 255, cv2.THRESH_OTSU)

    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    ret, drugi = cv2.threshold(s, 0, 255, cv2.THRESH_OTSU)

    # height, width = img_bin.shape[:2]
    # img_for_floodfill = img_bin.copy()
    # mask = np.zeros((height+2, width+2), np.uint8)
    # cv2.floodFill(img_for_floodfill, mask, (0,0), 255)
    # floodfill_inverted = cv2.bitwise_not(img_for_floodfill)
    # img_floodfilled = img_bin | floodfill_inverted

    imga = cv2.erode(img_bin2, kernel_cross, iterations=2)
    imga = cv2.erode(imga, kernel_ellipse, iterations=1)
    imga = cv2.dilate(imga, kernel_cross, iterations=1)

    img_cont, contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
    cv2.drawContours(img_cont, contours, -1, (255, 0, 0), 1)
    img_contoured_final, new_contours, wbc_area = removeInsideContours(image_copy, contours, hierarchy)
    plt.imshow(imga, 'gray')
    plt.show()
    plt.imshow(image_copy)
    plt.show()

    return len(new_contours)

#340/358 daje najbolji procenat iz nekog razloga
def increaseImage(img):
    scale_percent = 340
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dsize = (width, height)
    return cv2.resize(img, dsize)


def decreaseImage(img):
    scale_percent = 50
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dsize = (width, height)
    return cv2.resize(img, dsize)


#160 daje najbolji rezultat
def removeInsideContours(img, contours, hierarchy):
    i = 0
    new_contours = []
    cells_area = 0.0
    for contour in contours:
        if hierarchy[0, i, 3] == -1 and cv2.contourArea(contour) > 160.0:
            new_contours.append(contour)
            cells_area = cells_area + cv2.contourArea(contour)
        i = i + 1

    cv2.drawContours(img, new_contours, -1, (255, 0, 0), 1)
    return img, new_contours, cells_area


def calculateEdge(img):
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.dilate(img, kernel_ellipse, iterations=1) - cv2.erode(img, kernel_ellipse, iterations=1)
    return img
