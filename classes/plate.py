#!/usr/bin/python3
import re, time, sys, os
import cv2
import imutils
import numpy as np
from PIL import Image
# KERAS ocr
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model

# DISABLE Tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# SETUP GPU MEMORY GROWTH
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

# To ENABLE CPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

class plateUtils:

    @staticmethod
    def toGrayscale(frame, width=None):
        if width:
            frame = imutils.resize(frame, width=480)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to grey scale

    @staticmethod
    def grayToBinary(gray, inverse=False):
        if inverse:
            flags = cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV
        else:
            flags = cv2.THRESH_OTSU + cv2.THRESH_BINARY
        ret, threshold = cv2.threshold(gray, 127, 255, flags)
        return threshold

    @staticmethod
    def dilate(frame):
        ## Applied dilation 
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.morphologyEx(frame, cv2.MORPH_DILATE, kernel3)
        return dilated

    @staticmethod
    def erode(frame):
        ## Applied dilation 
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = cv2.morphologyEx(frame, cv2.MORPH_ERODE, kernel3)
        return eroded

    @staticmethod
    def sort_contours(cnts, reverse = False):
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
        return cnts

    @staticmethod
    def readImage(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        # img = cv2.resize(img, (620,480) )
        return img

    @staticmethod
    def readGrayImage(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # img = cv2.resize(img, (620,480) )
        return img

    @staticmethod
    def getLetterImages(gray, knownText=None):
        letters = []
        threshold = plateUtils.grayToBinary(gray, True)
        # predict letter locations
        cont, _  = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # define standard width and height of the final character set
        digit_w, digit_h = 32, 32
        finalRatio = digit_w / digit_h

        # index to find letter in plate
        contourCtr = 0
        for c in plateUtils.sort_contours(cont):
            # GET ROI coordinates
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h/w
            letterValue = ""
            # FILTER by h/w ratio
            if 1<=ratio<=5.1: # Only select contour with defined ratio
                # FILTER by complete letter/plate height ratio and min-width
                if h/gray.shape[0]>=0.3 and w > 12:
                    # Fill and resize character image to final ratio
                    if ratio > 1.0:
                        missingPixels = int((finalRatio*h - w)/2)
                        letter = np.zeros((h, w + missingPixels * 2), dtype=np.uint8)
                        letter[0:h, missingPixels:w+missingPixels] = threshold[y:y+h, x:x+w]
                    else:
                        letter = threshold[y:y+h,x:x+w]

                    # erode chars
                    letter = plateUtils.erode(letter)

                    if knownText:
                        try:
                            letterValue = knownText[contourCtr]
                        except:
                            # invalid contour found
                            print("Error character overflow")
                            continue

                    # letter image to final size
                    letter = cv2.resize(letter, dsize=(digit_w, digit_h))
                    letters.append({'img':letter, 'value': letterValue})
                    contourCtr += 1

        return letters

class PlateExtractor:

    platePattern = '[0-8][0-9] ?[A-Z]{1,3} ?[0-9]{2,4}'
    plateRegex = re.compile(platePattern)
    sampleCounter = 0
    letters = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
    '8': 8, '9': 9, '10': 'A', '11': 'B', '12': 'C', '13': 'D', '14': 'E',
    '15': 'F', '16': 'G', '17': 'H', '18': 'I', '19': 'J', '20': 'K', '21': 'L', 
    '22': 'M', '23': 'N', '24': 'O', '25': 'P', '26': 'R', '27': 'S', '28': 'T', 
    '29': 'U', '30': 'V', '31': 'Y', '32': 'Z'}

    def extractPlatesFromOCRResult(self, ocrResult):
        selection = []
        matches = self.plateRegex.findall(ocrResult)
        for match in matches:
            match = match.replace(" ", "")
            if len(match) >= 7 or len(match) <= 9:
                selection.append(match)
        return selection

    def __init__(self):
    
        # Define Path
        model_path = './models/conv-5-3-32-maxpool.h5'
        model_weights_path = './models/conv-5-3-32-maxpool-weights.h5'
        # Load the pre-trained models
        self.model = load_model(model_path)
        self.model.load_weights(model_weights_path)
        # Rectangles that fill sobel contours to locate plates - important!
        self.sobel_fill_dims = (30,11)
        # Minimum Plate area
        self.minPlateArea = 500

        # Define letter image dimensions
        self.img_width, self.img_height = 32, 32

    def isPlateRectangle(self, contour):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
        cntrArea = cv2.contourArea(contour)
        if cntrArea < self.minPlateArea:
            return False
        # check if the contour has 4 points (for a vehicle plate)
        if len(approx) == 4:
            x,y,w,h = cv2.boundingRect(approx)
            if w/h > 2.5 and w/h < 5.5:
                return approx
            # print(w/h)
        return False

    def auto_canny(self, image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)
    
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
    
        # return the edged image
        return edged

    def getTop3(self, array):
        result = array[0]
        # print('Result', result)
        # top1 = np.argmax(result)

        top1 = None
        top2 = None
        top3 = None

        for i, prob in enumerate(result):
            prob = prob.numpy()
            # print('PROB: ', prob, i)
            if top1 is None:
                top1 = (i, prob)
            elif prob > top1[1]:
                top2 = top1
                top1 = (i, prob)
            elif top2 is None or prob > top2[1]:
                top2 = (i, prob)
            elif top3 is None or prob > top3[1]:
                top3 = (i, prob)

        return (top1, top2, top3)

    def readPlate(self, image, minConf=0.2):
        text = ''
        conf = 1

        letterImages = plateUtils.getLetterImages(image)
        firstChar = None
        # invalid plate length
        if len(letterImages) < 7:
            return None, None

        for l in letterImages:
            letterImage = l['img'].copy()
            letterImage = img_to_array(letterImage) / 255
            letterImage = np.array([letterImage])
            result = self.model(letterImage)
            result = self.getTop3(result)
            
            # print(result)
            classIndex1 = result[0][0]
            # classIndex2 = result[1][0]
            # classIndex3 = result[2][0]

            firstChar = (str(self.letters[str(classIndex1)]), result[0][1])

            text = text + firstChar[0]
            conf = conf * firstChar[1]

        return text, conf

    def getPlateContours(self, gray):
        threshold = plateUtils.grayToBinary(gray)

        # Get simple contours (this is to help finding candidate plate regions)
        simpleContours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # BLUR
        bgray = cv2.blur(gray, (3,3)) # bgray = cv2.bilateralFilter(gray, 11, 17, 17) #Blur to reduce noise
        
        # SOBEL on x to find letters (differential on x axis)
        sobel = cv2.Sobel(bgray, cv2.CV_8U, 1, 0)
        # TO BINARY
        thresh = plateUtils.grayToBinary(sobel)
        # FILL around letters to detect text field
        element = cv2.getStructuringElement(cv2.MORPH_RECT, self.sobel_fill_dims)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, element)

        # Draw SIMPLE CONTOURS to close candidate area
        # So that RETR_EXTERNAL contour can work better - Against close sobel contours that connect normally separated regions
        for c in simpleContours:
            if not self.isPlateRectangle(c) is False:
                cv2.drawContours(thresh, [c], -1, 0, 5)

        # NOW FIND candidate plate regions
        cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # SORT contours by area
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
        # FILTER contours by norms (e.g. h/w ratio, area, number of edges)
        cntsFinal = []
        for c in cnts:
            if not self.isPlateRectangle(c) is False:
                cntsFinal.append(c)
        return cntsFinal

    def loop(self, frame):

        frameCopy = frame.copy()
        gray = plateUtils.toGrayscale(frameCopy)
        cnts = self.getPlateContours(gray)

        plates = []
        plateLocations = []
        # loop over our contours
        for c in cnts:
            # GET bounding rectangle
            x,y,w,h = cv2.boundingRect(c)
            # EXTRACT cropped plate
            grayCropped = imutils.resize(gray[max(y-4, 0):min(y+h+4, gray.shape[0]), max(x-4, 0):min(x+w+4, gray.shape[1])], height=120)
            
            # TO BINARY
            threshCropped = plateUtils.grayToBinary(grayCropped)

            ## READ the number plate
            text, conf = self.readPlate(threshCropped)
            # IGNORE low confidence or empty text
            if not conf or conf < 0.5 or not text or not text.strip():
                continue

            # EXTRACT plate text via plate regex
            cPlates = self.extractPlatesFromOCRResult(text)
            for plate in cPlates:
                plates.append({'plate':plate, 'conf':conf})
                plateLocations.append(c)
                # cv2.imshow('Plate {}'.format(plate), grayCropped)
                # cv2.waitKey(0)

        if len(plates) > 0:
            print("Detected :", plates)

        return plates, plateLocations


if __name__ == '__main__':

    # test plate extractor
    pe = PlateExtractor()

    img = plateUtils.readGrayImage(sys.argv[1])
    print(pe.readPlate(img.copy()))

    cv2.imshow("IMG", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
