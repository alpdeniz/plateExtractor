#!/usr/bin/python3
from Levenshtein import distance
import sys
from time import time
import cv2, imutils
from classes.plate import PlateExtractor, plateUtils

allPlates = {}
candidatePlates = {}

# meh
def mergeFrameReadings(plateArray):
    for newPlate in plateArray:
        if newPlate['plate'] in candidatePlates:
            candidatePlates[newPlate['plate']]['conf'] += max(candidatePlates[newPlate['plate']]['conf'], newPlate['conf']) + 0.05
        else:
            candidatePlates[newPlate['plate']] = newPlate
        for plate in candidatePlates:
            levenshteinDistance = distance(candidatePlates[plate]['plate'], newPlate['plate'])
            if levenshteinDistance < 4 and levenshteinDistance > 0:
                if abs(candidatePlates[plate]['conf'] - newPlate['conf']) > 5:
                    if candidatePlates[plate]['conf'] < newPlate['conf']:
                        del candidatePlates[plate]
                    else:
                        del candidatePlates[newPlate['plate']]
                    break
    for plate in candidatePlates:
        if candidatePlates[plate]['conf'] > 0.7:
            print("ACCEPTED PLATE: ", plate, " CONF: ", candidatePlates[plate]['conf'])
            allPlates[plate] = candidatePlates[plate]
    for plate in allPlates:
        if plate in candidatePlates:
            del candidatePlates[plate]

if __name__ == '__main__':

    # INIT VIDEO
    source = sys.argv[1]
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    newfh = 3.5*fh/4
    newfw = 3.5*fw/4

    # SET ROI
    print("FPS", fps, " Width: ", fw, " Height: ", fh)
    if newfw != fw:
        print('Reducing the dimensions to {}x{}'.format(newfw, newfh))
    
    # INIT platereader
    pe = PlateExtractor()

    # OPTIONS
    numFrames = 60 # every x frame show info
    processPerNFrames = 2 # every x frame analyze
    roi = [int(2*newfh/4), int(newfh), 0, int(newfw)] # select sub-region - TODO Get via args
    
    # READ FRAMES
    counter = 0
    skippedFrameCounter = 0
    previousFrame = None
    start = time()
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                raise Exception("No data")
            if newfw != fw:
                frame = imutils.resize(frame, width=int(newfw))
        except Exception as e:
            print("Exception at cap.read: ", e)
            cv2.waitKey(0)
            exit()

        # EXTRACT ROI from frame
        frameROI = frame[roi[0]:roi[1], roi[2]:roi[3]] # frameROI = imutils.resize(frame[roi[0]:roi[1],roi[2]:roi[3]], height=640)

        # DIFF gray, fixed-size frames
        smallGray = plateUtils.toGrayscale(frame, width=480)
        frameDiff = 1000000
        try:
            diffFrame = cv2.absdiff(smallGray, previousFrame)
            frameDiff = cv2.sumElems(diffFrame)[0]
        except Exception as e:
            print("EXCEPTION", e)
        # STORE current frame for frame diff
        previousFrame = smallGray.copy()

        # SKIP if scene is stationary
        if (frameDiff < 5000):
            skippedFrameCounter += 1
        else:
            # READ PLATES each processPerNFrames
            if counter % processPerNFrames == 0:
                plates, plateLocations = pe.loop(frameROI.copy())
                for pl in plateLocations:
                    cv2.drawContours(frame, [pl], -1, (0, 255, 0), 3)
                mergeFrameReadings(plates)
        
        # PRINT INFO
        counter += 1
        if counter % numFrames == 0:
            seconds = time()-start
            print('Took {} seconds'.format(seconds))
            print('Calculated FPS: {}'.format(numFrames/seconds))
            print('Accepted Plates: ', allPlates)
            print('Skipped {} frames'.format(skippedFrameCounter))
            start=time()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.rectangle(frame, (roi[2], roi[0]), (roi[3], roi[1]), (0,0,0), 4) # draw black around roi
        cv2.imshow('Original', frame)

    # When everything done, release the capture and show collected plates
    cap.release()
    print(candidatePlates)
    print(allPlates)
    cv2.waitKey(0)
    cv2.destroyAllWindows()