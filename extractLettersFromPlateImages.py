#!/usr/bin/python3
import sys, os, glob
from random import randint
import cv2
from classes.plate import plateUtils


def getLettersFromPlatesFolder(platesPath, samplesFolder):
    # Initialize a list which will be used to append charater image
    allLetters = []
    letterCount = 0
    # Get plate images e.g /samples/06T2768-122.jpg'
    platesPath = os.path.join(platesPath,'*g')
    files = glob.glob(platesPath)

    # create samples folder
    if not os.path.isdir(os.path.join('./{}'.format(samplesFolder))):
        os.makedirs(os.path.join('./{}'.format(samplesFolder)))

    for f1 in files:
        plate = f1.split('/')[-1].split('-')[0]
        print(f1, plate)
        img = cv2.imread(f1)
        img = plateUtils.toGrayscale(img)
        # cv2.imshow("Orig",img)
        letters = plateUtils.getLetterImages(img, plate)
        for l in letters:
            letterValue = l['value']
            letter = l['img']
            randcand = randint(0,100)
            if randcand > 50:
                print("TRAIN")
                if not os.path.isdir(os.path.join('./{}/train/{}'.format(samplesFolder, letterValue))):
                    os.makedirs(os.path.join('./{}/train/{}'.format(samplesFolder, letterValue)))
                cv2.imwrite(os.path.join('./{}/train/{}/{}-{}.jpg'.format(samplesFolder, letterValue, letterValue, letterCount)), letter)
            elif randcand > 30:
                print("TEST")
                if not os.path.isdir(os.path.join('./{}/test/{}'.format(samplesFolder, letterValue))):
                    os.makedirs(os.path.join('./{}/test/{}'.format(samplesFolder, letterValue)))
                cv2.imwrite(os.path.join('./{}/test/{}/{}-{}.jpg'.format(samplesFolder, letterValue, letterValue, letterCount)), letter)
            else: 
                print("VALID")
                if not os.path.isdir(os.path.join('./{}/valid/{}'.format(samplesFolder, letterValue))):
                    os.makedirs(os.path.join('./{}/valid/{}'.format(samplesFolder, letterValue)))
                cv2.imwrite(os.path.join('./{}/valid/{}/{}-{}.jpg'.format(samplesFolder, letterValue, letterValue, letterCount)), letter)
            
            allLetters.append(l)
            # cv2.imshow('Letter {}'.format(letterValue), letter)
            # cv2.waitKey(0)

            letterCount += 1
        # cv2.putText(test_roi, plate, (6,24), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),3)
        # cv2.imshow("Letters", test_roi)
        # cv2.imshow("Final", threshold)
        # cv2.waitKey(0)
    return allLetters

if __name__ == '__main__':
    letterImages = getLettersFromPlatesFolder(sys.argv[1], sys.argv[2])
    print("Detected {} letters...".format(len(letterImages)))