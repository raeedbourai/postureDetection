import cv2
import os
import random
import torch
from PIL import Image
from facenet_pytorch import MTCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=False, device=device)

def extractImages(pathIn, pathOut, imgCount):
    num = random.randint(1, 10)
    count = imgCount
    cap = cv2.VideoCapture(pathIn)
    cap.set(cv2.CAP_PROP_POS_MSEC,(num*1000))
    success, image = cap.read()  
    if success and image is not None:
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = mtcnn.detect(frame, landmarks = True)
        if boxes is not None:
            cv2.imwrite(pathOut + 'frame%d.jpg' % count, image)
            count += 1
    return count

def filterImages(pathIn, pathOut, imgCount):
    image = cv2.imread(pathIn)
    count = imgCount
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes, probs, landmarks = mtcnn.detect(frame, landmarks = True)
    if boxes is not None:
        cv2.imwrite(pathOut + 'frame%d.jpg' % count, image)
        count += 1
    return count


globalCount = 1

for filename in os.listdir('mp4/th'):
    filename = 'mp4/th/' + filename
    print(filename)
    globalCount = extractImages(filename, 'images/', globalCount)

for filename in os.listdir('mp4/th-bb'):
    filename = 'mp4/th-bb/' + filename
    print(filename)
    globalCount = extractImages(filename, 'images/', globalCount)

for filename in os.listdir('mp4/th-ob'):
    filename = 'mp4/th-ob/' + filename
    print(filename)
    globalCount = extractImages(filename, 'images/', globalCount)

for filename in os.listdir('Videos'):
    filename = 'Videos/' + filename
    print(filename)
    if filename.endswith('.mp4'):
        globalCount = extractImages(filename, 'images/', globalCount)

for filename in os.listdir('cropped_people'):
    filename = 'cropped_people/' + filename
    print(filename)
    globalCount = filterImages(filename, 'images/', globalCount)
