from threading import Thread
from queue import Queue
import numpy as np
import timeit
import shutil
import cv2
import os

import sys
sys.path.insert(0, 'E:\Documents\20211113_Projects\VTel\Yolov5_DeepSort_Pytorch')
from Yolov5_DeepSort_Pytorch.track import parse_opt, detect, run

ref = [127,128,129,130,131,132]
VIDEO_PATH = 'E:\\Documents\\20211113_Projects\\VTel\\source.mp4'
FRAME_PATH = 'E:\\Documents\\20211113_Projects\\VTel\\NewBugs'
q = Queue()
img_path = ''

if os.path.exists(FRAME_PATH):
    shutil.rmtree(FRAME_PATH)
os.mkdir(FRAME_PATH)

def main():
    opt = parse_opt()
    model, device = detect(opt)
    cap = cv2.VideoCapture(VIDEO_PATH)
    i = 1
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        thredReadFrame = Thread(target=readFrame, args=(frame, i))
        threadFilterBugs = Thread(target=filterBugs, args=(q.empty(),))
        threadYolo = Thread(target=runDetect, args=(opt, model, device, img_path))
        thredReadFrame.start()
        threadFilterBugs.start()
        threadYolo.start()
        thredReadFrame.join()
        threadFilterBugs.join()
        threadYolo.join()
        i = i + 1

    return

def readFrame(frame, i):    
    cv2.imwrite(filename=os.path.join(FRAME_PATH, 'frame' + str(i) + '.jpg'), img=frame)
    cv2.waitKey(1)
    q.put('frame' + str(i) + '.jpg')
    return

def filterBugs(flag):
    if not flag:
        global img_path
        img_path = os.path.join(FRAME_PATH, q.get())
        frame = cv2.imread(img_path,0)
        hist_full = cv2.calcHist([frame],[0],None,[256],[0,256])
        total = sum(hist_full)
        hist_full = hist_full / total * 100
        maxHist = max(hist_full)[0]
        index = hist_full.tolist().index(maxHist)

        if index in ref and maxHist >= 20:
            os.remove(img_path)
    return

def runDetect(opt, model, device, source):
    if os.path.exists(source):
        run(opt, model, device, source)

if __name__ == '__main__':
    start_time = timeit.default_timer()
    main()
    print(timeit.default_timer() - start_time)
