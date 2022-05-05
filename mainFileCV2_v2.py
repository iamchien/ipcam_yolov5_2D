from email.policy import default
from threading import Thread
from queue import Queue
import cv2
import os
import shutil
import timeit

import sys
sys.path.insert(0,'/home/hongthinh/haison98/ChienUET/Yolov5_DeepSort_Pytorch')
from track import parse_opt, detect, run

RTSP_URL = 'rtsp://admin:son123456@192.168.0.9:554/onvif1'
PATH_FRAME = '/home/hongthinh/haison98/ChienUET/5fps'
YOLO_FRAME = '/home/hongthinh/haison98/ChienUET/Yolov5_DeepSort_Pytorch/inference/output'
ref = [127,128,129,130,131,132]; i = 1; q = Queue(); q2 = Queue()
 
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport; udp'

if os.path.exists(PATH_FRAME):
    shutil.rmtree(PATH_FRAME)
os.mkdir(PATH_FRAME)

def main():
    opt = parse_opt()
    model, device = detect(opt)

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    default_fps = cap.get(cv2.CAP_PROP_FPS)
    print(default_fps)
    fps = 5
    gap = int(default_fps/fps)
    counter = 0

    while(cap.isOpened()):
        if counter % gap == 0:
            ret, frame = cap.read()
            if not ret:
                break

            thredReadFrame = Thread(target=readFrame, args=(frame,))
            threadYolo = Thread(target=runDetect, args=(q.empty(),opt, model, device))
            threadShowImg = Thread(target=showImg, args=(q2.empty(),))
            thredReadFrame.start()
            threadYolo.start()
            threadShowImg.start()
            thredReadFrame.join()
            threadYolo.join()
            threadShowImg.join()

            counter = 0
        else:
            ret = cap.grab()

        counter = counter + 1
    cap.release()
    return

def readFrame(frame):
    global i

    err = filterBugs(frame)
    if not err:
        encodeParam = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        cv2.imwrite(os.path.join(PATH_FRAME, 'frame' + str(i) + '.jpg'), frame, encodeParam)
        cv2.waitKey(1)
        q.put('frame' + str(i) + '.jpg')

        if i % 1500 == 0:
            i = 1
        else:
            i = i + 1

    return

def filterBugs(frame):
    err = False
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist_full = cv2.calcHist([frame],[0],None,[256],[0,256])
        total = sum(hist_full)
        hist_full = hist_full / total * 100
        maxHist = max(hist_full)[0]
        index = hist_full.tolist().index(maxHist)

        if index in ref and maxHist >= 20:
            err = True

    return err

def runDetect(empty, opt, model, device):
    # exist = os.path.exists(filter_path)
    if not empty:
        img = q.get()
        img_path = os.path.join(PATH_FRAME, img)
        run(opt, model, device, img_path)
        q2.put(img)

    return

def showImg(empty):
    if not empty:
        yolo_path = os.path.join(YOLO_FRAME, q2.get())
        frame = cv2.imread(yolo_path)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
    
    return

if __name__ == '__main__':
    # python3 backupMainFile.py --yolo_weights ./yolov5/weights/crowdhuman_yolov5m.pt --output ./inference/output/ --save-vid --device '0'
    start_time = timeit.default_timer()
    main()
    print(timeit.default_timer() - start_time)
