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
ref = [127,128,129,130,131,132]
q = Queue()
detect_path = Queue()
 
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
    counter, i = [0, 1]

    while(cap.isOpened()):
        if counter % gap == 0:
            ret, frame = cap.retrieve()
            if not ret:
                break

            thredReadFrame = Thread(target=readFrame, args=(frame,i))
            threadFilterBugs = Thread(target=filterBugs, args=(q.empty(),))
            threadYolo = Thread(target=runDetect, args=(detect_path.empty(),opt, model, device))
            thredReadFrame.start()
            threadFilterBugs.start()
            threadYolo.start()
            thredReadFrame.join()
            threadFilterBugs.join()
            threadYolo.join()

            if i % 1500 == 0:
                i = 1
            else:
                i = i + 1
            counter = 0
        else:
            ret = cap.grab()

        counter = counter + 1

    cap.release()
    return

def readFrame(frame, i):
    encodeParam = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
    cv2.imwrite(os.path.join(PATH_FRAME, 'frame' + str(i) + '.jpg'), frame, encodeParam)
    cv2.waitKey(1)
    q.put('frame' + str(i) + '.jpg')
    return

def filterBugs(empty):
    if not empty:
        filter_path = os.path.join(PATH_FRAME, q.get())
        frame = cv2.imread(filter_path,0)
        if frame is None:
            os.remove(filter_path)
        else:
            hist_full = cv2.calcHist([frame],[0],None,[256],[0,256])
            total = sum(hist_full)
            hist_full = hist_full / total * 100
            maxHist = max(hist_full)[0]
            index = hist_full.tolist().index(maxHist)

            if index in ref and maxHist >= 20:
                os.remove(filter_path)
            else:
                detect_path.put(filter_path)

    return

def runDetect(empty, opt, model, device):
    # exist = os.path.exists(filter_path)
    if not empty:
        yolo_img = detect_path.get()
        run(opt, model, device, yolo_img)
        # frame = cv2.imread(os.path.join(YOLO_FRAME, yolo_img))
        # cv2.imshow('frame', frame)
        # cv2.waitKey(1)

    return

if __name__ == '__main__':
    # python3 backupMainFile.py --yolo_weights ./yolov5/weights/crowdhuman_yolov5m.pt --output ./inference/output/ --save-vid --device '0'
    start_time = timeit.default_timer()
    main()
    print(timeit.default_timer() - start_time)
