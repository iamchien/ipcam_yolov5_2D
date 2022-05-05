# # from genericpath import exists
# # from multiprocessing.dummy import Process
# from threading import Thread
# from queue import Queue
# import subprocess
# import cv2
# import os
# import shutil
# import timeit

# import sys
# sys.path.insert(0,'/home/hongthinh/haison98/ChienUET/Yolov5_DeepSort_Pytorch')
# from track import parse_opt, detect, run

# RTSP_URL = 'rtsp://admin:son123456@192.168.0.9:554/onvif1'
# PATH_FRAME = '/home/hongthinh/haison98/ChienUET/5fps'
# ref = [127,128,129,130,131,132]
# i = 1
# q = Queue()

# if os.path.exists(PATH_FRAME):
#     shutil.rmtree(PATH_FRAME)

# os.mkdir(PATH_FRAME)

# def main():
#     opt = parse_opt()
#     model, device = detect(opt)

#     readFrame()

#     threadFilterBugs = Thread(target=filterBugs,args=())
#     threadYolo = Thread(target=runDetect, args=(q.empty(),opt, model, device))

#     threadFilterBugs.start()
#     threadYolo.start()
#     threadFilterBugs.join()
#     threadYolo.join()

#     return

# def readFrame():    
#     # command and params for ffmpeg
#     command = 'ffmpeg -y -i rtsp://admin:son123456@192.168.0.9:554/onvif1 -r 5 -frames 1500 /home/hongthinh/haison98/ChienUET/5fps/test%d.jpg'
#     # using subprocess and pipe to fetch frame data
#     subprocess.Popen(command,  shell = True)

# def filterBugs():
#     global i
#     img_path = os.path.join(PATH_FRAME, 'test'+str(i)+'.jpg')
#     flag = os.path.isfile(img_path)
#     if flag:
#         q.put(img_path)
#         frame = cv2.imread(os.path.join(img_path),0)
#         if frame is None:
#             os.remove(img_path)
#         else:
#             hist_full = cv2.calcHist([frame],[0],None,[256],[0,256])
#             total = sum(hist_full)
#             hist_full = hist_full / total * 100
#             maxHist = max(hist_full)[0]
#             index = hist_full.tolist().index(maxHist)

#             if index in ref and maxHist >= 20:
#                 os.remove(img_path)

#         i = i + 1
    
#     return

# def runDetect(flag, opt, model, device):
#     if not flag:
#         img_path = q.get()
#         run(opt, model, device, img_path)

#     return

# if __name__ == '__main__':
#     # python3 backupMainFile.py --yolo_weights ./yolov5/weights/crowdhuman_yolov5m.pt --output ./inference/output/ --save-vid --device 'cpu'
#     start_time = timeit.default_timer()
#     main()
#     print(timeit.default_timer() - start_time)

