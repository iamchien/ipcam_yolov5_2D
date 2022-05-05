import cv2
import time
import os
import numpy as np

#OUTPUT = 'E:\\Documents\\20211113_Projects\\VTel\\furniture\\output.avi'
PATH_FRAME = 'E:\\Documents\\20211113_Projects\\VTel\\furniture'

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    default_fps = cap.get(cv2.CAP_PROP_FPS)
    print("Default FPS by cap.get(cv2.CAP_PROP_FPS): " + str(default_fps))
    fps = 6
    gap = int(default_fps/fps)
    counter, i = [0, 1]

    start_time = time.time()
    i = 1
    while(cap.isOpened()):
        if counter % gap == 0:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(PATH_FRAME, 'test' + str(i) + '.jpg'), frame)
            cv2.waitKey(1)
            i = i + 1
            counter = 0
        else:
            ret = cap.grab()

        counter = counter + 1
        if (time.time() - start_time) >= 60.0:
            break

        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    cap.release()
    cv2.destroyAllWindows()
