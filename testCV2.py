import cv2
import os

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport; udp'

RTSP_URL = 'rtsp://admin:chien197@192.168.1.120:554/onvif1'

if __name__ == '__main__':
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    print("Begin!")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("test", frame)
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
        