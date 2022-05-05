import cv2
import numpy as np

#OUTPUT = 'E:\\Documents\\20211113_Projects\\VTel\\furniture\\output.avi'
PATH_FRAME = 'E:\\Documents\\20211113_Projects\\VTel\\test1.jpg'

if __name__ == '__main__':
    image = cv2.imread(PATH_FRAME)
    height, width, _ = np.shape(image)

    image = cv2.resize(image, (0,0), fx = 0.5, fy = 0.5)
    # image = cv2.GaussianBlur(image,(3,3),0)
    # image = cv2.medianBlur(image,3)
    image = cv2.blur(image,(3,3))
    # image = cv2.bilateralFilter(image,9,75,75)

    cv2.imshow('frame',image)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
