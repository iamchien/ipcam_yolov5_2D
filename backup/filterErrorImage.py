import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # Take each frame
    frame = cv2.imread('./NewBugs/frame28.jpg',0)

    hist_full = cv2.calcHist([frame],[0],None,[256],[0,256])
    total = sum(hist_full)
    hist_full = hist_full / total * 100
    print(max(hist_full), hist_full.tolist().index(max(hist_full)))
    plt.subplot(121), plt.imshow(frame)
    plt.subplot(122), plt.plot(hist_full)
    plt.show()