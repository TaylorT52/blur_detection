import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew
from matplotlib import pyplot as plt

if __name__ == "__main__":
    ASnoBlur = 0
    fails_no_blur = ["frame-06-02-2018-03-10-50.jpg",
                   "1NB.jpg",
                   "2NB.jpg",
                   "frame-06-02-2018-03-12-13.jpg",
                   "frame-06-02-2018-03-12-21.jpg",
                   "frame-06-02-2018-03-12-40.jpg",
                   "frame-06-02-2018-03-12-44.jpg",
                   "frame-06-02-2018-03-13-14.jpg",
                   "frame-06-02-2018-03-13-17.jpg",
                   "frame-06-02-2018-03-13-21.jpg",
                   "frame-06-02-2018-03-13-48.jpg",
                   "frame-06-02-2018-03-14-01.jpg",
                   "frame-06-02-2018-03-14-06.jpg",
                   "frame-06-02-2018-03-14-53.jpg",
                   "frame-06-02-2018-03-15-39.jpg",
                   "frame-06-02-2018-03-16-35.jpg",
                   "frame-06-02-2018-03-16-47.jpg",
                   "frame-06-02-2018-03-16-52.jpg",
                   "frame-06-02-2018-03-17-10.jpg",
                   "frame-06-02-2018-03-17-16.jpg",
                   "frame-06-02-2018-03-17-26.jpg",
                   "frame-06-02-2018-03-17-43.jpg",
                   "frame-06-02-2018-03-17-46.jpg",
                   "frame-06-02-2018-03-17-53.jpg",
                   "frame-06-02-2018-03-18-17.jpg",
                   "frame-06-02-2018-03-18-29.jpg",
                   "frame-06-02-2018-03-18-34.jpg",
                   "frame-06-02-2018-03-18-37.jpg",
                   "frame-06-02-2018-03-18-40.jpg",
                   "frame-06-02-2018-03-18-42.jpg",
                   "frame-06-02-2018-03-18-45.jpg",
                   "frame-10-02-2022-13-21-50.jpg",
                   "frame-10-02-2022-13-21-56.jpg",
                   "frame-10-02-2022-13-22-00.jpg",
                   "frame-10-02-2022-13-22-21.jpg",
                   "frame-10-02-2022-13-23-08.jpg",
                   "frame-10-02-2022-13-23-11.jpg",
                   "frame-10-02-2022-13-23-18.jpg",
                   "frame-10-02-2022-13-23-24.jpg",
                   "frame-10-02-2022-13-23-55.jpg",
                   "frame-10-02-2022-13-24-18.jpg",
                   "frame-10-02-2022-13-24-36.jpg",
                   "frame-10-02-2022-13-24-47.jpg",
                   "frame-10-02-2022-13-25-04.jpg",
                   "frame-10-02-2022-13-25-19.jpg",
                   "frame-10-02-2022-13-25-29.jpg",
                   "frame-10-02-2022-13-25-35.jpg",
                   "frame-21-01-2022-15-49-20.jpg"]
    fails_blur = ["frame-06-02-2018-03-08-20.jpg", "frame-06-02-2018-03-19-37.jpg",
                 "1B.jpg", "frame-06-02-2018-03-19-41.jpg",
                 "2B.jpg", "frame-06-02-2018-03-19-47.jpg",
                 "frame-06-02-2018-03-11-11.jpg", "frame-06-02-2018-03-19-54.jpg",
                 "frame-06-02-2018-03-11-16.jpg", "frame-06-02-2018-03-19-59.jpg",
                 "frame-06-02-2018-03-11-37.jpg", "frame-06-02-2018-03-20-05.jpg",
                 "frame-06-02-2018-03-11-53.jpg", "frame-06-02-2018-03-20-15.jpg",
                 "frame-06-02-2018-03-12-27.jpg", "frame-06-02-2018-03-20-23.jpg",
                 "frame-06-02-2018-03-12-48.jpg", "frame-10-02-2022-13-21-37.jpg",
                 "frame-06-02-2018-03-12-54.jpg", "frame-10-02-2022-13-22-25.jpg",
                 "frame-06-02-2018-03-13-01.jpg", "frame-10-02-2022-13-22-28.jpg",
                 "frame-06-02-2018-03-13-30.jpg", "frame-10-02-2022-13-22-31.jpg",
                 "frame-06-02-2018-03-13-34.jpg", "frame-10-02-2022-13-22-35.jpg",
                 "frame-06-02-2018-03-14-14.jpg", "frame-10-02-2022-13-22-41.jpg",
                 "frame-06-02-2018-03-14-29.jpg", "frame-10-02-2022-13-22-47.jpg",
                 "frame-06-02-2018-03-14-33.jpg", "frame-10-02-2022-13-22-54.jpg",
                 "frame-06-02-2018-03-14-36.jpg", "frame-10-02-2022-13-23-00.jpg",
                 "frame-06-02-2018-03-14-41.jpg", "frame-10-02-2022-13-23-03.jpg",
                 "frame-06-02-2018-03-14-47.jpg", "frame-10-02-2022-13-23-14.jpg",
                 "frame-06-02-2018-03-14-57.jpg", "frame-10-02-2022-13-24-03.jpg",
                 "frame-06-02-2018-03-15-00.jpg", "frame-10-02-2022-13-24-13.jpg",
                 "frame-06-02-2018-03-15-04.jpg", "frame-10-02-2022-13-24-22.jpg",
                 "frame-06-02-2018-03-15-23.jpg", "frame-10-02-2022-13-24-30.jpg",
                 "frame-06-02-2018-03-16-31.jpg", "frame-10-02-2022-13-25-45.jpg",
                 "frame-06-02-2018-03-16-39.jpg", "frame-10-02-2022-13-25-52.jpg",
                 "frame-06-02-2018-03-16-43.jpg", "frame-10-02-2022-13-26-03.jpg",
                 "frame-06-02-2018-03-16-56.jpg", "frame-10-02-2022-13-26-09.jpg",
                 "frame-06-02-2018-03-17-29.jpg", "frame-10-02-2022-13-26-24.jpg",
                 "frame-06-02-2018-03-17-33.jpg", "frame-21-01-2022-15-49-30.jpg",
                 "frame-06-02-2018-03-17-37.jpg", "frame-21-01-2022-15-49-34.jpg",
                 "frame-06-02-2018-03-17-40.jpg", "frame-21-01-2022-15-49-46.jpg",
                 "frame-06-02-2018-03-18-52.jpg", "frame-21-01-2022-15-49-55.jpg",
                 "frame-06-02-2018-03-19-32.jpg", "frame-21-01-2022-15-50-00.jpg"]
    failsDistributions = ["Not-blurry/frame-06-02-2018-03-17-10.jpg", "Not-blurry/frame-06-02-2018-03-17-16.jpg", "Not-blurry/frame-10-02-2022-13-21-50.jpg"]

    for fail in failsDistributions:
        text = fail
        img = cv2.imread(text, 0)
        # global thresholding
        ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        # Otsu's thresholding
        ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # plot all the images and their histograms
        images = [img, 0, th1,
                  img, 0, th2,
                  blur, 0, th3]
        titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
                  'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
                  'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]

        hist = cv2.calcHist([blur], [0], None, [256], [0, 256])
        print("Skew" + str(skew(hist)))
        ASnoBlur = ASnoBlur + skew(hist)
        print()

        for i in range(3):
            plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
            plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
            plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
            plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
            plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
            plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])

        plt.show()

    avgASNoBlur = ASnoBlur / len(fails_no_blur)
    print("average skew: " + str(avgASNoBlur))

