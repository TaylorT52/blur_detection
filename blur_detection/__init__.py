import cv2

class Blur:
    def __init__(self, file, show = False):
        self.filename = file
        self.show = show

    def blur(self, threshold = 36):
        isBlurry = False
        image = cv2.imread(self.filename)
        #converts to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #applies laplacian filter, takes variance
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()

        if fm < threshold:
            isBlurry = True

        if self.show:
            print(str(fm) + ", Blurry: " + str(isBlurry))
        return isBlurry