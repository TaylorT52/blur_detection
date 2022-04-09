import cv2
import argparse

class Blur:
    def __init__(self, file, show):
        self.filename = file
        self.show = show

    def blur(self):
        isBlurry = False
        threshold = 36
        image = cv2.imread(self.filename)
        #converts to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #applies laplacian filter, takes variance
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()

        if fm < threshold:
            isBlurry = True

        print(str(fm) + ", Blurry: " + str(isBlurry))
        return isBlurry

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Blurry photos')
    parser.add_argument('--file', type=str,
                        help='the file to read')
    parser.add_argument('--show', type=bool, default=False,
                        help='if true, show the image')
    args = parser.parse_args()
    #creates instance of blur class initialized with values from input args
    myblur = Blur(args.file, args.show)
    #calls blur function on obj with input args, sets isBlurry to boolean isBlurry
    isBlurry = myblur.blur()
