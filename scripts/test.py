import sys
from os import listdir
from os.path import isfile, join

sys.path.append('../output/blur_detection-1.0-py3.9.egg')

import blur_detection

blurry_dir = "../sample_images/blurry"
blurry_files = [join(blurry_dir, f) for f in listdir(blurry_dir) if isfile(join(blurry_dir, f))]

not_blurry_dir = "../sample_images/not_blurry"
not_blurry_files = [join(not_blurry_dir, f) for f in listdir(not_blurry_dir) if isfile(join(not_blurry_dir, f))]


if __name__ == "__main__":

    # process all blurry files
    for file in blurry_files:
        obj = blur_detection.Blur(file)
        result = obj.blur()
        print(file + " is blurry: " + str(result))

    # process all not-blurry files
    for file in not_blurry_files:
        obj = blur_detection.Blur(file)
        result = obj.blur()
        print(file + " is not blurry: " + str(result))

