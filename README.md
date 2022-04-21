# Blur Detection

## Setup
To install, please ensure you have python 3.9.1 or later!

### Install dependencies
Note! Some are only used for debugging the algorithm and can be removed if they don't install correctly on your computer! See the `requirements.txt` for notes.

```
pip3 install -r requirements.txt
```

### Create the egg
This will make a python egg in the folder called `output`:

```
python3 create_egg.py
```

### Add to your program
You can now add the module to your program:

```python
import sys
from os import listdir
from os.path import isfile, join

# import the blur_detection from our module
import blur_detection

# Note! This path is relative to your python script
# Here we assume the script is in the root of this project
sys.path.append('./output/blur_detection-1.0-py3.9.egg')

# this is the file we want to process
file="../sample_images/blurry/frame-10-02-2022-13-26-03.jpg"

# create a blur detection object
obj = blur_detection.Blur(file)

# process the file. Note! you can optionally pass a blurriness threshold as an argument
result = obj.blur()
print(file + " is blurry: " + str(result))
```

### Example program
There is an example program called `./scripts/test.py` folder that will read all the files in
the `./sample_images/blurry` and `./sample_images/not_blurry` folders and run them
through the algorithm, and output the filename and whether it was detected to be blurry or not
