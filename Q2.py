import sys
from PIL import Image
img = Image.open(sys.argv[1])
pixel = img.load()
for i in range(0, img.width-1):
    for j in range(0, img.height-1):
        pixel[i, j] = tuple(map(lambda x: round(x / 2), pixel[i, j]))
img.save('Q2.png')
