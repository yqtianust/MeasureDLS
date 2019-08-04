import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import cv2

def humansize(nbytes):
    '''From https://stackoverflow.com/questions/14996453/python-libraries-to-calculate-human-readable-filesize-from-bytes'''
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])


fns = os.listdir("data/img_val")
fns.sort()
fns = ["data/img_val/" + fn for fn in fns]


x_val = np.zeros((len(fns), 224, 224, 3), dtype=np.float32)
print(humansize(x_val.nbytes))

# Processing images
for i in range(len(fns)):
    if i %2000 == 0:
        print("%d/%d" % (i, len(fns)))

    # Load (as BGR)
    img = cv2.imread(fns[i])

    # Resize
    height, width, _ = img.shape
    new_height = height * 256 // min(img.shape[:2])
    new_width = width * 256 // min(img.shape[:2])
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Crop
    height, width, _ = img.shape
    startx = width//2 - (224//2)
    starty = height//2 - (224//2)
    img = img[starty:starty+224,startx:startx+224]
    assert img.shape[0] == 224 and img.shape[1] == 224, (img.shape, height, width)

    # Save (as RGB)
    x_val[i,:,:,:] = img[:,:,::-1]

# Save all val dataset
np.save("data/x_val.npy", x_val)
print('finish extraction')

