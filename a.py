import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

def humansize(nbytes):
    '''From https://stackoverflow.com/questions/14996453/python-libraries-to-calculate-human-readable-filesize-from-bytes'''
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])


fns = os.listdir("data/val")
fns.sort()
fns = ["data/val/" + fn for fn in fns]


x_val = np.zeros((len(fns), 224, 224, 3), dtype=np.float32)
print(humansize(x_val.nbytes))
