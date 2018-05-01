#!/usr/bin/env python

import os
from os.path import isfile

mydirstart='/home/jjtobin/srtn/data/sun/'
myfiledest='/home/jjtobin/RealTime-Sun/sun-current.all.dat'
temp='/tmp/sun-current.tmp'

if isfile(temp):
    os.system('rm -f {}'.format(temp))
for root, dirs, files in os.walk(mydirstart):
    for file in files:
        if isfile(file):
             os.system('cat {} >> {}'.format(os.path.join(root, file),temp))