#!/usr/bin/env python
'''
Filename: Reading Outputs, readingoutputs.py
Author:   Nickalas Reynolds
Desc:     This file is for reading out the parameters file and placing them into an
               easily readable column format.
'''

import argparse
from glob import glob
from sys import exit
from time import time 
TIME=time()


parser = argparse.ArgumentParser('Parses the output of specplot.py to just yield integrated intensities. '+\
                                 'Either input file(s) unique string or let the program decide: '+\
                                 'e.g. if all files are trial1_G10_parameters.txt, trial1_G20_parameters.txt '+\
                                 'only use -i trial1 and the program will try to find them')
parser.add_argument('-i', '--input', dest='i',default='*',help='String of input identifiers')
parser.add_argument('-o', '--output',dest='o',default='intensities_{}.txt'.format(TIME),help='Output file name')
parser.add_argument('-s', '--space', dest='s',default=15,type=int,help='Output file spacing of columns')
args = parser.parse_args()

# gather files
count = 0
while True:
    files = []
    try:
        if count == 0:
            files = glob(args.i+'parameters.txt')
        elif count == 1:
            files = glob(args.i+'*parameters.txt')
        elif count == 2:
            files = glob(args.i)
        elif count == 3:
            files = glob('*'+args.i)
        elif count == 4:
            files = glob(args.i+'*')
        elif count == 5:
            files = glob('*'+args.i+'*')
        elif count == 6:
            files = glob('*parameters.txt')
        elif count == 7:
            files = glob('*')

        assert len(files) > 0
        print('Using files: ',files)
        break
    except:
        count += 1
        if count > 10:
            print('Couldn\'t find the files. Try with a different name or set all to *parameters.txt suffix')
            exit(0)

# read in all files info
final = []
for x in files:
    with open(x,'r') as f:
        alllines = f.readlines()
    
    final.append(['G'+alllines[0].split('G')[1].strip('\n'),alllines[-1]])

# function for formatting
def addspace(sstring,spacing=args.s):
    while True:
        if len(sstring) >= spacing:
            sstring = sstring[:-1]
        elif len(sstring) < (spacing -1):
            sstring = sstring + ' '
        else: 
            break
    return sstring + ' '

# output 
with open(args.o,'w') as f:
    f.write('{}{}{}\n'.format(addspace('# Source'),\
                              addspace('Inten. K km/s'),\
                              addspace('Error')))
    for x in final:
        #print(x)
        formatted = '{}{}{}'.format(addspace(x[0]),\
                                    addspace(x[1].split(' +- ')[0].split(':')[1].strip(' ')\
                                                                  .strip('[').strip(']')),\
                                    addspace(x[1].split(' +- ')[1].split('(')[0]))
        f.write('{}\n'.format(formatted))



