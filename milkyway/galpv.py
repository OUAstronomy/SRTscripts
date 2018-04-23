#!/usr/bin/env python
'''
Name  : Galaxy Position-Velocity CMD creator, galpv.py
Author: Nickalas Reynolds
Date  : Fall 2017
Misc  : Will create an appropriate srt.cat and command file for 
        wanted to observe the milky way galaxy
'''

# imported standard modules
import sys
import argparse
import math as m
import numpy as np
from math import acos,sqrt,pi,cos,sin

def ang_vec(deg):
    rad = deg*pi/180.
    return (cos(rad),sin(rad))
def length(v):
    return sqrt(v[0]**2+v[1]**2)
def dot_product(v,w):
   return v[0]*w[0]+v[1]*w[1]
def determinant(v,w):
   return v[0]*w[1]-v[1]*w[0]
def inner_angle(v,w):
   cosx=dot_product(v,w)/(length(v)*length(w))
   rad=acos(cosx) # in radians
   return rad*180/pi # returns degrees
def angle_clockwise(A, B):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det>0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else: # if the det > 0 then A is immediately clockwise of B
        return 360-inner

parser = argparse.ArgumentParser('Will create the srt.cat and the srt command file for galaxy observing')
parser.add_argument('-r','--resolution',dest='r',default=5,type=float,help='Resolution Element (use >1 please) or set to Nyquist sampling for best results')
parser.add_argument('-b','--box',dest='b',action='store_true',help='Toggle the use of boxes vs tracks.')
parser.add_argument('-sd','--startdegree',dest='sd',required=True,help='Starting degree along galactic plane. Input either float (starting Long) or [float,float] (starting box corner)')
parser.add_argument('-ed','--enddegree',dest='ed',required=True,help='Ending degree along galactic plane. Input either float (ending Long) or [float,float] (ending box corner)')
parser.add_argument('-vo','--verticaloffset',type=float,dest='vo',default=0,help='The degree value above or below galactic plane to map (only does a single pass per vo). Only use with long tracks')
parser.add_argument('-d', '--direction',type=str, default="+",dest='d',help='Positive or negative direction of mapping plane (0->360 is + ; 180 -> 90 is -')
parser.add_argument('-i', '--integration',type=int, default=120,dest='i',help='Integration times')
parser.add_argument('-sp','--stowpos',type=str,default="90,20",dest='sp',help='Set the stow position to a new stow')
parser.add_argument('-tc','--tcal',type=int,default=1200,dest='tcal',help='Set the TCAL for the cat file')
parser.add_argument('--debug',action='store_true',help='Debug helper')
args = parser.parse_args()

assert args.d in ['-','+']
if ((360./args.r) % 1) != 0:
    print('Resolution is improper <1, will still generate file with 1deg res')
if args.d == '+':
    outname = "mw_{}_p_{}_{}".format(str(args.sd).replace('[','').replace(']','').replace(',','').replace('.',''),str(args.ed).replace('[','').replace(']','').replace(',','').replace('.',''),str(args.vo).replace('.',''))
else:
    outname = "mw_{}_m_{}_{}".format(str(args.sd).replace('[','').replace(']','').replace(',','').replace('.',''),str(args.ed).replace('[','').replace(']','').replace(',','').replace('.',''),str(args.vo).replace('.',''))

# make degree array
if not args.b :
    start = float(args.sd)%360
    end   = float(args.ed)%360
    alldegrees = [round(x*args.r,2) for x in range(int(360./args.r))]

    if args.d == "+":# positive direction
        diff = round(angle_clockwise(ang_vec(start),ang_vec(end)),2)
        numd = int(m.ceil(diff/args.r))+1
        final = [round((start + x*args.r),2)%360 for x in range(numd)]

    elif args.d == "-": # negative direction
        diff = round(360.-angle_clockwise(ang_vec(start),ang_vec(end)),2)
        numd = int(m.ceil(diff/args.r))+1
        final = [round((start - x*args.r),2)%360  for x in range(numd)]
 
    alldegrees = [[x,args.vo] for x in final]
    # CMD file creation
    with open(outname+'_cmd.txt','w') as f:
        f.write(': record \n')
        for i,x in alldegrees:
            f.write(':{} G{}\n'.format(args.i,x))
        f.write(':roff\n')
        f.write(':stow\n')
        f.write('')
else:
    startb = [x for x in map(float,args.sd.strip('[').strip(']').split(','))]
    startb[0] = startb[0]%360
    endb = [x for x in map(float,args.ed.strip('[').strip(']').split(','))]
    endb[0] = endb[0]%360
    start,end = startb[0],endb[0]
    vstart,vend = startb[1],endb[1]
    vdiff = round(angle_clockwise(ang_vec(vstart),ang_vec(vend)),2)
    numvp = int(m.ceil(vdiff/args.r))+1
    
    if startb[1] < endb[1]:
        verticalrange = [x for x in np.linspace(vstart, vend,endpoint=True,num=numvp)]
    else:
        verticalrange = [x for x in np.linspace(vstart, vend,endpoint=True,num=numvp)]
        verticalrange = verticalrange[::-1]

    alldegrees = []
    count = 0
    for i,lat in enumerate(verticalrange):
        if args.d == "+":# positive direction
            diff = round(angle_clockwise(ang_vec(start),ang_vec(end)),2)
            numd = int(m.ceil(diff/args.r))+1
            final = [round((start + x*args.r),2)%360 for x in range(numd)]

        elif args.d == "-": # negative direction
            diff = round(360.-angle_clockwise(ang_vec(start),ang_vec(end)),2)
            numd = int(m.ceil(diff/args.r))+1
            final = [round((start - x*args.r),2)%360  for x in range(numd)]

        if count%2 == 1: # negative direction
            final = final[::-1]

        for x in final:
            alldegrees.append([x,lat])
        count += 1

totaltime = round(len(alldegrees)*args.i,3)
if totaltime >= 21600:
    print('Please split up the desired tracks to smaller increments, this is a large program')

print("Total Time: {}s...with slew: {}s ".format(totaltime,round(totaltime*1.1,3)))
print('Made files {0}.cat {0}_cmd.txt'.format(outname))

# write command file
with open(outname+'_cmd.txt','w') as f:
    f.write(': record \n')
    for i,x in alldegrees:
        f.write(':{0} G{1}_{2}\n'.format(args.i,''.join(str(round(i,2)).split('.')),''.join(str(round(x,2)).split('.'))))
    f.write(':roff\n')
    f.write(':stow\n')
    f.write('')

# making srt.cat file
with open(outname+'.cat','w') as f:
    f.write('BIGRAS\n')
    f.write('CALMODE 20\n')
    f.write('STATION 35.207 97.44 Sooner_Station\n')
    f.write('SOU 00 00 00  00 00 00 Sun\n')
    f.write('SOU 02 23 17 61 38 54 W3 1950    // strongest OH line 1665.4 MHz -44 km/s \n')
    f.write('GALACTIC 132 -1 S7     // hydrogen line calibration region\n')
    f.write('GALACTIC 207 -15 S8    // hydrogen line calibration region\n')
    f.write('\n')
    for i,x in alldegrees:
        f.write("GALACTIC {0} {1} G{2}_{3}\n".format(round(i,2),round(x,2),''.join(str(round(i,2)).split('.')),''.join(str(round(x,2)).split('.'))))
    f.write('\n')
    f.write('NOPRINTOUT\n')
    f.write('BEAMWIDTH 5\n')
    f.write('NBSW 10\n')
    f.write('AZLIMITS 0 360\n')
    f.write('ELLIMITS 0 90.0\n')
    f.write('STOWPOS {} {}\n'.format(*args.sp.split(',')))
    f.write('TSYS 125    \n')
    f.write('*TCAL 1200    // conservative noise diode temp\n')
    f.write('*TCAL 1650   // possible new temp\n')
    f.write('*TCAL 300    // vane calibrator\n')
    f.write('TCAL {}   // set calibrator\n'.format(args.tcal))
    f.write('RECORD 5 SPEC\n')
    f.write('NUMFREQ 256    // good choice for dongle\n')
    f.write('BANDWIDTH 2.0\n')
    f.write('FREQUENCY 1420.406\n')
    f.write('RESTFREQ 1420.406\n')
    f.write('FREQCORR -0.05   // TV dongle correction\n')
    f.write('NBLOCK 5   // number of blocks per update - can be reduced for Beagle board with slow display for PCI it is hardwired to 20\n')
    f.write('COUNTPERSTEP 100 // to move with old SRT is steps\n')
    f.write('ROT2SLP 2  // change rot2 sleep time to 3 seconds - default is 1 second\n')
    f.write('NOISECAL 70 // default is 300\n')
    f.write('')

#############
# end of file

