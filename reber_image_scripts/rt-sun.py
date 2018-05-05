#!/usr/bin/env python
import numpy as np
import matplotlib as mlab
mlab.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import ScalarFormatter
from astropy.time import Time
import matplotlib.dates as mdates
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y-%B')
hoursFmt = mdates.DateFormatter('%H:%M')
from scipy.signal import medfilt as medfilt
from time import strftime, localtime,time
import argparse
TIME=time()

parser = argparse.ArgumentParser('Tries to pretty plot the current day and all sun data')
parser.add_argument('-sp',dest='sp',action='store_true',help='Only plot short plots of single day')
parser.add_argument('-lp',dest='lp',action='store_true',help='plot the long plots of all days')
parser.add_argument('-llp',dest='llp',action='store_true',help='plot the long plots of all days (regen from source)')
parser.add_argument('-rp',dest='rp',action='store_true',help='refresh the long plot')
args = parser.parse_args()
cwd = os.getcwd()
os.chdir('/home/jjtobin/RealTime-Sun/')
if (not args.sp) and (not args.lp):
    print('Requires -sp (single day plotting) or -lp (for all plotting)')
    exit(0)

# for general formatting of figure axis
def fig_formatter(ax,formatter='hours',locations=None):
    ticks_font = font_manager.FontProperties(size=16, weight='normal', stretch='normal')
    for axis in ['top','bottom','left','right']:
       ax.spines[axis].set_linewidth(2)
    if formatter=='months':
        myFmtmajor   = yearsFmt
        myLocmajor   = mdates.MonthLocator()
    elif formatter=='hours':
        myFmtmajor   = hoursFmt
        myLocmajor   = mdates.HourLocator()
    ax.xaxis.set_major_formatter(myFmtmajor)
    if not locations:
        ax.xaxis.set_major_locator(myLocmajor)
    else:
        ax.xaxis.set_major_locator()
    plt.gcf().autofmt_xdate()  # orient date labels at a slant
    ax.tick_params('both', which='major', length=15, width=1, pad=15)
    ax.tick_params('both', which='minor', length=7.5, width=1, pad=15)
    plt.tight_layout()
    plt.draw()

if args.sp:
    ##############################################################################
    # getting data
    ##############################################################################
    os.system('cp -L ~/sun-current.dat .')
    os.system('python metaparse.py -i sun-current.dat -o sun-current-reduced.dat --auto -l metaparse.log -v0')
    data = np.loadtxt("sun-current-reduced.dat",skiprows=2,dtype=str)

    t = Time(data[:,0], format='yday', scale='utc')
    tantdata = [float(x) for x in data[:,6]]

    ##############################################################################
    # short time plot
    ##############################################################################
    #ti=Time(strftime('%Y:%j:24:00:00', localtime()),format='yday',scale='utc')
    tstart=Time(strftime('%Y:%j:12:00:00', localtime()),format='yday',scale='utc')
    plt.clf()

    fig=plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)
    ax.set_xlim(tstart.mjd,tstart.mjd+0.042*14.0)
    ax.set_ylim(0,1.01*max(tantdata))

    Tant_smooth = medfilt(tantdata, 15)

    time_axis=t.mjd[0::15]
    resampled_data=np.interp(time_axis,t.mjd,tantdata)
    #print (t.mjd)

    lin1=ax.plot(t.mjd,tantdata,color='black',marker='o',linestyle='none')
    lin2=ax.plot(time_axis,resampled_data,color='red',marker='o',linestyle='none')
    #lin3=ax.plot(t.mjd,Tant_smooth,color='blue',marker='o',linestyle='none')
    #print("{},{},{}".format(data['DATE'],data['Tant'],resampled_data))
    ax.set_xlabel('Universal Time', fontsize=18)
    ax.set_ylabel('Radio Brightness (K)', fontsize=18)

    T_ant_sun=np.median(tantdata)

    #lin1=ax.plot([tmax.mjd-0.00,tmax.mjd-0.00],[0,10000],color='black',linestyle='dashed')
    #lin1=ax.plot([tstart.mjd-0.00,tstart.mjd-0.00],[0,10000],color='black',linestyle='dashed')
    #lin1=ax.plot([tend.mjd-0.00,tend.mjd-0.00],[0,10000],color='black',linestyle='dashed')

    #ax.text(tstart.mjd+0.001,400.0,'Eclipse Start - 11:37 am',rotation=90.0)
    #ax.text(tmax.mjd+0.001,400.0,'Eclipse Max - 1:06 pm',rotation=90.0)
    #ax.text(tend.mjd+0.001,400.0,'Eclipse End - 2:35 pm',rotation=90.0)
    showtime = strftime("%Y-%m-%d %H:%M:%S", localtime())
    plt.title('Realtime Solar Flux Data - '+showtime)
    #plt.title('Realtime Eclipse Data - '+showtime)
    fig_formatter(ax)
    plt.savefig('sun-current.png')

def list_files(dir):
    r = []
    subdirs = [x[0] for x in os.walk(dir)]
    for subdir in subdirs:
        files = os.walk(subdir).next()[2]
        if (len(files) > 0):
            for file in files:
                r.append(subdir + "/" + file)
    return r  

if args.lp:
    ##############################################################################
    # getting data
    ##############################################################################
    # to repopulate full plot, backup should already be stored if needed <sun-current-reduced.all.dat.bak>
    if args.llp:
        mydirstart='/home/jjtobin/srtn/data/sun/'
        myfiledest='/home/jjtobin/RealTime-Sun/sun-current.all.dat'
        allfiles= list_files(mydirstart)
        os.system('python metaparse.py -i "{}" -o sun-current-reduced.all.metaparse.dat ' \
                   '--auto -l metaparse.log -v 0'.format(allfiles))
        os.system('rm -f /tmp/sun-current-reduced.all.metaparse.dat')
        os.system('sed "1,2d" sun-current-reduced.all.metaparse.dat | awk \'{ print $1, $7 }\' '\
                  '>> /tmp/sun-current-reduced.all.dat ')
        os.system('mv -f /tmp/sun-current-reduced.all.dat ./')
        os.system('rm -f ./sun-current-reduced.all.metaparse.dat')
    # to just update the plot
    elif not args.rp:
        os.system('python metaparse.py -i sun-current.dat -o sun-current-reduced.all.metaparse.dat ' \
                   '--auto -l metaparse.log -v 0')
        os.system('rm -f /tmp/sun-current-reduced.all.metaparse.dat')
        os.system('sed "1,2d" sun-current-reduced.all.metaparse.dat | awk \'{ print $1, $7 }\' '\
                  '>> /tmp/sun-current-reduced.all.dat ')
        os.system('cat /tmp/sun-current-reduced.all.dat >> ./sun-current-reduced.all.dat')
        os.system('rm -f ./sun-current-reduced.all.metaparse.dat')

    data = np.loadtxt("sun-current-reduced.all.dat",dtype=str)
    t = Time(data[:,0], format='yday', scale='utc')
    tantdata = [float(x) for x in data[:,1]]

    ##############################################################################
    # long time plot
    ##############################################################################
    tstart =Time(strftime('2017:65:00:00:00'),format='yday',scale='utc')
    tfinish=Time(strftime('%Y:%j:23:59:59', localtime()),format='yday',scale='utc')
    plt.clf()
    fig=plt.figure(figsize=(45,7))
    ax = fig.add_subplot(111)
    ax.set_xlim(tstart.plot_date,tfinish.plot_date)
    ax.set_ylim(0,1.01*max(tantdata))

    sampling=31
    Tant_smooth = medfilt(tantdata, sampling)
    time_axis=t.plot_date[0::sampling]
    resampled_data=np.interp(time_axis,t.plot_date,tantdata)

    lin1=ax.plot(t.plot_date,tantdata,color='black',marker='o',linestyle='none')
    lin2=ax.plot(time_axis,resampled_data,color='red',marker='o',linestyle='none')

    ax.set_xlabel('Universal Time', fontsize=18)
    ax.set_ylabel('Radio Brightness (K)', fontsize=18)

    T_ant_sun=np.median(tantdata)
    showtime = strftime("%Y-%m-%d %H:%M:%S", localtime())
    plt.title('Realtime Solar Flux Data - '+showtime)
    fig_formatter(ax,'months')
    plt.savefig('sun-current.all.png',dpi=100)
    plt.savefig('sun-current.all.lowres.png',dpi=50)

print('[{}] Median Tant: {} across {} integrations'.format(showtime,round(np.median(tantdata),4),len(tantdata)))

os.chdir(cwd)
