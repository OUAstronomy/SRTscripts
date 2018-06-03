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
from sys import version_info
import temp as tf
tempd = tf.TemporaryDirectory
TIME=time()

parser = argparse.ArgumentParser('Tries to pretty plot the current day and all sun data')
parser.add_argument('-sp',dest='sp',action='store_true',help='Only plot short plots of single day')
parser.add_argument('-lp',dest='lp',action='store_true',help='plot the long plots of all days')
parser.add_argument('-llp',dest='llp',action='store_true',help='plot the long plots of all days (regen from source)')
parser.add_argument('-rp',dest='rp',action='store_true',help='refresh the long plot')
args = parser.parse_args()

# directory management
cwd  = os.getcwd()
ncwd = '/home/jjtobin/RealTime-Sun/'          # directory where daily solar data will be managed
scripts = '/home/jjtobin/github/SRTscripts'   # location of all scripts

# most recent solar obs
currentfile = '/home/jjtobin/sun-current.dat'   # this is the current data file, can be a symbolic link

# this is for complete generation
mydirstart  ='/home/jjtobin/srtn/data/sun/'     # all solar data location, this will step through all directories therein
myfiledest  =ncwd + '/sun-current.all.dat'      # the destination for the complete generations

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

# list all files within a directory and all sub dirs
def list_files(dir):
    r = []
    subdirs = [x[0] for x in os.walk(dir)]
    for subdir in subdirs:
        files = os.walk(subdir).next()[2]
        if (len(files) > 0):
            for file in files:
                r.append(subdir + "/" + file)
    return r  

# opening a temp directory
with tempd() as TMPD:
    if args.sp:
        ##############################################################################
        # getting data
        ##############################################################################
        # determine file 'newness'
        try:
            ndatafile   = os.readlink(currentfile)
            cdatafile   = os.path.join(ncwd,'sun-current-reduced.dat')
            newdata     = os.path.getmtime(ndatafile)
            currentdata = os.path.getmtime(cdatafile)
            #print(newdata,currentdata,ndatafile,cdatafile)
        except:
            exit(0)
        # make sure file isn't blank, assuming file > 10 bytes which is ~5 unicode characters
        if os.path.getsize(cdatafile) < 10:
            exit(0)
        # copy over file if newer than current data
        if (currentdata < newdata):
            pass
        else:
            exit(0)
        # now begin run
        os.system('python {}/metaparse.py -i {} -o {} --auto -v0'.format(scripts,ndatafile,cdatafile))
        data = np.loadtxt(cdatafile,skiprows=2,dtype=str)

        t = Time(data[:,0], format='yday', scale='utc')
        tantdata = [float(x) for x in data[:,6]]

        ##############################################################################
        # short time plot
        ##############################################################################
        #ti=Time(strftime('%Y:%j:24:00:00', localtime()),format='yday',scale='utc')
        tstart=Time(strftime('%Y:%j:11:00:00', localtime()),format='yday',scale='utc')
        plt.clf()

        fig=plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111)
        ax.set_xlim(tstart.mjd,tstart.mjd+0.042*15.0)
        ax.set_ylim(0,1.01*max(tantdata))

        sampling = 15 # 15 integrations
        Tant_smooth = medfilt(tantdata, sampling)
        time_axis=t.mjd[0::sampling]
        resampled_data=np.interp(time_axis,t.mjd,tantdata)
        #print (t.mjd)

        lin1=ax.plot(t.mjd,tantdata,color='black',marker='o',linestyle='none',label='raw')
        lin2=ax.plot(time_axis,resampled_data,color='red',marker='o',linestyle='none',label='Resample: {}'.format(sampling))
        #lin3=ax.plot(t.mjd,Tant_smooth,color='blue',marker='o',linestyle='none')
        #print("{},{},{}".format(data['DATE'],data['Tant'],resampled_data))
        ax.set_xlabel('Universal Time', fontsize=18)
        ax.set_ylabel('Radio Brightness (K)', fontsize=18)

        T_ant_sun=np.median(tantdata)
        lin3=ax.plot([time_axis[0],time_axis[-1]],[T_ant_sun,T_ant_sun],\
                     color='yellow',label='Median: {}'.format(round(T_ant_sun,1)))

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
        plt.legend()
        plt.savefig(ncwd + '/sun-current.png')

    if args.lp:
        ##############################################################################
        # getting data
        ##############################################################################
        # to repopulate full plot, backup should already be stored if needed <sun-current-reduced.all.dat.bak>
        if args.llp:
            allfiles= list_files(mydirstart)
            print('Number of files: {}. Time est: {}s'.format(len(allfiles),len(allfiles)*3))
            os.system('python "{}/metaparse.py" -i "{}" -o "{}/sun-current-reduced.all.metaparse.dat" ' \
                       '--auto -v 0'.format(scripts,allfiles,TMPD))
            os.system('sed "1,2d" "{}/sun-current-reduced.all.metaparse.dat"'.format(TMPD) + \
                " | awk \'{ print $1, $7 }\' >> " + "{}/sun-current-reduced.all.dat".format(TMPD))
            os.system('mv -f "{}/sun-current-reduced.all.dat" "{}/sun-current-reduced.all.dat"'.format(TMPD,ncwd))
        # to just update the plot
        elif args.rp:
            os.system('python "{}/metaparse.py" -i "{}" -o "{}/sun-current-reduced.all.metaparse.dat" ' \
                       '--auto -v 0'.format(scripts,currentfile,TMPD))
            os.system('sed "1,2d" "{}/sun-current-reduced.all.metaparse.dat"'.format(TMPD) + \
                " | awk \'{ print $1, $7 }\' >> " + "{}/sun-current-reduced.all.dat".format(TMPD))
            os.system('cat "{}/sun-current-reduced.all.dat" >> "{}/sun-current-reduced.all.dat"'.format(TMPD,ncwd))

        data = np.loadtxt(ncwd + "/sun-current-reduced.all.dat",dtype=str)
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

        lin1=ax.plot(t.plot_date,tantdata,color='black',marker='o',linestyle='none',label='Raw')
        lin2=ax.plot(time_axis,resampled_data,color='red',marker='o',linestyle='none',label='Resample: {}'.format(sampling))

        ax.set_xlabel('Universal Time', fontsize=18)
        ax.set_ylabel('Radio Brightness (K)', fontsize=18)

        T_ant_sun=np.median(tantdata)
        lin3=ax.plot([tstart.plot_date,tfinish.plot_date],[T_ant_sun,T_ant_sun],\
                     color='yellow',label='Median: {}'.format(round(T_ant_sun,1)))
        # add markers for solar flares
        # get sigma value if values > 3 sigma then mark location with tick
        showtime = strftime("%Y-%m-%d %H:%M:%S", localtime())
        plt.title('Realtime Solar Flux Data - '+showtime)
        plt.legend()
        fig_formatter(ax,'months')
        plt.savefig(ncwd + '/sun-current.all.png',dpi=100)
        plt.savefig(ncwd + '/sun-current.all.lowres.png',dpi=50)

print('[{}] Median Tant: {} across {} integrations'.format(showtime,round(np.median(tantdata),4),len(tantdata)))

os.chdir(cwd)
