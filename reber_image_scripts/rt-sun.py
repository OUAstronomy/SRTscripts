import numpy as np
from matplotlib import font_manager
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import ScalarFormatter
from astropy.time import Time
import matplotlib.dates as mdates
from scipy.signal import medfilt as medfilt
from time import gmtime, strftime, localtime

##############################################################################
# getting data
##############################################################################
os.system('cp ~/sun-current.dat .')
os.system('python metaparse.py -i sun-current.dat -o sun-current-reduced.dat --auto -l metaparse.log -v0')
data = np.loadtxt("sun-current-reduced.dat",skiprows=2,dtype=str)

t = Time(data[:,0], format='yday', scale='utc')
tantdata = [float(x) for x in data[:,6]]

##############################################################################
# short time plot
##############################################################################
#tmax=Time(strftime('%Y:%j:18:06:09.3', localtime()),format='yday',scale='utc')
tstart=Time(strftime('%Y:%j:16:37:24.1', localtime()),format='yday',scale='utc')
#tend=Time(strftime('%Y:%j:19:35:21.5', localtime()),format='yday',scale='utc')
tstart=Time(strftime('%Y:%j:13:00:00', localtime()),format='yday',scale='utc')

tbegin=Time('2017:222:00:00:00',format='yday',scale='utc')
tstop=Time('2017:223:00:00:00',format='yday',scale='utc')

decimalyearstart=tbegin.decimalyear
decimalyearstop=tstop.decimalyear

decimalyear=np.linspace(decimalyearstart,decimalyearstop,24)
timepsace=Time(decimalyear,format='decimalyear',scale='utc')


fig=plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
ax.set_xlim(tstart.mjd,tstart.mjd+0.042*12.0)
ax.set_ylim(0,1.01*max(tantdata))

Tant_smooth = medfilt(tantdata, 15)

time_axis=t.mjd[0::15]
resampled_data=np.interp(time_axis,t.mjd,tantdata)
#print (t.mjd)

lin1=ax.plot(t.mjd,tantdata,color='black',marker='o',linestyle='none')
lin2=ax.plot(time_axis,resampled_data,color='red',marker='o',linestyle='none')
#lin3=ax.plot(t.mjd,Tant_smooth,color='blue',marker='o',linestyle='none')
#print("{},{},{}".format(data['DATE'],data['Tant'],resampled_data))
ax.tick_params('both', which='major', length=15, width=1, pad=15)
ax.tick_params('both', which='minor', length=7.5, width=1, pad=15)

ticks_font = font_manager.FontProperties(size=16, weight='normal', stretch='normal')
#for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#            label.set_fontproperties(ticks_font)
#for axis in [ax.xaxis, ax.yaxis]:
#    axis.set_major_formatter(ScalarFormatter())
#    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
for axis in ['top','bottom','left','right']:
   ax.spines[axis].set_linewidth(2)

ax.set_xlabel('Universal Time', fontsize=18)
ax.set_ylabel('Radio Brightness (K)', fontsize=18)

print('Median Tant: ',np.median(tantdata))

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
plt.plot_date(timepsace.plot_date, decimalyear)
plt.gcf().autofmt_xdate()  # orient date labels at a slant
plt.draw()

myFmt = mdates.DateFormatter('%H:%M')
ax.xaxis.set_major_formatter(myFmt)
plt.tight_layout()
plt.savefig('sun-current.png')

##############################################################################
# getting data
##############################################################################
os.system('python metaparse.py -i sun-current.all.dat -o sun-current-reduced.all.dat --auto -l metaparse.log -v0')
data = np.loadtxt("sun-current-reduced.all.dat",skiprows=2,dtype=str)

t = Time(data[:,0], format='yday', scale='utc')
tantdata = [float(x) for x in data[:,6]]

##############################################################################
# long time plot
##############################################################################
tstart=Time(strftime('2017:01:00:00:00'),format='yday',scale='utc')
tfinish=Time(strftime('%Y:%j:23:59:59', localtime()),format='yday',scale='utc')
tbegin=Time('2017:222:00:00:00',format='yday',scale='utc')
tstop=Time('2017:223:00:00:00',format='yday',scale='utc')

decimalyearstart=tbegin.decimalyear
decimalyearstop=tstop.decimalyear

decimalyear=np.linspace(decimalyearstart,decimalyearstop,24)
timepsace=Time(decimalyear,format='decimalyear',scale='utc')


fig=plt.figure(figsize=(25,7))
ax = fig.add_subplot(111)
ax.set_xlim(tstart.mjd,tfinish.mjd)
ax.set_ylim(0,1.01*max(tantdata))

Tant_smooth = medfilt(tantdata, 15)

time_axis=t.mjd[0::15]
resampled_data=np.interp(time_axis,t.mjd,tantdata)

lin1=ax.plot(t.mjd,tantdata,color='black',marker='o',linestyle='none')
lin2=ax.plot(time_axis,resampled_data,color='red',marker='o',linestyle='none')
ax.tick_params('both', which='major', length=15, width=1, pad=15)
ax.tick_params('both', which='minor', length=7.5, width=1, pad=15)

ticks_font = font_manager.FontProperties(size=16, weight='normal', stretch='normal')
for axis in ['top','bottom','left','right']:
   ax.spines[axis].set_linewidth(2)

ax.set_xlabel('Universal Time', fontsize=18)
ax.set_ylabel('Radio Brightness (K)', fontsize=18)

print('Median Tant: ',np.median(tantdata))

T_ant_sun=np.median(tantdata)
showtime = strftime("%Y-%m-%d %H:%M:%S", localtime())
plt.title('Realtime Solar Flux Data - '+showtime)
plt.plot_date(timepsace.plot_date, decimalyear)
plt.gcf().autofmt_xdate()  # orient date labels at a slant
plt.draw()

myFmt = mdates.DateFormatter('%H:%M')
ax.xaxis.set_major_formatter(myFmt)
plt.tight_layout()
plt.savefig('sun-current.all.png')





