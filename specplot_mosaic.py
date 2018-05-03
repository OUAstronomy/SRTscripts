#!/usr/bin/env python
'''
Name  : Spectrum Reduction, specreduc.py
Author: Nickalas Reynolds
Date  : Fall 2017
Misc  : Will reduce the 1d spectra data from the specparse program
        Will output numerous plots along the way and ask if you want to delete the intermediate steps at the end
'''

# import standard modules
from sys import version_info,exit
from os import system as _SYSTEM_
from os import getcwd
from os.path import isfile
from copy import deepcopy
from glob import glob
from argparse import ArgumentParser
import time

# import nonstandard modules
import numpy as np
from astropy.table import Table
from astropy.io import ascii
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from scipy.optimize import curve_fit
from scipy.integrate import trapz
ticks_font = mpl.font_manager.FontProperties(size=16, weight='normal', stretch='normal')

# import custom modules
from colours import colours
from constants import constants
import utilities
from version import *

# checking python version
assert assertion()
__version__ = package_version()

####################################################################################
# prepare mask lasso command
####################################################################################
class SelectFromCollection(object):
    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, self.Npts).reshape(self.Npts, -1)

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero([path.contains_point(xy) for xy in self.xys])[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

####################################################################################
# plotting command
####################################################################################
class plotter(object):
    def __init__(self,title,logger=None,size=[10,7]):
        self.size   = size
        self.title  = title
        self.logger = logger
        self.data   = {}

    def open(self,numsubs=(1,1),xlabel=None,ylabel=None):
        self.numsubs = numsubs
        self.f = plt.subplots(nrows=numsubs[0], ncols=numsubs[1],figsize=self.size)
        self.formats(xlabel,ylabel)

    def formats(self,xlabel=None,ylabel=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.f[1].tick_params('both', which='major', length=15, width=1, pad=15)
        self.f[1].tick_params('both', which='minor', length=7.5, width=1, pad=15)
        self.f[1].set_ylabel(self.ylabel, fontsize=18)
        self.f[1].set_xlabel(self.xlabel, fontsize=18)
        self.f[1].set_title(self.title)

    def scatter(self,x,y,datalabel,**kwargs):
        self.data[datalabel] = self.f[1].scatter(x,y,**kwargs)

    def plot(self,x,y,datalabel,**kwargs):
        self.data[datalabel] = self.f[1].plot(x,y,**kwargs)

    def int(self):
        plt.ion()

    def draw(self):
        plt.legend()
        plt.draw()

    def selection(self,label):
        temp      = []
        msk_array = []
        while True:
            selector = SelectFromCollection(self.f[1], self.data[label],0.1)
            self.logger.header2("Draw mask regions around the non-baseline features...")
            self.draw()
            self.logger.pyinput('[RET] to accept selected points')
            temp = selector.xys[selector.ind]
            msk_array = np.append(msk_array,temp)
            selector.disconnect()
            # Block end of script so you can check that the lasso is disconnected.
            answer = self.logger.pyinput("(y or [SPACE]/n or [RET]) Want to draw another lasso region")
            plt.show()
            if ((answer.lower() == "n") or (answer == "")):
                self.save('{}_PLOT.pdf'.format(_TEMPB_))
                break
        self.logger.waiting(auto)
        return msk_array

    def save(self,name):
        plt.savefig(name)

    def resetplot(self,title):
        self.title = title
        self.data = {}
        self.f[1].cla()
        self.formats(self.xlabel,self.ylabel)
        self.limits()

    def limits(self,xlim=None,ylim=None):
        if xlim:
            self.f[1].set_xlim(xlim[0],xlim[1])
        if ylim:
            self.f[1].set_ylim(ylim[0],ylim[1])

####################################################################################
# create fitting code for gauss bimodal lines etc
####################################################################################
def gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2./sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)
def bimodal2(x,mu1,sigma1,A1,mu2,sigma2,A2,C):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2) + C
def binning(data,width=3):
    return data[:(data.size // width) * width].reshape(-1, width).mean(axis=1)
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return [idx,array[idx]]
####################################################################################
# main function
####################################################################################
if __name__ == "__main__":
    # -----------------------
    # Argument Parser Setup
    # -----------------------
    description = 'Reads in masterfile output from specparse.py and reduces. ' \
                  'Will flatten baselines, remove RFI, and find the integrated intensity.\n' \
                  'This code isn\'t generalized. the process is as follows: reduce 1 source\n'\
                  '{} Version: {} {}'.format(colours.WARNING,__version__,colours._RST_)

    in_help   = 'name of the file to parse'
    f_help    = 'The output file identifying string'
    rfi       = np.array(
                [[1420.949875,1420.9405],
                 [1420.08,1420.035]])
    log_help  = 'name of logfile with extension'
    v_help    = 'Integer 1-5 of verbosity level'
    stdhelp   = 'standard region multiplication value (float): get this by reducing the data via specplot normally, '\
                'find the integrated intensity normally and compare.'
    rfihelp   = 'Will try to remove the rfi points that are known: {}'.format(rfi)

    # Initialize instance of an argument parser
    #############################################################################
    parser = ArgumentParser(description=description)
    parser.add_argument('-i', '--input',  type=str, help=in_help, dest='fin',required=True)
    parser.add_argument('-s', '--stdreg', type=float, help=stdhelp, dest='spec',default = 1.)
    parser.add_argument('-p','--plot',  action='store_true', help='Plot all sources (warning slows computer'\
                                     ,dest='plot')
    parser.add_argument('-r','--rfi',action="store_true", help=rfihelp,dest='rfi')
    parser.add_argument('-o','--output',  type=str, help=f_help,dest='fout',required=True)
    parser.add_argument('-l', '--logger', type=str, help=log_help,dest='log')
    parser.add_argument('-v','--verbosity', help=v_help,default=2,dest='verb',type=int)

    # Get the arguments
    #############################################################################
    args = parser.parse_args()
    orig_datafile = args.fin
    ooutfilename  = 'specplot.' + args.fout
    logfile       = args.log
    verbosity     = args.verb
    auto = False
    retry = -99

    # Set up message logger       
    #############################################################################     
    if not logfile:
        logfile = ('{}/{}_{}.log'.format(getcwd(),__file__[:-3],time.time()))
    logger = utilities.Messenger(verbosity=verbosity, add_timestamp=True,logfile=logfile)
    logger.header1("Starting {}....".format(__file__[:-3]))
    logger.debug("Commandline Arguments: {}".format(args))
    
    # handle files
    #############################################################################
    files = [f for f in glob('*'+ooutfilename+'*') if isfile(f)]
    if files == []:
        files = ['None',]
    logger.failure("Will remove these files: {}\n".format(' | '.join(files)))
    logger.warn('Move these to a directory if you don\'t want these deleted')

    _TEMP_ = str(time.time())
    datafile = 'TEMPORARY_FILE_SPECREDUC_{}_0.txt'.format(_TEMP_) # holds orig data
    _TEMPB_ = 'TEMPORARY_FILE_SPECREDUC_{}'.format(_TEMP_)        # temp file name format
    _TEMP0_ = '{}.txt'.format(_TEMPB_)                            # another data backup for manip    
    _TEMP1_ = '{}_1.txt'.format(_TEMPB_)                          # holds current data and eventually final
    _TEMP2_ = '{}_2.txt'.format(_TEMPB_)                          # holds parameters
    _TEMP3_ = []

    logger.waiting(auto)
    logger._REMOVE_(_TEMP_)
    _SYSTEM_('cp -f ' + orig_datafile + ' ' + datafile)

    # getting firstlines
    #############################################################################
    _SYSTEM_('head -n 2 ' + datafile + ' > ' + _TEMP0_)
    with open(_TEMP0_,'r') as f:
        first = ''.join(f.readlines())
    _SYSTEM_("sed -i '1d' " + datafile)
    with open(datafile, 'r') as f:
        first_line=f.readline().strip('\n').split(" ")
    _SYSTEM_("sed -i '1d' " + datafile)
    data = ascii.read(datafile)

    # to verify correct input
    #############################################################################
    logger.header2("Will reduce these ({}) sources: {}".format(len(first_line),"|".join(first_line)))

    # actual plotting now
    #############################################################################
    divisor     = '' # this will hold the dividing function for normalizing spectra
    rfi_regions = '' # this will hold the x values for the rfi to remove
    fullrms     = '' # holds the rms 
    total_num = 0

    # starting at non-zero source
    #############################################################################
    acstart = ''
    countings = 0
    while True:
        try:
            newstart = logger.pyinput('(y or [SPACE]/[RET] or n) Do you wish to start at a source')
            if(newstart == ' ' ) or (newstart.lower() == 'y'):
                acstart = logger.pyinput('Input source exactly')
            else:
                break
            if acstart in first_line:
                countings = 1
                break
            else:
                logger.debug('Try again')
                continue
        except ValueError:
            continue

    # actual plotting now
    #############################################################################
    total_num = 0
    while total_num < len(first_line):
        if countings == 1:
            total_num = first_line.index(acstart)
            retry = 0
            countings = 0
        if total_num == 0:  
            countings = 0  
            col1 = "vel"
            col2 = "Tant"
            col0 = "vel_vlsr"
            col3 = 'freq'
        else:
            col1 = "vel_{}".format(total_num)
            col2 = "Tant_{}".format(total_num)
            col0 = "vel_vlsr_{}".format(total_num)
            col3 = "freq_{}".format(total_num)

        outfilename = ooutfilename + "_" + first_line[total_num]
        logger.warn('Working on: {}'.format(outfilename))
        with open(_TEMP2_,'w') as _T_:
            _T_.write('Working on: {}\n'.format(outfilename ))
        minvel = min(data[col1])
        maxvel = max(data[col1])
        data.sort([col1])
        spectra_x = deepcopy(data[col1])
        spectra_y = deepcopy(data[col2])      
        minvel = min(spectra_x)
        maxvel = max(spectra_x)
        found = []
        if args.rfi:
            for l in rfi:
                start = find_nearest(data[col3],l[0])[0] - 2
                end = find_nearest(data[col3],l[1])[0]  + 2
                fit = np.polyfit(spectra_x[start:end],spectra_y[start:end],1)
                fit_fn = np.poly1d(fit)
                spectra_y[start:end] = fit_fn(spectra_x[start:end])
                #print(start,end)
                #print(spectra_x[start],spectra_x[end],spectra_y[start],spectra_y[end])
        #print('RFI')
        # plot raw data
        #########################################################################
        if (total_num == 0) or (retry != -99):
            x2label = ''
            x1label = r'V$_{lsr}$ (km/s)'
            ylabel = 'Antenna Temperature (K)'

            interactive = plotter('Raw Data Lasso',logger)
            interactive.int()
            interactive.open((1,1),x1label,ylabel)
            interactive.scatter(spectra_x,spectra_y,'scatter raw')
            interactive.plot(spectra_x,spectra_y,'line raw',color='red',linestyle='steps')
            # prepare mask
            interactive.draw()
            # baseline
            baseline_med=np.median(spectra_y)/1.02
            baseline_ul=baseline_med*1.02
            logger.message('Median of baseline: {} and 2sigma baseline {}'.format(baseline_med,baseline_ul))
            with open(_TEMP2_,'a') as _T_:
                _T_.write('Median of baseline: {} and 2sigma baseline {}'.format(baseline_med,baseline_ul))

            # actual defining mask
            msk_array = interactive.selection('scatter raw')

            # draw and reset
            
            mainplot = plotter('Raw Data',logger)
            mainplot.open((1,1),x1label,ylabel)
            mainplot.plot(spectra_x,spectra_y,'raw data',color='black',linestyle='steps')
            mainplot.draw()
            outfilename_iter =0
            _TEMPNAME = "{}_{}.pdf".format(outfilename,outfilename_iter)
            _TEMP3_.append(_TEMPNAME)
            mainplot.save(_TEMPNAME)

            # need to invert mask to polyfit region
            mask_inv = []
            for i in range(len(msk_array)):
                mask_inv = np.append(mask_inv,np.where(spectra_x == msk_array[i]))
            mask_tot = np.linspace(0,len(spectra_x)-1,num=len(spectra_x))
            mask = np.delete(mask_tot,mask_inv)
            mask = [int(x) for x in mask]

            # show projected baselines
            mainplot.resetplot('Projected Baselines')
            mainplot.plot(spectra_x,spectra_y,'raw',color='black',linestyle='steps')
            mainplot.plot([minvel,maxvel],[baseline_med,baseline_med],'lower',color='red',linestyle='steps')
            mainplot.plot([minvel,maxvel],[baseline_ul,baseline_ul],'upper',color='red',linestyle='steps')
            mainplot.draw()
            outfilename_iter +=1
            _TEMPNAME = "{}_{}.pdf".format(outfilename,outfilename_iter)
            _TEMP3_.append(_TEMPNAME)
            mainplot.save(_TEMPNAME)

            # fitting baseline to higher order polynomial
            newask = ' '
            while (newask.lower() == 'n')or (newask == ' '):
                polyfit = ''
                asking = 0
                while True:
                    try:
                        asking = logger.pyinput('what order polynomial do you want to fit to the baseline (integer) or [RET] for 4? ')
                        if asking == '':
                            polynumfit = 4
                            break
                        polynumfit = int(asking)
                    except ValueError:
                        logger.message('Please input an integer.')
                        continue
                    if polynumfit:
                        break

                # fitting polynomial 4th order to baseline
                fit = np.polyfit(spectra_x[mask],spectra_y[mask],polynumfit)
                fit_fn = np.poly1d(fit)

                # plotting fitted baseline to original image
                
                mainplot.resetplot('Plotting fitted baseline')
                mainplot.plot(spectra_x,spectra_y,'data',color='black',linestyle='steps',label='data')
                mainplot.plot(spectra_x,fit_fn(spectra_x),'model',color='red',linestyle='steps',label='model')
                mainplot.draw()
                newask = logger.pyinput('(y or [RET]/n or [SPACE]) Was this acceptable? ')
                if (newask.lower() == 'y') or (newask == ''):
                    with open(_TEMP2_,'a') as _T_:
                        _T_.write("The polynomial is: \n {}\n".format(fit_fn))
                    break
            divisor = deepcopy(fit_fn)


            outfilename_iter +=1
            _TEMPNAME = "{}_{}.pdf".format(outfilename,outfilename_iter)
            _TEMP3_.append(_TEMPNAME)
            mainplot.save(_TEMPNAME)

        # defining corrected spectra
        spectra_blcorr=args.spec * (deepcopy(spectra_y)-divisor(spectra_x))
        maxt = max(spectra_blcorr)
        mint = min(spectra_blcorr)
        #print('RMS')
        # defining RMS
        if (total_num == 0) or (retry != -99):
            rms=np.std(spectra_blcorr[mask])
            fullrms = rms
            logger.message('RMS Noise: {}K'.format(rms))
            with open(_TEMP2_,'a') as _T_:
                _T_.write('RMS Noise: {}K\n'.format(rms))

        # plotting the corrected baseline
        if (total_num == 0) or (retry != -99):
            mainplot.resetplot('Plotting the corrected baseline')
            mainplot.plot(spectra_x,spectra_blcorr,'data',color='black',linestyle='steps',label='data')
            mainplot.plot([minvel,maxvel],[0,0],'baseline',color='red',linestyle='steps',label='flat baseline')
            mainplot.draw()
            outfilename_iter +=1
            _TEMPNAME = "{}_{}.pdf".format(outfilename,outfilename_iter)
            _TEMP3_.append(_TEMPNAME)
            mainplot.save(_TEMPNAME)

            # define the RFI
            print('Only select noise not falling on the signal, only on baselines...')
            interactive.resetplot('Lasso selection:')
            interactive.formats(x1label,ylabel)
            interactive.scatter(spectra_x,spectra_blcorr,'data',color='black',label='datapoints')
            interactive.plot(spectra_x,spectra_blcorr,'rfi',color='blue',linestyle='steps',label='rfi')
            interactive.plot([minvel,maxvel],[0,0],'flat',color='red',linestyle='steps',label='flat baseline')
            interactive.draw()

            temp = []
            rfi_mask_array = interactive.selection('data')
            rfi_mask = []

            newask = ' '
            _TRY_ =1
            for i in range(len(rfi_mask_array)):
                rfi_mask = np.append(rfi_mask,np.where(spectra_x == rfi_mask_array[i]))
            rfi_mask = [int(x) for x in rfi_mask]
            logger.debug('RFI mask region: {}'.format(','.join(map(str,rfi_mask))))
            rfi_regions = deepcopy(rfi_mask)

        # remove rfi
        if (total_num == 0) or (retry != -99):
            logger.message("Will try fitting with simple polynomial, gaussian, bimodal, or fail")
            rfi_fit_fn_ans=''
            while ((newask.lower() == 'n')or (newask == ' ')) and (len(rfi_mask) > 0):
                _TEMPSPEC_ = spectra_blcorr
                FITX    = np.delete(spectra_x,rfi_mask)
                FITSPEC = np.delete(_TEMPSPEC_,rfi_mask)
                mu = spectra_x[np.where(spectra_blcorr == max(spectra_blcorr))][0]
                gaussrms = abs(spectra_x[rfi_mask[len(rfi_mask)-1]] - spectra_x[rfi_mask[0]])*2.
                # fitting polynomial nth order to baseline
                try:
                    if _TRY_ == 1:
                        logger.warn('Polynomial fit...')
                        rfi_fit = np.polyfit(FITX,FITSPEC,20)
                        rfi_poly_fn = np.poly1d(rfi_fit)
                        rfi_fit_fn = rfi_poly_fn
                        function = rfi_poly_fn(spectra_x)

                    # fit Gaussian
                    elif _TRY_ == 2:
                        logger.warn('Gaussian fit...')
                        _expected1=[mu,gaussrms,np.max(_TEMPSPEC_)]
                        logger.debug("Input params: {}".format(_expected1))
                        _params1,_cov1=curve_fit(gauss,FITX,FITSPEC,_expected1)
                        logger.debug("Fit params: {}".format(_params1))
                        _sigma1=np.sqrt(np.diag(_cov1))
                        function = gauss(spectra_x,*_expected1)

                        rfi_fit_fn = 'gauss(x,mu1,sigma1,A1)' + ','.join(map(str,_params1))

                    elif _TRY_ == 3:
                        logger.warn('Bimodal Gaussian fit...')
                        _expected2=[mu,gaussrms,np.max(_TEMPSPEC_),mu,gaussrms,np.max(_TEMPSPEC_)]
                        logger.debug("Input params: {}".format(_expected2))
                        _params2,_cov2=curve_fit(bimodal,FITX,FITSPEC,_expected2)
                        _sigma2=np.sqrt(np.diag(_cov2))
                        logger.debug("Fit params: {}".format(_params2))
                        function = bimodal(spectra_x,*_expected2)

                        rfi_fit_fn = 'gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)' + ','.join(map(str,_params2))

                    elif _TRY_ >= 4:
                        logger.failure('Auto fitting RFI failed...')
                        functions = ['polynomial','gaussian','bimodal']
                        ans = logger.pyinput("(integer or [RET]) name of better fit {} or set values to zero with [RET]".format(functions))
                        if ans.lower() in functions:
                            _TRY_ = int(functions.index(ans)+1)
                        else:
                            _TEMPSPEC_[rfi_mask] = 0.0
                            break

                    # plotting fitted baseline to original image
                    
                    mainplot.resetplot('Plotting RFI removal')
                    if _TRY_ == 1:
                        for _RFI_ in rfi_mask:
                            logger.debug("Region of RFI: {}".format(_TEMPSPEC_[_RFI_]))
                            _TEMPSPEC_[_RFI_] = rfi_poly_fn(spectra_x[_RFI_]) 
                            logger.debug("Region of RFI after fit: {}".format(_TEMPSPEC_[_RFI_]))
                        mainplot.plot(spectra_x,rfi_poly_fn(spectra_x),'polyfit',color='yellow',linestyle='steps',label='Poly model')
                    elif _TRY_ == 2:
                        for _RFI_ in rfi_mask:
                            logger.debug("Region of RFI: {}".format(_TEMPSPEC_[_RFI_]))
                            _TEMPSPEC_[_RFI_] = gauss(spectra_x[_RFI_],*_params1)
                            logger.debug("Region of RFI after fit: {}".format(_TEMPSPEC_[_RFI_]))
                        mainplot.plot(spectra_x,gauss(spectra_x,*_params1),'gauss',color='red',linestyle='steps',label='Gauss model')
                    elif _TRY_ == 3:
                        for _RFI_ in rfi_mask:
                            logger.debug("Region of RFI: {}".format(_TEMPSPEC_[_RFI_]))
                            _TEMPSPEC_[_RFI_] = bimodal(spectra_x[_RFI_],*_params2)
                            logger.debug("Region of RFI after fit: {}".format(_TEMPSPEC_[_RFI_]))
                        mainplot.plot(spectra_x,bimodal(spectra_x,*_params2),'bimodal',color='orange',linestyle='steps',label='Bimodal model')
                except RuntimeError:
                    logger.failure('Couldn\'t converge on try {}, setting values to zero...'.format(_TRY_))
                    rfi_fit_fn = "Fitter failed...."
                    _TEMPSPEC_[rfi_mask] = 0.0            
                mainplot.plot(spectra_x,_TEMPSPEC_,'data',color='black',linestyle='steps',label='data')
                mainplot.limits(ylim=(-1,1.2*max(spectra_blcorr)))
                mainplot.draw()
                newask = logger.pyinput('(y or [RET]/n or [SPACE]) Is this acceptable? ')
                if (newask.lower() == 'y') or (newask == ''):
                    with open(_TEMP2_,'a') as _T_:
                        _T_.write("The function is: \n{}\n".format(rfi_fit_fn))
                    break
                else:
                    _TRY_ +=1
        else:
            _TEMPSPEC_ = spectra_blcorr
        # draw and reset
        try:
            spectra_blcorr = _TEMPSPEC_
        except:
            pass

        if (total_num == 0) or (retry != -99):
            mainplot.formats(x1label,ylabel)
            mainplot.plot(spectra_x,spectra_blcorr,'corrected',color='black',linestyle='steps',label='corrected')
            mainplot.plot([minvel,maxvel],[0,0],'flat',color='red',linestyle='steps',label='flat baseline')
            mainplot.limits(ylim=(-1,1.2*max(spectra_blcorr)))
            mainplot.draw()
            outfilename_iter +=1
            _TEMPNAME = "{}_{}.pdf".format(outfilename,outfilename_iter)
            _TEMP3_.append(_TEMPNAME)
            mainplot.save(_TEMPNAME)

            # Final correction plot 
            
            mainplot.resetplot('Final corrected plot')
            mainplot.formats(x1label,ylabel)
            mainplot.limits(xlim=(minvel,maxvel),ylim=(mint-1,maxt * 1.1))
            mainplot.plot(spectra_x,spectra_blcorr,'data',color='black',linestyle='steps',label='data')
            mainplot.draw()
            outfilename_iter +=1
            _TEMPNAME = "{}_{}.pdf".format(outfilename,outfilename_iter)
            _TEMP3_.append(_TEMPNAME)
            mainplot.save(_TEMPNAME)

            # intensity estimate
            while True:
                try:
                    intensity_answer = logger.pyinput('Sigma value for Gaussian (integers * rms) or [RET] for default 5 sigma or "none" to skip')
                    if intensity_answer == '':
                        intensity_answer = 5.0
                    elif str(intensity_answer).lower() == 'none':
                        break
                    intensity_answer = float(intensity_answer)
                except ValueError:
                    logger.warn('Please input integer or float.')
                    continue
                if intensity_answer <= 3.:
                    logger.warn('Low signal Gaussian, result maybe incorrect.')
                    logger.warn('Gaussian signal: {}*rms'.format(intensity_answer))
                    break
                if intensity_answer > 3.:
                    logger.message('Gaussian signal: {}*rms'.format(intensity_answer))
                    break
                if str(intensity_answer).lower() != 'none':
                    with open(_TEMP2_,'a') as _T_:
                        _T_.write('Sigma value for Gaussian: {}\n'.format(intensity_answer))
        if (total_num != 0) or (retry != -99):
            med= (np.median(spectra_blcorr)/3.)
            intensity_answer = 5.0
        intensity_mask_guess = []
        while True:
            try:
                if len(intensity_mask_guess) == 0:
                    #print('Guessing intensity')
                    intensity_mask_guess = np.where((spectra_blcorr >= np.abs(intensity_answer * rms)))
                    minint=min(spectra_x[intensity_mask_guess])
                    maxint=max(spectra_x[intensity_mask_guess])
                if intensity_answer == 0:
                    intensity_mask_guess = np.linspace(len(spectra_x)/4-1,3*len(spectra_x)/4-1, num = len(spectra_x)/2)
                if len(intensity_mask_guess) > 0:
                    break
            except ValueError:
                intensity_answer -=1
                continue

        #print('Made it to intensity')
        autoask = 'y'
    
        try:
            if minint > 0:
                pass
        except:
            minint = min(spectra_x)
        try:
            if maxint > 0:
                pass
        except:
            maxint = max(spectra_x)
        try:
            if maxt > 0:
                pass
        except:
            maxint = max(spectra_y)


        if (total_num == 0) or (retry != -99):
            # Intensity line estimate
            mainplot.resetplot('Intensity Line Estimate')
            mainplot.formats(x1label,ylabel)
            mainplot.limits(xlim=(minvel,maxvel),ylim=(mint-1,maxt * 1.1))
            mainplot.plot(spectra_x,spectra_blcorr,'data',color='black',linestyle='steps',label='data')
            mainplot.plot(spectra_x[intensity_mask_guess],np.zeros(len(spectra_x[intensity_mask_guess])),'est',color='blue',linestyle='dotted')
            mainplot.plot([minint,minint],[0,maxt],'lower',color='blue',linestyle='dotted')
            mainplot.plot([maxint,maxint],[0,maxt],'upper',color='blue',linestyle='dotted')
            mainplot.draw()
            outfilename_iter +=1
            _TEMPNAME = "{}_{}.pdf".format(outfilename,outfilename_iter)
            _TEMP3_.append(_TEMPNAME)
            mainplot.save(_TEMPNAME)

            while True:
                try:
                    answer_ok = logger.pyinput("(y or [RET]/n or [SPACE]) Is region guess for the line intensity is okay")
                    if ((answer_ok.lower() == "y") or (answer_ok == "")):
                        intensity_mask = intensity_mask_guess
                        break
                    else:
                        # define the Intensity
                        interactive.resetplot('Lasso selection:')
                        interactive.scatter(spectra_x,spectra_blcorr,'data',color='black')
                        interactive.plot(spectra_x,spectra_blcorr,'dataselect',color='blue',linestyle='steps')
                        interactive.plot([minvel,maxvel],[0,0],'int',color='red',linestyle='steps')
                        interactive.draw()
                        # recovering intensity of line 
                        temp = []
                        intensity_mask_array = lasso.selection('data')
                        intensity_mask = []

                        for i in range(len(intensity_mask_array)):
                            intensity_mask = np.append(intensity_mask,np.where(spectra_x == intensity_mask_array[i]))
                        intensity_mask = [int(x) for x in intensity_mask]

                        # draw and reset
                        try:
                            if minint > 0:
                                pass
                        except:
                            minint = min(spectra_x)
                        try:
                            if maxint > 0:
                                pass
                        except:
                            maxint = max(spectra_x)
                        try:
                            if maxt > 0:
                                pass
                        except:
                            maxint = max(spectra_y)

                        minint=min(spectra_x[intensity_mask])
                        maxint=max(spectra_x[intensity_mask])
                        
                        mainplot.resetplot('With Line Intensity Mask')
                        mainplot.plot(spectra_x,spectra_blcorr,'data',color='black',linestyle='steps')
                        mainplot.plot(spectra_x[intensity_mask],np.zeros(len(spectra_x[intensity_mask])),'bottom',color='blue',linestyle='dotted')
                        mainplot.plot([minint,minint],[0,maxt],'lower',color='blue',linestyle='dotted')
                        mainplot.plot([maxint,maxint],[0,maxt],'upper',color='blue',linestyle='dotted')                
                        mainplot.draw()
                        break
                except ValueError:
                    continue

            mainplot.resetplot('Intensity Mask')
            mainplot.formats(x1label,ylabel)
            mainplot.limits(xlim=(minvel,maxvel),ylim=(mint,maxt * 1.1))
            mainplot.plot(spectra_x,spectra_blcorr,'data',color='black',linestyle='steps',label='Data')
            mainplot.plot(spectra_x[intensity_mask],np.zeros(len(spectra_x[intensity_mask])),'bottom',color='blue',linestyle='dotted')
            mainplot.plot([minint,minint],[0,maxt],'lower',color='blue',linestyle='dotted')
            mainplot.plot([maxint,maxint],[0,maxt],'upper',color='blue',linestyle='dotted')
            mainplot.draw()
            outfilename_iter +=1
            _TEMPNAME = "Final.{}_{}.pdf".format(outfilename,outfilename_iter)
            mainplot.save(_TEMPNAME)
            mainplot.draw()
            logger.waiting(auto)
            plt.show()
            if retry != -99:
                logger.pyinput("[RET]")
                plt.close('all')
                plt.clf()
                plt.close()
            retry = -99

            # showing Intensity Mask
        else:
            retry = -99
            intensity_mask = intensity_mask_guess
            x = [int(x) for x in range(len(spectra_x)) if (x < 30) or (x > 180)]
            rfi_fit = np.polyfit(spectra_x[x],spectra_blcorr[x],2)
            rfi_poly_fn = np.poly1d(rfi_fit)
            rfi_fit_fn = rfi_poly_fn
            spectra_blcorr = spectra_blcorr - rfi_fit_fn(spectra_x)
            holder = []
            for i,j in enumerate(spectra_blcorr):
                if 2<i<len(spectra_blcorr)-2:
                    if (j  >= intensity_answer * rms) and (spectra_blcorr[i+1] >= intensity_answer * rms) and (spectra_blcorr[i-1] >= intensity_answer * rms) :
                        holder.append(i)

            intensity_mask_guess = np.ndarray(len(holder),dtype=int)
            try:
                for i,j in enumerate(holder):
                    intensity_mask_guess[i] = int(j)
                minint=np.min(spectra_x[intensity_mask_guess])
                maxint=np.max(spectra_x[intensity_mask_guess])
                intensity_mask = intensity_mask_guess
            except:
                minint = np.min(spectra_x)
                maxint = np.max(spectra_x)
                intensity_mask = holder
            maxt = np.max(spectra_blcorr)

            if (args.plot):
                mainplot.resetplot('Intensity Mask')
                mainplot.formats(x1label,ylabel)
                mainplot.plot(spectra_x,spectra_blcorr,'data',color='black',linestyle='steps',label='Data')
                mainplot.plot([min(spectra_x),max(spectra_x)],[0,0],'bottom',color='red',linestyle='dotted',label='Baseline')
                mainplot.plot([minint,minint],[0,maxt],'lower',color='blue',linestyle='dotted')
                mainplot.plot([maxint,maxint],[0,maxt],'upper',color='blue',linestyle='dotted')
                mainplot.draw()
                outfilename_iter +=1
                _TEMPNAME = "Final.{}_{}.pdf".format(outfilename,outfilename_iter)
                plt.savefig(_TEMPNAME)
                plt.show()
                autoask = logger.pyinput('(y or [RET]/n or [SPACE]) Is this acceptable? ')
                if (autoask.lower() != 'y') and (autoask != ''):
                    retry = total_num
                    total_num = total_num -1
                    logger.failure("retrying....")
                plt.clf()
            
        if retry == -99:
            # intensity
            intensity=trapz(spectra_blcorr[intensity_mask],intensity_mask)
            chanwidth=abs(max(spectra_x)-min(spectra_x))/len(spectra_x)
            intensity_rms=rms*chanwidth*(float(len(intensity_mask)))**0.5
            logger.message("Intensity: ")
            logger.message("{} +- {} (K km/s)".format(intensity,intensity_rms))
            with open(_TEMP2_,'a') as _T_:
                _T_.write('Intensity: {} +- {} (K km/s)'.format(intensity,intensity_rms))

            # write to file
            try:
                spec_final = Table([data[col3],data[col0],spectra_x,spectra_y,spectra_blcorr], names=('freq','vel_sub', 'vel', 'Tant_raw', 'Tant_corr'))
            except KeyError:
                spec_final = Table([data[col3],spectra_x,spectra_y,spectra_blcorr], names=('freq', 'vel', 'Tant_raw', 'Tant_corr'))           
            ascii.write(spec_final,_TEMP1_,overwrite=True)
            _SYSTEM_('cp -f ' + _TEMP1_ + ' ' + outfilename + "_spectra_corr.txt")
            _SYSTEM_('cp -f ' + _TEMP2_ + ' ' + outfilename + "_parameters.txt")

        if (total_num == 0):
            # close and reset
            ans = ''
            ans = logger.pyinput("[RET] to continue to complete this source or [SPACE] to cancel out...")

            plt.close("all")
            plt.clf()
            plt.close()
        total_num +=1

    logger.pyinput("[RET] to exit")

    # finished
    logger._REMOVE_(_TEMPB_)

    logger.header2("#################################")
    logger.success("Finished with all.")
    logger.message("These are the sources processed: {}".format(' | '.join(first_line)))
    logger.message("These are the files processed: {}".format(orig_datafile))
    files = [f for f in glob(outfilename+'*') if isfile(f)]
    logger.header2("Made the following files: {} and logfile: {}".format(', '.join(files),logfile))
    ans = logger.pyinput("(y or [RET] / n or [SPACE]) if you would like to delete the intermediate files")
    if ans == "" or ans.lower() == 'y':
        for delfile in _TEMP3_:
            logger._REMOVE_(delfile)

    plt.close()

    #############
    # end of code
