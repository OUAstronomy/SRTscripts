#!/usr/bin/env python
'''
Name  : Metadata Parser, metaparse.py
Author: Nick Reynolds
Date  : Fall 2017
Misc  :
  Command line tool to format SRT metadata files to a human-readable format.
  The current output format is given by the example below:
<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    Version:0.2...Made from files: data.txt
    DATE obsn az el freq_MHz Tsys Tant vlsr glat glon azoff eloff source Fstart fstop spacing bw fbw nfreq nsam npoint integ sigma bsw
    2015:218:16:30:49 0 165 63 1421.5000 162.000 1125.714 8.19 36.980 211.928 0.00 0.00 Sun 1420.497 1422.503 0.009375 2.400 2.000 256 1048576 214 5 0.781 0
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
'''
# import modules
from argparse import ArgumentParser
from os import system as _SYSTEM_
from os.path import isfile as _ISFILE_
from glob import glob
import time

# self defined modules
from version import *
from constants import constants
from colours import colours
import utilities

# checking python version
assert assertion()
__version__ = package_version()

# preps the data file, returns the data from the file
def prep(orig,inputfile,returndata=False):
    _SYSTEM_('cp -f {} {}'.format(orig,inputfile))
    _SYSTEM_("sed -i '/entered/d' {}".format(inputfile))
    _SYSTEM_("sed -i '/cmd out of limits/d' {}".format(inputfile))
    _SYSTEM_("sed -i '/Scan/d' {}".format(inputfile))
    _SYSTEM_("sed -i 's/ MHz / /g' {}".format(inputfile))
    _SYSTEM_("sed -i '/*/d' {}".format(inputfile))

    if returndata:
        with open(inputfile,'r') as f:
            f.seek(0)
            allines = [[x for x in line.strip('\n').split(' ') if x != ''] for line in f.readlines()]
        return allines

def info_parse(input_file, output_file):
    with open(input_file, 'r') as f:
        input_data = [line.strip('\n') for line in f.readlines()]

    # full header
    def get_fullheader(section):
        '''
        Assuming section (list,4) list that is the 4 row section of each integration readout
        will output just the headers of the first 3 columns. eg
        DATE***
        FSTART***
        Spectrum***
         582***
        '''
        header=[]
        headcols=[]
        headvals=[]
        for _T_ in range(4):
            if _T_%4 != 3:
                section[_T_] = ' '.join([x.strip(' ') for x in section[_T_].split(' ')\
                                if ((x != ' ') and (x != ''))])
                for i,x in enumerate(section[_T_].split(' ')):
                    if ((x != ' ') and (x != '') and (i%2==0)):
                        headcols.append(x)
                    elif ((x != ' ') and (x != '') and (i%2==1)):
                        headvals.append(x)
        return ' '.join(headcols),' '.join(headvals)
        
    headervals=[]
    i,j = 0,''
    while i < (len(input_data) - 3):
        start,stop = i,i+4
        j = input_data[start:stop]
        if 'azoff' not in j[0]:
            j[0] = j[0].replace("source","azoff 0.00 eloff 0.00 source")
        if i==0:
            headercols,x=get_fullheader(j)
            headervals.append(x)
        else:
            ignore,x=get_fullheader(j)
            headervals.append(x)
        i = stop

    with open(output_file,'w') as f:
        f.write(headercols+'\n')
        for _I_ in headervals:
            f.write("{}\n".format(_I_))

def _main_(args):
    instring = args.infile
    tmpname = args.fout
    auto = args.auto
    logfile = args.log
    verbosity = args.verb
    # Set up message logger            
    if not logfile:
        logfile = ('{}_{}.log'.format(__file__[:-3],time.time()))
    if verbosity >= 3:
        logger = utilities.Messenger(verbosity=verbosity, add_timestamp=True,logfile=logfile)
    else:
        logger = utilities.Messenger(verbosity=verbosity, add_timestamp=False,logfile=logfile)
    logger.header2("Starting {}....".format(__file__[:-3]))

    logger.header2('This program will create and remove numerous temporary files for debugging.')
    logger.debug("Commandline Arguments: {}".format(args))

    _TEMP_ = str(time.time())
    _TEMP0_ = 'TEMPORARY_RM_ERROR_'+_TEMP_+'.txt'
    _TEMP1_ = 'TEMPORARY_METADATA_'+_TEMP_+'.txt'
    _TEMP2_ = 'TEMPORARY_METADATA2_'+_TEMP_+'.txt'
    logger._REMOVE_(_TEMP_)

    # Read in the files
    if _ISFILE_(tmpname):
        logger.warn("Will overwrite:  {}".format(tmpname))
    logger.waiting(auto,seconds=0)

    if len(instring.split(',')) < 2:
        origfiles = [f.strip('[').strip(']').strip(' ') for f in glob(instring+'*') if _ISFILE_(f)]
        if origfiles == []:
            origfiles.append(instring)
    else:
        origfiles = [x.strip('[').strip(']').strip(' ') for x in instring.split(',')]

    logger.success('Files to be analyzed: {}'.format(','.join(origfiles)))
    logger.waiting(auto,seconds=0)

    logger.waiting(auto,seconds=0)

    for _NUM_,_FILE_ in enumerate(origfiles):
        # starting
        print(_FILE_)
        logger.header2('#################################')
        logger.header2("Running file: {}".format(_FILE_))
        prep(_FILE_,_TEMP0_)

        # running parse
        info_parse(_TEMP0_,_TEMP1_)
        logger.success("Finished file: {}".format(_FILE_))
        logger.header2('#################################')
        if _NUM_ == 0:
            _SYSTEM_('cat {} >> {}'.format(_TEMP1_,_TEMP2_))
        else:
            _SYSTEM_('sed -i "1d" {}'.format(_TEMP1_))
            _SYSTEM_('cat {} >> {}'.format(_TEMP1_,_TEMP2_))

    with open(_TEMP2_, 'r') as original: data = original.read()
    with open(_TEMP2_, 'w') as modified: modified.write('Version: {}...Made from files: {}\n{}'.format(__version__,origfiles ,data))
    _SYSTEM_("mv -f " + _TEMP2_ + " " + tmpname)

    logger.success("Finished with all files: {}".format(' | '.join(origfiles)))
    logger.header2("Made file: {} and logfile: {}".format(tmpname,logfile)) 
    logger._REMOVE_(_TEMP_)

# main function
if __name__ == "__main__":
    # -----------------------
    # Argument Parser Setup
    # -----------------------
    description = 'Metadata parser to format srtn metadata into a human-readable form.\n' \
                  '{} Version: {} {}'.format(colours.WARNING,__version__,colours._RST_)

    in_help = 'unique string for name of the file/s to parse, no extension'
    f_help    = 'The output file identifying string'
    a_help    = 'If toggled will run the script non interactively'
    log_help  = 'name of logfile with extension'
    v_help    = 'Integer 1-5 of verbosity level'

    # Initialize instance of an argument parser
    parser = ArgumentParser(description=description)
    parser.add_argument('-i', '--input', type=str, help=in_help, dest='infile',required=True)
    parser.add_argument('-o','--output',type=str, help=f_help,dest='fout',\
        default='master_metaparse_{}_v{}.txt'.format(time.time(),__version__))
    parser.add_argument('--auto',action="store_true", help=a_help,dest='auto')
    parser.add_argument('-l', '--logger',type=str, help=log_help,dest='log')
    parser.add_argument('-v','--verbosity', help=v_help,default=2,dest='verb',type=int)

    # Get the arguments
    args = parser.parse_args()
    _main_(args)

#############
# end of code
