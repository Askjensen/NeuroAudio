#!/usr/bin/python
# -*- coding: latin-1 -*-

import IOhandler, plotting, functions,heartbeat
import os, sys, logging
from definitions import *
from ROOT import *

if __name__ == '__main__':
    #chose debug level:
    #debuglevel = logging.DEBUG
    debuglevel = logging.INFO

    IOhandler.ensure_dir("./out/")
#    f = open(os.path.join(sys.path[0], './out/output.txt'), 'w')
    logging.basicConfig(filename='./out/output.txt', level=debuglevel,filemode='w')

    logging.info('Reading in data')
    df,respondents = IOhandler.read_data(datacolumns,folder_path,filename_ext,delimiter)

    logging.info('Choosing events')
    if Events:
        df = IOhandler.select_data(df,Events)
    logging.debug(df)

    #logging.info('Handling r-peak distance (must be between 600 and 1200 ms cite: Psycho.book')
    #df=functions.HRFilter(df, respondents) - needs corrections

    logging.info('Manually calculating HR in bpm')
    dftest = heartbeat.calcHeartbeat(df.ix[:,['position',EKG_data,'tag__info_StudioEventData']],respondents)
    dftest['position'] = dftest.index
    dftest = dftest.set_index('resp')

    #logging.info('Handling r-peak distance (must be between 600 and 1200 ms cite: Psycho.book')
 #th
    logging.info('Normalizing each respondent to their level in baseline')
    df=functions.norm2baseline(df,respondents,eda_data)
    dftest=functions.norm2baseline(dftest,respondents,HR_data)

    logging.info('Plotting EDA')
    plotting.plot_sequence(df,respondents,Events,eda_data,phasic=True)  
    #plotting.plot_total(df,respondents,Events,eda_data)

    logging.info('Plotting HR')
    plotting.plot_sequence(dftest, respondents, Events, HR_data)
    #plotting.plot_total(dftest, respondents, Events, HR_data)


    #logging.info('calculate EDA tonic and phasic EDA')
    #functions.calcPhasicEDA(df,respondents,Events,eda_data)


    test = 0

