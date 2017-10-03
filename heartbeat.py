#from http://www.paulvangent.com/2016/03/30/analyzing-a-discrete-heart-rate-signal-using-python-part-3/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from definitions import *
import plotting,functions

measures = {}


def get_data(filename):
    dataset = pd.read_csv(filename)
    return dataset


def rolmean(dataset, hrw, fs, aveAmp):
    #mov_avg = pd.rolling_mean(dataset[EKG_data], window=(hrw * fs))
    mov_avg=dataset[EKG_data].rolling(window=(hrw * fs), center=False).mean()
    avg_hr = (np.mean(dataset[EKG_data]))
    mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
    mov_avg = [x * aveAmp for x in mov_avg]
    dataset['rollingmean'] = mov_avg


def detect_peaks(dataset, clean=True):
    window = []
    peaklist = []
    indexlist = []
    listpos = 0
    findPeaksAboveMovAve(dataset, indexlist, listpos, peaklist, window)

    measures['peaklist'] = peaklist
    measures['ybeat'] = [dataset[EKG_data][x] for x in indexlist]
    ybeats = measures['ybeat']
    if clean:
        rpeakDistanceFilter2(measures, dataset, ybeats, indexlist, peaklist)


def findPeaksAboveMovAve(dataset, indexlist, listpos, peaklist, window):
    for datapoint in dataset[EKG_data]:
        rollingmean = dataset.rollingmean[listpos]
        if (datapoint < rollingmean) and (len(window) < 1):
            listpos += 1
        elif (datapoint >= rollingmean):
            window.append(datapoint)
            listpos += 1
        else:
            try:
                maximum = max(window)
            except ValueError,e:
                test=0
            beatposition = dataset['position'][listpos - len(window) + (window.index(max(window)))]
            indexlist.append(listpos - len(window) + (window.index(max(window))))
            peaklist.append(beatposition)
            window = []
            listpos += 1


def rpeakDistanceFilter(measures,dataset,ybeats, indexlist,peaklist):
    #print 'handling r-peak distance '
    # cleaning of peaks to select best estimates for r-peaks.
    # 1) find peaks within definitions.minpeakdistance and select second last - exception: if amplitude is significantly smaller - select last within top 2 in amplitude.
    ibeat=0

    while ibeat < len(indexlist):
        upper=min(len(indexlist),ibeat+10)
        nextbeats = [x for x in range(ibeat+1,upper) if peaklist[x]-peaklist[ibeat]<minpeakdistance]
        if not nextbeats:
            ibeat+=1
        else:
            #impliment amp check here
            if len(nextbeats)>1 and (ibeat+len(nextbeats))<len(indexlist):
                if ybeats[nextbeats[-1]] >= sorted(ybeats[ibeat:nextbeats[-1]])[-2]: #if last is at least second-largest in range
                    del peaklist[ibeat:nextbeats[len(nextbeats)-1]]
                    del indexlist[ibeat:nextbeats[len(nextbeats)-1]]
                    del ybeats[ibeat:nextbeats[len(nextbeats)-1]]
                else:  # case where last peak is smaller than the two larges in the range - find last of two peaks with greatest amplitude
                    peaksinrange = [[ybeats[x], peaklist[x]] for x in
                                    range(ibeat, ibeat + len(nextbeats))]
                    peaksinrange.sort()
                    lastInTopTwoByAmplitde = peaksinrange[-1][1] if peaksinrange[-1][1] > peaksinrange[-2][1] else \
                    peaksinrange[-2][1]
                    for i2beat in range(min(ibeat + len(nextbeats),len(indexlist)-1), ibeat, -1):
                        if peaklist[i2beat] != lastInTopTwoByAmplitde:
                            del peaklist[i2beat]
                            del indexlist[i2beat]
                            del ybeats[i2beat]
                        else:
                            test=0
            else:
                ibeat += 1
                # #if only one after orignal within range - look at amplitude ratio. Select last if amp_last / amp_2nd > 0.5 - from plot of values
                # #print ybeats[ibeat+1]/ybeats[ibeat]
                #
                # if ybeats[ibeat+1]/ybeats[ibeat] > 0.5:
                #     del peaklist[ibeat]
                #     del indexlist[ibeat]
                #     del ybeats[ibeat]
                # else:
                #     del peaklist[ibeat+1]
                #     del indexlist[ibeat+1]
                #     del ybeats[ibeat+1]
            #ibeat+=1

    # write this to the measures peak and ybeat lists
    measures['ybeat'] = [dataset[EKG_data][indexlist[x]] for x in range(0, len(indexlist))]
    if len(measures['ybeat'])!=len(ybeats):
        print 'check'
    measures['peaklist'] = [peaklist[x] for x in range(0, len(indexlist))]

    if True:
        #handle special case of first peaks
        ## start with special case where first peak is a dobule peak
        loopfirst = True
        while loopfirst:
            firstdiff = measures['peaklist'][1] - measures['peaklist'][0]
            seconddiff = measures['peaklist'][2] - measures['peaklist'][1]
            if firstdiff < minpeakdistance:
                del measures['peaklist'][0]
                del measures['ybeat'][0]
            elif firstdiff < 600 and (seconddiff - firstdiff) > 0.2:
                del measures['peaklist'][0]
                del measures['ybeat'][0]
            else:
                loopfirst = False

        # 2) find peaks further than maxpeakdistance apart (corresponding to HR below a minimum set in definitions - start from second peak to have a difference
        modelPeaks(measures) # peaks are inserted if HR is below thsi minimum set - assumed missing in reconstruction

        #3) double peaks might be left where p-peaks have been reconstructed instead of r-peaks or where distance from last peak exceeded minpeakdistance.
        ##  now remove first of two peaks if they are closer than minpeakdistance
        #or closer than 0.6 seconds together AND the distance between last three peaks is greater than 0.8 seconds.
        removePpeaks(measures,only333=True)

        # 4) after final removal of peaks run once more to model peaks that where not reconstructed
        modelPeaks(measures)
        removePpeaks(measures, only333=True)

def rpeakDistanceFilter2(measures, dataset, ybeats, indexlist, peaklist):
    # print 'handling r-peak distance '
    # cleaning of peaks to select best estimates for r-peaks.
    # 1) find peaks within definitions.minpeakdistance and select second last of two largest.
    ibeat = 0

    while ibeat < len(indexlist):
        upper = min(len(indexlist), ibeat + 10)
        nextbeats = [x for x in range(ibeat + 1, upper) if peaklist[x] - peaklist[ibeat] < minpeakdistance]
        if not nextbeats:
            ibeat += 1
        else:
            if len(nextbeats) > 1 and (ibeat + len(nextbeats)) < len(indexlist):
              # find the two peaks with greatest amplitude in the range and select second last unless amplitude of last is significantly higher
                peaksinrange = [[ybeats[x], peaklist[x]] for x in
                                range(ibeat, ibeat + len(nextbeats)+1)]
                peaksinrange.sort()
                secondlastInTopTwoByAmplitde = peaksinrange[-1][1] if peaksinrange[-1][1] < peaksinrange[-2][1] else peaksinrange[-2][1]
                biggestandorsecondlargest = peaksinrange[-1][1] if (peaksinrange[-1][0] / peaksinrange[-2][0]) > 1.5 else secondlastInTopTwoByAmplitde
                for i2beat in range(min(ibeat + len(nextbeats), len(indexlist) - 1), ibeat-1, -1): #from ibeat+len + 1, -1 steps ending at ibeat (-1)
                    if peaklist[i2beat] != biggestandorsecondlargest:
                        del peaklist[i2beat]
                        del indexlist[i2beat]
                        del ybeats[i2beat]
            else:
                ibeat += 1

    # write this to the measures peak and ybeat lists
    measures['ybeat'] = [dataset[EKG_data][indexlist[x]] for x in range(0, len(indexlist))]
    if len(measures['ybeat']) != len(ybeats):
        print 'check'
    measures['peaklist'] = [peaklist[x] for x in range(0, len(indexlist))]

    if True:
        # handle special case of first peaks
        ## start with special case where first peak is a dobule peak
        loopfirst = True
        while loopfirst:
            firstdiff = measures['peaklist'][1] - measures['peaklist'][0]
            seconddiff = measures['peaklist'][2] - measures['peaklist'][0]
            if firstdiff < minpeakdistance:
                del measures['peaklist'][1]
                del measures['ybeat'][1]
            #elif firstdiff < 600 and (seconddiff - firstdiff) > 0.2:
            #    del measures['peaklist'][1]
            #    del measures['ybeat'][1]
            else:
                loopfirst = False

        # 2) find peaks further than maxpeakdistance apart (corresponding to HR below a minimum set in definitions - start from second peak to have a difference
        modelPeaks(
            measures)  # peaks are inserted if HR is below thsi minimum set - assumed missing in reconstruction

        # 3) double peaks might be left where p-peaks have been reconstructed instead of r-peaks or where distance from last peak exceeded minpeakdistance.
        ##  now remove LAST of two peaks if they are closer than minpeakdistance
        # or closer than 0.6 seconds together AND the distance between last three peaks is greater than 0.8 seconds.
        removePpeaks(measures, only333=True)

        # 4) after final removal of peaks run once more to model peaks that where not reconstructed
        modelPeaks(measures)
        removePpeaks(measures, only333=True)
    '''
    newbeats = [dataset[EKG_data][indexlist[x]] for x in range(0, len(indexlist) - 1) if
                (peaklist[x + 1] - peaklist[x]) > 15 or (dataset[EKG_data][indexlist[x]] - dataset[EKG_data][
                    indexlist[x + 1]]) > 1000]  # [measures['ybeat'][0]]+
    newpeaks = [peaklist[x] for x in range(0, len(indexlist) - 1) if (peaklist[x + 1] - peaklist[x]) > 15 or (
    dataset[EKG_data][indexlist[x]] - dataset[EKG_data][indexlist[x + 1]]) > 1000]  # [peaklist[0]]
    indexlist = [indexlist[x] for x in range(0, len(indexlist) - 1) if (peaklist[x + 1] - peaklist[x]) > 15 or (
    dataset[EKG_data][indexlist[x]] - dataset[EKG_data][indexlist[x + 1]]) > 1000]  # [peaklist[0]]
    measures['ybeat'] = newbeats
    measures['peaklist'] = newpeaks
    peaklist = measures['peaklist']
    # select beats greater than
    newbeats = [dataset[EKG_data][indexlist[x]] for x in range(0, len(peaklist) - 1) if
                (peaklist[x + 1] - peaklist[x]) > 10]  # [measures['ybeat'][0]]+
    newpeaks = [peaklist[x] for x in range(0, len(peaklist) - 1) if
                (peaklist[x + 1] - peaklist[x]) > 10]  # [peaklist[0]]
    measures['ybeat'] = newbeats
    measures['peaklist'] = newpeaks
    '''


def removePpeaks(measures,only333):
    ibeat = 3
    while ibeat < len(measures['peaklist']) - 1:
        last3diff = (measures['peaklist'][ibeat - 1] - measures['peaklist'][ibeat - 3]) / 2.
        nextdiff = measures['peaklist'][ibeat + 1] - measures['peaklist'][ibeat]
        #if nextdiff < minpeakdistance:
        #    del measures['peaklist'][ibeat+1]
        #    del measures['ybeat'][ibeat+1]
        #    ibeat += 1
        #if(nextdiff < 600 and last3diff > 800) and only333:
        if nextdiff < 600 and only333:
            del measures['peaklist'][ibeat+1]
            del measures['ybeat'][ibeat+1]
            ibeat += 1
        else:
            ibeat += 1

def modelPeaks(measures):
    ibeat = 1
    while ibeat < len(measures['peaklist']) - 1:
        lastdiff = measures['peaklist'][ibeat] - measures['peaklist'][ibeat - 1]
        nextdiff = measures['peaklist'][ibeat + 1] - measures['peaklist'][ibeat]
        if nextdiff >= maxpeakdistance:
            if nextdiff < maxpeakdistance*2.0 and lastdiff>minpeakdistance:
                measures['peaklist'].insert(ibeat + 1, measures['peaklist'][ibeat] + lastdiff)
                measures['ybeat'].insert(ibeat + 1, measures['ybeat'][ibeat])
                ibeat += 1
            else:
                if nextdiff > maxpeakdistance*3.0: ibeat+=1
                else:
                    # find distance and divide by maxpeakdistance rounding down
                    peak_diff = int((measures['peaklist'][ibeat + 1] - measures['peaklist'][ibeat]) / maxpeakdistance)
                    # then create equidistant peaks and insert all
                    for ipeak in range(1, peak_diff):
                        measures['peaklist'].insert(ibeat + ipeak, measures['peaklist'][ibeat] + ipeak * lastdiff)
                        measures['ybeat'].insert(ibeat + ipeak, measures['ybeat'][ibeat])
                    ibeat += 1
        else:
            ibeat += 1


def calc_RR(fs):
    measures['RR_diff'] = np.diff(measures['peaklist'])#[measures['peaklist'][x+1]-measures['peaklist'][x] for x in range(len(measures['peaklist'])-1)]
    measures['RR_sqdiff'] = [math.pow(measures['peaklist'][x+1]-measures['peaklist'][x],2) for x in range(len(measures['peaklist'])-1)]

def calc_ts_measures():
    RR_list = measures['peaklist']
    RR_diff = measures['RR_diff']
    RR_sqdiff = measures['RR_sqdiff']
    measures['bpm'] = 60000 / np.mean(RR_diff)
    measures['ibi'] = np.mean(RR_list)
    measures['sdnn'] = np.std(RR_list)
    measures['sdsd'] = np.std(RR_diff)
    measures['rmssd'] = np.sqrt(np.mean(RR_sqdiff))
    NN20 = [x for x in RR_diff if (x>20)]
    NN50 = [x for x in RR_diff if (x>50)]
    measures['nn20'] = NN20
    measures['nn50'] = NN50
    measures['pnn20'] = float(len(NN20)) / float(len(RR_diff))
    measures['pnn50'] = float(len(NN50)) / float(len(RR_diff))



def fft(measures,dataset,fs):
    from scipy.interpolate import interp1d  # Import the interpolate function from SciPy

    peaklist = measures['peaklist']  # First retrieve the lists we need
    RR_list = measures['RR_list']

    RR_x = peaklist[1:]  # Remove the first entry, because first interval is assigned to the second beat.
    RR_y = RR_list  # Y-values are equal to interval lengths

    RR_x_new = np.linspace(RR_x[0], RR_x[-1], RR_x[
        -1])  # Create evenly spaced timeline starting at the second peak, its endpoint and length equal to position of last peak

    f = interp1d(RR_x, RR_y, kind='cubic')  # Interpolate the signal with cubic spline interpolation
    print f(250)
    #Returns 997.619845418, the Y value at x=250
    # plt.title("Original and Interpolated Signal")
    # plt.plot(RR_x, RR_y, label="Original", color='blue')
    # plt.plot(RR_x_new, f(RR_x_new), label="Interpolated", color='red')
    # plt.legend()
    # plt.show()
    # # Now to find the frequencies that make up the interpolated signal,
    # use numpys fast fourier transform np.fft.fft() method, calculate sample spacing, convert sample bins to Hz and plot:
    # Set variables
    n = len(dataset[EKG_data])  # Length of the signal
    frq = np.fft.fftfreq(len(dataset[EKG_data]), d=((1 / fs)))  # divide the bins into frequency categories
    frq = frq[range(n / 2)]  # Get single side of the frequency range

    # Do FFT
    Y = np.fft.fft(f(RR_x_new)) / n  # Calculate FFT
    Y = Y[range(n / 2)]  # Return one side of the FFT

    # Plot
    plt.title("Frequency Spectrum of Heart Rate Variability")
    plt.xlim(0, 0.6)  # Limit X axis to frequencies of interest (0-0.6Hz for visibility, we are interested in 0.04-0.5)
    plt.ylim(0, 100)  # Limit Y axis for visibility
    plt.plot(frq, abs(Y))  # Plot it
    plt.xlabel("Frequencies in Hz")
    plt.show()



    #plt.xlabel("Frequencies in Hz")
   # plt.show()
  #  plt.plot(dataset[EKG_data], alpha=0.5, color='blue', label="raw signal")
  #  plt.plot(dataset.rollingmean, color='green', label="moving average")
  #  plt.scatter(peaklist, ybeat, color='red', label="average: %.1f BPM" % measures['bpm'])
  #  plt.legend(loc=4, framealpha=0.6)
    # The module dict now contains all the variables computed over our signal:
    ##bpm = measures['bpm']  # beats per minute
    #ibi = measures['ibi']  # mean Inter Beat Interval
    #sdnn = measures['sdnn']  # standard deviation of all R-R intervals
    #sdsd = measures['sdsd']  # standard deviation of the differences between all subsequent R-R intervals
    #rmssd = measures['rmssd']  # root of the mean of the list of squared differences


    # RR_list = measures['RR_list']
    # RR_diff = measures['RR_diff']
    # RR_sqdiff = measures['RR_sqdiff']
    # measures['bpm'] = 60000 / np.mean(RR_list)
    # measures['ibi'] = np.mean(RR_list)
    # measures['sdnn'] = np.std(RR_list)
    # measures['sdsd'] = np.std(RR_diff)
    # measures['rmssd'] = np.sqrt(np.mean(RR_sqdiff))

def process(iresp,dataset, hrw, fs): # Remember; hrw was the one-sided window size (we used 0.75) and fs was the sample rate (sample file is recorded at 100Hz - ours at 32ms intervals = 31.25 Hz)
        global measures
        measures = {}
        rolmean(dataset, hrw, fs,mov_ave_amplification)
        detect_peaks(dataset, clean=True)
        calc_RR(fs)
        if 60000./measures['RR_diff'].mean() > 100:
            for movAve in np.arange(mov_ave_amplification, 2.0, 0.1):
                rolmean(dataset, hrw, fs, movAve)
                detect_peaks(dataset, clean=True)
                calc_RR(fs)
                print 'used a moving average multiplication of ' + str(movAve) + ' for dataset ' + str(iresp)
                print 60000./measures['RR_diff'].mean()
                if 60000./measures['RR_diff'].mean() < 100: break

        calc_ts_measures()
        #plotting.plotHR(measures)
        measures['HR'] = [60000 / x for x in measures['RR_diff']]
        measures['HR'].append(measures['HR'][-1])
        #filter max and min allowed HR after reconstruction of peaks and HR calculation:
        measures['HR'] = [limit(x,measures['HR']) for x in range(len(measures['HR']))]
        dataset=dataset.set_index('position',drop=True)
        returndf = dataset.loc[measures['peaklist']]
        returndf['HR'] = measures['HR']
        #returndf['HRman']=functions.filterHR(returndf, 'HRman')
        return returndf

def limit(x,measures):
    if measures[x] > 100:
        if x>0:
            return measures[x-1]
        else:
            return 100
    elif measures[x]<40:
        if x>0:
            return measures[x-1]
        else:
            return 40
    else:
        return measures[x]
def calcHeartbeat(fulldataset,respondents):#dataset = get_data("fullsample_rollercoster_ask.csv") #noisy
    summed_df = pd.DataFrame()
    for iresp in respondents:
        dataset = fulldataset.loc[iresp]
        #dataset.index = dataset['position']
        dataset.reset_index(drop=True,inplace=True)
        HRdataset = process(iresp,dataset, hrw, fs)
        # #The module dict now contains all the variables computed over our signal:
        # bpm = measures['bpm'] #beats per minute
        # ibi = measures['ibi'] #mean Inter Beat Interval
        # sdnn = measures['sdnn'] # standard deviation of all R-R intervals
        # sdsd = measures['sdnn'] # standard deviation of the differences between all subsequent R-R intervals
        # rmssd = measures['rmssd'] #root of the mean of the list of squared differences

        #Remember that you can get a list of all dictionary entries with "keys()":
        #print measures.keys()
        #plotHR()
        dataset = dataset.set_index('position')
        plotting.plotter(dataset,"heart rate",measures,iresp)
        print 'concatting'
        HRdataset['resp'] = iresp
        summed_df = pd.concat([summed_df, HRdataset])

        #fft(measures,dataset,fs)
    return summed_df
    
    
