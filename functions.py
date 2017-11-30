import pandas as pd, numpy as np,matplotlib.pyplot as plt
from pandas import Series, DataFrame
import pandas.tools.plotting as plots
import pandas.core.common as com
from pandas.compat import lrange
from matplotlib.artist import setp
import definitions


def norm2baseline(df,respondents,variable):
    #bl_mean_EDA = {}
    vardf = pd.DataFrame()
    for iresp in respondents:
        print 'normalising on ' + iresp
        summed_df = df.loc[iresp]
        try:
            bl_mean_EDA = summed_df.loc[summed_df['tag__info_StudioEventData'] == definitions.baseline_seq][[variable]].values.mean()
            #bl_min_EDA = summed_df.loc[summed_df['tag__info_StudioEventData'] == definitions.baseline_seq][[variable]].values.min()
            #bl_mean_EDA = bl_mean_EDA if bl_mean_EDA>0 else bl_mean_EDA*(-1.0)
            #if df.loc[iresp][variable] < bl_mean_EDA:
            #    print bl_mean_EDA
            #    print df.loc[iresp][variable]
            dfvar = df.loc[iresp][variable] - bl_mean_EDA
            #print dfvar
            vardf = pd.concat([vardf, dfvar])
        except KeyError as e:
            print 'There was a problem with the specified key/index - make sure they match the columns in the input dataset (ECG or EKG or EDA/GSR etc)'
            print e.message


    df[variable]=vardf.values
    return df


def HRFilter(df, respondents):
    master_df = pd.DataFrame()
    for iresp in respondents:
        summed_df = filterHR(df, 'HR', iresp)

        master_df = pd.concat([master_df,summed_df])
    return master_df

# def calcPhasicEDA(df,respondents,Events,variable):
#     s = TSpectrum()
#     master_df = pd.DataFrame()
#     #loop over sequences
#     phasichist = TH1F("phasichist", "phasic part per sequence", nbins, 0, nbins)
#
#     #loop over respondents
#     hist = TH1F(str(respondent) + "hist", str(respondent) + "hist", int((respdataarray.index[-1] * binscale)),
#                 respdataarray.index[0], respdataarray.index[-1])
#
#     #fill hist with iresp'th values
#     for i in range(0, len(respdataarray) - 1):
#         hist.Fill(respdataarray.index[i], respdataarray.values[i])
#
#     accumulated=0
#     first = True
#     fig, ax = plt.subplots(nrows=1)
#     for evt in Events:
#         summed_df = df.loc[respondents[0]]
#         summed_df = summed_df.loc[summed_df['tag__info_StudioEventData'] ==evt][['position', variable]]
#         # tags might occour at different positions if there is a user defined wait between movies - thus the first position in the sequence should be set to zero for plotting.
#         summed_df['position'] = (summed_df['position'] - summed_df['position'].min()) / 1000. +accumulated
#         summed_df = summed_df.set_index('position')
#         ax = summed_df.plot(kind="line",cmap=sns.cubehelix_palette(as_cmap=True), label="first", legend=True, ax=ax, title=variable+"total length")
#         if first: ax.set_xlim([0,df['position'].max()/1000-df['position'].min()/1000])
#         fig.show()
#         #loop over respondents and sum the variable to be plottet
#         if len(respondents) > 1:
#             for i in range(1, len(respondents)):
#                 df1 = df.loc[respondents[i]]
#                 df1 = df1.loc[df1['tag__info_StudioEventData'] == evt][['position', variable]]
#                 #position is incremented (by 'accumulated' for each event/sequence being plotted in total plot:
#                 df1['position'] = (df1['position'] - df1['position'].min()) / 1000. + accumulated
#                 df1 = df1.set_index('position') #color=(i/float(len(respondents)),i/float(len(respondents)),i/float(len(respondents))),
#                 df1.plot(label=respondents[i].decode('latin'), ax=ax, legend=True, title=variable+"Total measurement")  # , c=i/len(respondents))
#                 summed_df = summed_df.add(df1, fill_value=0)
#         accumulated += summed_df.index.max() - summed_df.index.min()
#         master_df = pd.concat([master_df,summed_df])
#         first=False
#     ax.legend([x.decode('latin') for x in respondents])
#     fig = ax.get_figure()
#     fig.set_size_inches(40, 10, forward=True)
#     fig.savefig('./out/'+variable+'Total_length_respondent.png')
#     plt.close()
#     first=False
#     del fig, ax
#     #ax_sum = master_df.plot(kind='area', label="first", legend=True,stacked=False)#cmap=sns.cubehelix_palette(as_cmap=True),
#     fig, ax_sum = plt.subplots(nrows=1)
#     functions.Profile(x=master_df.index.values, y=master_df[variable].values, nbins=int(master_df.index.max()+1
#                                                                                         ),
#                                xmin=master_df.index.min(), xmax=master_df.index.max(),ax=ax_sum,variable=variable)
#     #fig = ax_sum.get_figure()
#     ##fig.set_size_inches(40, 10, forward=True)
#     #fig.savefig('./out/'+variable+'Total_length_summed.png')
#     #ax_sum = sns.regplot(x=master_df.index.values, y=master_df[variable].values, x_bins=master_df.index.max(), fit_reg=None)
#     fig = ax_sum.get_figure()
#     fig.set_size_inches(40, 10, forward=True)
#     fig.savefig('./out/'+variable+'_Total_length_summed.png')
#     #master_df[variable] = master_df[variable].mean()
#     #ax_mean = master_df.plot(kind='area', label="first", legend=True,stacked=False)
#     #fig = ax_mean.get_figure()
#     #fig.savefig('./out/'+variable+'Total_length_average.png')
#     plt.close()
#     del fig
#     fig, ax_sum = plt.subplots(nrows=1)
#     functions.Profile(x=master_df.index.values, y=master_df[variable].values, nbins=int(master_df.index.max()/3),
#                       xmin=master_df.index.min(), xmax=master_df.index.max(), ax=ax_sum, variable=variable)
#     # fig = ax_sum.get_figure()
#     ##fig.set_size_inches(40, 10, forward=True)
#     # fig.savefig('./out/'+variable+'Total_length_summed.png')
#     # ax_sum = sns.regplot(x=master_df.index.values, y=master_df[variable].values, x_bins=master_df.index.max(), fit_reg=None)
#     fig = ax_sum.get_figure()
#     fig.set_size_inches(40, 10, forward=True)
#     fig.savefig('./out/' + variable + '_Total_length_summed_3secbins.png')
#     fig, ax_sum = plt.subplots(nrows=1)
#     functions.Profile(x=master_df.index.values, y=master_df[variable].values, nbins=int(master_df.index.max() / 10),
#                       xmin=master_df.index.min(), xmax=master_df.index.max(), ax=ax_sum, variable=variable)
#     # fig = ax_sum.get_figure()
#     ##fig.set_size_inches(40, 10, forward=True)
#     # fig.savefig('./out/'+variable+'Total_length_summed.png')
#     # ax_sum = sns.regplot(x=master_df.index.values, y=master_df[variable].values, x_bins=master_df.index.max(), fit_reg=None)
#     fig = ax_sum.get_figure()
#     fig.set_size_inches(40, 10, forward=True)
#     fig.savefig('./out/' + variable + '_Total_length_summed_10secbins.png')
#     # master_df[variable] = master_df[variable].mean()
#     # ax_mean = master_df.plot(kind='area', label="first", legend=True,stacked=False)
#     # fig = ax_mean.get_figure()
#     # fig.savefig('./out/'+variable+'Total_length_average.png')
#     plt.close()
#     del fig

def filterHR(df, variable, iresp=""):
    print 'handling HR' + iresp
    summed_df =  df.loc[iresp] if iresp else df
    # a maximum of 1,2s between r-peaks is allowed resulting in a minimum of 50 bpm. Set all value below 50bpm to be the mininal measured hearth rate above 50bpm
    minval = summed_df[variable][summed_df[variable] > 50].values.min()
    summed_df.loc[summed_df[variable] < 50, variable] = minval
    # summed_df[summed_df[variable]<50] = summed_df[variable][summed_df[variable]>50].values.min()
    # A maximum of 180 bpm is allowed, so higher is set to the global max. Set all value above 180 bpm to be the maximal measured hearth rate below 180 bpm
    maxval = summed_df[variable][summed_df[variable] < 180].values.max()
    summed_df.loc[summed_df[variable] > 180, variable] = maxval
    # summed_df[summed_df[variable]>180] = summed_df[variable][summed_df[variable]<180].values.max()
    # loop over values in HR and check if variation is greater than 20%
    dfrange = len(summed_df[variable])
    # fuck it arrays:
    this = summed_df[variable].values
    # test = summed_df.HR.apply(lambda x: x.shift(-1) if abs(x.shift(1) - x) > 0.2 else x)
    for i in range(2, dfrange):
        now = this[i]
        last = this[i - 1]
        diff = abs((now - last) / now)
        if diff > 0.20:  # if greater than 20%
            this[i] = last
            # summed_df.loc[i,variable]=last[i]
    return summed_df


def Profile(x,y,nbins,xmin,xmax,ax,variable):
    if nbins == 0: nbins=1
    df = DataFrame({'x' : x , 'y' : y})

    binedges = xmin + ((xmax-xmin)/nbins) * np.arange(nbins+1)
    df['bin'] = np.digitize(df['x'],binedges)

    bincenters = xmin + ((xmax-xmin)/nbins)*np.arange(nbins) + ((xmax-xmin)/(2*nbins))
    ProfileFrame = DataFrame({'bincenters' : bincenters, 'N' : df['bin'].value_counts(sort=False)},index=range(1,nbins+1))

    bins = ProfileFrame.index.values
    for bin in bins:
        ProfileFrame.ix[bin,'ymean'] = df.ix[df['bin']==bin,'y'].mean()
        ProfileFrame.ix[bin,'yStandDev'] = df.ix[df['bin']==bin,'y'].std(skipna=True)
        ProfileFrame.ix[bin,'yMeanError'] = ProfileFrame.ix[bin,'yStandDev'] / np.sqrt(ProfileFrame.ix[bin,'N'])
    try:
        test=ProfileFrame['ymean']
    except KeyError:
        test=0

    return ax.errorbar(x=ProfileFrame['bincenters'], y=ProfileFrame['ymean'], yerr=np.array(ProfileFrame['yMeanError']), linestyle='-', marker='.')
    #ax.errorbar(ProfileFrame['bincenters'], ProfileFrame['ymean'])
    #ax.errorbar(ProfileFrame['bincenters'], ProfileFrame['ymean'], yerr=ProfileFrame['yMeanError'], xerr=(xmax-xmin)/(2*nbins), fmt=None)
    #return ax


def Profile_Matrix(frame):
  #Much of this is stolen from https://github.com/pydata/pandas/blob/master/pandas/tools/plotting.py

    range_padding=0.05

    df = frame._get_numeric_data()
    n = df.columns.size

    fig, axes = plots._subplots(naxes=n*n, squeeze=False)

    # no gaps between subplots
    fig.subplots_adjust(wspace=0, hspace=0)

    mask = com.notnull(df)

    boundaries_list = []
    for a in df.columns:
        values = df[a].values[mask[a].values]
        rmin_, rmax_ = np.min(values), np.max(values)
        rdelta_ext = (rmax_ - rmin_) * range_padding / 2.
        boundaries_list.append((rmin_ - rdelta_ext, rmax_+ rdelta_ext))

    for i, a in zip(lrange(n), df.columns):
        for j, b in zip(lrange(n), df.columns):

            common = (mask[a] & mask[b]).values
            nbins = 100
            (xmin,xmax) = boundaries_list[i]

            ax = axes[i, j]
            Profile(df[a][common],df[b][common],nbins,xmin,xmax,ax)

            ax.set_xlabel('')
            ax.set_ylabel('')

            #plots._label_axis(ax, kind='x', label=b, position='bottom', rotate=True)
            #plots._label_axis(ax, kind='y', label=a, position='left')

            if j!= 0:
                ax.yaxis.set_visible(False)
            if i != n-1:
                ax.xaxis.set_visible(False)

    for ax in axes.flat:
        setp(ax.get_xticklabels(), fontsize=8)
        setp(ax.get_yticklabels(), fontsize=8)

    return axes