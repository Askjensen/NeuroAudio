import matplotlib.pyplot as plt
import seaborn as sns, pandas as pd
import functions, definitions
import pandas.core.common as com
import pandas.tools.plotting as plots
from pandas.compat import lrange
from matplotlib.artist import setp
from ROOT import TH1F,TSpectrum,TCanvas, TFile

def plot_sequence(df,respondents,Events,variable,phasic=False):
    for evt in Events:
        # ROOT hist for calculating phsic component

        summed_df = df.loc[respondents[0]]
        summed_df = summed_df.loc[summed_df['tag__info_StudioEventData']== evt][['position',variable]]
        #tags might occour at different positions if there is a user defined wait between movies - thus the first position in the sequence should be set to zero for plotting.
        summed_df['position'] = (summed_df['position'] - summed_df['position'].min())/1000.
        maxbin = int(summed_df['position'].max()/2)*2
        summed_df = summed_df.set_index('position')
        if variable == definitions.eda_data:
            # add constant to avoid negative values:
            summed_df[variable] = summed_df[variable] + 100
            # normalise to an integral of 1:
            summed_df[variable] = summed_df[variable] / summed_df[variable].sum()
        ax = summed_df.plot(cmap=sns.cubehelix_palette(as_cmap=True),label="first",legend=True, title=evt+variable)
        #max is per sequence and first bin is set to zero, so that max i length of given sequence

        rawedahist = TH1F("rawedahist", "sum of normalized EDA per sequence", maxbin+1, 0, maxbin)
        phasichist = TH1F("phasichist", "phasic part per sequence", maxbin+1, 0, maxbin)
        tonichist = TH1F("tonichist", "tonic part per sequence", maxbin+1, 0, maxbin)
        # make sure errors are sum of squares of weights - wwell actually dont as hists are 'added'
        #phasichist.Sumw2()
        #tonichist.Sumw2()

        if phasic:
            print len(summed_df[variable])
            for ix in summed_df.index:
                rawedahist.Fill(ix, summed_df[variable][ix])
                rawedahist.GetXaxis().SetRangeUser(summed_df.index[1],summed_df.index[-2])
            # add constant to handle negative values. BSS calibrates EDA signal but can introduce a systematic error that sets the lowest arrousal level as negative.
            for i in range(0, rawedahist.GetNbinsX()):
                rawedahist.SetBinContent(i, rawedahist.GetBinContent(i) + 100)
            rawedahist.Scale(1./rawedahist.Integral())
            s = TSpectrum()
            #cmark = TCanvas("c", "c", 1200, 800)
            #cmark.cd()
            #phasichist.Draw()

            #np = s.Search(phasichist, definitions.sigmapeaksinterval, "noMarkov same nobackground", definitions.peakamplitude)
            bg = s.Background(rawedahist, definitions.bgiter, "Compton same")
            #cmark.Update()
            #cmark.SaveAs('./out/PhasicModelling_'+respondents[0]+'_'+evt+'_respondents.png')
            #cmark.SaveAs('./out/PhasicModelling_'+respondents[0]+'_'+evt+'_respondents.root')
            #cmark.Close()
            tonichist.Add(bg)
            phasichist.Add(rawedahist)
            phasichist.Add(bg, -1)

        #loop over respondents
        if len(respondents) > 1:
            for i in range(1,len(respondents)):
                df1 = df.loc[respondents[i]]
                df1 = df1.loc[df1['tag__info_StudioEventData']== evt][['position',variable]]
                df1['position'] = (df1['position'] - df1['position'].min())/1000.
                df1 = df1.set_index('position')
                df1.plot(label=respondents[i].decode('latin'), ax=ax,legend=True, title=evt+variable)#, c=i/len(respondents))
                #resp=respondents[i]
                if phasic:
                    hist = TH1F("loophist", "loophisthist", maxbin+1, 0, maxbin)
                    for ix in df1.index:
                        hist.Fill(ix, df1[variable][ix])
                    # add constant to handle negative values. BSS calibrates EDA signal but can introduce a systematic error that sets the lowest arrousal level as negative.
                    for ix in range(0, hist.GetNbinsX()):
                        hist.SetBinContent(i, hist.GetBinContent(ix) + 100)
                    hist.Scale(1./hist.Integral())
                    cmark = TCanvas("c", "c", 1200, 800)
                    cmark.cd()
                    hist.Draw()
                    s = TSpectrum()
                    np = s.Search(hist, definitions.sigmapeaksinterval, "noMarkov same nobackground", definitions.peakamplitude)
                    bg = s.Background(hist, definitions.bgiter, "Compton same")
                    cmark.Update()
                    cmark.SaveAs('./out/respondents/PhasicModelling_'+respondents[i]+'_'+evt+'_respondents.png')
                    cmark.SaveAs('./out/respondents/PhasicModelling_'+respondents[i]+'_'+evt+'_respondents.root')
                    rawedahist.Add(hist)
                    tonichist.Add(bg)
                    hist.Add(bg,-1)
                    phasichist.Add(hist)
                    #phasichist.Draw()
                    #cmark.Update()
                    #cmark.Close()
                    cmark.Close()
                    del hist
                    #cmark.Close()
                    if variable==definitions.eda_data:
                        #add constant to avoid negative values:
                        df1[variable] = df1[variable]+100
                        #normalise to an integral of 1:
                        df1[variable] = df1[variable]/df1[variable].sum()
                summed_df = summed_df.add(df1, fill_value=0)
        test=0
        if phasic:
            cmark = TCanvas("c", "c", 1200, 800)
            cmark.cd()
            tonichist.Draw()
            cmark.Update()
            cmark.SaveAs('./out/TonicEda_'+evt+'_respondents.png')
            cmark.SaveAs('./out/rootfiles/TonicEda_'+evt+'_respondents.root')
            tonichist.SaveAs('./out/rootfiles/TonicEdaHist_'+evt+'_respondents.root')
            print 'tonic mean of sequence: ' + str(evt) + str(tonichist.GetMean())

            cmark.cd()
            phasichist.Draw()
            cmark.Update()
            cmark.SaveAs('./out/PhasicEda_' + evt + '_respondents.png')
            cmark.SaveAs('./out/rootfiles/PhasicEda_' + evt + '_respondents.root')
            phasichist.SaveAs('./out/rootfiles/PhasicEdaHist_' + evt + '_respondents.root')
            print 'phasic mean of sequence: ' + str(evt) + str(phasichist.GetMean())


            rawedahist.Draw()
            cmark.Update()
            cmark.SaveAs('./out/RawEda_' + evt + '_respondents.png')
            cmark.SaveAs('./out/rootfiles/RawEda_' + evt + '_respondents.root')
            rawedahist.SaveAs('./out/rootfiles/RawEdaHist_' + evt + '_respondents.root')
            print 'raw mean of sequence: ' + str(evt) + str(phasichist.GetMean())


            if definitions.sequence_splitting.has_key(evt):
                try:
                    nx = len(definitions.sequence_names[evt])
                    meanphasic = TH1F("meanphasic", "mean phasic EDA per sequence", nx, 0, nx)
                    meaneda = TH1F("meaneda", "mean EDA per sequence", nx, 0, nx)
                    for i in range(1, nx+1):
                        meanphasic.GetXaxis().SetBinLabel(i,definitions.sequence_names[evt][i-1])
                        meaneda.GetXaxis().SetBinLabel(i, definitions.sequence_names[evt][i-1])
                    for irange in range(1,len(definitions.sequence_splitting[evt])):
                        min=definitions.sequence_splitting[evt][irange-1]
                        max=definitions.sequence_splitting[evt][irange]
                        meanphasic.Fill(definitions.sequence_names[evt][irange-1],phasichist.Integral(phasichist.GetXaxis().FindBin(min),phasichist.GetXaxis().FindBin(max))/float(max-min))
                        meaneda.Fill(definitions.sequence_names[evt][irange-1], rawedahist.Integral(rawedahist.GetXaxis().FindBin(min),rawedahist.GetXaxis().FindBin(max))/float(max-min))
                except IndexError,e:
                    test=0
                cmark.cd()
                meanphasic.Draw()
                meanphasic.SaveAs('./out/rootfiles/PhasicEdaHist_splitMean_' + evt+'.root')
                cmark.Update()
                cmark.SaveAs('./out/PhasicEda_splitMean_' + evt+'.png')
                cmark.SaveAs('./out/rootfiles/PhasicEda_splitMean_' + evt+'.root')
                meaneda.Draw()
                cmark.Update()
                cmark.SaveAs('./out/RawEda_splitMean_' + evt+'.png')
                cmark.SaveAs('./out/rootfiles/RawEda_splitMean_' + evt+'.root')
                meaneda.SaveAs('./out/rootfiles/RawEdaHist_splitMean_' + evt+'.root')
            cmark.Close()
            #del tonichist,phasichist,rawedahist
        ax.legend([x.decode('latin') for x in respondents])
        fig = ax.get_figure()
        fig.set_size_inches(40, 10, forward=True)
        fig.savefig('./out/'+variable+evt+'_respondents.png')
        #h = Hist(10, 0, 1, name="some name", title="some title")
        #ax_sum = sns.regplot(x=summed_df.index.values, y=summed_df[variable].values, x_bins=summed_df.index.b(), fit_reg=None)
        #from https://stackoverflow.com/questions/23709403/plotting-profile-hitstograms-in-python
        plt.close()

        if variable==definitions.eda_data:
            #3 sec bins
            fig1, ax_sum = plt.subplots(nrows=1)
            functions.Profile(x=summed_df.index.values,y=summed_df[variable].values,nbins=int(summed_df.index.max()/3),xmin=summed_df.index.min(),xmax=summed_df.index.max(),ax=ax_sum,variable=variable)
            #fig = ax_sum.get_figure()
            fig1.set_size_inches(40, 10, forward=True)
            fig1.savefig('./out/'+variable + evt + '_summed_3secbins.png')

            #summed_df[variable] = summed_df[variable].mean()
            #ax_mean = summed_df.plot(cmap=sns.cubehelix_palette(as_cmap=True), title=evt, legend=True)
            #fig = ax_mean.get_figure()
            #fig.savefig('./out/'+variable + evt + '_average.png')
            plt.close()

            # 10 sec bins
            fig1, ax_sum = plt.subplots(nrows=1)
            functions.Profile(x=summed_df.index.values, y=summed_df[variable].values, nbins=int(summed_df.index.max() / 10),
                              xmin=summed_df.index.min(), xmax=summed_df.index.max(), ax=ax_sum, variable=variable)
            # fig = ax_sum.get_figure()
            fig1.set_size_inches(40, 10, forward=True)
            fig1.savefig('./out/' + variable + evt + '_summed_10secbins.png')
            plt.close()

        fig1, ax_sum = plt.subplots(nrows=1)
        functions.Profile(x=summed_df.index.values, y=summed_df[variable].values, nbins=int(summed_df.index.max()),
                          xmin=summed_df.index.min(), xmax=summed_df.index.max(), ax=ax_sum, variable=variable)
        # fig = ax_sum.get_figure()
        fig1.set_size_inches(40, 10, forward=True)
        fig1.savefig('./out/' + variable + evt + '_summed.png')
        # summed_df[variable] = summed_df[variable].mean()
        # ax_mean = summed_df.plot(cmap=sns.cubehelix_palette(as_cmap=True), title=evt, legend=True)
        # fig = ax_mean.get_figure()
        # fig.savefig('./out/'+variable + evt + '_average.png')
        plt.close()

        del fig, fig1


def plot_total(df, respondents, Events,variable):
    master_df = pd.DataFrame()
    #ax = pd.DataFrame().plot()
    accumulated=0
    first = True
    fig, ax = plt.subplots(nrows=1)
    for evt in Events:
        summed_df = df.loc[respondents[0]]
        summed_df = summed_df.loc[summed_df['tag__info_StudioEventData'] ==evt][['position', variable]]
        # tags might occour at different positions if there is a user defined wait between movies - thus the first position in the sequence should be set to zero for plotting.
        summed_df['position'] = (summed_df['position'] - summed_df['position'].min()) / 1000. +accumulated
        summed_df = summed_df.set_index('position')
        ax = summed_df.plot(kind="line",cmap=sns.cubehelix_palette(as_cmap=True), label="first", legend=True, ax=ax, title=variable+"total length")
        maxbin=df['position'].max()/1000-df['position'].min()/1000
        if first: ax.set_xlim([0,maxbin])
        fig.show()

        #loop over respondents and sum the variable to be plottet
        if len(respondents) > 1:
            for i in range(1, len(respondents)):
                df1 = df.loc[respondents[i]]
                df1 = df1.loc[df1['tag__info_StudioEventData'] == evt][['position', variable]]
                #position is incremented (by 'accumulated' for each event/sequence being plotted in total plot:
                df1['position'] = (df1['position'] - df1['position'].min()) / 1000. + accumulated
                df1 = df1.set_index('position') #color=(i/float(len(respondents)),i/float(len(respondents)),i/float(len(respondents))),
                df1.plot(label=respondents[i].decode('latin'), ax=ax, legend=True, title=variable+"Total measurement")  # , c=i/len(respondents))
                summed_df = summed_df.add(df1, fill_value=0)
        accumulated += summed_df.index.max() - summed_df.index.min()
        master_df = pd.concat([master_df,summed_df])
        first=False
    ax.legend([x.decode('latin') for x in respondents])
    fig = ax.get_figure()
    fig.set_size_inches(40, 10, forward=True)
    fig.savefig('./out/'+variable+'Total_length_respondent.png')
    plt.close()
    first=False
    del fig, ax
    #ax_sum = master_df.plot(kind='area', label="first", legend=True,stacked=False)#cmap=sns.cubehelix_palette(as_cmap=True),
    fig, ax_sum = plt.subplots(nrows=1)
    functions.Profile(x=master_df.index.values, y=master_df[variable].values, nbins=int(master_df.index.max()+1
                                                                                        ),
                               xmin=master_df.index.min(), xmax=master_df.index.max(),ax=ax_sum,variable=variable)
    #fig = ax_sum.get_figure()
    ##fig.set_size_inches(40, 10, forward=True)
    #fig.savefig('./out/'+variable+'Total_length_summed.png')
    #ax_sum = sns.regplot(x=master_df.index.values, y=master_df[variable].values, x_bins=master_df.index.max(), fit_reg=None)
    fig = ax_sum.get_figure()
    fig.set_size_inches(40, 10, forward=True)
    fig.savefig('./out/'+variable+'_Total_length_summed.png')
    #master_df[variable] = master_df[variable].mean()
    #ax_mean = master_df.plot(kind='area', label="first", legend=True,stacked=False)
    #fig = ax_mean.get_figure()
    #fig.savefig('./out/'+variable+'Total_length_average.png')
    plt.close()
    del fig
    if variable == definitions.eda_data:
        fig, ax_sum = plt.subplots(nrows=1)
        functions.Profile(x=master_df.index.values, y=master_df[variable].values, nbins=int(master_df.index.max()/3),
                          xmin=master_df.index.min(), xmax=master_df.index.max(), ax=ax_sum, variable=variable)
        # fig = ax_sum.get_figure()
        ##fig.set_size_inches(40, 10, forward=True)
        # fig.savefig('./out/'+variable+'Total_length_summed.png')
        # ax_sum = sns.regplot(x=master_df.index.values, y=master_df[variable].values, x_bins=master_df.index.max(), fit_reg=None)
        fig = ax_sum.get_figure()
        fig.set_size_inches(40, 10, forward=True)
        fig.savefig('./out/' + variable + '_Total_length_summed_3secbins.png')
        fig, ax_sum = plt.subplots(nrows=1)
        functions.Profile(x=master_df.index.values, y=master_df[variable].values, nbins=int(master_df.index.max() / 10),
                          xmin=master_df.index.min(), xmax=master_df.index.max(), ax=ax_sum, variable=variable)
        # fig = ax_sum.get_figure()
        ##fig.set_size_inches(40, 10, forward=True)
        # fig.savefig('./out/'+variable+'Total_length_summed.png')
        # ax_sum = sns.regplot(x=master_df.index.values, y=master_df[variable].values, x_bins=master_df.index.max(), fit_reg=None)
        fig = ax_sum.get_figure()
        fig.set_size_inches(40, 10, forward=True)
        fig.savefig('./out/' + variable + '_Total_length_summed_10secbins.png')
        # master_df[variable] = master_df[variable].mean()
        # ax_mean = master_df.plot(kind='area', label="first", legend=True,stacked=False)
        # fig = ax_mean.get_figure()
        # fig.savefig('./out/'+variable+'Total_length_average.png')
        plt.close()
        del fig

def plotHR(measures):
    # Plot
    fig, axes =plt.subplots(nrows=1)
    plt.title("Beats per minute")
    peaklist = measures['peaklist']  # First retrieve the lists we need
    measures['HR'] = [60000 / x for x in measures['RR_list']]
    axes[0].plot(peaklist[1:], measures['HR'], label="BMP", color='red')
    axes[1].scatter(peaklist, measures['ybeat'], label="ybeats", color='blue')
    plt.show()
    plt.close()
    del fig, axes

def plotter(dataset, title,measures,iresp):
    peaklist = measures['peaklist']
    ybeat = measures['ybeat']
    #peaklist = measures[definitions.EKG_data]
    #ybeat = measures['rollingmean']
    #axes.title(title)
    #summed_df.plot(kind="line", cmap=sns.cubehelix_palette(as_cmap=True), label="first", legend=True, ax=ax, title=variable + "total length")
    #if first: ax.set_xlim([0, df['position'].max() / 1000 - df['position'].min() / 1000])
    fig, ax = plt.subplots(nrows=1)
    plt.ion()
    ax = dataset.plot(cmap=sns.cubehelix_palette(as_cmap=True), ax=ax, label="raw signal",legend=True,title=iresp.decode('latin'))
    #ax = dataset.plot(x=dataset['position'], y=dataset[definitions.EKG_data], cmap=sns.cubehelix_palette(as_cmap=True), ax=ax, label="raw signal",legend=True)
    #ax = plt.plot(x=dataset['position'], y=dataset[definitions.EKG_data], alpha=0.5, color='blue', label="raw signal",ax=ax,legend=True)
    #plt.plot(x=dataset['position'], y=dataset.rollingmean, alpha=0.5, cmap=sns.cubehelix_palette(as_cmap=True), label="moving average",ax=ax)
    plt.scatter(peaklist[0:len(ybeat)], ybeat, color='red', label="average: %.1f BPM" % measures['bpm'])
    #plt.scatter(peaklist, ybeat, color='red', label="average: %.1f BPM" % measures['bpm'],ax=ax)
    #plt.plot(peaklist, measures['HR'], label="BMP", color='red')
    # measures['HR'] = [60000 / x for x in measures['RR_list']]
    # plt.plot(measures['HR'], label="BMP", color='red')
    ax.legend(loc=4, framealpha=0.6)
    ax.set_xlim(285000, 315000)
    fig = ax.get_figure()
    fig.show()
    #print len(measures['peaklist'])/(measures['peaklist'][-1]-measures['peaklist'][0])*60000
    fig.set_size_inches(40, 10, forward=True)
    fig.show()
    fig.savefig('./out/HRrespondents/'+iresp + title + 'hearthrate.png')
    plt.close()
    del fig, ax
    try:
        maxbin = int(max(dataset.index))
        minbin = int(min(dataset.index))
        datahist = TH1F("datahist", "datahist", int(maxbin-minbin), minbin, maxbin)
        datahist
        ybeathist = TH1F("ybeathist", "ybeathist", int(maxbin-minbin), 0, int(maxbin-minbin))
        for i in range(len(measures['peaklist'])):
            ybeathist.Fill(measures['peaklist'][i], measures['ybeat'][i])
        for i in range(len(dataset.index)):
            datahist.Fill(dataset.index[i],dataset[definitions.EKG_data][dataset.index[i]])
        cmark = TCanvas("c", "c", 1200, 800)
        cmark.cd()
        datahist.SetMarkerColor(2)
        datahist.SetLineColor(2)
        ybeathist.SetLineColor(1)
        ybeathist.SetMarkerColor(1)
        ybeathist.SetMarkerSize(1.0)
        datahist.Draw("C")
        ybeathist.Draw("same P")
        cmark.Update()
        ybeathist.SaveAs('./out/HRrespondents/'+iresp + title + 'Rpeaks.root')
        datahist.SaveAs('./out/HRrespondents/'+iresp + title + 'ekg.root')
        cmark.SaveAs('./out/HRrespondents/'+iresp + title + 'ekgpeaks.root')
    except KeyError, e:
        test=e
    test=0
    '''
    x = range(100)
    y = range(100, 200)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(x[:4], y[:4], s=10, c='b', marker="s", label='first')
    ax1.scatter(x[40:], y[40:], s=10, c='r', marker="o", label='second')
    plt.legend(loc='upper left');
    plt.show()


    df.loc[df['tag__info_StudioEventData'] == definitions.baseline_seq]
    df.plot(x='position' , y=variable)
    plt.plot(df[variable].values, df['position'].values)
    plt.show()
    df.loc[df['tag__info_StudioEventData'] == 'Baseline.avi'].cumsum().plot(x='position', y=variable)
    test = 0
    '''