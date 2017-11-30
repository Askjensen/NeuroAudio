#!/usr/bin/python
# -*- coding: latin-1 -*-

#### datafile tag definitions - these match naming scheme of Biometric Software Suite output-files
eda_data = 'GSR'
EKG_data = 'ECG'
HR_data = 'HR'
datacolumns = ['position', eda_data, EKG_data, 'tag__info_StudioEventData']
sync_pos = 'position'

pupil_data = ['PupilLeft', 'PupilRight']
event_data = 'tag__info_StudioEventData'
delimiter = ';'

###### Vælg sti til datafilerne eksporteret fra Tobii (Stien skal slutte med '/' )
#folder_path = 'C:/Data/HistorienOmDK/test/'
#folder_path = 'C:/LokaleProjekter/RadioNeuro/trial/'
#folder_path = 'C:/data/RadioNeuro/test/raw/'
#folder_path ='C:/Data/RadioNeuro/RadioHR/gode/'
#folder_path ='C:/Data/RadioNeuro/RadioHR/alle/'

#første relle måling ligger her: (hvor dårlige hjerterytme datasæt er sorteret fra:
#folder_path ='C:/Data/RadioNeuro/250917_radionyhederP4/datasæt/godeHR/' <- brug denne for målingen fra 250917
#folder_path ='C:/Data/RadioNeuro/250917_radionyhederP4/datasæt/alle/'
#folder_path ='C:/Data/RadioNeuro/250917_radionyhederP4/datasæt/test/'
#folder_path ='C:/Data/RadioNeuro/250917_radionyhederP4/datasæt/gode/'

#Anden relle måling ligger her:
#folder_path ='C:/Data/radio26102017alle/VersionA/'
#folder_path ='C:/Data/radio26102017alle/VersionB/'
folder_path ='C:/Data/radio26102017alle/alle/'
#folder_path ='C:/Data/radio26102017alle/test/'

#folder_path = 'C:/data/RadioNeuro/test/smoothed/'
#C:\Data\Bankerotdata
filename_ext = '.txt'
#vælg hvilke scener/sekvenser, der skal analyseres. Hvis Events er tom benyttes alle.
#Events = ['01_Indledning.avi','Baseline.avi','18_Afslutning.avi']
#Events = ['81a1bee6-bba6-443b-b456-f0a2da887352.avi','Baseline 1','20_20Dragon_20Coaster_20Part_204.avi']
#Events = ['81a1bee6-bba6-443b-b456-f0a2da887352.avi','Baseline 1','20_20Dragon_20Coaster_20Part_204.avi']
#Events = ['raindrops.avi','baseline.avi','baseline2','Kl07.avi','Musik.avi','Kl08.avi']

#fra Første reelle test var sekvensnavnene i Tobii:
#Events = ['regn.avi','baseline.avi','Pre_ Klokken07.avi','Klokken07.avi','Pre_Klokken08.avi','Klokken08.avi']

#fra anden reelle måling var sekvensnavnene i Tobii:
#Events = ['regn.avi','baseline.avi','Pre07.avi','Radioavis07.avi','Pre08.avi','Radioavis08.avi']

#For all in 7:
Events = ['regn.avi','baseline.avi','Pre07.avi','Radioavis07.avi','Pre08.avi']

#baseline_seq benyttes til at sætte baseline niveau - atlså hjerterytme og
#baseline_seq = 'Pre_Klokken08.avi'
baseline_seq = 'Pre08.avi'

#HR
mov_ave_amplification = 1.1 #1.11.07
fs = 31.25
hrw = 0.75 #0.75
minpeakdistance=545 #ms,600~100bmp 545 ~ 110 BPM, 500 corresponding to 120 BPM, 400 corresponding to 150 BPM
maxpeakdistance=1500 #ms, corresponding to 40 BPM

constanttoadd=100

#phasic modelling
bgiter = 10 # the number of interations for finding background - this is highly influences timescale granularity. The higher the number the lower the smoother (and lower) the background estimate.
            # Too smooth and low is not desired for short interval (radio) analysis.
sigmapeaksinterval=3
peakamplitude = 0.01

##### herunder KAN navne på individuelle indslag i programmet samt tidspunkter indsættes - på den måde kan der laves gennemsnit af enkelte sekvenser:
#sequence splitting is used to split sequences into periodes (stories) where mean EDA and mean phasic component is calculated. Syntex is a dict with key: sequence, value: timeintervals in that story
#sequence_splitting = {'Kl07.avi':[0,130,245,305,332,349],'Kl08.avi':[0,101,160,210,298,328,342]}
#sequence_names = {'Kl07.avi':['Bestyrelse','Obamacare','DF','Macau','Vejret'],'Kl08.avi':['Gengrug','Bestyrelse','Perm. make-up','Moderater','Lotto','Vejret']}

#Version A
#sequence_splitting = {'Radioavis07.avi':[0,123,188,293,319,330],'Radioavis08.avi':[0,114,257,298,334,343]}
#sequence_names = {'Radioavis07.avi':['Kejsersnit','Kenya','Post','Kom. folkeafst.','Vejret'],'Radioavis08.avi':['Kejsersnit','Kenya','Nordea','Kom. Folkafs.','Vejret']}
#Version B
sequence_splitting = {'Radioavis07.avi':[0,123,188,293,319,330],'Radioavis08.avi':[0,161,258,286,306,326,332]}
sequence_names = {'Radioavis07.avi':['Kejsersnit','Kenya','Post','Kom. folkeafst.','Vejret'],'Radioavis08.avi':['Kenya','Kejsersnit','Telegram: Nordea','Telegram: Kom. afst.','Telegram: Post','Vejret']}
