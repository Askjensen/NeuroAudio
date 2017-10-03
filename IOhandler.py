import os, pandas as pd, logging

#create dir if not exist
def ensure_dir(flocal):
    d = os.path.dirname(flocal)
    if not os.path.exists(d):
        os.makedirs(d)

def read_data(datacolumns,folder_path,filename_ext,delimiter):
    filelist = []
    # Walk through folder creating file list
    for roots, dirs, files in os.walk(folder_path):
        for files_to_compute in files:
            if files_to_compute.endswith(filename_ext):
                filelist.append(files_to_compute)
    logging.debug('Number of datasets to compute: '+ str(len(filelist)))
    master_df = pd.DataFrame()
    # Extract data from files
    for masterfile in filelist:
        print 'reading in ' + masterfile
        respondent_df = pd.read_csv(filepath_or_buffer=folder_path+masterfile,sep=delimiter,usecols=datacolumns,decimal=",")#,index_col="position")
        respondent_df["resp"]=masterfile
        #print respondent_df.head()
        master_df = pd.concat([master_df,respondent_df])
        #logging.debug(master_df.head())
        #print master_df.memory_usage()

    #set respodents to be index
    master_df = master_df.set_index("resp")
    master_df['tag__info_StudioEventData'] = master_df['tag__info_StudioEventData'].fillna(method='ffill')
    master_df = master_df[pd.notnull(master_df['position'])]

    #master_df.fillna(method='ffill',inplace=True)
    logging.debug(master_df.head())
    return master_df,filelist

def select_data(df,Event):
    master_df = pd.DataFrame()
    for evt in Event:
        tmp_df = df.loc[df['tag__info_StudioEventData'] == evt]
        master_df = pd.concat([master_df,tmp_df])
    return master_df

'''

def dataextract(files, label):
    fhandle = open(folder_path + files, 'r')

    temp_data = list()
    n = 0

    for p in fhandle:

        p1 = p.split(delimiter)

        if n == 0:
            label_lookup = p1.index(label)

        elif n != 0:

            if p1[label_lookup] != '':
                temp_data.append(float(p1[label_lookup].replace(',', '.')))

            elif p1[label_lookup] == '':
                if(len(temp_data)==0):
                    temp_data.append(0)
                else:
                    temp_data.append(temp_data[len(temp_data) - 1])

        n = n + 1

    fhandle.close()
    return (temp_data)


    while 1>0:
        print >> f, "Processing dataset: ", masterfile
        index_list = dataextract(masterfile, sync_pos)
        eda_data_list = dataextract(masterfile, eda_data)
        pupil_data_array = []
        pupil_data_list = []
        for i in range(0, len(pupil_data)):
            pupil_data_array.append(dataextract(masterfile, pupil_data[i]))
        for i in range(0, len(pupil_data_array[0])):
            pupil_data_list.append((pupil_data_array[0][i] + pupil_data_array[1][i]) / 2.0)

        print masterfile
        event_names, event_datapoints = event_hz_pointers(masterfile, Events_list, event_data,
                                                          sync_pos)  # syntax: file, list of events to look for, data tag in data set describing events, sync positions, delimiter
        id_name = (masterfile.split('.'))[0]

        if len(index_list) != len(eda_data_list):
            "WARNING! in file", id_name, " a difference between index lenght and value lenght has been encounted."
        if len(index_list) != len(pupil_data_list):
            "WARNING! in file", id_name, " a difference between index lenght and pupil diameter lenght has been encounted."
        n1 = 0
        id_name_list = list()

        while (len(index_list)) > n1:
            id_name_list.append(id_name)
            n1 = n1 + 1

        # Clean data focusing solely on defined events. Other data = NaN.

        xn = 0
        xn1 = 0
        xn2 = 0

        eda_data1 = list()
        pupil_data1 = list()
        event_hz_markers1 = event_datapoints
        event_hz_markers1.sort()

        for xp in index_list:

            if len(event_hz_markers1) != xn and event_hz_markers1[xn] == xp:
                # Switching - when in event data array add data otherwise add numpy.nan
                xn = xn + 1
                xn1 = 0 if xn1 else 1

            if xn1 == 0:
                eda_data1.append(np.nan)  # Fill NaN when outside event boundaries
                pupil_data1.append(np.nan)

            elif xn1 == 1:
                eda_data1.append(eda_data_list[xn2])
                pupil_data1.append(pupil_data_list[xn2])

            xn2 = xn2 + 1

        # All data are extended to three lists, essentially creating a MultiIndex (index1+2) and referring values
        index1_names.extend(id_name_list)
        index2_syncpos.extend(index_list)
        eda_values_list.extend(eda_data1)
        pupil_diameter_values_list.extend(pupil_data1)

        # This list keeps track of all data sets in order to compute event means
        dataset_index.append(id_name)
        dataset_event_names.append(event_names)
        dataset_event_datapoints.append(event_datapoints)
        EventBinsPos[id_name] = event_datapoints
        EventBinsNames[id_name] = event_names

def create_dataframes():
    #global eda_data_series, pupil_data_series
    print >> f, "Creating dataframe..."
    # Make MultiIndex a Tuple zipping two lists together
    index3 = list(zip(index1_names, index2_syncpos))
    # Create 2D MultiIndex
    index4 = pd.MultiIndex.from_tuples(index3, names=['Names', 'Syncpos'])
    # Parse the Index to create a new dataframe named DF
    df = pd.DataFrame(index=index4)
    # Create a new series with the original data
    eda_data_series = pd.Series(eda_values_list, index=index4)
    # resultdf = eda_data_series.to_frame()
    # print resultdf.head
    # print resultdf.dropna(axis=0).head
    pupil_data_series = pd.Series(pupil_diameter_values_list, index=index4)
    return eda_data_series.to_frame().dropna(axis=0),pupil_data_series.to_frame().dropna(axis=0)
'''