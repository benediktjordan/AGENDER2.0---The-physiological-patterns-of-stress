


#TODO Create loop for all data folders
#os.chdir(r"C:\\Users\\Ben Ali Kokou\\Documents\\Master\\05_WS 2020.21\\07_MPI\\05_Scripts\\Python\\01_Data\\01_Data\\")
#folders = os.listdir()
#for i in folders:
#    path = r"C:\\Users\\Ben Ali Kokou\\Documents\\Master\\05_WS 2020.21\\07_MPI\\05_Scripts\\Python\\01_Data\\01_Data\\"+ i + r"\\Rohdaten\\Sensor")

#region load bio data (return = data_ecg, data_hrvrmssd, data_hrvisvalid, starting_time)
def load_biodata(path):
    os.chdir(path + r"\\Rohdaten\\Sensor")
    folder_bio = os.listdir()
    u = unisens.Unisens(folder_bio[0])  # folder containing the unisens data
    #print(u.entries)

    #acc = u.acc_bin
    #data_acc= acc.get_data()
    ecg = u.ecg_bin
    data_ecg = ecg.get_data()
    #press = u.press_bin
    #data_press= press.get_data()
    #temp = u.temp_bin
    #data_temp = temp.get_data()

    HRVisvalid = u.hrvisvalid_bin
    data_hrvisvalid = HRVisvalid.get_data()

    HRVrmssd = u.hrvrmssd_bin
    data_hrvrmssd = HRVrmssd.get_data()

    #ecgrrfiltered = u.ecgrrfiltered_csv
    #data_ecgrrfiltered = ecgrrfiltered.get_data()

    #marker = u.marker_csv
    #data_marker = marker.get_data()

    # Read starttime from xml file
    root = ET.parse(folder_bio[0]+"/unisens.xml").getroot()
    starting_time = root.attrib['timestampStart']

    return data_ecg, data_hrvrmssd, data_hrvisvalid, starting_time


#endregion

#region Create UNIX Microseconds
def unix_time_microseconds(data):
    # dt: string containing datetime in this format: '2015-03-16T10:58:51.437'
    time_pandas = pd.Timestamp(data)
    epoch = dt.datetime.utcfromtimestamp(0)
    return (time_pandas - epoch).total_seconds() * 1000000
#endregion

#region Create MNE raw ECG object
def create_MNE_raw(data_ecg, starting_time):

    # Initialize an info structure
    info = mne.create_info(
        ch_names=['ECG'],
        ch_types=['ecg'],
        sfreq=1024,
    )

    # Create raw ECG object
    raw_ecg = mne.io.RawArray(data_ecg, info)

    # Add starting time to info
    starting_time_inseconds = unix_time_microseconds(starting_time)/1000000
    raw_ecg.set_meas_date(starting_time_inseconds)
    starting_time = raw_ecg.info["meas_date"]

    return raw_ecg
    # raw_ecg.info["meas_date"]
#endregion

#region Create MNR raw HRV object
def create_MNE_raw_HRV(data_hrvrmssd, starting_time):

    # Initialize an info structure
    info = mne.create_info(
        ch_names=['ECG'],
        ch_types=['ecg'],
        sfreq=0.0166666666666666666666666666666,
    )

    # Create raw HRVRMSSD object
    raw_hrvrmssd = mne.io.RawArray(data_hrvrmssd, info)

    # Add starting time to info
    starting_time_inseconds = unix_time_microseconds(starting_time)/1000000
    raw_hrvrmssd.set_meas_date(starting_time_inseconds)
    starting_time = raw_hrvrmssd.info["meas_date"]

    return raw_hrvrmssd

#endregion

#region Converting from annotations to events
def annot_to_events(raw_object):

    custom_mapping = {"1.0": 1, "2.0":2, "3.0":3}#only looking at stress events (not at no-stress events
    (events_from_annot,
     event_dict) = mne.events_from_annotations(raw_object, custom_mapping)
    return events_from_annot, event_dict
#endregion

#region convert from epochs to df
def epochs_to_df(epochs):

    # epochs into dataframe
    index,  scalings = ['epoch', 'time'], dict(ecg=1, mag=1, grad=1) #scaling is surpressed
    df_epochs = epochs.to_data_frame(picks=None,time_format = None, scalings=scalings,index=['condition', 'epoch', 'time'])
    df_epochs.sort_index(inplace=True)

    # Select epochs & create df with time in rows and epochs in columns
    count = 0
    df = pd.DataFrame()
    for epoch, new_df in df_epochs.groupby(level=1):
        if count ==0:
            count = count + 1
            new_df.index = new_df.index.droplevel(0)
            new_df.index = new_df.index.droplevel(0)
            df = new_df
        else:
            count = count + 1
            this_column = str(count)
            new_df.index = new_df.index.droplevel(0)
            new_df.index = new_df.index.droplevel(0)
            df[this_column] = new_df
        df.rename(columns={'ECG': '1'}, inplace=True)

    return df
#endregion

#region load label data
# load csv
os.chdir(r"C:\\Users\\Ben Ali Kokou\\Documents\\Master\\05_WS 2020.21\\07_MPI\\05_Scripts\\Python\\01_Data\\")
files = glob.glob("*.csv")
files

def load_label(path):
    os.chdir(path + r"\\Rohdaten\\Report")
    file_label = glob.glob("*.csv")
    df = pd.read_csv(file_label[0], sep = ";")

    return df

#endregion

#region transform label
def create_stress_columns(df):
    stress_events = []
    stress_durations = []
    stress_intensities = []
    for idx, entry in df["Stress_Stunde"].iteritems():
        starttime = df["Form_start"][idx]
        starttime = dt.datetime.strptime(starttime, '%Y-%m-%dT%H:%M:%S')

        if entry == 2:
            stress_event = starttime - dt.timedelta(minutes = 60)

            stress_duration = 60 #minutes

            stress_intensity = 0 #since they are not stressed at all

        elif entry == 1:
            if df["Wann_Stress"][idx] == 1:
                stress_event = starttime- dt.timedelta(minutes=10)
            elif df["Wann_Stress"][idx] == 2:
                stress_event = starttime - dt.timedelta(minutes=20)
            elif df["Wann_Stress"][idx] == 3:
                stress_event = starttime - dt.timedelta(minutes=30)
            elif df["Wann_Stress"][idx] == 4:
                stress_event = starttime - dt.timedelta(minutes=40)
            elif df["Wann_Stress"][idx] == 5:
                stress_event = starttime - dt.timedelta(minutes=50)
            elif df["Wann_Stress"][idx] == 6:
                stress_event = starttime - dt.timedelta(minutes=60)
            else:
                stress_event = np.nan

            stress_duration = 600 #second (an arbitrary duration)

            stress_intensity = df["Wie_Stress_categorical"][idx]


        elif pd.isnull(entry):
            stress_event = np.nan

            stress_duration = np.nan

            stress_intensity = np.nan

        else:
            stress_event = np.nan

            stress_duration = np.nan

            stress_intensity = np.nan

        stress_events.append(stress_event)
        stress_durations.append(stress_duration)
        stress_intensities.append(stress_intensity)

    df["Stress_Event"] = stress_events
    df["Stress_Duration"] = stress_durations
    df["Stress_Intensity"] = stress_intensities

    return df

def transform_label(df):
    # select only interesting variables
    df = df[['Form', 'Form_start_date', 'Form_start_time', 'Form_finish_date', 'Form_finish_time',
             'Missing', 'Stress_Stunde', 'Wann_Stress', 'Wie_Stress', 'Wie_momentan', 'Confounder_1',
             'Confounder_2', 'Confounder_3', 'Confounder_4', 'Confounder_5', 'Confounder_wann']]

    # Delete rows which contain "Abend" or "Schlaf" in column "Form"
    df = df[df['Form'] == "Tag"]

    # Merge columns which contain date and time into datetime column
    df["Form_start"] = df["Form_start_date"] + "T" + df["Form_start_time"]
    df["Form_finish"] = df["Form_finish_date"] + "T" + df["Form_finish_time"]

    # Create Stress intensity in categorical form
    ## Intensity-Score:
    ## 1 if 0  <= x <= 33
    ## 2 if 34 <= x <= 66
    ## 3 if 67 <= x <= 100
    ## np.nan if else

    df["Wie_Stress_categorical"] = [
        1 if x <= 33 else 2 if x >= 34 and x <= 66 else 3 if x >= 67 and x <= 100 else np.nan \
        for x in df["Wie_Stress"]]

    # compute new stress columns: "WhenStress", "StressDuration" and "WhichStress"
    ## if 2 (no stress in last hour) -> start stress marker ("no stress") at 60 minutes before & mark 60 minutes duration
    df = create_stress_columns(df)

    return df
#endregion

#region annotate MNE raw with labels
def create_timedifference(df, raw_object):
    deltatime_array = []
    starting_time = raw_object.info["meas_date"]
    starting_time = starting_time.replace(tzinfo=None)
    for entry in df["Stress_Event"]:
        #datetime = dt.datetime.strptime(entry, '%Y-%m-%dT%H:%M:%S')
        deltatime = entry-starting_time
        deltatime = deltatime.total_seconds()
        deltatime_array.append(deltatime)
    df["timedifference"] = deltatime_array
    return(df)

def annotate(df, raw_object):
    raw_total_seconds = raw_object.n_times/raw_object.info["sfreq"] #total seconds in raw object
    df = df[df['timedifference'] > 0]
    df = df[df['timedifference'] < raw_total_seconds]

    # create arrays which are accepted by MNE annotate
    stress_onset = []
    stress_duration = []
    stress_intensity = []

    for idx, entry in df['timedifference'].iteritems():
        stress_onset.append(entry)
        stress_duration.append(df["Stress_Duration"][idx])
        stress_intensity.append(df['Stress_Intensity'][idx])


    #region Annotate MNE raw (outdated)
    my_annot = mne.Annotations(onset=stress_onset,
                               duration=stress_duration,
                               description=stress_intensity)

    raw_object.set_annotations(my_annot)
    print("There are " + str(len(my_annot)) + " annotations")

    return raw_object

#endregion




#region Pipeline for one ECG database
path = [
    r"C:\\Users\\Ben Ali Kokou\\Documents\\Master\\05_WS 2020.21\\07_MPI\\03_Data\\01_Stress\\01_Roehner\\02_RawData\\Preprocessed\\AGENDER11_1" ]
path = path[0]

data_ecg, data_hrvrmssd, data_hrvisvalid, starting_time = load_biodata(path)
df = load_label(path)
print("Size of ECG data loaded is " + str(data_ecg.shape))
print("Starting time of this ECG measurement was " + str(starting_time))

raw_ecg = create_MNE_raw(data_ecg, starting_time)
print("The metadata of this MNE raw object is: " + str(raw_ecg.info))

df = transform_label(df)
print("The number of people in the different \"stress categories\" is: " + str(df["Stress_Intensity"].value_counts()))

df= create_timedifference(df, raw_ecg)
raw_ecg = annotate(df, raw_ecg)
raw_ecg.plot()

#endregion

#region Pipeline for HRV database

data_ecg, data_hrvrmssd, data_hrvisvalid, starting_time = load_biodata(path)
raw_hrvrmssd = create_MNE_raw_HRV(data_hrvrmssd, starting_time)

df = load_label(path)
df = transform_label(df)
print("The number of people in the different \"stress categories\" is: " + str(df["Stress_Intensity"].value_counts()))

df= create_timedifference(df, raw_hrvrmssd)
raw_hrvrmssd = annotate(df, raw_hrvrmssd)
#raw_hrvrmssd.plot()

#Create events from annotations
events_from_annot, event_dict = annot_to_events(raw_hrvrmssd)
# print(event_dict)
# print(events_from_annot)
events_from_annot.shape

#Creating epochs from events
event_types = pd.DataFrame(events_from_annot)[2].value_counts()
event_types #Check which type of annotations are there

event_dict = {'low stress': 1, 'medium stress': 2, 'high stress': 3} #adapt depending on data
epochs = mne.Epochs(raw_hrvrmssd, events_from_annot, tmin=-1200, tmax=1200, event_id=event_dict, picks= "ecg", event_repeated = "drop")
epochs.events

# Epochs to Dataframe
df = epochs_to_df(epochs)

fig = df.plot(subplots=True, figsize=(10,12), title = "Stress Moments \ Participant 14").get_figure()
#fig.savefig('test.pdf')

#endregion








#region OUTDATED Create Epochs & convert to dataframe

#Converting from annotations to events
custom_mapping = {"1.0": 1, "2.0":2, "3.0":3}#only looking at stress events (not at no-stress events
(events_from_annot,
 event_dict) = mne.events_from_annotations(raw_ecg, custom_mapping)
#print(event_dict)
#print(events_from_annot)
events_from_annot.shape

#Creating epochs from events
event_types = pd.DataFrame(events_from_annot)[2].value_counts()
event_types #Check which type of annotations are there

event_dict = {'high stress': 3} #adapt depending on data
epochs = mne.Epochs(raw_ecg, events_from_annot, tmin=-3, tmax=3, event_id=event_dict, picks= "ecg", event_repeated = "drop")

# epochs into dataframe
index,  scalings = ['epoch', 'time'], dict(ecg=1e13)
df_epochs = epochs.to_data_frame(picks=None,time_format = "ms", scalings=scalings,index=['condition', 'epoch', 'time'])
df_epochs.sort_index(inplace=True)

#Select epochs & create df with time in rows and epochs in columns
count = 0
df = pd.DataFrame()
for epoch, new_df in df_epochs.groupby(level=1):
    if count ==0:
        count = count + 1
        new_df.index = new_df.index.droplevel(0)
        new_df.index = new_df.index.droplevel(0)
        df = new_df
    else:
        count = count + 1
        this_column = str(count)
        new_df.index = new_df.index.droplevel(0)
        new_df.index = new_df.index.droplevel(0)
        df[this_column] = new_df
    df.rename(columns={'ECG': '1'}, inplace=True)

fig = df.plot(subplots=True, figsize=(10,12), title = "Stress Moments \ Participant 14").get_figure()
#fig.savefig('test.pdf')

#endregion


#region some trials to fix problems
data, times = raw_ecg.get_data(return_times=True)

#export epochs data
## into numpy array
epochs_data = epochs.get_data(

## into dataframe
epochs_dataframe = epochs.to_data_frame(picks=None,time_format = "ms", scalings=scalings,index=['condition', 'epoch', 'time'])
epochs_dataframe.sort_index(inplace=True)
#print(df.loc[('auditory/left', slice(0, 10), slice(100, 107)),
#             'EEG 056':'EEG 058'])
epochs_dataframe['index1'] = epochs_dataframe.index

#into dataframe method 2
index,  scalings = ['epoch', 'time'], dict(ecg=1e10)
df_epochs =  epochs.to_data_frame(picks=None,time_format = "ms", scalings=scalings, index=index)
df_epochs.sort_index(inplace=True)
df_epochs["ECG"].plot()

# create arrays which are accepted by MNE annotate
stress_onset = [0,4.5]
stress_duration = [1,2]
stress_intensity = [1,2]

my_annot = mne.Annotations(onset=stress_onset,
                           duration=stress_duration,
                           description=stress_intensity)

raw_ecg.set_annotations(my_annot)
print("There are " + str(len(my_annot)) + " annotations")

#endregion


#region visualizing only stressed periods
#Converting from annotations to events
custom_mapping = {"0.0": 0, "1.0": 1, "2.0":2, "3.0":3}
(events_from_annot,
 event_dict) = mne.events_from_annotations(raw_ecg, custom_mapping)
print(event_dict)
print(events_from_annot)
events_from_annot.shape

#Creating epochs from events
raw_ecg.annotations #Check which type of annotations are there

event_dict = {'no stress': 0,'low stress': 1, 'medium stress': 2} #adapt depending on data
epochs = mne.Epochs(raw_ecg, events_from_annot, tmin=-1, tmax=1, event_id=event_dict, picks= "ecg", event_repeated = "drop")

picks = mne.pick_types(raw_ecg.info)

tmin, tmax = -0.1, 0.1
raw_ecg.del_proj()
epochs = mne.Epochs(raw_ecg, events_from_annot, event_dict, tmin, tmax)
data = epochs.get_data()



print(epochs.event_id)
print(epochs)

epochs.plot()

epochs['auditory'].plot_image(picks=['MEG 0242', 'MEG 0243'])


test = df["Form_start"][2]
datetime = dt.datetime.strptime(test, '%Y-%m-%dT%H:%M:%S')

#endregion















    print(entry)
starting_time
starting_time = dt.datetime.strptime(starting_time, '%Y-%m-%dT%H:%M:%S.%f')
test = dt.datetime.strptime(df["Form_start"][1], '%Y-%m-%dT%H:%M:%S')

delta = test-starting_time




#Calculate period
period = data_ecg.shape[1]/1024 #in seconds; sampling rate 1024 Hz
print(str(period) + "seconds")



########################################################################
# Convert unix milisecond format back into datetime format
ts = int("1426503532479")/1000

# if you encounter a "year is out of range" error the timestamp
# may be in milliseconds, try `ts /= 1000` in that case
print(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S:%f'))

########################################################################

# Import necessary packages
# Convert ECG row dataframe into columns
data_ecg = data_ecg.transpose()
data_ecg = pd.DataFrame(data_ecg, columns = ["ECG"])





# Create function which assigns correct unix microseconds time to ECG



def unix_time_microseconds(dt):
    epoch = datetime.datetime.utcfromtimestamp(0)
    return (dt - epoch).total_seconds() * 1000000

def assign_time(df,variable, sampling_rate, starting_time):
    # Parameters:
    ## df: the dataframe containing the physiological data
    ## variable: the variable for which the time column should be created
    ## sampling_rate: sampling rate of the physiological variable
    ## starting_time: the starting time as a pandas Timestamp (pd.Timestamp)

    # Assign correct unix time for starttime
    start_time_unix = unix_time_millis(starting_time)

    # create list with size of ECG df & increased unix timestamp
    time = [start_time_unix] #first value in list is starttime
    microseconds_SR = 1000000 / sampling_rate #how much microseconds have to be increased for each sampled datapoint
    len = df[variable].shape[0]-1 #length of the list (has to match dataframe
    for i in range(len):
        start_time_unix = start_time_unix + microseconds_SR
        time.append(start_time_unix)

    # add array to dataframe as time column
    df["time"] = time

    return(df)

df = data_ecg
variable = "ECG"
sampling_rate = 1024
starting_time = starting_time

df = assign_time(df, variable, sampling_rate, starting_time)



#region oldcode
data_transposed = data.transpose()
data_transposed = pd.DataFrame(data_transposed)
data_transposed = data_transposed.astype(str)
test = data_transposed[0].str.contains("66529687")

#endregion

