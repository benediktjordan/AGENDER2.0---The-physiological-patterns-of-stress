#region Create Epochs

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

