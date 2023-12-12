


#region Testrun (to check for errors)
path = r"C:\Users\BJ\Documents\07_MPI\03_Data\01_Stress\01_Roehner\02_RawData\Preprocessed"
folders = os. listdir(path)

#Choose proband
proband = folders[1]

path = [r"C:\\Users\\BJ\\Documents\\07_MPI\\03_Data\\01_Stress\\01_Roehner\\02_RawData\\Preprocessed\\" + proband]
path = path[0]

# Load and transform label data
stress_threshold = 33 #the number above which stress is counted as such (below it will not be considered as stress); between 0-100
df_label = load_label(path)
df_label = transform_label(df_label, stress_threshold)

# Load ECG data
data_ecg, data_hrvrmssd, data_hrvisvalid, starting_time = load_biodata(path)

# Convert ECG data into df
df_ecg = create_ecg_df(data_ecg)

# Combine Label & ECG data and annotate raw object
df_label = create_timedifference(df_label, starting_time)
frequency = 1024
events = create_events(df_label, df_ecg, frequency)
# plt.plot(df_ecg['ECG']) #Plot ECG signal
# plot = nk.events_plot(events, df_ecg) #Visualize Events & Data

### ANALYZE ### Are there NaNs in events
Counter(events["condition"])

# Balance the data
stress_percentage = 50  # Percentage of stress values requiered in dataset
events = balance(events, stress_percentage)

# Creating Epochs
if len(events["condition"]) == 0:
    print("Proband " + proband + " has no events")
    continue
counter_epochs = counter_epochs + len(events["condition"])
epochs_length = 600  # in seconds
epochs = nk.epochs_create(df_ecg, events, sampling_rate=1024, epochs_start=-(epochs_length / 2),
                          epochs_end=epochs_length / 2)  # start and end times are in seconds

#endregion


#region PreProcessing 1: load & transform (including filtering) all ECG & ACC data (until filtered epochs)
# this code is based on the script "loading_data_and_initial_transformations.py"

t0 = time.time()
path = r"C:\Users\BJ\Documents\07_MPI\03_Data\01_Stress\01_Roehner\02_RawData\Preprocessed"
folders = os. listdir(path)
counter_epochs = 0
counter_peaks = 0
proband_overview = []
stress_event_number = []
ecg_starting_time = []
ecg_ending_time = []
sampling_rate = 1024        #Hz

for proband in folders:
    path = [r"C:\\Users\\BJ\\Documents\\07_MPI\\03_Data\\01_Stress\\01_Roehner\\02_RawData\\Preprocessed\\" + proband]
    path = path[0]

    # Load and transform label data
    stress_threshold = 0  # the number above which stress is counted as such (below it will not be considered as stress); between 0-100
    df_label = load_label(path)
    df_label = transform_label(df_label, stress_threshold)

    # Load ECG data
    data_ecg, data_acc, data_hrvrmssd, data_hrvisvalid, starting_time = load_biodata(path)

    # Convert ECG data into df
    df_ecg = create_ecg_df(data_ecg)

    # Combine Label & ECG data and annotate raw object
    df_label = create_timedifference(df_label, starting_time)
    frequency = 1024
    list_of_times = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7]  # days; where to look for no-stress events
    events = create_events(proband, df_label, df_ecg, frequency, list_of_times)
    # plt.plot(df_ecg['ECG']) #Plot ECG signal
    # plot = nk.events_plot(events, df_ecg) #Visualize Events & Data

    # Create overview over events
    proband_overview.append(proband)
    stress_event_number.append(len(events["condition"]))
    ecg_starting_time.append(starting_time)
    ecg_ending_time.append(dt.datetime.strptime(starting_time, '%Y-%m-%dT%H:%M:%S.%f')+dt.timedelta(0, len(df_ecg["ECG"])/(1024)))

    # Creating Epochs for ECG
    if len(events["condition"]) == 0:
        continue

    counter_epochs = counter_epochs + len(events["condition"])
    epochs_length = 660  # in seconds
    epochs = nk.epochs_create(df_ecg, events, sampling_rate=1024, epochs_start=-660,
                              epochs_end=600)  # start and end times are in seconds; we take -11 minutes & +10 minutes

    # Create Epochs for ACC
    epochs_length_acc = 600
    ## transform ACC data to dataframe
    df_acc = pd.DataFrame({
        "x1": data_acc[0],
        "x2": data_acc[1],
        "x3": data_acc[2]
    })
    ## downsample events location
    events_acc = copy.deepcopy(events)
    newList = [x/16 for x in events_acc["onset"]]
    events_acc["onset"] = newList        #adapting events onset location to ACC sampling rate (which is
                                                        # at 64Hz which is 16x less frequent than ECG sampling rate
    epochs_acc = nk.epochs_create(df_acc, events_acc, sampling_rate=64, epochs_start=-600,
                              epochs_end=600)  # start and end times are in seconds; we take -10 minutes & +10 minutes; note:
    #in difference to the approach for epochs of ECG, we calculate the ACC epochs directly for 20 minutes in total (instead of 21 minutes)


    # Delete NaN values in Epochs (NaN values occur if events are close to the border)
    for i in epochs:
        epochs[i] = epochs[i][epochs[i]["ECG"].notna()]

    # Filter data
    method_filtering = "neurokit"
    for l in epochs:
        signals = pd.DataFrame({"ECG": nk.ecg_clean(epochs[l]["ECG"], sampling_rate=sampling_rate,
                                                             method=method_filtering)})
        signals.index = epochs[l]["ECG"].index  # need to set the index of signals the same as index of original ECG series
        epochs[l]["ECG"] = signals["ECG"]
    print("Proband" + str(proband) +"Is done now ")

t1 = time.time()
t1 - t0
    # Save Epochs ECG and ACC
    save_obj(epochs, "epochs_" + str(epochs_length) + "seconds_filtered_" + proband)
    save_obj(epochs_acc, "epochs_acc_" + str(epochs_length_acc) + "seconds_" + proband)

#create df with overview over stress events
df_overview = pd.DataFrame()
df_overview["proband"] = proband_overview
df_overview["number of events"] = stress_event_number
df_overview["ECG starting_time"] = ecg_starting_time
df_overview["ECG end time"] = ecg_ending_time
df_overview.to_csv('C:\\Users\\BJ\\PycharmProjects\\AGENDER2.0\\obj\\probands_events_overview.csv')

#endregion

#region PreProcessing 2: Peak detection, correction & noise detection
# this code-region is based on functions in two scripts: "peak_detection_and_peakNoise_detection.py" & "ECGNoise_detection.py"

    # Peak detection
path = r"C:\Users\BJ\PycharmProjects\AGENDER2.0\obj"
files = os.listdir(path)
condition = "epochs_660seconds_filtered"
method_peakdetection = "neurokit"
#This is temporarily created so I don´t have to run the main script every time again
proband_withoutevents =['AGENDER07',
 'AGENDER11',
 'AGENDER11_1',
 'AGENDER13',
 'AGENDER18',
 'AGENDER26']
path = r"C:\Users\BJ\Documents\07_MPI\03_Data\01_Stress\01_Roehner\02_RawData\Preprocessed"
folders = os. listdir(path)

peaks = create_peaks(files, folders, condition, method_peakdetection,proband_withoutevents)

    # Peak Correction
    ## Importing the .txt files which have been outputed from Brown website
location = 'C:\\Users\\BJ\\PycharmProjects\\AGENDER2.0\\obj\\peaks\\corrected'
peaks_corrected = correct_peaks_brown(peaks, location)

    # Cutting-off first minute of epochs & peaks
#path = r"C:\Users\BJ\PycharmProjects\AGENDER2.0\obj"
#files = os.listdir(path)
#condition = "epochs_660seconds_filtered"
#epochs_cutoff(files,condition, number_of_minutes)           #cuts-off first minute of every epoch & saves resulting
                                                            # epoch in obj folder

number_of_minutes = 1
peaks_corrected_cutoff = peaks_cutoff(peaks_corrected, number_of_minutes)      #cuts-off first minute of every peak
save_obj(peaks_corrected_cutoff, "peaks_corrected_cutoff")
peaks_corrected_cutoff = load_obj("peaks_corrected_cutoff.pkl")


    #Noise treatment
    ##Exlude epochs which have more than 10% peaks corrected
df_perc_corr = noise_percentage_peakcorrection(peaks_corrected)     #Calculate percentage of noise
df_perc_corr.to_csv('C:\\Users\\BJ\\PycharmProjects\\AGENDER2.0\\obj\\PeakCorrection_CorrectionPercentages_overview.csv')

df_epochs_noisy = df_perc_corr[df_perc_corr["Percentage of Artifacts"] >= 0.1]#Create list of epochs with over 10% peaks corrected

    # Manual noise treatment
noise = load_obj("noise_labels.pkl")
df_noise = noise_percentage(noise, peaks_corrected_cutoff)  #calculates how much noise there is
df_noise_above10 = df_noise[df_noise["Percentage of Noise"] >= 0.10]#Create list of epochs with over 10% peaks corrected
df_noise_above10below30 = df_noise_above10[df_noise_above10["Percentage of Noise"] <= 0.30]
df_noise_above30 = df_noise[df_noise["Percentage of Noise"] >= 0.30]#Create list of epochs with over 30% peaks corrected
df_noise_above25 = df_noise[df_noise["Percentage of Noise"] >= 0.25]#Create list of epochs with over 25% peaks corrected

df_noise.to_csv('C:\\Users\\BJ\\PycharmProjects\\AGENDER2.0\\obj\\Noise_overview.csv')
df_noise_above30.to_csv('C:\\Users\\BJ\\PycharmProjects\\AGENDER2.0\\obj\\Noise_overview_above30%.csv')

    #Exclude epochs with too much noise
df_noise = pd.read_csv('C:\\Users\\BJ\\PycharmProjects\\AGENDER2.0\\obj\\Noise_overview_above30%.csv')
peaks_corrected_cutoff_noisecorrected = copy.deepcopy(peaks_corrected_cutoff)
for index, proband in enumerate(df_noise["Proband"]):
    epoch = df_noise["Epoch"][index]
    del peaks_corrected_cutoff_noisecorrected[proband][epoch]
save_obj(peaks_corrected_cutoff_noisecorrected, "peaks_corrected_cutoff_noisecorrected")
peaks_corrected_cutoff_noisecorrected = load_obj("peaks_corrected_cutoff_noisecorrected.pkl")

#endregion

#region PreProcessing 3: Segment peaks & feature creation
# this code-region is based on functions in the script "segmentation_and_feature_creation.py"

    # Create Peak segments
segment_duration = 30           #seconds
peaks_corrected_cutoff_noisecorrected_segmented = create_segments(peaks_corrected_cutoff_noisecorrected, segment_duration)
save_obj(peaks_corrected_cutoff_noisecorrected_segmented, "peaks_corrected_cutoff_noisecorrected_segmented")
peaks_corrected_cutoff_noisecorrected_segmented = load_obj("peaks_corrected_cutoff_noisecorrected_segmented.pkl")


    # Feature Creation
    ## HRV features
segment_duration = 30
peaks_hrvfeatures = hrv_feature_creation(peaks_corrected_cutoff_noisecorrected_segmented, segment_duration)
save_obj(peaks_hrvfeatures, "peaks_hrvfeatures")
peaks_hrvfeatures = load_obj("peaks_hrvfeatures.pkl")


    #Resampling & adding ACC to feature dict
segment_duration = 30
resampling_method = "FFT"
peaks_hrvfeatures_acc=load_obj("peaks_hrvfeatures_acc.pkl")
peaks_hrvfeatures_acc = ACC_resample(peaks_hrvfeatures, peaks_hrvfeatures_acc, segment_duration, resampling_method)
save_obj(peaks_hrvfeatures_acc, "peaks_hrvfeatures_acc")


    #Create df with all relevant data for ensemble classification
t0 = time.time()
df_class_allfeatures = create_classdf(peaks_hrvfeatures_acc)
t1 = time.time()
print("Time of whole df creation was " + str(t1-t0))
save_obj(df_class_allfeatures, "df_class_allfeatures")
df_class_allfeatures = load_obj("df_class_allfeatures.pkl")



#region Subpipeline 1.1 Segment data based on individual noise labels
# this code-region is based on functions in the script "segmentation_and_feature_creation.py"

    # Segment
peaks_corrected_cutoff = load_obj("peaks_corrected_cutoff.pkl")
noise = load_obj("noise_labels.pkl")

peaks_corrected_cutoff_noisecorrectedIMROVED_segmented = create_segments_excludingnoise(peaks_corrected_cutoff, noise)
save_obj(peaks_corrected_cutoff_noisecorrectedIMROVED_segmented, "peaks_corrected_cutoff_noisecorrectedIMROVED_segmented")
peaks_corrected_cutoff_noisecorrectedIMROVED_segmented = load_obj("peaks_corrected_cutoff_noisecorrectedIMROVED_segmented.pkl")

    # Feature Creation
    ## HRV features
segment_duration = 30
peaks_hrvfeatures_noiseIMPROVED = hrv_feature_creation(peaks_corrected_cutoff_noisecorrectedIMROVED_segmented, segment_duration)
save_obj(peaks_hrvfeatures_noiseIMPROVED, "peaks_hrvfeatures_noiseIMPROVED")


    #Resampling & adding ACC feature (one row per segment) to feature dict
segment_duration = 30
resampling_method = "FFT"
peaks_hrvfeatures_noiseIMPROVED = load_obj("peaks_hrvfeatures_noiseIMPROVED.pkl")
peaks_hrvfeatures_noiseIMPROVED_acc = ACC_resample(peaks_hrvfeatures_noiseIMPROVED, segment_duration, resampling_method)
save_obj(peaks_hrvfeatures_noiseIMPROVED_acc, "peaks_hrvfeatures_noiseIMPROVED_acc")

    #Create df with all relevant data for ensemble classification
t0 = time.time()
peaks_hrvfeatures_noiseIMPROVED_acc = load_obj("peaks_hrvfeatures_noiseIMPROVED_acc.pkl")
df_class_allfeatures_noiseIMPROVED = create_classdf(peaks_hrvfeatures_noiseIMPROVED_acc)
t1 = time.time()
print("Time of whole df creation was " + str(t1-t0))
save_obj(df_class_allfeatures_noiseIMPROVED, "df_class_allfeatures_noiseIMPROVED")
df_class_allfeatures_noiseIMPROVED = load_obj("df_class_allfeatures_noiseIMPROVED.pkl")

#endregion
#endregion


#region Classification Pipeline 1: LSTM classification
#region LSTM classification preparation:  segment epochs & add noise & ACC features to ECG epochs
# this code-region is based on functions in the script "segmentation_and_feature_creation.py"

    #Add noise to epochs
noise = load_obj("noise_labels.pkl")
vis_period = 15
sampling_rate = 1024
ECG_epochs_addnoise(noise, vis_period, sampling_rate)

    #add ACC features to epochs
condition = "epochs_600seconds_noise_filtered_"
path = r"C:\Users\BJ\PycharmProjects\AGENDER2.0\obj"
files = os.listdir(path)
upsample_method = "interpolation"
ECG_epochs_addACC(files, condition, upsample_method)

    #segment epochs
condition = "epochs_600seconds_noise_acc_filtered_"
path = r"C:\Users\BJ\PycharmProjects\AGENDER2.0\obj"
files = os.listdir(path)
segment_duration = 10 #seconds
ECG_epochs_segment(files, condition, segment_duration)


    # Train & Test data of all participants for LSTM
segment_duration = 10
split_percentage = 0.1 #percentage of test data
sampling_rate = 1024
path = r"C:\Users\BJ\PycharmProjects\AGENDER2.0\obj"
files = os.listdir(path)
condition = "epochs_600seconds_noise_acc_filtered_segmented_"
x_train_all, y_train_all, x_test_all, y_test_all = split_train_test_forall(files,condition, split_percentage, segment_duration, sampling_rate)

save_obj(x_train_all,"x_train_epochs_length_600_segment_duration_30_train_split_0.1" )
save_obj(y_train_all,"y_train_all_epochs_length_600_segment_duration_30_train_split_0.1" )
save_obj(x_test_all,"x_test_all_epochs_length_600_segment_duration_30_train_split_0.1" )
save_obj(y_test_all,"y_test_all_epochs_length_600_segment_duration_30_train_split_0.1" )

x_train_all = load_obj("x_train_epochs_length_600_segment_duration_30_train_split_0.1.pkl")
y_train_all = load_obj("y_train_all_epochs_length_600_segment_duration_30_train_split_0.1.pkl")
x_test_all = load_obj("x_test_all_epochs_length_600_segment_duration_30_train_split_0.1.pkl")
y_test_all = load_obj("y_test_all_epochs_length_600_segment_duration_30_train_split_0.1.pkl")

# Visualize my training data
plt.hist(y_train_all)
plt.show()

#Train LSTM model
model, history, enc, y_train_encoded, y_test_encoded = LSTM(x_train_all, y_train_all, x_test_all, y_test_all)

model.save('C:\\Users\\lysan\\PycharmProjects\\AGENDER2.0\\models\\LSTM')
save_obj(history, "LSTM_history")

model = keras.models.load_model('C:\\Users\\lysan\\PycharmProjects\\AGENDER2.0\\models\\LSTM')
history = load_obj("LSTM_history.pkl")

# Visualize model results
## Plot history of loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend();

# Evaluate model
## Temporarily (until I run the model function again & without NaNs):
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
enc = enc.fit(y_train_all)
y_train_encoded = enc.transform(y_train_all)
y_test_encoded = enc.transform(y_test_all)
enc.categories_[0][2] = 999 #Label NaNs as 999

## Evaluate model
model.evaluate(x_test_all, y_test_encoded)
y_pred = model.predict(x_test_all)

#Show confusion matrix
def plot_cm(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(18, 16))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=sns.diverging_palette(220, 20, n=7),
        ax=ax
    )

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.show() # ta-da!


plot_cm(
    enc.inverse_transform(y_test_encoded),
    enc.inverse_transform(y_pred),
    enc.categories_[0]
)


#endregion

#region LSTM execution and evaluation
# this code-region is based on functions in the script "lstm.py" as well as on "visualizations_functions.py"

# Train & Test data of all participants
train_split = 0.8 #Percentage of training data
condition = "epochs_segmented_600seconds"
path = r"C:\Users\BJ\PycharmProjects\AGENDER2.0\obj"
files = os.listdir(path)
x_train_all, y_train_all, x_test_all, y_test_all = split_train_test_forall(files,condition, train_split, segment_duration, sampling_rate)

x_train_all = load_obj("x_train_epochs_length_600_segment_duration_30_train_split_0.1.pkl")
y_train_all = load_obj("y_train_all_epochs_length_600_segment_duration_30_train_split_0.1.pkl")
x_test_all = load_obj("x_test_all_epochs_length_600_segment_duration_30_train_split_0.1.pkl")
y_test_all = load_obj("y_test_all_epochs_length_600_segment_duration_30_train_split_0.1.pkl")

# Visualize my training data
plt.hist(y_train_all)

#Train LSTM model
model, history, enc, y_train_encoded, y_test_encoded = LSTM(x_train_all, y_train_all, x_test_all, y_test_all)

model.save('C:\\Users\\lysan\\PycharmProjects\\AGENDER2.0\\models\\LSTM')
save_obj(history, "LSTM_history")

model = keras.models.load_model('C:\\Users\\lysan\\PycharmProjects\\AGENDER2.0\\models\\LSTM')
history = load_obj("LSTM_history.pkl")

# Visualize model results
## Plot history of loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend();

# Evaluate model
## Temporarily (until I run the model function again & without NaNs):
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
enc = enc.fit(y_train_all)
y_train_encoded = enc.transform(y_train_all)
y_test_encoded = enc.transform(y_test_all)
enc.categories_[0][2] = 999 #Label NaNs as 999

## Evaluate model
model.evaluate(x_test_all, y_test_encoded)
y_pred = model.predict(x_test_all)

#Show confusion matrix
plot_cm(
    enc.inverse_transform(y_test_encoded),
    enc.inverse_transform(y_pred),
    enc.categories_[0]
)

#endregion
#endregion

#region Classification Pipeline 2-8: all supervised-learning models except LSTM

#region Preparation for all supervised learning models (except LSTM)
# the included steps were performed in varying combinations to test different data preprocessing approaches

df_class_allfeatures = load_obj("df_class_allfeatures.pkl")

df_class_allfeatures = load_obj("df_class_allfeatures_noiseIMPROVED.pkl") #temporary

#region Improvement pipeline 1.3: reducing epoch length to 10 minutes (from 20 minutes)
#deleting first 5 minutes (segments 1-10) and last 5 minutes (segments 31-40)
index_del = [1,2,3,4,5,6,7,8,9,10,31,32,33,34,35,36,37,38,39,40]
for i in index_del:
	df_class_allfeatures = df_class_allfeatures.drop(df_class_allfeatures[df_class_allfeatures.Segment == i].index)

#endregion

#region find out which columns contain NaN values
for i in list(df_class_allfeatures):
	print("The column" + str(i) + " contains following number of Nan "+ str(df_class_allfeatures[i].isnull().sum()))
#endregion

#region delete features which are not necessary (low frequency HRV)

features_del = ["HRV_LF", "HRV_LFHF", "HRV_LFn", "HRV_ULF", "HRV_VLF", "HRV_SampEn", "Proband", "Segment", "Epoch"]
df_class_allfeatures_del = df_class_allfeatures.drop(features_del, axis = 1)

#following two lines for LOSOCV (proband information remains included)
features_del_new = ["HRV_LF", "HRV_LFHF", "HRV_LFn", "HRV_ULF", "HRV_VLF", "HRV_SampEn", "Segment", "Epoch"]
df_class_allfeatures_del = df_class_allfeatures.drop(features_del_new, axis = 1)

#endregion

#region drop NaN rows
# Note: only HRV_HFn,HRV_LnHF, HRV_HF contain each 1 NaN value -> only 1 row is dropped
df_class_allfeatures_del_NaN = df_class_allfeatures_del.dropna()
#endregion

#region convert label: Subpipeline 01: convert into binary
df_class_allfeatures_del_NaN_binary = df_class_allfeatures_del_NaN.copy()
df_class_allfeatures_del_NaN_binary.loc[df_class_allfeatures_del_NaN_binary['Label'] > 0, 'Label'] = 1
#save_obj(df_class_allfeatures_del_NaN_binary, "df_class_allfeatures_del_NaN_binary")
#df_class_allfeatures_del_NaN_binary = load_obj("df_class_allfeatures_del_NaN_binary.pkl")

#endregion

#region convert label: subpipeline 01.2: convert into binary but only taking "high stress" into account)
df_class_allfeatures_del_NaN_binary = df_class_allfeatures_del_NaN.copy()

df_class_allfeatures_del_NaN_binary.loc[df_class_allfeatures_del_NaN_binary['Label'] > 66, 'Label'] = 1
#drop all rows were label column contains value above 1
df_class_allfeatures_del_NaN_binary = df_class_allfeatures_del_NaN_binary.drop(df_class_allfeatures_del_NaN_binary[df_class_allfeatures_del_NaN_binary.Label >1].index)

#balance the data again
df_class_allfeatures_del_NaN_binary = df_class_allfeatures_del_NaN_binary.sample(frac=1)#shuffle
number_to_drop = df_class_allfeatures_del_NaN_binary[df_class_allfeatures_del_NaN_binary["Label"]==0].shape[0]-df_class_allfeatures_del_NaN_binary[df_class_allfeatures_del_NaN_binary["Label"]==1].shape[0]
counter = 0
index = 0
while counter < number_to_drop:
	try:
		df_class_allfeatures_del_NaN_binary["Label"].loc[index]
	except:
		index = index+1
		continue
	else:
		if df_class_allfeatures_del_NaN_binary["Label"].loc[index] == 0:
			df_class_allfeatures_del_NaN_binary = df_class_allfeatures_del_NaN_binary.drop(labels = index, axis = 0)
			counter = counter + 1
			index = index +1
		else:
			index = index + 1

#endregion

#region shuffle dataset
df_class_allfeatures_del_NaN_binary_shuffled = df_class_allfeatures_del_NaN_binary.sample(frac=1)
df_class_allfeatures_del_NaN_binary_shuffled = df_class_allfeatures_del_NaN_binary_shuffled.reset_index(drop=True)
#save_obj(df_class_allfeatures_del_NaN_binary_shuffled, "df_class_allfeatures_del_NaN_binary_shuffled")
#save_obj(df_class_allfeatures_del_NaN_binary_shuffled, "df_class_allfeatures_del_NaN_binary_shuffled_includingProband")

#endregion

#region split training & test data
# the splitting function is included in the script "preprocessing_for_modeling.py"

#Note: better use SKlearn splitting function

#X = df_class_allfeatures_del_NaN_binary_shuffled.drop("Label", axis = 1)
#y = df_class_allfeatures_del_NaN_binary_shuffled["Label"]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=3)

#endregion


#region different datasets from different (sub)pipelines
#Basic Pipelind
df_class_allfeatures_del_NaN_binary_shuffled = load_obj("df_class_allfeatures_del_NaN_binary_shuffled.pkl")

#Basic Pipeline INCLUDING "Proband" column
df_class_allfeatures_del_NaN_binary_shuffled_includingProband = load_obj("df_class_allfeatures_del_NaN_binary_shuffled_includingProband.pkl")

#Subpipeline 1.1: Noise improved (noise excluded on level of noisy segments NOT whole epochs)
df_class_allfeatures_del_NaN_binary_shuffled_NOISEIMPROVED = load_obj("df_class_allfeatures_del_NaN_binary_shuffled_NOISEIMPROVED.pkl")

# Subpipeline 1.2.1: Noise improved AND only high stress included!
save_obj(df_class_allfeatures_del_NaN_binary ,"df_class_allfeatures_del_NaN_binary_shuffled_NOISEIMPROVED_HIGHSTRESS")
df_class_allfeatures_del_NaN_binary_shuffled_NOISEIMPROVED_HIGHSTRESS = load_obj("df_class_allfeatures_del_NaN_binary_shuffled_NOISEIMPROVED_HIGHSTRESS.pkl")

#Subpipeline 1.2.2: Noise NOT improved (excluded on epoch level) and only high stress
save_obj(df_class_allfeatures_del_NaN_binary ,"df_class_allfeatures_del_NaN_binary_shuffled_HIGHSTRESS")
df_class_allfeatures_del_NaN_binary_shuffled_HIGHSTRESS = load_obj("df_class_allfeatures_del_NaN_binary_shuffled_HIGHSTRESS.pkl")

# Subpipeline 1.3: epoch length only 10 minutes
save_obj(df_class_allfeatures_del_NaN_binary_shuffled ,"df_class_allfeatures_del_NaN_binary_shuffled_EPOCH10MIN")
df_class_allfeatures_del_NaN_binary_shuffled_EPOCH10MIN = load_obj("df_class_allfeatures_del_NaN_binary_shuffled_EPOCH10MIN.pkl")

# Subpipeline 1.3.2 only high stress events & epoch length only 10 minutes
save_obj(df_class_allfeatures_del_NaN_binary ,"df_class_allfeatures_del_NaN_binary_shuffled_HIGHSTRESS_EPOCH10MIN")
df_class_allfeatures_del_NaN_binary_shuffled_HIGHSTRESS_EPOCH10MIN = load_obj("df_class_allfeatures_del_NaN_binary_shuffled_HIGHSTRESS_EPOCH10MIN.pkl")

# Subpipeline 1.3.3 improved noise & only high stress events & epoch length only 10 minutes
save_obj(df_class_allfeatures_del_NaN_binary ,"df_class_allfeatures_del_NaN_binary_NOISEIMPROVED_HIGHSTRESS_EPOCH10MIN")
df_class_allfeatures_del_NaN_binary_NOISEIMPROVED_HIGHSTRESS_EPOCH10MIN = load_obj("df_class_allfeatures_del_NaN_binary_NOISEIMPROVED_HIGHSTRESS_EPOCH10MIN.pkl")

# Subpipeline: only predictive features: meanNN and meadian NN Interval with NORMAL STRESS
# normal stress (basic pipeline)
df_class_allfeatures_del_NaN_binary_shuffled = load_obj("df_class_allfeatures_del_NaN_binary_shuffled.pkl")
df_class_allfeatures_del_NaN_binary_shuffled_ONLYPREDICTIVE = df_class_allfeatures_del_NaN_binary_shuffled[["HRV_MeanNN", "HRV_MedianNN", "Label", "x1", "x2", "x3"]]
save_obj(df_class_allfeatures_del_NaN_binary_shuffled_ONLYPREDICTIVE ,"df_class_allfeatures_del_NaN_binary_shuffled_ONLYPREDICTIVE")

# Subpipeline: only predictive features: meanNN and meadian NN Interval with HIGH STRESS
df_class_allfeatures_del_NaN_binary_shuffled_HIGHSTRESS = load_obj("df_class_allfeatures_del_NaN_binary_shuffled_HIGHSTRESS.pkl")
df_class_allfeatures_del_NaN_binary_shuffled_HIGHSTRESS_ONLYPREDICTIVE = df_class_allfeatures_del_NaN_binary_shuffled_HIGHSTRESS[["HRV_MeanNN", "HRV_MedianNN", "Label", "x1", "x2", "x3"]]
save_obj(df_class_allfeatures_del_NaN_binary_shuffled_HIGHSTRESS_ONLYPREDICTIVE ,"df_class_allfeatures_del_NaN_binary_shuffled_HIGHSTRESS_ONLYPREDICTIVE")

# Subpipeline: all features except MeanNN and Median NN Interval with NORMAL STRESS
df_class_allfeatures_del_NaN_binary_shuffled = load_obj("df_class_allfeatures_del_NaN_binary_shuffled.pkl")
df_class_allfeatures_del_NaN_binary_shuffled_EXCEPTPREDICTIVE = df_class_allfeatures_del_NaN_binary_shuffled.copy()
del df_class_allfeatures_del_NaN_binary_shuffled_EXCEPTPREDICTIVE["HRV_MeanNN"]
del df_class_allfeatures_del_NaN_binary_shuffled_EXCEPTPREDICTIVE["HRV_MedianNN"]
save_obj(df_class_allfeatures_del_NaN_binary_shuffled_EXCEPTPREDICTIVE ,"df_class_allfeatures_del_NaN_binary_shuffled_EXCEPTPREDICTIVE")

# Subpipeline: only using 10 worst performing features (based on SHAP on DF which was trained on all HRV features without ACC)
df_class_allfeatures_del_NaN_binary_shuffled = load_obj("df_class_allfeatures_del_NaN_binary_shuffled.pkl")
df_class_allfeatures_del_NaN_binary_shuffled_Only10worstpredictive = df_class_allfeatures_del_NaN_binary_shuffled.copy()
choosecolumns = ["HRV_GI", "HRV_AI", "HRV_SI", "HRV_RMSSD", "HRV_C2d", "HRV_C2a", "HRV_SD2", "HRV_SD1", "HRV_Cd", "HRV_pNN50", "x1", "x2", "x3", "Label"]
df_class_allfeatures_del_NaN_binary_shuffled_Only10worstpredictive = df_class_allfeatures_del_NaN_binary_shuffled_Only10worstpredictive[choosecolumns]
save_obj(df_class_allfeatures_del_NaN_binary_shuffled_Only10worstpredictive ,"df_class_allfeatures_del_NaN_binary_shuffled_Only10worstpredictive")

# Subpipeline: only RMSSD and HF since we want to compare with Roehner findings
df_class_allfeatures_del_NaN_binary_shuffled = load_obj("df_class_allfeatures_del_NaN_binary_shuffled.pkl")
df_class_allfeatures_del_NaN_binary_shuffled_OnlyRMSSDandHF = df_class_allfeatures_del_NaN_binary_shuffled.copy()
choosecolumns = ["HRV_RMSSD", "HRV_HF" , "x1", "x2", "x3", "Label"]
df_class_allfeatures_del_NaN_binary_shuffled_OnlyRMSSDandHF = df_class_allfeatures_del_NaN_binary_shuffled_OnlyRMSSDandHF[choosecolumns]
save_obj(df_class_allfeatures_del_NaN_binary_shuffled_OnlyRMSSDandHF ,"df_class_allfeatures_del_NaN_binary_shuffled_OnlyRMSSDandHF")

# Subpipeline: only physical movement features
df_class_allfeatures_del_NaN_binary_shuffled = load_obj("df_class_allfeatures_del_NaN_binary_shuffled.pkl")
df_class_allfeatures_del_NaN_binary_shuffled_OnlyMovement = df_class_allfeatures_del_NaN_binary_shuffled.copy()
choosecolumns = [ "x1", "x2", "x3", "Label"]
df_class_allfeatures_del_NaN_binary_shuffled_OnlyMovement = df_class_allfeatures_del_NaN_binary_shuffled_OnlyMovement[choosecolumns]
save_obj(df_class_allfeatures_del_NaN_binary_shuffled_OnlyMovement ,"df_class_allfeatures_del_NaN_binary_shuffled_OnlyMovement")

#endregion

#endregion

#region Classification Pipeline 1: Stacking Ensemble Classifier
# all the code for this pipeline is included in the script "stacking.py"
#endregion

#region Classification Pipeline 2: Random Forest Classifier
# all the code for this pipeline is included in the script "random_forest.py"
#endregion

#region Classification Pipeline 3: Support Vector Machine Classifier (SVM)
# all the code for this pipeline is included in the script "support_vector_machine_(SVM).py"
#endregion

#region Classification Pipeline 4: Decision Tree Classifier
# all the code for this pipeline is included in the script "decision_tree.py"
#endregion

#region Classification Pipeline 5: Deep Neural Multilayer Perceptron Classifier (MLP)
# all the code for this pipeline is included in the script "deep_multilayer_perceptron_(MLP).py"
#endregion

#region Classification Pipeline 6: Logistic Regression
# all the code for this pipeline is included in the script "logistic_regression.py"
#endregion


#region possibly outdated Feature Creation
# Calculate HRV features
df_ecg_features = feature_creation(epochs_segmented)

#TODO: Calculate "Condition" (Stress Intensity) again & add to feature dataframe (its the label)
#TODO: FInd out minimal period needed for HRV feature computation!

#endregion

#TODO: Unbedingt peak-saving problem lösen: Speichere die Peaks in dem epochs-dict.

