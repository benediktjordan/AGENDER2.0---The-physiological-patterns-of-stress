#TODO: Unbedingt peak-saving problem lösen: Speichere die Peaks in dem epochs-dict.



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



#region PreStep 1: load & transform (including filtering) all ECG & ACC data (until filtered epochs)
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

#region PreStep 2: Peak detection, correction & noise detection
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


#region Pipeline 1: Ensemble classification preparation: segment peaks & feature creation
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

#endregion

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


#region Pipeline 2: LSTM classification preparation:  segment epochs & add noise & ACC features to ECG epochs
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





#region Pipeline 01 Feature Creation
# Calculate HRV features
df_ecg_features = feature_creation(epochs_segmented)

#TODO: Calculate "Condition" (Stress Intensity) again & add to feature dataframe (its the label)
#TODO: FInd out minimal period needed for HRV feature computation!

#endregion

#region Pipeline 02: LSTM
# this code-region is based on functions in the script "lstm.py"

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
