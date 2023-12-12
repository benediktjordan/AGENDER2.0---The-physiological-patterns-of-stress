#region for main pipeline
#region Create segments for peaks (with noise exclusion on epoch level)
def create_segments(peaks, segment_duration):
    peaks_segmented = {}

    for i in peaks:
        proband = peaks[i]
        epochs_acc = load_obj("epochs_acc_600seconds_" + i + ".pkl")
        for l in proband:
            epoch_peaks = proband[l]["peaks_corr"]
            segments = {}
            counter = 1

            #Creating ECG segments: Iterating through every epoch and "taking out" segments; note: I don´t check if last segment has full length
            # -> could become a problem!
            for j in range(1, peaks[i][l]["ECG length"] + 1, segment_duration * 1024):
                begin = j - 1
                end = j + segment_duration * 1024

                if end > (peaks[i][l]["ECG length"] + 1):
                    break  # exits the for loop and continues with next epoch

                events = []
                events = copy.deepcopy(epoch_peaks)
                # next three lines filter out all peaks which are not relevant for this segment
                events = events - begin
                events = events[events >= 0]
                events = events[events <= (end - begin)]
                segments[counter] = events
                counter = counter+1


            #Creating ACC segments
            epoch_acc = epochs_acc[l]
            segments_acc = {}
            counter_acc = 1
            for j in range(1, (int(peaks[i][l]["ECG length"]/16) + 1), segment_duration * 64):
                begin = j - 1
                end = j + segment_duration * 64

                if end > (peaks[i][l]["ECG length"]/16 + 1):
                    break  # exits the for loop and continues with next epoch

                events_acc = []
                events_acc = copy.deepcopy(epoch_acc)
                events_acc_cropped = events_acc.iloc[begin:end, :]
                segments_acc[counter_acc] = events_acc_cropped
                counter_acc = counter_acc + 1

            peaks[i][l]["segments_" + str(segment_duration)] = copy.deepcopy(segments)
            peaks[i][l]["segments_acc_" + str(segment_duration)] = copy.deepcopy(segments_acc)
    return peaks

#endregion

#region create segments for peaks (with noise exclusion on 15-second segment level)
peaks_corrected_cutoff = load_obj("peaks_corrected_cutoff.pkl")
noise = load_obj("noise_labels.pkl")
#peaks_test = create_segments_excludingnoise(peaks_corrected_cutoff, noise)

def create_segments_excludingnoise(peaks, noise, segment_duration = 30):
    peaks_segmented = {}

    for proband in noise:
        #load ACC epochs
        epochs_acc = load_obj("epochs_acc_600seconds_" + str(proband) + ".pkl")

        for epoch in noise[proband]:
            #load peaks & ACC epoch
            epoch_peaks = peaks[proband][epoch]["peaks_corr"]
            epoch_acc = epochs_acc[epoch]
            segments = {}
            segments_acc = {}
            counter = 1

            #iterate through noise segments and chose adjacent no-noise segments
            segment_length = len(noise[proband][epoch])-1
            vis_segment = 0
            while vis_segment < segment_length:
                if noise[proband][epoch][vis_segment] != 2:
                    if noise[proband][epoch][vis_segment+1] != 2: #case 1: both segments don´t contain noise
                        print("Add segment " + str(vis_segment) + " and segment " + str(vis_segment+1))

                        # Creating ECG segments: Iterating through every epoch and "taking out" segments; note: I don´t check if last segment has full length
                        # -> could become a problem!
                        begin_ecg = vis_segment * 15 * 1024
                        end_ecg = begin_ecg + 30*1024
                        events = []
                        events = copy.deepcopy(epoch_peaks)
                        # next three lines filter out all peaks which are not relevant for this segment
                        events = events - begin_ecg
                        events = events[events >= 0]
                        events = events[events <= (end_ecg - begin_ecg)]
                        segments[counter] = events

                        # Creating ACC segments
                        begin_acc = vis_segment * 15 * 64
                        end_acc = begin_acc + 30*64
                        events_acc = []
                        events_acc = copy.deepcopy(epoch_acc)
                        events_acc_cropped = events_acc.iloc[begin_acc:end_acc, :]
                        segments_acc[counter] = events_acc_cropped

                        counter = counter + 1
                        vis_segment = vis_segment + 2
                    else: #case 2: first segment contains no noise but second segment contains noise
                        vis_segment = vis_segment + 2
                else: #case 3: first segment contains noise
                    vis_segment = vis_segment+1

            peaks[proband][epoch]["segments_30"] = copy.deepcopy(segments)
            peaks[proband][epoch]["segments_acc_30"] = copy.deepcopy(segments_acc)
    return peaks


#testrun
"""
peaks = peaks_corrected_cutoff
epochs_acc = load_obj("epochs_acc_600seconds_" + str(proband) + ".pkl")
epoch_peaks = peaks[proband][epoch]["peaks_corr"]
epoch_acc = epochs_acc[epoch]
segments = {}
segments_acc = {}
counter = 1
segment_length = len(noise[proband][epoch]) - 1
vis_segment = 0



while vis_segment < segment_length:
    if noise[proband][epoch][vis_segment] != 2:
        if noise[proband][epoch][vis_segment + 1] != 2:  # case 1: both segments don´t contain noise
            print("Add segment " + str(vis_segment) + " and segment " + str(vis_segment + 1))

            # Creating ECG segments: Iterating through every epoch and "taking out" segments; note: I don´t check if last segment has full length
            # -> could become a problem!
            begin_ecg = vis_segment * 15 * 1024
            end_ecg = begin_ecg + 30 * 1024
            events = []
            events = copy.deepcopy(epoch_peaks)
            # next three lines filter out all peaks which are not relevant for this segment
            events = events - begin_ecg
            events = events[events >= 0]
            events = events[events <= (end_ecg - begin_ecg)]
            segments[counter] = events

            # Creating ACC segments
            begin_acc = vis_segment * 15 * 64
            end_acc = begin_acc + 30 * 64
            events_acc = []
            events_acc = copy.deepcopy(epoch_acc)
            events_acc_cropped = events_acc.iloc[begin_acc:end_acc, :]
            segments_acc[counter] = events_acc_cropped

            counter = counter + 1
            vis_segment = vis_segment + 2
        else:  # case 2: first segment contains no noise but second segment contains noise
            vis_segment = vis_segment + 2
    else:  # case 3: first segment contains noise
        vis_segment = vis_segment + 1


"""
#endregion

#region HRV Feature Creation
def hrv_feature_creation(peaks_segmented, segment_duration):
    """
    Computing HRV features & adding them to peaks dict
    :param peaks_segmented:
    :return:
    """
    peaks_hrvfeatures = copy.deepcopy(peaks_segmented)
    for proband in peaks_hrvfeatures:
        for epoch in peaks_hrvfeatures[proband]:
            segments_hrv = {}
            for segment in peaks_hrvfeatures[proband][epoch]["segments_"+str(segment_duration)]:
                try:
                    hrv_indices = nk.hrv(peaks_hrvfeatures[proband][epoch]["segments_"+str(segment_duration)][segment], sampling_rate=1024)
                except Exception:
                    print("Exception for proband "+ str(proband) +"Epoch " + str(epoch) + "Segment" + str(segment) )
                else:
                    hrv_indices = nk.hrv(peaks_hrvfeatures[proband][epoch]["segments_"+str(segment_duration)][segment], sampling_rate=1024)
                    segments_hrv[segment] = hrv_indices
                    #print("Proband" + str(proband) + "Epoch " + str(epoch) + "Segment" + str(segment))
            peaks_hrvfeatures[proband][epoch]["Features_HRV"] = copy.deepcopy(segments_hrv)

    return peaks_hrvfeatures


#endregion

#region resample ACC to segment period

def ACC_resample(peaks_hrvfeatures,segment_duration, resampling_method):
    peaks_hrvfeatures_acc = copy.deepcopy(peaks_hrvfeatures)
    for proband in peaks_hrvfeatures_acc:
        for epoch in peaks_hrvfeatures_acc[proband]:
            peaks_hrvfeatures[proband][epoch]["segments_acc_single" + str(segment_duration)] = {}
            for segment_acc in peaks_hrvfeatures_acc[proband][epoch]["segments_acc_" + str(segment_duration)]:
                x1 = nk.signal_resample(peaks_hrvfeatures_acc[proband][epoch]["segments_acc_" + str(segment_duration)][segment_acc]["x1"],
                                                     method=resampling_method, sampling_rate=64, desired_sampling_rate=(1/segment_duration))
                x2 = nk.signal_resample(peaks_hrvfeatures_acc[proband][epoch]["segments_acc_" + str(segment_duration)][segment_acc]["x2"],
                                                     method=resampling_method, sampling_rate=64, desired_sampling_rate=(1/segment_duration))
                x3 = nk.signal_resample(peaks_hrvfeatures_acc[proband][epoch]["segments_acc_" + str(segment_duration)][segment_acc]["x3"],
                                                     method=resampling_method, sampling_rate=64, desired_sampling_rate=(1/segment_duration))

                peaks_hrvfeatures[proband][epoch]["segments_acc_single" + str(segment_duration)][segment_acc] = pd.DataFrame({"x1": x1,"x2": x2,"x3": x3})
                print("PRoband"+str(proband)+ " epoch" + str(epoch) + "segment" + str(segment_acc) +" is done now " )


    return peaks_hrvfeatures

#endregion

#region transfer data from dict to dataframe
def create_classdf(peaks_hrvfeatures_acc):
    df = pd.DataFrame()
    for proband in peaks_hrvfeatures_acc:
        for epoch in peaks_hrvfeatures_acc[proband]:
            t0_epoch = time.time()
            label = peaks_hrvfeatures_acc[proband][epoch]["Stress label"]
            for segment in peaks_hrvfeatures_acc[proband][epoch]["Features_HRV"]:
                row = pd.Series([],dtype=pd.StringDtype())
                row["Proband"] = proband
                row["Epoch"] = epoch
                row["Segment"] = segment
                row = row.append(peaks_hrvfeatures_acc[proband][epoch]["Features_HRV"][segment].iloc[0])
                row = row.append(peaks_hrvfeatures_acc[proband][epoch]["segments_acc_single30"][segment].iloc[0])
                row["Label"] = label
                df = df.append(row, ignore_index = True)
            t1_epoch = time.time()
            print("Proband" + str(proband) + " Epoch " + str(epoch) + " are done now and df has now shape " +str(df.shape) +
                  " time is " + str(t1_epoch-t0_epoch))
    return df
#endregion

#endregion

#Note: the following regions are used for the preparation for the LSTM algorithm

#region Add manual noise labels to ECG epochs
def ECG_epochs_addnoise(noise_labels, vis_period, sampling_rate):
    for proband in noise:
        name = "epochs_600seconds_filtered_"+str(proband)+".pkl"
        epochs = load_obj(name)
        for epoch in noise[proband]:

            # create list with noise for every epoch
            noise_segments = []
            for vis_segment in noise_labels[proband][epoch]:
                noise_segment = np.repeat(noise_labels[proband][epoch][vis_segment], sampling_rate*vis_period)
                noise_segments = np.concatenate((noise_segments, noise_segment))

            #add noise data to epoch
            epochs[epoch] = epochs[epoch].reset_index()
            epochs[epoch].loc[:,"Noise_Labels"] = pd.Series(noise_segments)
            print("Success for proband" +str(proband) + " epoch " + str(epoch))
        save_obj(epochs, "epochs_600seconds_noise_filtered_"+str(proband))





    for name in files:
        # load the epochs objects
        if condition not in name:
            continue
        proband = name[-13:-4]
        epochs = load_obj(name)
        epochs_new = {}
        for epoch in epochs:
            epoch = epochs[i]

            for j in range(1, epochs[epoch].shape[0] + 1, vis_period * sampling_rate):
                begin = j - 1
                end = j + vis_period * sampling_rate

                # Only select segments of length "vis_period" in epochs
                if end < epochs[epoch].shape[0]:

                    df_cropped = []
                    df_cropped = df.iloc[begin:end, :]
                else:
                    break  # exits the for loop and continues with next epoch

                ecg = df_cropped["ECG"]
            epochs_new[i] = epoch.iloc[1024*60*number_of_minutes:]        #selects everything except first minute
        save_obj(epochs_new, "epochs_600seconds_filtered_" + proband)

#endregion

#region Add resampled ACC to ECG epochs
def ECG_epochs_addACC(files, condition,upsample_method):
    for name in files:
        if condition not in name:
            continue

        proband = name[-13:-4]

        # load the epochs for the proband
        epochs = load_obj(name)
        epochs_acc = load_obj("epochs_acc_600seconds_"+ str(proband)+".pkl")

        # iterate through epochs & add ACC
        for epoch in epochs:
            epochs[epoch]["x1"] = nk.signal_resample(epochs_acc[epoch]["x1"],method=upsample_method, sampling_rate=64, desired_sampling_rate=1024)
            epochs[epoch]["x2"] = nk.signal_resample(epochs_acc[epoch]["x2"],method=upsample_method, sampling_rate=64, desired_sampling_rate=1024)
            epochs[epoch]["x3"] = nk.signal_resample(epochs_acc[epoch]["x3"],method=upsample_method, sampling_rate=64, desired_sampling_rate=1024)

        save_obj(epochs, "epochs_600seconds_noise_acc_filtered_"+ str(proband))
        print("Proband " + str(proband) + " Epoch " + str(epoch) +" is done")

#endregion

#region create segments for epochs
def ECG_epochs_segment(files, condition, segment_duration):
    for name in files:
        if condition not in name:
            continue

        proband = name[-13:-4]

        # load the epochs for the proband
        epochs = load_obj(name)
        epochs_segmented = {}

        for epoch in epochs:
            epochs_segmented[epoch] = {}
            counter = 0
            for j in range(1, epochs[epoch].shape[0] + 1, segment_duration * 1024):
                counter = counter +1
                begin = j - 1
                end = j + segment_duration * 1024
                if end > (epochs[epoch].shape[0] + 1):
                    break  # exits the for loop and continues with next epoch

                epochs_segmented[epoch][counter] = epochs[epoch].iloc[begin:end].copy()

        save_obj(epochs_segmented, "epochs_600seconds_noise_acc_filtered_segmented_"+str(proband))
        print("Proband " + str(proband) + " is now done")





    epochs_segmented = {}

    for i in peaks:
        proband = peaks[i]
        epochs_acc = load_obj("epochs_acc_600seconds_" + i + ".pkl")
        for l in proband:
            epoch_peaks = proband[l]["peaks_corr"]
            segments = {}
            counter = 1

            #Creating ECG segments: Iterating through every epoch and "taking out" segments; note: I don´t check if last segment has full length
            # -> could become a problem!
            for j in range(1, peaks[i][l]["ECG length"] + 1, segment_duration * 1024):
                begin = j - 1
                end = j + segment_duration * 1024

                if end > (peaks[i][l]["ECG length"] + 1):
                    break  # exits the for loop and continues with next epoch

                events = []
                events = copy.deepcopy(epoch_peaks)
                # next three lines filter out all peaks which are not relevant for this segment
                events = events - begin
                events = events[events >= 0]
                events = events[events <= (end - begin)]
                segments[counter] = events
                counter = counter+1


            #Creating ACC segments
            epoch_acc = epochs_acc[l]
            segments_acc = {}
            counter_acc = 1
            for j in range(1, (int(peaks[i][l]["ECG length"]/16) + 1), segment_duration * 64):
                begin = j - 1
                end = j + segment_duration * 64

                if end > (peaks[i][l]["ECG length"]/16 + 1):
                    break  # exits the for loop and continues with next epoch

                events_acc = []
                events_acc = copy.deepcopy(epoch_acc)
                events_acc_cropped = events_acc.iloc[begin:end, :]
                segments_acc[counter_acc] = events_acc_cropped
                counter_acc = counter_acc + 1

            peaks[i][l]["segments_" + str(segment_duration)] = copy.deepcopy(segments)
            peaks[i][l]["segments_acc_" + str(segment_duration)] = copy.deepcopy(segments_acc)
    return peaks
#endregion



def feature_creation(epochs_segmented):
    df_ecg_features = pd.DataFrame()
    for key in epochs_segmented:
        epoch_segmented = epochs_segmented[key]
        features = nk.ecg_intervalrelated(epoch_segmented)
        features["Epoch"] = key
        df_ecg_features = df_ecg_features.append(features)

    return df_ecg_features

##TODO Create HR features
#endregion

