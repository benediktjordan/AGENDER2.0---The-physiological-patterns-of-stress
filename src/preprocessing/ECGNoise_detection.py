#TODO manual labeling of noise: continue (adapt starting point; now its 318)

#Description: this script focusses on creating a manual noise detection system

#region For main pipelin
#region calculate percentage of noise for every epoch

def noise_percentage(noise, peaks):
    '''
    Calculates percentages of noise and includes also the label of the epoch
    :param noise:
    :param peaks:
    :return:
    '''
    probands = []
    epochs_name = []
    numbers_segments = []
    numbers_noise = []
    percentages = []
    stress_labels = []

    for proband in noise:

        for epoch in noise[proband]:
            number_segments = len(noise[proband][epoch])
            number_noise = noise[proband][epoch].count(2)
            percentage = number_noise / number_segments
            stress_label = peaks[proband][epoch]["Stress label"]

            probands.append(proband)
            epochs_name.append(epoch)
            numbers_segments.append(number_segments)
            numbers_noise.append(number_noise)
            percentages.append(percentage)
            stress_labels.append(stress_label)

    df = pd.DataFrame(
        {"Proband": probands,
         "Epoch": epochs_name,
         "Number of Segments": numbers_segments,
         "Number of Noise": numbers_noise,
         "Percentage of Noise": percentages,
         "Stress Label": stress_labels}
    )
    return df

#df_noise = noise_percentage(noise,peaks)
#endregion
#endregion


#region ECG: labeling noise

# iterate through different probands; check also if I "treated" that proband already
peaks_corrected = load_obj("peaks_corrected_cutoff.pkl")
vis_period = 15
sampling_rate = 1024
condition = "epochs_600seconds_filtered"
path = r"C:\Users\BJ\PycharmProjects\AGENDER2.0\obj"
files = os.listdir(path)
#alreadydone = []
#noise = {}

alreadydone = load_obj("noise_ProbandsAlreadyDone.pkl")
noise = load_obj("noise_labels.pkl")

for name in files:
    proband = name[-13:-4]
    if condition not in name:
        continue
    if proband in alreadydone: #check if noise has been already detected for this proband
        continue
    noise[proband] = {}

    #load the epochs for the proband
    epochs = load_obj(name)

    #iterate through epochs & visualize
    for epoch in epochs:
        df = epochs[epoch]
        labels = [] #initialize noise label list

        for j in range(1, df.shape[0] + 1, vis_period * sampling_rate):
            begin = j - 1
            end = j + vis_period * sampling_rate

            # Only select segments of length "vis_period" in epochs
            if end < df.shape[0]:
                df_cropped = []
                df_cropped = df.iloc[begin:end, :]
            else:
                break #exits the for loop and continues with next epoch

            ecg = df_cropped["ECG"]

           # Plot
            nrow = 2;
            ncol = 1;
            fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(15, 8))
            fig.suptitle('Peaks and Artifacts')

            #this creates the ECG and peak visualization
            axs[0].plot(np.array(ecg))
            axs[0].set_xlim((0, 16000))

            events = []
            events = peaks_corrected[proband][epoch]["peaks_corr"].copy()
            # next three lines filter out all events which are not relevant for this vis_period
            events = events - begin
            events = events[events >= 0]
            events = events[events <= (end - begin)]
            for p in events:
                axs[0].axvline(p, color="red", linestyle="--")
            axs[0].set_title("ECG & Peaks")

            # with the following lines Brown artifact visualization is created
            artifacts = peaks_corrected[proband][epoch]["artifacts"].copy()
            artifacts = artifacts.drop(artifacts[artifacts.type == 0].index) #all 0 (where no change was made) are dropped
            # next three lines filter out all artifacts which are not relevant for this vis_period
            artifacts["location"] = artifacts["location"] - begin
            artifacts = artifacts.drop(artifacts[artifacts.location < 0].index)
            artifacts = artifacts.drop(artifacts[artifacts.location > (end - begin)].index)
            scatter = axs[1].scatter(artifacts["location"],
                                      artifacts["type"])
            axs[1].set_title("Artifacts")
            axs[1].set_xlim((0, 16000))
            axs[1].set_ylim((0.5, 5.5))
            # classes = ["ectopic", "missed", "extra", "longshort"]
            # axs[2].legend([1,2,3,4], classes)

            plt.xlim(0, 16000)
            plt.show()
            # Adjust plots
            fig.tight_layout()

            plt.pause(0.8)
            # close figure
            plt.close()  # closes figure

            #Check for noise system
            label = input("Is there noise?")
            if len(label) == 0:
                labels.append(0)
            else:
                labels.append(int(label))


        noise[proband][epoch] = labels #add noise list of epoch to noise dict
        print("Proband" + str(proband) +" and Epoch " + str(epoch) + "is done now")
    alreadydone.append(proband)
    save_obj(noise, "noise_labels")
    save_obj(alreadydone, "noise_ProbandsAlreadyDone")
    print("Proband "+str(proband) + " is done")
    plt.pause(15)

#endregion



#region visualize epochs with noise
peaks_corrected = load_obj("peaks_corrected_cutoff.pkl")
vis_period = 15
sampling_rate = 1024
storage_path = "C:/Users/BJ/PycharmProjects/AGENDER2.0/01_DataCleaning/Noise/ManualDetection/above10%below30%/"

def visualize_corrected(df_noise, peaks_corrected, vis_period, sampling_rate, storage_path ):
    for index, proband in enumerate(df_noise["Proband"]):
        #get name of epoch
        epoch = df_noise["Epoch"][index]

        # load the epochs for the proband
        name = "epochs_600seconds_filtered_"+proband+".pkl"
        epochs = load_obj(name)

        df = epochs[epoch]
        labels = []  # initialize noise label list

        for j in range(1, df.shape[0] + 1, vis_period * sampling_rate):
            begin = j - 1
            end = j + vis_period * sampling_rate

            # Only select segments of length "vis_period" in epochs
            if end < df.shape[0]:
                df_cropped = []
                df_cropped = df.iloc[begin:end, :]
            else:
                break  # exits the for loop and continues with next epoch

            ecg = df_cropped["ECG"]

            # Plot
            nrow = 2;
            ncol = 1;
            fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(15, 8))
            fig.suptitle('Peaks and Artifacts')

            # this creates the ECG and peak visualization
            axs[0].plot(np.array(ecg))
            axs[0].set_xlim((0, 16000))

            events = []
            events = peaks_corrected[proband][epoch]["peaks_corr"].copy()
            # next three lines filter out all events which are not relevant for this vis_period
            events = events - begin
            events = events[events >= 0]
            events = events[events <= (end - begin)]
            for p in events:
                axs[0].axvline(p, color="red", linestyle="--")
            axs[0].set_title("ECG & Peaks")

            # with the following lines Brown artifact visualization is created
            artifacts = peaks_corrected[proband][epoch]["artifacts"].copy()
            artifacts = artifacts.drop(
                artifacts[artifacts.type == 0].index)  # all 0 (where no change was made) are dropped
            # next three lines filter out all artifacts which are not relevant for this vis_period
            artifacts["location"] = artifacts["location"] - begin
            artifacts = artifacts.drop(artifacts[artifacts.location < 0].index)
            artifacts = artifacts.drop(artifacts[artifacts.location > (end - begin)].index)
            scatter = axs[1].scatter(artifacts["location"],
                                     artifacts["type"])
            axs[1].set_title("Artifacts")
            axs[1].set_xlim((0, 16000))
            axs[1].set_ylim((0.5, 5.5))
            # classes = ["ectopic", "missed", "extra", "longshort"]
            # axs[2].legend([1,2,3,4], classes)

            plt.xlim(0, 16000)
            plt.show()
            # Adjust plots
            fig.tight_layout()

            # Save
            starting = j / sampling_rate

            fig.savefig(
                storage_path+"Participant_" +
                str(proband) + "_Epoch_" + str(epoch) + "_StartingTime (second)_" + str(starting) + ".png")
            plt.close()  # closes figure

storage_path = "C:/Users/BJ/PycharmProjects/AGENDER2.0/01_DataCleaning/Noise/ManualDetection/above10%below30%/"
visualize_corrected(df_noise_above10below30, peaks_corrected, vis_period, sampling_rate, storage_path)

storage_path = "C:/Users/BJ/PycharmProjects/AGENDER2.0/01_DataCleaning/Noise/ManualDetection/above30%/"
visualize_corrected(df_noise_above30, peaks_corrected, vis_period, sampling_rate, storage_path)
#endregion





#Load all segmented data into one array: x for ECG and y for stress labels
condition = "epochs_segmented_600secondsEpochsDuration_5_secondsSegmentsDuration_filtered"
segment_duration = 5  # s
sampling_rate = 1024 #Hz
path = r"C:\Users\BJ\PycharmProjects\AGENDER2.0\obj"
files = os.listdir(path)
x_all, y_all = create_array_allepochs_segmented(files,condition, segment_duration, sampling_rate)

## Save arrays
save_obj(x_all, "x_all_" + condition)
save_obj(y_all, "y_all_" + condition)

# Iterate through data & label
#noise_labels = []
peak_detection_method ="kalidas2017"       #kalidas2017 is the method which has been proven to be best in script "02_PeakDetection"

for i in range(318,len(x_all),1):
    events = nk.ecg_findpeaks(np.squeeze(x_all[i], axis=-1), method=peak_detection_method)
    plt.plot(x_all[i])
    if len(events) != 0:
        for p in events["ECG_R_Peaks"]:
            plt.axvline(p, color="red", linestyle="--")
    plt.pause(1)
    plt.close()
    a = int(input("Is there noise?"))
    noise_labels.append(a)

#TEMPORARY
events = nk.ecg_findpeaks(np.squeeze(x_all[317], axis=-1), method=peak_detection_method)
plt.plot(x_all[317])
for p in events["ECG_R_Peaks"]:
    plt.axvline(p, color="red", linestyle="--")
plt.show()
#endregion

test = np.squeeze(x_all[1], axis=-1)