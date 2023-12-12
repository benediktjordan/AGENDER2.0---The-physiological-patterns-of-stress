#TODO: adjust: create visualisations for all probands and all epochs to really compare the performance of the peak detection algorithsm

#Description: Idea of this script is to compare the different peak detection algorithms implemented in NK2 and


#Progress: completed; (result: neurokit is the best method);


#region Comparison of different NK2 peak detection algorithms; peaks are computed for vis_periods (approach 01)


#region Variables
methods = ["neurokit", "pantompkins1985", "nabian2018", "hamilton2002", "martinez2003", "christov2004",
           "gamboa2008", "elgendi2010", "engzeemod2012","kalidas2017", "rodrigues2020", "promac" ]
vis_period = 15
sampling_rate = 1024
path = r"C:\Users\BJ\PycharmProjects\AGENDER2.0\obj"
condition = "epochs_660seconds_filtered"
#endregion

#region Function
def peakdetection_comparison(path, condition, methods, vis_period, sampling_rate):
    """
    This function creates visualisations which compare the 12 peak detection algorithms by comparing visualisations
    of peaks detected by them on 15 seconds intervalls of ECG data;
    peaks are computed FOR EACH OF THE 15 SECOND INTERVALS INDIVIDUALLY

    :param path: path where the saved epochs are located
    :param condition: condition which clearly identifies the names of the epoch-files to be used
    :param methods: The peak detection methods which are compared
    :param vis_period: The period for the individual visualisations (in seconds)
    :param sampling_rate: The sampling rate of the ECG data
    :return:
    """
    files = os.listdir(path)
    counter = 0

    for name in files:
        # load the epochs objects
        if condition not in name:
            continue
        epochs = load_obj(name)

        # Detect peaks
        proband = name[-13:-4]

        # Iterate through different epochs & then iterate through different segments within the epochs
        for epoch in epochs:
            df = epochs[epoch]
            for j in range(1, df.shape[0] + 1, vis_period * sampling_rate):
                begin = j - 1
                end = j + vis_period * sampling_rate

                # Only select segments of length "vis_period" in epochs
                if end < df.shape[0]:
                    df_cropped = []
                    df_cropped = df.iloc[begin:end, :]
                else:
                    break

                ecg = df_cropped["ECG"]

                # Apply Peak detection methods; only run them if they are actually working
                for l in methods:
                    try:
                        globals()[l] = nk.ecg_findpeaks(ecg, method=l)
                    except Exception:
                        globals()[l] = []
                    else:
                        globals()[l] = nk.ecg_findpeaks(ecg, method=l)

               # Plot
                nrow = 4;
                ncol = 3;
                fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(15, 8))
                fig.suptitle('Different Peak Detection Algorithms')

                for ax, method in zip(axs.reshape(-1), methods):
                    ax.plot(np.array(ecg))
                    events = globals()[method]
                    if len(events) != 0:
                        for p in events["ECG_R_Peaks"]:
                            ax.axvline(p, color="red", linestyle="--")
                    ax.set_title(method)

                plt.show()

                # Adjust plots
                fig.tight_layout()

                # Save
                starting = j / sampling_rate
                fig.savefig("C:/Users/BJ/PycharmProjects/AGENDER2.0/01_DataCleaning/Peaks/PeakDetectionComparison_control/Participant_" +
                            str(proband) + "_Epoch_" + str(epoch) + "_StartingTime (second)_" + str(starting) + ".png")
                plt.close()  # closes figure

                # Update counter
                counter = counter + 1

#endregion

#region Run Function
peakdetection_comparison(path, condition, methods, vis_period, sampling_rate)
#endregion



## Run functions (archive?)
lowpass_epochs(epochs, lowcut, vis_period, sampling_rate)

#endregion


#region Comparison of different NK2 peak detection algorithms; peaks are computed for epochs (approach 02)

#Variables
## Variables for peak detection
path = r"C:\\Users\\BJ\\Documents\\07_MPI\\03_Data\\01_Stress\\01_Roehner\\02_RawData\\Preprocessed\\"
folders = os.listdir(path)

path = r"C:\Users\BJ\PycharmProjects\AGENDER2.0\obj"
files = os.listdir(path)
condition = "epochs_660seconds_filtered"
method_peakdetection = "kalidas2017"
#This is temporarily created so I don´t have to run the main script every time again
proband_withoutevents =['AGENDER07',
 'AGENDER11',
 'AGENDER11_1',
 'AGENDER13',
 'AGENDER18',
 'AGENDER26']
methods = ["neurokit", "pantompkins1985", "nabian2018", "hamilton2002", "martinez2003", "christov2004",
           "gamboa2008", "elgendi2010", "engzeemod2012","kalidas2017", "rodrigues2020", "promac" ]


##Variables for visualization of peak comparison images
##TODO Visualize peaks
#Approach: use peak_vis function from script "03-2RRR Peak Noise Detection"

vis_period = 15                     #in seconds
sampling_rate = 1024
condition = "epochs_660seconds_filtered"
storage_path = "C:/Users/BJ/PycharmProjects/AGENDER2.0/01_DataCleaning/Peaks/PeakDetectionComparison_epochs/"
nrow = 4
ncol = 3



#Function
## Peak Detection
def create_peaks_allNK2(folders, files, condition, methods, proband_withoutevents):
    """
    This function creates peaks with the kalidas2017 algorithm. Peaks are created ON THE LEVEL OF EPOCHS!
    This is a difference to the peak_detection_algorithm_comparison where peaks have been detected ON THE LEVEL
    OF 15 seconds visualization periods
    :param files:
    :param folders:
    :param condition:
    :param method_peakdetection:
    :param proband_withoutevents:
    :return:
    """

    # Iterate through different peak detection methods
    for method in methods:

        # Iterate through different probands epoched data
        for name in files:
            #load the epochs objects
            if condition not in name:
                continue
            epochs = load_obj(name)

            # Detect peaks by iterating through epochs
            proband = name[-13:-4]
            for i in epochs:
                ecg = np.array(epochs[i]["ECG"])

                #Only use peak detection methods which don´t result in an error
                try:
                    peaks = nk.ecg_findpeaks(ecg, method=method)
                except Exception:
                    peaks = []
                    globals()["epoch_" + str(i)] = peaks
                else:
                    peaks = nk.ecg_findpeaks(ecg, method=method)
                    globals()["epoch_"+str(i)] = peaks["ECG_R_Peaks"]

            # store the detected peaks in dict with PROBAND -> EPOCH -> PEAKS structure
            globals()[proband] = {}
            for i in epochs:
                globals()[proband][i] = globals()["epoch_"+str(i)]

        #Create dict which contains all peaks from participants
        globals()["peaks_" + str(method)] = {}
        for proband in folders:
            if proband in proband_withoutevents:
                continue
            globals()["peaks_" + str(method)][proband] = globals()[proband]

    #Create one dict in which the results for all peak detection algorithms are stored
    peaks_all = {}
    for m in methods:
        peaks_all[m] = globals()["peaks_" + str(m)]

    return peaks_all



#Run
## Peak detection
peaks_all = create_peaks_allNK2(folders, files, condition, methods, proband_withoutevents)
save_obj(peaks_all, "peakdetection_comparisonNK2_epochs")
peaks_all = load_obj("peakdetection_comparisonNK2_epochs.pkl")

## Peak Visualization
peak_vis(condition, peaks_all, vis_period, sampling_rate, storage_path, nrow, ncol)


#endregion

#region BUG: Find out why some peakdetection algorithms don´t work
#Solution: the ecg data has to be a numpy ndarray (not a dataframe as before) in order to be accepted by some of th
# peak detection algorithms

import neurokit2 as nk
ecg2 = nk.ecg_simulate(duration=10, sampling_rate=1000)
cleaned = nk.ecg_clean(ecg, sampling_rate=1000)

neurokit = nk.ecg_findpeaks(nk.ecg_clean(ecg, method="neurokit"), method="neurokit")
pantompkins1985 = nk.ecg_findpeaks(nk.ecg_clean(ecg, method="pantompkins1985"), method="pantompkins1985")
nabian2018 = nk.ecg_findpeaks(cleaned, method="nabian2018")
hamilton2002 = nk.ecg_findpeaks(nk.ecg_clean(ecg, method="hamilton2002"), method="hamilton2002")
martinez2003 = nk.ecg_findpeaks(ecg, method="martinez2003")
christov2004 = nk.ecg_findpeaks(ecg, method="christov2004")
gamboa2008 = nk.ecg_findpeaks(nk.ecg_clean(ecg2, method="gamboa2008"), method="gamboa2008")
elgendi2010 = nk.ecg_findpeaks(nk.ecg_clean(ecg, method="elgendi2010"), method="elgendi2010")
engzeemod2012 = nk.ecg_findpeaks(nk.ecg_clean(ecg, method="engzeemod2012"), method="engzeemod2012")
kalidas2017 = nk.ecg_findpeaks(nk.ecg_clean(ecg, method="kalidas2017"), method="kalidas2017")
rodrigues2020 = nk.ecg_findpeaks(cleaned, method="rodrigues2020")

test = np.array(ecg)

neurokit = nk.ecg_findpeaks(ecg, method="neurokit")
pantompkins1985 = nk.ecg_findpeaks(ecg, method="pantompkins1985")
hamilton2002 = nk.ecg_findpeaks(ecg, method="hamilton2002")
martinez2003 = nk.ecg_findpeaks(ecg, method="martinez2003")
christov2004 = nk.ecg_findpeaks(ecg, method="christov2004")
kalidas2017 = nk.ecg_findpeaks(ecg, method="kalidas2017")


nabian2018 = nk.ecg_findpeaks(test, method="nabian2018")
gamboa2008 = nk.ecg_findpeaks(test, method="gamboa2008")
elgendi2010 = nk.ecg_findpeaks(test, method="elgendi2010")
engzeemod2012 = nk.ecg_findpeaks(test, method="engzeemod2012")
rodrigues2020 = nk.ecg_findpeaks(test, method="rodrigues2020")
promac = nk.ecg_findpeaks(test, method="promac")

ecg = test
"elgendi2010"
name
epochs = load_obj(name)
ecg = epochs[1]["ECG"]


    >>> # Visualize
nk.events_plot([neurokit["ECG_R_Peaks"],
                pantompkins1985["ECG_R_Peaks"],
                nabian2018["ECG_R_Peaks"],
                hamilton2002["ECG_R_Peaks"],
                christov2004["ECG_R_Peaks"],
                gamboa2008["ECG_R_Peaks"],
                elgendi2010["ECG_R_Peaks"],
                engzeemod2012["ECG_R_Peaks"],
                kalidas2017["ECG_R_Peaks"],
                martinez2003["ECG_R_Peaks"],
                rodrigues2020["ECG_R_Peaks"]], cleaned) #doctest: +ELLIPSIS
#endregion



#region Archive
#Try out di
import neurokit2 as nk
ecg = nk.ecg_simulate(duration=10, sampling_rate=1000)
cleaned = nk.ecg_clean(ecg, sampling_rate=1024)
signals, info = nk.ecg_peaks(cleaned, correct_artifacts= False)
artifact_location, signal_corrected = nk.signal_fixpeaks(signals, sampling_rate=1024, iterative=True, method="Kubios", show = True )
nk.events_plot(info["ECG_R_Peaks"], cleaned)  # doctest: +ELLIPSIS

#TODO Think of how to wisely compare them: how to measure success?
#TODO Compare them



neurokit = nk.ecg_findpeaks(ecg, method="neurokit")
pantompkins1985 = nk.ecg_findpeaks(ecg, method="pantompkins1985")
hamilton2002 = nk.ecg_findpeaks(ecg, method="hamilton2002")
martinez2003 = nk.ecg_findpeaks(ecg, method="martinez2003")
christov2004 = nk.ecg_findpeaks(ecg, method="christov2004")
kalidas2017 = nk.ecg_findpeaks(ecg, method="kalidas2017")

promac = []
nabian2018 = []
gamboa2008 = []
engzeemod2012 = []
elgendi2010 = []
rodrigues2020 = []

#Plot

nrow = 4; ncol = 3;
fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(15, 8))
fig.suptitle('Different Peak Detection Algorithms')

for ax, method in zip(axs.reshape(-1), methods):
  ax.plot(np.array(ecg))
  events = globals()[method]
  if len(events) != 0:
      for i in events["ECG_R_Peaks"]:
          ax.axvline(i, color="red", linestyle="--")
  ax[0, 0].set_title(method)

axs[0, 0].plot(np.array(ecg))
if len(neurokit) != 0:
    for i in neurokit["ECG_R_Peaks"]:
        axs[0, 0].axvline(i, color="red", linestyle="--")
axs[0, 0].set_title('Neurokit')

axs[0, 1].plot(np.array(ecg))
if len(pantompkins1985) != 0:
    for i in pantompkins1985["ECG_R_Peaks"]:
        axs[0, 1].axvline(i, color="red", linestyle="--")
axs[0, 1].set_title('pantompkins1985')

axs[1, 0].plot(np.array(ecg))
if len(hamilton2002) != 0:
    for i in hamilton2002["ECG_R_Peaks"]:
        axs[1, 0].axvline(i, color="red", linestyle="--")
axs[1, 0].set_title('hamilton2002')

axs[1, 1].plot(np.array(ecg))
if len(martinez2003) != 0:
    for i in martinez2003["ECG_R_Peaks"]:
        axs[1, 1].axvline(i, color="red", linestyle="--")
axs[1, 1].set_title('martinez2003')


axs[1,0].set_title('Frequencies raw ECG signal')
axs[1, 1].set_title('Frequencies filtered ECG signal')
# axs[0].set_xlabel('Samples')
axs[1,0].set_xlabel('Frequencies (Hz)')
axs[1, 1].set_xlabel('Frequencies (Hz)')


axs[0, 1].plot(ecg_filtered)
axs[1,0].plot(xf, np.abs(yf))
axs[1,1].plot(xf, np.abs(yf))

plots
plt.show()

# axs[0].set_xlabel('Samples')
axs[1, 0].set_xlabel('Frequencies (Hz)')
axs[1, 1].set_xlabel('Frequencies (Hz)')


#endregion