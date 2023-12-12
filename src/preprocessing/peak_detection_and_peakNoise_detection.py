#TODO If time: transfer this noise-detection algorithm from Matlab to Python: https://github.com/ufopcsilab/qrs-better-heartbeat-segmentation
# (its a Nature paper of 2020)

#TODO then: Impelement my decision (if I use short periods: a) think about: in which format do I need the peaks for the feature creation & further analysis
# b) somehow merge the peaks of the short periods so I get reasonable number of files to upload for website; c) adapt visualisation function

#Description: this script focusses on noise detection on RR-peak level
# It compares two noise/artifact-detection algorithms: the NK2 standard one and the one implemented on this website:

#region For main pipeline

#region Creating peaks for all probands (returns a dict with probands -> epochs -> peaks structure)
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

def create_peaks(files, folders, condition, method_peakdetection,proband_withoutevents):
    """
    This function creates peaks with the chosen peak detection algorithm. Peaks are created ON THE LEVEL OF EPOCHS!
    This is a difference to the FIRST (a) peak_detection_algorithm_comparison where peaks have been detected ON THE LEVEL
    OF 15 seconds visualization periods
    :param files:
    :param folders:
    :param condition:
    :param method_peakdetection:
    :param proband_withoutevents:
    :return:
    """
    for name in files:
        #load the epochs objects
        if condition not in name:
            continue
        epochs = load_obj(name)

        # Detect peaks
        proband = name[-13:-4]
        for i in epochs:
            ecg = np.array(epochs[i]["ECG"])
            peaks = nk.ecg_findpeaks(ecg, method=method_peakdetection)
            globals()["epoch_"+str(i)] = peaks["ECG_R_Peaks"]

        globals()[proband] = {}

        for i in epochs:
            globals()[proband][i] = {}
            globals()[proband][i]["Peaks"] = globals()["epoch_"+str(i)]
            globals()[proband][i]["Stress label"] = epochs[i]["Condition"].iloc[0]
            globals()[proband][i]["ECG length"] = epochs[i].shape[0]


    #Create dict which contains all peaks from participants
    peaks = {}
    for proband in folders:
        if proband in proband_withoutevents:
            continue
        peaks[proband] = globals()[proband]

    return peaks

#peaks = create_peaks(files, folders, condition, method_peakdetection,proband_withoutevents)
#endregion

#region imports the processed peak epochs & includes them into dict
location = 'C:\\Users\\BJ\\PycharmProjects\\AGENDER2.0\\obj\\peaks\\corrected'
def correct_peaks_brown(peaks, location):
    """ In this function, the .txt files, which have been downloaded from the website, are imported and treated:
    the format of the Brown peaks is reconverted from seconds to samples

    :param peaks_all: the dict which contains the peaks sorted by proband -> epochs
    :param location: location of the .txt files which contain the results from the processing by http://www.saipai-hrv.com/
    :return: dict which contains the corrected peaks sorted by proband -> epochs
    """

    os.chdir(location)
    files = os.listdir(location)
    peaks_corrected_brown = {}
    for proband in peaks:
        epoch_peaks = peaks[proband]
        globals()[proband] = {}
        for epoch in epoch_peaks:
            filename = "ppa_peaks_" + proband + "_epoch_" + str(epoch) + ".txt"
            if filename not in files:
                continue
            with open(filename) as f:
                location = f.readlines()
            with open(filename) as f:
                type = f.readlines()
            for i in range(len(location)):
                location[i] = float(location[i][0:7])   #extracts the location of the corrected peak
                if len(type[i]) != 0:
                    type[i] = float(type[i][-3:])            #extracts the type/label of the artifact
            location = np.array(location)
            location = (location * 1024).round()      #multiply by 1024 to transform from seconds to #samples & round

            df_artifacts = pd.DataFrame({"location": location, "type": type})  # create df from two lists
            df_artifacts["type"] = df_artifacts["type"].replace([1,2,16,31,32], [1,2,3,4,5]) #replaces the initial legend values by
            # natural number which are better for plotting
            globals()[proband][epoch] = {}
            globals()[proband][epoch]["peaks_corr"] = location
            globals()[proband][epoch]["artifacts"] = df_artifacts
            globals()[proband][epoch]["ECG length"] = copy.deepcopy(peaks[proband][epoch]["ECG length"])
            globals()[proband][epoch]["Stress label"] = copy.deepcopy(peaks[proband][epoch]["Stress label"])
        peaks_corrected_brown[proband] = globals()[proband]

    return peaks_corrected_brown

#peaks_corrected_brown = correct_peaks_brown(peaks, location)

#endregion

#region cut-off first minute of every epoch
path = r"C:\Users\BJ\PycharmProjects\AGENDER2.0\obj"
files = os.listdir(path)
condition = "epochs_660seconds_filtered"
number_of_minutes = 1
#epochs_cutoff(files,condition, number_of_minutes)

def epochs_cutoff(files, condition, number_of_minutes):
    for name in files:
        # load the epochs objects
        if condition not in name:
            continue
        proband = name[-13:-4]
        epochs = load_obj(name)
        epochs_new = {}
        for i in epochs:
            epoch = epochs[i]
            epochs_new[i] = epoch.iloc[1024*60*number_of_minutes:]        #selects everything except first minute
        save_obj(epochs_new, "epochs_600seconds_filtered_" + proband)
#endregion

#region cut-off first minute of every peak
def peaks_cutoff(peaks, number_of_minutes):
    peaks_new = copy.deepcopy(peaks)
    for l in peaks_new:
        proband = peaks_new[l]
        for i in proband:
            peaks_corr = proband[i]["peaks_corr"]
            peaks_corr = peaks_corr-(1024*60*number_of_minutes)      #substract #minutes from peaks
            peaks_corr = peaks_corr[peaks_corr >= 0]
            peaks_new[l][i]["peaks_corr"] = peaks_corr

            artifacts = proband[i]["artifacts"]
            artifacts["location"] = artifacts["location"] -(1024*60*number_of_minutes)      #substract #minutes from artifacts
            artifacts = artifacts[artifacts["location"] >= 0]
            peaks_new[l][i]["artifacts"] = artifacts

            ECG_length = proband[i]["ECG length"]
            ECG_length = ECG_length-1024*60*number_of_minutes
            peaks_new[l][i]["ECG length"] = ECG_length
    return peaks_new
#endregion

#region Check how much artifacts in every epoch and exclude if above 10%

##TODO Calculate percentage of corrected
def noise_percentage_peakcorrection(peaks):
    probands = []
    epochs = []
    numbers_peaks = []
    numbers_artifacts = []
    percentages = []

    for proband in peaks:
        for epoch in peaks[proband]:
            number_peaks = len(peaks[proband][epoch]["peaks_corr"])
            number_artifacts = len(
                peaks[proband][epoch]["artifacts"]["type"][peaks[proband][epoch]["artifacts"]["type"] != 0])
            percentage = number_artifacts / number_peaks
            probands.append(proband)
            epochs.append(epoch)
            numbers_peaks.append(number_peaks)
            numbers_artifacts.append(number_artifacts)
            percentages.append(percentage)

    df = pd.DataFrame(
        {"Proband": probands,
         "Epoch": epochs,
         "Number of Peaks": numbers_peaks,
         "Number of Artifacts": numbers_artifacts,
         "Percentage of Artifacts": percentages}
    )
    return df

#df_perc_corr = noise_percentage_peakcorrection(peaks_corrected)

##TODO Exclude all epochs whcih are above 10%




#endregion

#endregion


#region Correct peaks with NK2 method (returns a dict with probands -> epochs -> peaks_corrected & artifact locations structure)
sampling_rate = 1024
method_peakcorrection ="Kubios"
def correct_peaks_NK(peaks, sampling_rate, method_peakcorrection):
    peaks_corrected_NK = {}
    for proband in peaks:
        epoch_peaks = peaks[proband]
        globals()[proband] = {}
        for epoch in epoch_peaks:
            globals()[proband][epoch] = {}
            #compute corrected peaks and locations of artifacts (corr)
            peaks_temp = epoch_peaks[epoch]
            corr, peaks_clean = nk.signal_fixpeaks(peaks_temp, sampling_rate, iterative=True, method=method_peakcorrection)
            globals()[proband][epoch]["peaks_corr"] = peaks_clean

            #compute df which includes location and type of artifacts
            loc = []
            type = []
            for index, i in enumerate(corr):
                for l in corr[i]:
                    loc.append(peaks_clean[l])
                    type.append(index+1)        #here the following "legend" is used:
                    # 1 = ectopic
                    # 2 = missed
                    # 3 = extra
                    # 4 = longshort
            df = pd.DataFrame({"location": loc, "type": type}) #create df from two lists
            df = df.sort_values(by=['location'])
            globals()[proband][epoch]["artifacts"] = df

        peaks_corrected_NK[proband] = globals()[proband]
    return peaks_corrected_NK

peaks_corrected_NK = correct_peaks_NK(peaks, sampling_rate, method_peakcorrection)
#endregion



#region NOT USED ANYMORE plot peaks & NK2 corrected peaks jointly (create class to try out that)
#Description: I created this section when comparing the peaks detected for vis_period and for whole epochs (since
# I realized that they don´t quite match); since this problem is solved, this region isn´t necessary anymore)
class PeakCorrection:
    def __init__(self, a, b):
        self.a = a
        self.b = b


#peaks_all = load_obj("peaks_original_and_corrected.pkl")

vis_period = 15
sampling_rate = 1024
condition = "epochs_660seconds_filtered"
storage_path = "C:/Users/BJ/PycharmProjects/AGENDER2.0/01_DataCleaning/Peaks/PeakCorrection_NK2/"
nrow = 2
ncol = 1

peaks_and_NKcorrection = {}
peaks_and_NKcorrection["peaks"] = peaks
peaks_and_NKcorrection["peaks_corrected_NK"] = peaks_corrected_NK
save_obj(peaks_and_NKcorrection, "peaks_original_and_NK2")
#peaks_and_NKcorrection = load_obj("peaks_original_and_NK2.pkl")

def peak_vis(condition, peaks_collection, vis_period, sampling_rate, storage_path, nrow, ncol):
    """

    :param condition:
    :param peaks_collection:
    :param vis_period:
    :param sampling_rate:
    :param storage_path:
    :param nrow: number of rows of the visualization
    :param ncol: number of columns of the visualization
    :return:
    """
    counter = 0

    #iterate through different probands
    path = r"C:\Users\BJ\PycharmProjects\AGENDER2.0\obj"
    files = os.listdir(path)
    for name in files:
        if condition not in name:
            continue

        #load the epochs for the proband
        proband = name[-13:-4]
        epochs = load_obj(name)

        #iterate through epochs & visualize
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
                    break #exits the for loop and continues with next epoch

                ecg = df_cropped["ECG"]

               # Plot
                nrow = nrow;
                ncol = ncol;
                fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(15, 8))
                fig.suptitle('Different Peak Noise Correction Algorithms')

                for ax, method in zip(axs.reshape(-1), peaks_collection):
                    ax.plot(np.array(ecg))
                    events = []
                    # Check if there is actually something in events. If not, just continue
                    try:
                        events = peaks_collection[method][proband][epoch]
                    except Exception:
                        continue
                    else:
                        events = np.array(peaks_collection[method][proband][epoch].copy())      #convert into nd.array
                        # since for rodrigues2020 the peaks are stored in a list
                        if len(events) != 0:
                            #next three lines filter out all events which are not relevant for this vis_period
                            events = events - begin
                            events = events[events >= 0]
                            events = events[events <= (end-begin)]
                            for p in events:
                                ax.axvline(p, color="red", linestyle="--")
                    ax.set_title(method)

                plt.show()

                # Adjust plots
                fig.tight_layout()

                # Save
                starting = j / sampling_rate
                fig.savefig(storage_path + "Participant_" +
                            str(proband) + "_Epoch_" + str(epoch) + "_StartingTime (second)_" + str(starting) + ".png")
                plt.close()  # closes figure

                # Update counter
                counter = counter + 1
        print("Proband " + str(proband) + " is now done")

peak_vis(condition, peaks_and_NKcorrection, vis_period, sampling_rate, storage_path, nrow, ncol)

#endregion

#region Correct peaks with "website method"  (returns a dict with probands -> epochs -> peaks_corrected structure)

#region Save all peak_epochs as .txt
location = 'C:\\Users\\BJ\\PycharmProjects\\AGENDER2.0\\obj\\peaks'

def save_peaks_txt(peaks, location):
    """
    This function saves the peaks contained in "peaks" (structure: proband -> epochs) into the location as .txt files
    Usage: these .txt files can be used for the peak-artifact-correction of this website: http://www.saipai-hrv.com/

    :param peaks_all: the dict containing the peaks sorted by proband -> epochs
    :param location: location where
    :return: returns nothing
    """
    os.chdir(location)

    for proband in peaks:
        epoch_peaks = peaks[proband]
        #globals()[proband] = {}
        for epoch in epoch_peaks:
            pks = epoch_peaks[epoch]
            pks = pks / 1024      #peaks need to be identified by the second (prerequisite of website)
            #save peaks
            np.savetxt("peaks_" + str(proband) + "_epoch_"+ str(epoch) + ".txt", pks, delimiter=',')

save_peaks_txt(peaks, location)
#endregion


#region process the peaks through the website
# Automatizing the up- and downloading of the files
## Using selenium
### Note: this code loads the webpage saipai-hrv.com and iputs the .txt file, but I still have to click the "Submit" button manually
### (tried to solve this problem but the server prevents it)
path = r"C:\Users\BJ\PycharmProjects\AGENDER2.0\obj\peaks"
def upload_peaks(path):
    """
    This function opens the website saipai-hrv.com and inputs the files in the folder "path" into the "upload" field sequentially
    :param path: path for the .txt files containing the peaks of the different epochs
    :return:
    """
    files = os. listdir(path)
    for file in files:
        driver = webdriver.Chrome(executable_path=r"C:\Users\BJ\PycharmProjects\Driver\chromedriver.exe")
        driver.get("http://www.saipai-hrv.com/")
        driver.maximize_window()
        driver.find_element_by_id("subarrhythmia").send_keys("C:\\Users\\BJ\\PycharmProjects\\AGENDER2.0\\obj\\peaks\\"+file)
        time.sleep(8)      #10 seconds delay so I can click on submit & wait for the processing of the file

upload_peaks(path)

#region try to click on submit
#click on submit (didnt work for some sever protection reason
driver.find_element_by_css_selector('.submit-button').click()
time.sleep(3)
#driver.find_element_by_css_selector('.submit-button').click()
button = driver.find_element_by_class_name(u"submit-button")
time.sleep(3)
ActionChains(driver).move_to_element(button).click(button).perform()

#endregion


#endregion



#endregion



#region add all peaks into one dict called "peaks_all" & saves them
peaks_all = {}
peaks_all["peaks"] = peaks
peaks_all["peaks_corrected_NK"] = peaks_corrected_NK
peaks_all["peaks_corrected_brown"] = peaks_corrected_brown

save_obj(peaks_all, "peaks_original_NK2_brown")
#endregion

#region Visualize comparatively the peaks returned from the three methods
peaks_all = load_obj("peaks_original_NK2_brown.pkl")
vis_period = 15
sampling_rate = 1024
condition = "epochs_660seconds_filtered"

def peak_noise_comparison_vis(condition, peaks_all, vis_period, sampling_rate):
    counter = 0

    #iterate through different probands
    path = r"C:\Users\BJ\PycharmProjects\AGENDER2.0\obj"
    files = os.listdir(path)
    for name in files:
        if condition not in name:
            continue

        #load the epochs for the proband
        proband = name[-13:-4]
        epochs = load_obj(name)

        #iterate through epochs & visualize
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
                    break #exits the for loop and continues with next epoch

                ecg = df_cropped["ECG"]

               # Plot
                nrow = 5;
                ncol = 1;
                fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(15, 8))
                fig.suptitle('Different Peak Noise Correction Algorithms')

                #this loop creates the ECG and peak visualization for peaks, NK2 corrected and Brown corrected
                axes = [0,1,3]
                for i, method in zip(axes, peaks_all):
                    ax = axs[i]
                    ax.plot(np.array(ecg))
                    ax.set_xlim((0, 16000))
                    if method == "peaks_corrected_NK" or method == "peaks_corrected_brown":   #treat these methods different since their dict structure is
                        #different (one layer deeper since also the artifacts are stored)
                        events = []
                        events = peaks_all[method][proband][epoch]["peaks_corr"].copy()
                        # next three lines filter out all events which are not relevant for this vis_period
                        events = events - begin
                        events = events[events >= 0]
                        events = events[events <= (end - begin)]
                        for p in events:
                            ax.axvline(p, color="red", linestyle="--")
                        ax.set_title(method)

                    else:
                        events = []
                        events = peaks_all[method][proband][epoch].copy()
                        #next three lines filter out all events which are not relevant for this vis_period
                        events = events - begin
                        events = events[events >= 0]
                        events = events[events <= (end-begin)]
                        for p in events:
                            ax.axvline(p, color="red", linestyle="--")
                        ax.set_title(method)


                #with the following lines NK2 artifact visualization is created
                artifacts_NK = []
                artifacts_NK = peaks_all["peaks_corrected_NK"][proband][epoch]["artifacts"].copy()
                # next three lines filter out all artifacts which are not relevant for this vis_period
                artifacts_NK["location"]= artifacts_NK["location"] - begin
                artifacts_NK = artifacts_NK.drop(artifacts_NK[artifacts_NK.location < 0].index)
                artifacts_NK = artifacts_NK.drop(artifacts_NK[artifacts_NK.location > (end - begin)].index)
                scatter1 = axs[2].scatter(artifacts_NK["location"],
                            artifacts_NK["type"])
                axs[2].set_title("Artifacts NK2")
                axs[2].set_xlim((0,16000))
                axs[2].set_ylim((0.5, 4.5))
                #classes = ["ectopic", "missed", "extra", "longshort"]
                #axs[2].legend([1,2,3,4], classes)

                # with the following lines Brown artifact visualization is created
                artifacts_brown = peaks_all["peaks_corrected_brown"][proband][epoch]["artifacts"].copy()
                artifacts_brown = artifacts_brown.drop(artifacts_brown[artifacts_brown.type == 0].index) #all 0 (where no change was made) are dropped
                # next three lines filter out all artifacts which are not relevant for this vis_period
                artifacts_brown["location"] = artifacts_brown["location"] - begin
                artifacts_brown = artifacts_brown.drop(artifacts_brown[artifacts_brown.location < 0].index)
                artifacts_brown = artifacts_brown.drop(artifacts_brown[artifacts_brown.location > (end - begin)].index)
                scatter2 = axs[4].scatter(artifacts_brown["location"],
                                          artifacts_brown["type"])
                axs[4].set_title("Artifacts Brown")
                axs[4].set_xlim((0, 16000))
                axs[4].set_ylim((0.5, 5.5))
                # classes = ["ectopic", "missed", "extra", "longshort"]
                # axs[2].legend([1,2,3,4], classes)

                plt.xlim(0, 16000)
                plt.show()

                # Adjust plots
                fig.tight_layout()

                # Save
                starting = j / sampling_rate
                if len(artifacts_NK["location"]) != 0 or len(artifacts_brown["location"] != 0):
                    fig.savefig(
                        "C:/Users/BJ/PycharmProjects/AGENDER2.0/01_DataCleaning/Peaks/PeakCorrection_NK2&Brown2/QualityComparison_noise cases/Participant_" +
                        str(proband) + "_Epoch_" + str(epoch) + "_StartingTime (second)_" + str(starting) + ".png")
                else:
                    fig.savefig("C:/Users/BJ/PycharmProjects/AGENDER2.0/01_DataCleaning/Peaks/PeakCorrection_NK2&Brown2/Participant_" +
                                str(proband) + "_Epoch_" + str(epoch) + "_StartingTime (second)_" + str(starting) + ".png")
                plt.close()  # closes figure

                # Update counter
                counter = counter + 1
        print("Proband " + str(proband) + " is now done")

peak_noise_comparison_vis(condition, peaks_all, vis_period, sampling_rate)
#endregion






#region Finding errors
#Problem: there are quite different results between the "peak detection comparison" and the "peak noise detection" visualisations.
# This region tries to spot the reason for these errors

# Approach: in a) peaks are computed for whole epoch and for small vis_period segments and both are visualized in one plot (for small segments)
# in b) the peaks for whole epoch (AGENDER02->epoch 01) are computed & visualized. Than, manually, I compared them with
# the small segments computed in a) (reason: double check if I did something wrong in a)

# Solution: I think the outcome here was that the peak detection for the short periods (15 seconds) works much better
# than for the whole epochs (but I am not sure if thats correct)

#a) Visualising peaks calculated for each period AND for whole epoch together to see if there are differences

#Analysing how much of stress-data is usable
path = r"C:\Users\BJ\PycharmProjects\AGENDER2.0\obj"
files = os.listdir(path)
condition = "epochs_600seconds_filtered"

epochs = load_obj('epochs_600seconds_filtered_AGENDER14.pkl')
epoch = 1
df = epochs[epoch]
condition = df["Condition"].iloc[0]

# Creating peaks for whole epoch
ecg = df["ECG"]

# Plot
plt.plot(np.array(ecg))
plt.show()

# Adjust plots
plt.tight_layout()
#endregion




vis_period = 15
sampling_rate = 1024
condition = "epochs_600seconds_filtered"
def peak_noise_comparison_vis(condition, vis_period, sampling_rate):
    counter = 0

    #iterate through different probands
    path = r"C:\Users\BJ\PycharmProjects\AGENDER2.0\obj"
    files = os.listdir(path)
    for name in files:
        if condition not in name:
            continue

        #load the epochs for the proband
        proband = name[-13:-4]
        epochs = load_obj(name)

        #iterate through epochs & visualize
        for epoch in epochs:
            df = epochs[epoch]

            #Creating peaks for whole epoch
            ecg_epoch = df["ECG"]
            events_epoch = nk.ecg_findpeaks(ecg_epoch, method="kalidas2017")

            #Iterating through different short time periods
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
                fig.suptitle('Different Peak Noise Correction Algorithms')

                #Plotting for peaks calculated for small segments
                axs[0].plot(np.array(ecg))
                events = nk.ecg_findpeaks(ecg, method="kalidas2017")
                for p in events["ECG_R_Peaks"]:
                    axs[0].axvline(p, color="red", linestyle="--")
                axs[0].set_title("Peaks_calculated in short period")

                #Plotting for peaks calculated for whole epohc
                axs[1].plot(np.array(ecg))
                # next three lines filter out all events which are not relevant for this vis_period
                events_epoch_thissegment = events_epoch.copy()
                events_epoch_thissegment = events_epoch_thissegment["ECG_R_Peaks"]
                events_epoch_thissegment = events_epoch_thissegment - begin
                events_epoch_thissegment = events_epoch_thissegment[events_epoch_thissegment >= 0]
                events_epoch_thissegment = events_epoch_thissegment[events_epoch_thissegment <= (end - begin)]
                for p in events_epoch_thissegment:
                    axs[1].axvline(p, color="red", linestyle="--")
                axs[1].set_title("Peaks_calculated for whole epoch")

                plt.show()

                # Adjust plots
                fig.tight_layout()

                # Save
                starting = j / sampling_rate
                fig.savefig("C:/Users/BJ/PycharmProjects/AGENDER2.0/01_DataCleaning/Peaks/Troubleshooting/PeakDetection_kalidas2017_comparing for whole epoch & single segments/Participant_" +
                            str(proband) + "_Epoch_" + str(epoch) + "_StartingTime (second)_" + str(starting) + ".png")
                plt.close()  # closes figure

                # Update counter
                counter = counter + 1
        print("Proband " + str(proband) + " is now done")

peak_noise_comparison_vis(condition, vis_period, sampling_rate)


#b) visualizing whole epoch (AGENDER02->epoch01)

name = "epochs_600seconds_filtered_AGENDER02.pkl"
epoch = 1

proband = name[-13:-4]
epochs = load_obj(name)

df = epochs[epoch]

# Creating peaks for whole epoch
ecg = df["ECG"]
events = nk.ecg_findpeaks(ecg, method="kalidas2017")

# Plot
plt.plot(np.array(ecg))
for p in events["ECG_R_Peaks"]:
    plt.axvline(p, color="red", linestyle="--")
plt.show()

# Adjust plots
plt.tight_layout()
#endregion










#Test: Does peaks defined in this way work with website-based artifact correction?
## Result: it works with this setup!
peaks_test = peaks_all["AGENDER02"][1]
peaks_test = peaks_test/1024
os.chdir('C:\\Users\\BJ\\PycharmProjects\\AGENDER2.0\\obj\\peaks')
np.savetxt('peaks_test.txt', peaks_test, delimiter=',')

# Test2: how does NK2 implemented artifact correction work?
peaks_test2 = peaks_all["AGENDER02"][1]     #note: needs to contain the index-places of peaks

artifacts, peaks_clean = nk.signal_fixpeaks(peaks_test2, sampling_rate, iterative=True, show = True, method = "Kubios")

#TODO Organize the peaks in dictionaries which are structured in the same way as the epochs dictionaries are
#endregion

#region Comparison of noise/artifact correction algorithms at RR-peak level
#region Variables
methods = ["neurokit", "pantompkins1985", "nabian2018", "hamilton2002", "martinez2003", "christov2004",
           "gamboa2008", "elgendi2010", "engzeemod2012","kalidas2017", "rodrigues2020", "promac" ]
vis_period = 15
sampling_rate = 1024
participant = "AGENDER05"
#endregion

#region Function
def RR_noisecorrection_comparison(participant, epochs, methods, vis_period, sampling_rate):
    counter = 0

    # Iterate through different epochs & then iterate through different segments within the epochs
    for i in epochs:
        df = epochs[i]
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
            fig.savefig(
                'C:/Users/lysan/PycharmProjects/AGENDER2.0/01_DataCleaning/PeakDetection/' "Participant " +
                str(participant) + " Epoch " + str(i) + " StartingTime (second) " + str(starting) + ".png")
            plt.close()  # closes figure

            # Update counter
            counter = counter + 1

#endregion

#region Run Function
condition = "epochs_segmented_600seconds"
path = r"C:\Users\BJ\PycharmProjects\AGENDER2.0\obj"
files = os.listdir(path)
path = files[4]
epochs = load_obj(path)

peakdetection_comparison(participant, epochs, methods, vis_period, sampling_rate)

#endregion



## Run functions (archive?)
lowpass_epochs(epochs, lowcut, vis_period, sampling_rate)

#endregion

#region Archive
#region Understand artifact correction algorith,
import neurokit2 as nk
ecg = nk.ecg_simulate(duration=10, sampling_rate=1000)
cleaned = nk.ecg_clean(ecg, sampling_rate=1000)
signals, info = nk.ecg_peaks(cleaned, correct_artifacts=False)
nk.events_plot(info["ECG_R_Peaks"], cleaned)
#endregion


#region Testing how to upload peak data to http://www.saipai-hrv.com/: DIDNT WORK!
# PROBLEM WITH THIS APPROACH: I sepnt 4 hours to develop this code, but it doesnt work BECAUSE somehow
# the peak detection algorithm doesnt detect peaks if I through all ECG data after each other. Therefore,
# the next approach is to detect peaks for each epoch & then merge these peak-lists together
# Builddata: load a single epochs element
epochs = load_obj("epochs_600seconds_AGENDER08.pkl")
data = epochs[1]
ecg = data["ECG"]

# Create one array with all ECG and all label data
path = r"C:\Users\BJ\PycharmProjects\AGENDER2.0\obj"
files = os.listdir(path)
condition = "epochs_600seconds_filtered"
x_all, y_all = create_array_allepochs(files,condition)
x_all_array = np.array(x_all)

# Create peaks of this ECG array
peaks = nk.ecg_findpeaks(x_all_array, method="kalidas2017")
peaks = peaks["ECG_R_Peaks"]
test = peaks/1024                                               #create peaks in "seconds" format

# Save the peaks
os.chdir('C:\\Users\\BJ\\PycharmProjects\\AGENDER2.0\\obj\\peaks')
np.savetxt('peaks.txt', peaks, delimiter=',')

#endregion: ,


#endregion


