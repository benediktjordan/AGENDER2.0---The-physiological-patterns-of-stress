#TODO Compare the Low, High and Bandpass filter with the NK2 filter. Maybe they are better

# Progress: Completed; decided to use Neurokit NK2 filtering function to filter


# region Comparing Neurokit filtering functions
# Description: Comparison of different signal processing methods implemented in NeuroKit

#region Variables
vis_period = 10         #s
sampling_rate = 1024    #Hz
filtering_methods = ["ECG_Raw", "ECG_NeuroKit", "ECG_BioSPPy", "ECG_PanTompkins", "ECG_Hamilton", "ECG_Elgendi", "ECG_EngZeeMod"]
peak_detection_method ="kalidas2017"       #kalidas2017 is the method which has been proven to be best in script "02_PeakDetection"

#endregion

#region Functions
## Create different filterings within epochs
def nk2_filter_comparison(epochs, sampling_rate):
    for i in epochs:
        ecg = epochs[i]["ECG"]
        signals = pd.DataFrame({"ECG_Raw" : ecg,
                                "ECG_NeuroKit" : nk.ecg_clean(ecg, sampling_rate=sampling_rate, method="neurokit"),
                                "ECG_BioSPPy" : nk.ecg_clean(ecg, sampling_rate=sampling_rate, method="biosppy"),
                                "ECG_PanTompkins" : nk.ecg_clean(ecg, sampling_rate=sampling_rate, method="pantompkins1985"),
                                "ECG_Hamilton" : nk.ecg_clean(ecg, sampling_rate=sampling_rate, method="hamilton2002"),
                                "ECG_Elgendi" : nk.ecg_clean(ecg, sampling_rate=sampling_rate, method="elgendi2010"),
                                "ECG_EngZeeMod" : nk.ecg_clean(ecg, sampling_rate=sampling_rate, method="engzeemod2012")})
        epochs[i] = signals
    return epochs

## visualize these filterings parallely (INCLUDING PEAKS DETECTED)
def cleaning_visualisation(epochs, vis_period, sampling_rate):
    counter = 0
    for i in epochs:
        df = epochs[i]
        for j in range(1, df.shape[0]+1, vis_period*sampling_rate):
            begin = j-1
            end = j+vis_period*sampling_rate

            # Only select segments of length "vis_period" in epochs
            if end < df.shape[0]:
                df_cropped = []
                df_cropped = df.iloc[begin:end, :]
            else:
                break

            # Apply Peak detection methods; only run them if they are actually working
            for l in filtering_methods:
                ecg = df_cropped[l]
                try:
                    globals()[l] = nk.ecg_findpeaks(ecg, method=peak_detection_method)
                except Exception:
                    globals()[l] = []
                else:
                    globals()[l] = nk.ecg_findpeaks(ecg, method=peak_detection_method)

            # Plot
            nrow = 3;
            ncol = 3;
            fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(15, 8))
            fig.suptitle('Different Filtering Algorithms')

            for ax, method in zip(axs.reshape(-1), filtering_methods):
                ax.plot(np.array(df_cropped[method]))
                events = globals()[method]
                if len(events) != 0:
                    for p in events["ECG_R_Peaks"]:
                        ax.axvline(p, color="red", linestyle="--")
                ax.set_title(method)

            plt.show()

            # Adjust plots
            fig.tight_layout()


            # Set labels
            #axs[3,0].set_xlabel('sample')
            #axs[3,1].set_xlabel('sample')

            #Save
            starting = j/sampling_rate
            fig.savefig(
                'C:/Users/BJ/PycharmProjects/AGENDER2.0/01_DataCleaning/02_NeurokitFiltersComparison/' + "Participant_"+name[-13:-4] + "_Epoch_" + str(i) + "_StartingTime (second)_" + str(
                    starting) + ".png")
            plt.close()  # closes figure

            # Update counter
            counter=counter+1

    return print("There have been " + str(counter) + " visualizations created")
#endregion

#region Run

## Run for single epoch
participant = "AGENDER01"
name = r"epochs_600seconds_"
name = name+participant+".pkl"
#files = os.listdir(path)
#path = files[4]
epochs = load_obj(name)

epochs_filtered = nk2_filter_comparison(epochs, sampling_rate)
cleaning_visualisation(epochs_filtered, vis_period, sampling_rate)

#run for all epochs
condition = "epochs_600seconds"
path = r"C:\Users\BJ\PycharmProjects\AGENDER2.0\obj"
files = os.listdir(path)

for name in files:
    if condition not in name:
        continue
    epochs = load_obj(name)
    epochs_filtered = nk2_filter_comparison(epochs, sampling_rate)
    cleaning_visualisation(epochs_filtered, vis_period, sampling_rate)
    print("Finished participant " + str(name[-13:-4]))

#endregion

#endregion






# region Creating Visualization function
vis_period = 60         #s
sampling_rate = 1024    #Hz
def cleaning_visualisation(epochs, epochs_cleaned, vis_period, sampling_rate):
    counter = 0
    for i in range(1,len(epochs)+1,1):
        df = epochs[i]
        df_cleaned = epochs_cleaned[i]
        for j in range(1, df.shape[0]+1, vis_period*sampling_rate):
            begin = j-1
            end = j+vis_period*sampling_rate
            if df.shape[0] != df_cleaned.shape[0]:
                print("Failure: Shapes are not the same")
                break

            # Only select segments of length "vis_period" in epochs
            if end < df.shape[0]:
                df_cropped = []
                df_cleaned_cropped = []
                df_cropped = df.iloc[begin:end, :]
                df_cleaned_cropped = df_cleaned.iloc[begin:end, :]
            else:
                break

            # Plot
            fig, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

            # Plt
            axs[0].plot(df_cropped["ECG"])
            axs[1].plot(df_cleaned_cropped["ECG"])

            # Set titles
            fig.suptitle('Comparison original and cleaned ECG')
            axs[0].set_title('Original ECG')
            axs[1].set_title('Cleaned ECG')
            axs[1].set_xlabel('seconds')

            plt.close(fig)

            #Save
            starting = j/sampling_rate
            fig.savefig(
                'C:/Users/lysan/PycharmProjects/AGENDER2.0/01_DataCleaning/' + "Epoch " + str(i) + " StartingTime (second) " + str(
                    starting) + ".png",
                bbox_inches="tight", dpi=300)
            plt.close()  # closes figure

            # Update counter
            counter=counter+1

    return print("There have been " + str(counter) + "visualizations created")

#endregion

#region Frequency-Analysis: Fast Fourier transform
## Generating Frequency-overview of normal ECG-data


sampling_rate = 1024
duration = 10
ecg = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate, noise = 0, heart_rate = 100)
frequency_analysis(ecg, 40, 50, 1024)

N = sampling_rate * duration
yf = fft(ecg)
xf = fftfreq(N, 1 / sampling_rate)

plt.plot(xf, np.abs(yf))
plt.suptitle("Frequencies of normal ECG signal")
plt.xlabel('Frequency (Hz)')
plt.show()

## Check Frequencies of abnormal parts of ECG: defined segments
### Variables
ecg = epochs[1]["ECG_Raw"][-50:-30]
ecg = np.array(ecg)
ecg = normalize_mean(ecg)

### Function
def plot_frequencies_singlesegment(ecg_array, start, end, sampling_rate):
    sampling_rate = 1024
    duration = abs(end-start)

    #Apply Fast Fourier Transform
    N = sampling_rate * duration
    yf = fft(ecg)
    xf = fftfreq(N, 1 / sampling_rate)

    #Visualize initial ECG signal and Frequencies
    fig, axs = plt.subplots(2, 1, figsize=(15, 8))
    axs[0].plot(ecg)
    axs[1].plot(xf, np.abs(yf))
    axs[1].set_xlim(-100, 100)
    fig.suptitle('Frequencies of abnormal ECG signal')
    axs[0].set_title('ECG signal')
    axs[1].set_title('Frequencies')
    #axs[0].set_xlabel('Samples')
    axs[1].set_xlabel('Frequencies (Hz)')
    plt.show()

### Applying function
plot_frequencies_singlesegment(ecg, 50, 30, 1024)

## Check Frequencies of abnormal parts of ECG: epochs (divided into segments)
### Variables
vis_period = 15
sampling_rate = 1024

### Function
def plot_frequencies_epochs(epochs, vis_period, sampling_rate):
    duration = vis_period

    counter = 0
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

            # Subtract mean
            ecg = normalize_mean(df_cropped["ECG"])
            ecg = np.array(ecg)

            # Apply Fast Fourier Transform
            N = sampling_rate * duration
            yf = fft(ecg)
            xf = fftfreq(N+1, 1 / sampling_rate)

            # Visualize initial ECG signal and Frequencies
            fig, axs = plt.subplots(2, 1, figsize=(15, 8))
            axs[0].plot(ecg)
            axs[1].plot(xf, np.abs(yf))
            axs[1].set_xlim(-100, 100)
            fig.suptitle('Frequencies of  ECG signal')
            axs[0].set_title('ECG signal')
            axs[1].set_title('Frequencies')
            # axs[0].set_xlabel('Samples')
            axs[1].set_xlabel('Frequencies (Hz)')
            plt.show()

            # Adjust plots
            #plt.subplots_adjust(bottom=-0.9)
            fig.tight_layout()

            # Save
            starting = j / sampling_rate
            fig.savefig(
                'C:/Users/lysan/PycharmProjects/AGENDER2.0/01_DataCleaning/01_FrequencyAnalysis/' + "Epoch " + str(
                    i) + " StartingTime (second) " + str(
                    starting) + ".png")
            plt.close()  # closes figure

            # Update counter
            counter = counter + 1

### Run Function
plot_frequencies_epochs(epochs, vis_period, sampling_rate)


#endregion

#region HighPass Filter
## Variables
lowcut = 0.5
vis_period = 15
sampling_rate = 1024

## Functions
def highpass_epochs(epochs,lowcut, vis_period, sampling_rate):
    '''
    Apply Lowpass filter on ECG data
    :param ecg: np array or pandas series
    :return:
    '''

    counter = 0
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

            # Subtract mean from ECG signal
            ecg = normalize_mean(df_cropped["ECG"])

            # Apply Highpass Filter
            ecg_filtered = nk.signal_filter(ecg, lowcut=lowcut, method='butterworth', order=2)
            ecg = np.array(ecg)

            # Apply Fast Fourier Transform on ECG
            N = sampling_rate * vis_period
            yf = fft(ecg)
            xf = fftfreq(N + 1, 1 / sampling_rate)

            # Apply Fast Fourier Transform on ECG filtered
            yf_filtered = fft(ecg_filtered)
            xf_filtered = fftfreq(N + 1, 1 / sampling_rate)

            # Visualize initial ECG signal and Frequencies
            fig, axs = plt.subplots(2, 2, figsize=(15, 8))

            #Plot
            axs[0,0].plot(ecg)
            axs[0, 1].plot(ecg_filtered)
            axs[1,0].plot(xf, np.abs(yf))
            axs[1,1].plot(xf_filtered, np.abs(yf_filtered))

            axs[1,0].set_xlim(-100, 100)
            axs[1, 1].set_xlim(-100, 100)

            fig.suptitle('Frequencies of  raw and filtered ECG signal')
            axs[0,0].set_title('raw ECG signal')
            axs[0, 1].set_title('filtered ECG signal')
            axs[1,0].set_title('Frequencies raw ECG signal')
            axs[1, 1].set_title('Frequencies filtered ECG signal')
            # axs[0].set_xlabel('Samples')
            axs[1,0].set_xlabel('Frequencies (Hz)')
            axs[1, 1].set_xlabel('Frequencies (Hz)')
            plt.show()

            # Adjust plots
            # plt.subplots_adjust(bottom=-0.9)
            fig.tight_layout()

            # Save
            starting = j / sampling_rate
            fig.savefig(
                'C:/Users/lysan/PycharmProjects/AGENDER2.0/01_DataCleaning/03_HighpassFilter/' + "Epoch " + str(
                    i) + " StartingTime (second) " + str(
                    starting) + ".png")
            plt.close()  # closes figure

            # Update counter
            counter = counter + 1

## Run functions
highpass_epochs(epochs, cutoff_value,vis_period, sampling_rate)

#endregion

#region LowPass Filter
## Variables
highcut = 45
vis_period = 15
sampling_rate = 1024

## Functions
def lowpass_epochs(epochs,highcut, vis_period, sampling_rate):
    '''
    Apply Lowpass filter on ECG data
    :param ecg: np array or pandas series
    :return:
    '''

    counter = 0
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

            # Subtract mean from ECG signal
            ecg = normalize_mean(df_cropped["ECG"])

            # Apply Highpass Filter
            ecg_filtered = nk.signal_filter(ecg,  highcut = highcut, method='butterworth', order=2)
            ecg = np.array(ecg)

            # Apply Fast Fourier Transform on ECG
            N = sampling_rate * vis_period
            yf = fft(ecg)
            xf = fftfreq(N + 1, 1 / sampling_rate)

            # Apply Fast Fourier Transform on ECG filtered
            yf_filtered = fft(ecg_filtered)
            xf_filtered = fftfreq(N + 1, 1 / sampling_rate)

            # Visualize initial ECG signal and Frequencies
            fig, axs = plt.subplots(2, 2, figsize=(15, 8))

            #Plot
            axs[0,0].plot(ecg)
            axs[0, 1].plot(ecg_filtered)
            axs[1,0].plot(xf, np.abs(yf))
            axs[1,1].plot(xf_filtered, np.abs(yf_filtered))

            axs[1,0].set_xlim(-100, 100)
            axs[1, 1].set_xlim(-100, 100)

            fig.suptitle('Frequencies of  raw and filtered ECG signal')
            axs[0,0].set_title('raw ECG signal')
            axs[0, 1].set_title('filtered ECG signal')
            axs[1,0].set_title('Frequencies raw ECG signal')
            axs[1, 1].set_title('Frequencies filtered ECG signal')
            # axs[0].set_xlabel('Samples')
            axs[1,0].set_xlabel('Frequencies (Hz)')
            axs[1, 1].set_xlabel('Frequencies (Hz)')
            plt.show()

            # Adjust plots
            # plt.subplots_adjust(bottom=-0.9)
            fig.tight_layout()

            # Save
            starting = j / sampling_rate
            fig.savefig(
                'C:/Users/lysan/PycharmProjects/AGENDER2.0/01_DataCleaning/04_LowpassFilter/' + "Epoch " + str(
                    i) + " StartingTime (second) " + str(
                    starting) + ".png")
            plt.close()  # closes figure

            # Update counter
            counter = counter + 1

## Run functions
lowpass_epochs(epochs, lowcut, vis_period, sampling_rate)

#endregion

#region Bandpass Filter
## Variables
lowcut = 0.5
highcut = 45
vis_period = 15
sampling_rate = 1024

## Functions
def bandpass_epochs(epochs,lowcut, highcut, vis_period, sampling_rate):
    '''
    Apply Lowpass filter on ECG data
    :param ecg: np array or pandas series
    :return:
    '''

    counter = 0
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

            # Subtract mean from ECG signal
            ecg = normalize_mean(df_cropped["ECG"])

            # Apply Highpass Filter
            ecg_filtered = nk.signal_filter(ecg, lowcut=lowcut, highcut = highcut, method='butterworth', order=2)
            ecg = np.array(ecg)

            # Apply Fast Fourier Transform on ECG
            N = sampling_rate * vis_period
            yf = fft(ecg)
            xf = fftfreq(N + 1, 1 / sampling_rate)

            # Apply Fast Fourier Transform on ECG filtered
            yf_filtered = fft(ecg_filtered)
            xf_filtered = fftfreq(N + 1, 1 / sampling_rate)

            # Visualize initial ECG signal and Frequencies
            fig, axs = plt.subplots(2, 2, figsize=(15, 8))

            #Plot
            axs[0,0].plot(ecg)
            axs[0, 1].plot(ecg_filtered)
            axs[1,0].plot(xf, np.abs(yf))
            axs[1,1].plot(xf_filtered, np.abs(yf_filtered))

            axs[1,0].set_xlim(-100, 100)
            axs[1, 1].set_xlim(-100, 100)

            fig.suptitle('Frequencies of  raw and filtered ECG signal')
            axs[0,0].set_title('raw ECG signal')
            axs[0, 1].set_title('filtered ECG signal')
            axs[1,0].set_title('Frequencies raw ECG signal')
            axs[1, 1].set_title('Frequencies filtered ECG signal')
            # axs[0].set_xlabel('Samples')
            axs[1,0].set_xlabel('Frequencies (Hz)')
            axs[1, 1].set_xlabel('Frequencies (Hz)')
            plt.show()

            # Adjust plots
            # plt.subplots_adjust(bottom=-0.9)
            fig.tight_layout()

            # Save
            starting = j / sampling_rate
            fig.savefig(
                'C:/Users/lysan/PycharmProjects/AGENDER2.0/01_DataCleaning/05_Low and Highpass Filter/' + "Epoch " + str(
                    i) + " StartingTime (second) " + str(
                    starting) + ".png")
            plt.close()  # closes figure

            # Update counter
            counter = counter + 1

## Run functions
bandpass_epochs(epochs, lowcut,highcut, vis_period, sampling_rate)

#endregion



#region: Different plotting functions #
# PLotting functions
## Plot ECG signals
# nk.ecg_plot(epochs[1]["ECG"]) #INput must be df returned from ecg_process

# plt.plot(df_ecg['ECG']) #Plot ECG signal

## Plot ECG and events
# plot = nk.events_plot(events, df_ecg) #Visualize Events & Data

signals.plot()  # plots different ECG signals above each other

#endregion

#region Archive
#region Normalisation
# Note: not needed since bandpass filter will take  care of this!
def normalize_mean(ecg):
    '''
    Description: subtracts mean from ECG signal

    :param ecg: Pandas Series containing ECG signal

    :return: ecg: Pandas Series containing ECG signal
    '''

    mean = np.mean(ecg)
    ecg = np.array(ecg)
    # "normalize" (subtract mean from ECG signal)
    for i in range(len(ecg)):
        value = ecg[i]
        l = value - mean
        ecg[i] = l
    ecg = pd.Series(ecg)
    return ecg

#endregion

#Creating toy events and epochs
events = create_events(df_label, df_ecg, frequency)
events["onset"]  = [350000,600000]
# Creating Epochs
epochs = nk.epochs_create(df_ecg, events, sampling_rate=1024, epochs_start=-300, epochs_end=300) #start and end times are in seconds
epochs_cleaned = epochs
#endregion