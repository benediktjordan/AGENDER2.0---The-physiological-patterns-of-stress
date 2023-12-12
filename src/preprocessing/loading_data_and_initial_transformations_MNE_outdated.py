#region load label data & initial transformations

# Function: loading label data
def load_label(path):
    os.chdir(path + "/Rohdaten/Report")
    file_label = glob.glob("*.csv")
    df = pd.read_csv(file_label[0], sep=";")
    return df

# Function: Create stress column
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

#Function: Transform label
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

#region load ECG data & initial transformations

# load bio data (return = data_ecg, data_hrvrmssd, data_hrvisvalid, starting_time)
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

# Function: Create Unix Microseconds
def unix_time_microseconds(data):
    # Input: data: string containing datetime in this format: '2015-03-16T10:58:51.437'
    # Output: unix timestamp in Microseconds
    time_pandas = pd.Timestamp(data)
    epoch = dt.datetime.utcfromtimestamp(0)
    return (time_pandas - epoch).total_seconds() * 1000000

# Function: Inserting ECG into MNE raw object
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


#endregion

#region Joining label & ECG data (resulting in annotated data)
# Create time-difference column (between time of sampling in ES and starting time of ECG measurement)
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

# Create MNE ANnotate object

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

#endregion (resulting in anno

#region Pipeline
# Pipeline
path = [
    r"C:\\Users\\lysan\\PycharmProjects\\AGENDER2.0\\AGENDER12"]
path = path[0]

#Load and transform label data
df = load_label(path)
df = transform_label(df)

# Load ECG data
data_ecg, data_hrvrmssd, data_hrvisvalid, starting_time = load_biodata(path)

# Convert ECG data into MNE raw dataformat

raw_ecg = create_MNE_raw(data_ecg, starting_time)

#Combine Label & ECG data and annotate raw object
df= create_timedifference(df, raw_ecg)
raw_ecg = annotate(df, raw_ecg)

# Creating Epochs
events_from_annot, event_dict = mne.events_from_annotations(raw_ecg)

#endregion