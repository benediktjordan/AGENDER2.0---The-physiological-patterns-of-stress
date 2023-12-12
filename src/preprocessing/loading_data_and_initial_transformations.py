#region Import
import unisens
import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
plt.rcParams['figure.figsize'] = [15, 5]  # Bigger images
from sklearn.metrics import confusion_matrix

import time
import datetime as dt
import xml.etree.ElementTree as ET
import pandas as pd
import pytz
import glob, os
import numpy as np
#!pip install mne
import mne

from scipy.fft import fft, fftfreq

from sklearn.preprocessing import OneHotEncoder
#import tensorflow as tf
#import keras
import pickle
from collections import Counter
import random
from sklearn.metrics import confusion_matrix

import selenium
from selenium import webdriver
import copy
import random

#Plotting
from matplotlib import pyplot
import seaborn as sns
import matplotlib.cm as cm
import matplotlib as mpl

#Classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

#MLP
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

#SVM
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

#Ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

#nested CV
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#Feature Importance
import shap

#for every proband a model
import itertools

#Statistical Testing
from scipy.stats import binom_test
from sklearn.model_selection import permutation_test_score

#endregion

#region General Functions
# Save & Load Data in obj folder
def save_obj(obj, name ):
    os.chdir('C:\\Users\\Ben Ali Kokou\\PycharmProjects\\AGENDER2.0')
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    os.chdir('C:\\Users\\Ben Ali Kokou\\PycharmProjects\\AGENDER2.0')
    with open('obj/' + name, 'rb') as f:
        return pickle.load(f)
#endregion

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
    #stress_intensities = []
    for idx, entry in df["Stress_Stunde"].iteritems():
        starttime = df["Form_start"][idx]
        starttime = dt.datetime.strptime(starttime, '%Y-%m-%dT%H:%M:%S')

        if entry == 2:
            stress_event = starttime - dt.timedelta(minutes = 60)

            stress_duration = 600 #second (an arbitrary duration)

            #stress_intensity = 0 #since they are not stressed at all

        elif entry == 1:
            if df["Wann_Stress"][idx] == 1:
                stress_event = starttime  - dt.timedelta(minutes=5)                          #reason why I take here -5:
                # the stress period here should span from 16 minutes until -5 minutes; since can´t insert individual
                # periods in the create_epochs function I have to adapt the stress timepoint here
            elif df["Wann_Stress"][idx] == 2:
                stress_event = starttime - dt.timedelta(minutes=15)
            elif df["Wann_Stress"][idx] == 3:
                stress_event = starttime - dt.timedelta(minutes=25)
            elif df["Wann_Stress"][idx] == 4:
                stress_event = starttime - dt.timedelta(minutes=35)
            elif df["Wann_Stress"][idx] == 5:
                stress_event = starttime - dt.timedelta(minutes=45)
            elif df["Wann_Stress"][idx] == 6:
                stress_event = starttime - dt.timedelta(minutes=55)         #reason why I enter here -55 minutes:
                # stress period should span around 55 minutes
            else:
                stress_event = np.nan

            stress_duration = 600 #second (an arbitrary duration)


        elif pd.isnull(entry):
            stress_event = np.nan

            stress_duration = np.nan

            #stress_intensity = np.nan

        else:
            stress_event = np.nan

            stress_duration = np.nan

            #stress_intensity = np.nan

        stress_events.append(stress_event)
        stress_durations.append(stress_duration)
        #stress_intensities.append(stress_intensity)

    df["Stress_Event"] = stress_events
    df["Stress_Duration"] = stress_durations

    return df

#Function: Transform label
def transform_label(df, stress_threshold):
    # select only interesting variables
    df = df[['Form', 'Form_start_date', 'Form_start_time', 'Form_finish_date', 'Form_finish_time',
             'Missing', 'Stress_Stunde', 'Wann_Stress', 'Wie_Stress', 'Wie_momentan', 'Confounder_1',
             'Confounder_2', 'Confounder_3', 'Confounder_4', 'Confounder_5', 'Confounder_wann']]

    # Delete rows which contain "Abend" or "Schlaf" in column "Form"
    df = df[df['Form'] == "Tag"]

    # Merge columns which contain date and time into datetime column
    df["Form_start"] = df["Form_start_date"] + "T" + df["Form_start_time"]
    df["Form_finish"] = df["Form_finish_date"] + "T" + df["Form_finish_time"]

    #ARCHIVE Create Form_start column in unix total seconds (necessary to choose no_stress_events later on
    #Form_start_unixtotalseconds = []
    #for event in df["Form_start"]:
    #    event_converted = dt.datetime.strptime(event, '%Y-%m-%dT%H:%M:%S')
    #    unix_time = time.mktime(event_converted.timetuple())
    #    Form_start_unixtotalseconds.append(unix_time)
    #df["Form_start_unixtotalseconds"] = Form_start_unixtotalseconds

    ## Create Stress intensity in categorical form
    ## Intensity-Score:
    ## 1 if 0  <= x <= 33
    ## 2 if 34 <= x <= 66
    ## 3 if 67 <= x <= 100
    ## np.nan if else

    #df["Wie_Stress_categorical"] = [
    #    1 if x <= 33 else 2 if x >= 34 and x <= 66 else 3 if x >= 67 and x <= 100 else np.nan \
    #    for x in df["Wie_Stress"]]

    ## Create Stress intensity in binary form
    ## Intensity-Score:
    ## 1 if 0  <= x <= 100
    ## np.nan if else

    df["Stress (binary)"] = [
        1 if x >= stress_threshold and x <= 100 else 0 \
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

    acc = u.acc_bin
    data_acc= acc.get_data()
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

    return data_ecg, data_acc, data_hrvrmssd, data_hrvisvalid, starting_time

# Function: Create Unix Microseconds
def unix_time_microseconds(data):
    # Input: data: string containing datetime in this format: '2015-03-16T10:58:51.437'
    # Output: unix timestamp in Microseconds
    time_pandas = pd.Timestamp(data)
    epoch = dt.datetime.utcfromtimestamp(0)
    return (time_pandas - epoch).total_seconds() * 1000000

# Function: Converting biodata into df
def create_ecg_df(data_ecg):
    return pd.DataFrame(data=data_ecg.transpose(), columns=["ECG"])

#endregion

#region Joining label & ECG data (resulting in events)
# Create time-difference column (between time of sampling in ES and starting time of ECG measurement)
def create_timedifference(df, starting_time):
    deltatime_array = []
    starting_time = dt.datetime.strptime(starting_time, '%Y-%m-%dT%H:%M:%S.%f')

    #create column indicating timedifference between starting_time & stress_event IN SECONDS
    for entry in df["Stress_Event"]:
        deltatime = entry-starting_time
        deltatime = deltatime.total_seconds()
        deltatime_array.append(deltatime)
    df["timedifference"] = deltatime_array

    # create columng indicating timedifference between starting_time & form_start time
    deltatime_array = []
    for entry2 in df["Form_start"]:
        entry2 = dt.datetime.strptime(entry2, '%Y-%m-%dT%H:%M:%S')
        deltatime = entry2-starting_time
        deltatime = deltatime.total_seconds()
        deltatime_array.append(deltatime)
    df["Form_start_timedifference"] = deltatime_array
    return(df)

# Create event dictionary (which can be used to create NK2 epochs)
def create_events(proband, df_label, df_biodata,frequency, list_of_times):
    """ This function creates stress and no_stress events by iterating through the df_label dataframe and a) first
    extracting stress_events and b) second finding corresponding no_stress events (which occured at a similar time another day)

    :param df_label:
    :param df_biodata:
    :param frequency: sampling frequency
    :param list_of_times: the times (in hours) through which algorithm should iterate when searching for corresponding no_stress
    event for every stress event
    :return: event locations (in number of samples)
    """
    total_seconds = df_biodata.shape[0]/frequency #total seconds in raw object
    df_label = df_label[df_label['timedifference'] > 0]
    df_label = df_label[df_label['timedifference'] < total_seconds]

    # create arrays which are accepted by MNE annotate
    stress_onset = []
    stress_duration = []
    #stress_existance = []
    stress_intensity = []

    #create stress and no_stress events; no_stress events have to be at the same time at another day
    for idx, entry in df_label['timedifference'].iteritems():
        if df_label['Stress (binary)'][idx] == 0:
            continue  # skip no_stress events

        # add no_stress events: for every stress event find a no_stress event at similar time another day
        # iterate through list of times: +1 day, -1 day, etc.
        counter = 0
        for day in list_of_times:
            time_in_seconds = day*24*60*60                          #converting days into seconds
            # iterate through "Form_start_timedifference" variable, since here it can be assured that looking in a +-
            # 30 minute range a) will return a no_stress period and b) no no_stress period is used twice
            for index, event in df_label["Form_start_timedifference"].items():
                # the no-stress period has to be in in the time range of of +-30 minutes in the days after or before the stress period
                if (event <= (df_label["Form_start_timedifference"][idx] + time_in_seconds+ 30*60) and event >= (df_label["Form_start_timedifference"][idx] + time_in_seconds - 30*60) and df_label["Stress (binary)"][index] == 0):
                    stress_onset.append(df_label["timedifference"][index]*1024)         # x1024 since the timedifference
                    # values are in seconds (and not in number of sample)
                    stress_duration.append(df_label["Stress_Duration"][index])
                    #stress_existance.append(df_label['Stress (binary)'][index])
                    stress_intensity.append(0)
                    counter = counter+1
                if counter !=0:
                    break                                               # this is to break the for-loop as soon as there has been one suitable no-stress period found
            if counter != 0:
                break  # this is to break the for-loop as soon as there has been one suitable no-stress period found

        if counter != 0:
            #add stress_events in case there has been a respective no_stress event found (which is not the case in 9 cases)
            stress_onset.append(entry*1024)
            stress_duration.append(df_label["Stress_Duration"][idx])
            #stress_existance.append(df_label['Stress (binary)'][idx])
            stress_intensity.append(df_label["Wie_Stress"][idx])

        else:
            print("For Proband" + str(proband) + "and  Index: " + str(
                idx) + "no no_stress event can be found and therefore this stress event will not be added")

    # create events dict
    events = {}
    events["condition"] = stress_intensity
    events["label"] = range(1,len(stress_onset)+1,1)        #this label is used as the index/name of the epoch!
    events["onset"] = stress_onset


    #print("There are " + str(len(my_annot)) + " annotations")

    return events
# plt.plot(df_ecg['ECG']) #Plot ECG signal
# plot = nk.events_plot(events, df_ecg) #Visualize Events & Data
#endregion


print("hi")

#region merging epochs
##Current progress: never finished working on this since we cancelled the idea of merging epochs shortly after I started

condition = "epochs_600seconds_filtered"
def merge_epochs(condition):
    df = pd.DataFrame()

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

participant = pd.Series(dtype='float64')
epoch_number = pd.Series(dtype='float64')
ECG = pd.Series(dtype='float64')
label = pd.Series(dtype='float64')
cond = pd.Series(dtype='float64')

df = pd.DataFrame()
for epoch in epochs:
    length = len(epochs[epoch]["Condition"])
    participant = participant.append(pd.Series([proband]).repeat(length))
    epoch_number = epoch_number.append(pd.Series([epoch]).repeat(length))
    ECG = ECG.append(epochs[epoch]["ECG"])
    label = label.append(epochs[epoch]["Label"])
    condition = condition.append(epochs[epoch]["Condition"])
participant = participant.reindex(df.index)

participant = participant.reset_index()

df["Participant"] = participant
df["Epoch"] = epoch_number
df["ECG"] = ECG.reindex(df.index)
df["label"] = label
df["Condition"] = condition
#endregion


#region create array which contains all segments (one array for ECG & onbe array for labels)
# Note: the data here is create in an array format with which the LSTM algorithm can deal

def create_array_epoch(epochs):
    x = []
    y = []
    for key in epochs:
        epoch = epochs[key]
        x_segment = epochs[key]["ECG"]
        y_segment = epochs[key]["Condition"]

        x_segment = x_segment.tolist()
        y_segment = y_segment.tolist()

        x.extend(x_segment)
        y.extend(y_segment)

    return x, y

def create_array_allepochs(files,condition):
    """

    :param condition: the substring which has to be contained within PKL file name so that its considered
    :param train_split:
    :param segment_duration:
    :param sampling_rate:
    :return:
    """
    x_all = []
    y_all = []
    for name in files:
        if condition not in name:
            continue
        epochs = load_obj(name)

        x, y = create_array_epoch(epochs)
        x_all.extend(x)
        y_all.extend(y)

    #save_obj(x_train_all, "x_train_epochs_lenght_" + str(epochs_length) + "_segment_duration_" + str(segment_duration) + "_train_split_" + str(train_split))
    #save_obj(y_train_all, "y_train_epochs_lenght_" + str(epochs_length) + "_segment_duration_" + str(segment_duration) + "_train_split_" + str(train_split))
    #save_obj(x_test_all, "x_test_epochs_lenght_" + str(epochs_length) + "_segment_duration_" + str(segment_duration) + "_train_split_" + str(train_split))
    #save_obj(y_test_all, "y_test_epochs_lenght_" + str(epochs_length) + "_segment_duration_" + str(segment_duration) + "_train_split_" + str(train_split))

    return x_all, y_all

#endregion
#region create array which contains all segments (one array for ECG & onbe array for labels)
# Note: the data here is create in an array format with which the LSTM algorithm can deal

def create_array_epoch_segmented(epochs_segmented,  segment_duration, sampling_rate):
    x = np.empty((0,segment_duration*sampling_rate,1))
    y = np.empty((0,1))
    for key in epochs_segmented:
        epoch = epochs_segmented[key]
        for key2 in epoch:
            x_segment = epoch[key2]["ECG"]
            y_segment = epoch[key2]["Condition"]

            # Creating 3D array (needed by LSTM algorithm)
            x_segment = np.array(x_segment)
            x_segment = np.reshape(x_segment, (len(x_segment), 1))
            x_segment = np.expand_dims(x_segment, axis=(0))

            x = np.vstack([x, x_segment])
            y = np.vstack([y, y_segment.iloc[0]])

    return x, y

def create_array_allepochs_segmented(files,condition, segment_duration, sampling_rate):
    """

    :param condition: the substring which has to be contained within PKL file name so that its considered for training & testing data
    :param train_split:
    :param segment_duration:
    :param sampling_rate:
    :return:
    """
    x_all = np.empty((0, segment_duration * sampling_rate, 1))
    y_all = np.empty((0, 1))
    for name in files:
        if condition not in name:
            continue
        epochs_segmented = load_obj(name)

        x, y = create_array_epoch_segmented(epochs_segmented, segment_duration, sampling_rate)
        x_all = np.concatenate([x_all, x], -3)
        y_all = np.concatenate([y_all, y], -2)

    #save_obj(x_train_all, "x_train_epochs_lenght_" + str(epochs_length) + "_segment_duration_" + str(segment_duration) + "_train_split_" + str(train_split))
    #save_obj(y_train_all, "y_train_epochs_lenght_" + str(epochs_length) + "_segment_duration_" + str(segment_duration) + "_train_split_" + str(train_split))
    #save_obj(x_test_all, "x_test_epochs_lenght_" + str(epochs_length) + "_segment_duration_" + str(segment_duration) + "_train_split_" + str(train_split))
    #save_obj(y_test_all, "y_test_epochs_lenght_" + str(epochs_length) + "_segment_duration_" + str(segment_duration) + "_train_split_" + str(train_split))

    return x_all, y_all

#endregion

#region Archive

## Old create_events function: this function creates events from the df_label dataframe & the starting_time; it just
## creates all available events, not choosing the no_stress events based on conditions (which was why it was
## replaced by newer create_events function afterwards
"""
def create_events(df_label, df_biodata,frequency):

    :param df_label:
    :param df_biodata:
    :param frequency:
    :return: event locations (in seconds; NOT in number of samples)

    total_seconds = df_biodata.shape[0]/frequency #total seconds in raw object
    df_label = df_label[df_label['timedifference'] > 0]
    df_label = df_label[df_label['timedifference'] < total_seconds]

    # create arrays which are accepted by MNE annotate
    stress_onset = []
    stress_duration = []
    stress_intensity = []

    for idx, entry in df_label['timedifference'].iteritems():
        stress_onset.append(entry)
        stress_duration.append(df_label["Stress_Duration"][idx])
        stress_intensity.append(df_label['Stress_Intensity'][idx])

    events = {}
    events["condition"] = stress_intensity
    #events["duration"] = stress_duration
    events["label"] = range(1,len(stress_onset)+1,1)
    events["onset"] = stress_onset


    #print("There are " + str(len(my_annot)) + " annotations")

    return events
"""

## Old balance_data function: this function chooses no_stress events randomly from all no_stress events given an event
## dictionary which contains stress and no_stress events and a stress_percentage; this function was replaced by the new
## create_events function in which the no_stress events are chosen based so that they correspond to similar time as stres_events
"""
def balance(events, stress_percentage):
    total_labels = len(events["condition"])
    actual_no_stress =Counter(events["condition"])[0]
    actual_stress = total_labels-actual_no_stress
    required_no_stress = round( (actual_stress/stress_percentage)*(100-stress_percentage))
    rnd_number = list(range(required_no_stress))
    random.shuffle(rnd_number)

    counter_general = 0
    counter_nostress = 0
    condition = []
    label = []
    onset = []
    for i in events["condition"]:
        if i == 0:
            if counter_nostress in rnd_number:
                condition.append(events["condition"][counter_general])
                label.append(events["label"][counter_general])
                onset.append(events["onset"][counter_general])
                counter_nostress = counter_nostress+1
                counter_general = counter_general + 1
            else:
                counter_general = counter_general + 1
        else:
            condition.append(events["condition"][counter_general])
            label.append(events["label"][counter_general])
            onset.append(events["onset"][counter_general])
            counter_general = counter_general+1

    events_balanced = {}
    events_balanced["condition"]= condition
    events_balanced["label"] = label
    events_balanced["onset"] = onset

    return events_balanced

"""

#endregion