
#region Split in Train and Test data


# Splitting function

def split_train_test(epochs_segmented, split_percentage, segment_duration, sampling_rate):
    x_train = np.empty((0,segment_duration*sampling_rate,5))
    y_train = np.empty((0,1))
    x_test = np.empty((0,segment_duration*sampling_rate,5))
    y_test = np.empty((0,1))
    for epoch in epochs_segmented:
        epoch_length = len(epochs_segmented[epoch])
        test_indices = random.sample(range(1, epoch_length + 1), int(split_percentage * epoch_length))

        for segment in epochs_segmented[epoch]:
            #check if segment has segment_duration*sampling_rate number of rows; if not, drop last one
            segment_data = epochs_segmented[epoch][segment]
            if epochs_segmented[epoch][segment].shape[0] != segment_duration*sampling_rate:
                segment_data.drop(segment_data.tail(1).index, inplace =True)

            #if segment not in test_indices -> add to training data
            if segment not in test_indices:
                x_train_segment = segment_data[["ECG", "Noise_Labels", "x1", "x2", "x3"]]
                y_train_segment = segment_data["Condition"]

                # Creating 3D array (needed by LSTM algorithm)
                x_train_segment = np.array(x_train_segment)
                x_train_segment = np.reshape(x_train_segment, (len(x_train_segment), 5))
                x_train_segment = np.expand_dims(x_train_segment, axis=(0))

                x_train = np.vstack([x_train, x_train_segment])
                y_train = np.vstack([y_train, y_train_segment.iloc[0]])

            # else add to training data
            else:
                x_test_segment = segment_data[["ECG", "Noise_Labels", "x1", "x2", "x3"]]
                y_test_segment = segment_data["Condition"]

                # Creating 3D array (needed by LSTM algorithm)
                x_test_segment = np.array(x_test_segment)
                x_test_segment = np.reshape(x_test_segment, (len(x_test_segment), 5))
                x_test_segment = np.expand_dims(x_test_segment, axis=(0))

                x_test= np.vstack([x_test, x_test_segment])
                y_test = np.vstack([y_test, y_test_segment.iloc[0]])

            print("Epoch " +str(epoch) + " and segment "+str(segment)+ " is done now")
    return x_train, y_train, x_test, y_test

#Split for all participants
def split_train_test_forall(files,condition, train_split, segment_duration, sampling_rate):
    """

    :param condition: the substring which has to be contained within PKL file name so that its considered for training & testing data
    :param train_split:
    :param segment_duration:
    :param sampling_rate:
    :return:
    """
    x_train_all = np.empty((0, segment_duration * sampling_rate, 5))
    y_train_all = np.empty((0, 1))
    x_test_all = np.empty((0, segment_duration * sampling_rate, 5))
    y_test_all = np.empty((0, 1))

    for name in files:
        if condition not in name:
            continue
        epochs_segmented = load_obj(name)

        x_train, y_train, x_test, y_test = split_train_test(epochs_segmented, split_percentage, segment_duration, sampling_rate)
        x_train_all = np.concatenate([x_train_all, x_train], -3)
        y_train_all = np.concatenate([y_train_all, y_train], -2)
        x_test_all = np.concatenate([x_test_all, x_test], -3)
        y_test_all = np.concatenate([y_test_all, y_test], -2)

        print("Proband " +str(name) +"is done now")

    return x_train_all, y_train_all, x_test_all, y_test_all

#endregion

#region LSTM model
# Based on this code. https://towardsdatascience.com/time-series-classification-for-human-activity-recognition-with-lstms-using-tensorflow-2-and-keras-b816431afdff
def LSTM(x_train, y_train, x_test, y_test):

# Encode the categories
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc = enc.fit(y_train)
    y_train = enc.transform(y_train)
    y_test = enc.transform(y_test)
    y_train
    # Create LSTM model
    model = keras.Sequential()
    model.add(
        keras.layers.Bidirectional(
          keras.layers.LSTM(
              units=128,
              input_shape=[x_train.shape[1],x_train.shape[2] ]
          )
        )
    )
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(units=128, activation='relu'))
    model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))

    model.compile(
      loss='categorical_crossentropy',
      optimizer='adam',
      metrics=['acc']
    )

    # Run LSTM model
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=25,
        validation_split=0.1,
        shuffle=False
    )

    return model, history, enc, y_train, y_test

#endregion