df_class_allfeatures = load_obj("df_class_allfeatures.pkl")

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

features_del = ["HRV_LF", "HRV_LFHF", "HRV_LFn", "HRV_ULF", "HRV_VLF", "HRV_SampEn"]
df_class_allfeatures_del = df_class_allfeatures.drop(features_del, axis = 1)

#endregion

#region drop NaN rows
# Note: only HRV_HFn,HRV_LnHF, HRV_HF contain each 1 NaN value -> only 1 row is dropped
df_class_allfeatures_del_NaN = df_class_allfeatures_del.dropna()
save_obj(df_class_allfeatures_del_NaN, "df_singleproband_class_allfeatures_del_NaN")

#endregion

#region convert label: Subpipeline 01: convert into binary
df_class_allfeatures_del_NaN_binary = df_class_allfeatures_del_NaN.copy()
df_class_allfeatures_del_NaN_binary.loc[df_class_allfeatures_del_NaN_binary['Label'] > 0, 'Label'] = 1
save_obj(df_class_allfeatures_del_NaN_binary, "df_singleproband_class_allfeatures_del_NaN_binary")
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

#endregion


#region different datasets from different (sub)pipelines
#Basic Pipelind
df_class_allfeatures_del_NaN_binary_shuffled = load_obj("df_class_allfeatures_del_NaN_binary_shuffled.pkl")

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

#endregion



#region Trial02: Random Forest classifier


#get data: train & test split
#df_class_allfeatures_del_NaN_binary_shuffled = load_obj("df_class_allfeatures_del_NaN_binary_shuffled.pkl")
df_class_allfeatures_del_NaN_binary_shuffled = load_obj("df_singleproband_class_allfeatures_del_NaN_binary.pkl")


all_including_probandinformation = df_class_allfeatures_del_NaN_binary_shuffled
X_acc_all_df = df_class_allfeatures_del_NaN_binary_shuffled.drop(["Label", "Epoch", "Proband", "Segment"], axis = 1)
X_all_df = df_class_allfeatures_del_NaN_binary_shuffled.drop(["Label", "x1", "x2", "x3", "Epoch", "Proband", "Segment"], axis = 1)
y_all_df = df_class_allfeatures_del_NaN_binary_shuffled["Label"]

#Choose data for this iteration
#X_train = X_all_df
#y_train = y_all_df
#X_test = X_test_df
#y_test = y_test_df

X_all = X_acc_all_df
y_all = y_all_df
X_all_df = X_acc_all_df
#y_all_df = y_all_df

#region Pipeline 2: using nested crossvalidation & feature interpretation with SHAP!
#create pairwise iteration function
import itertools
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    #a, b = itertools.tee(iterable)
    #next(b, None)
    a = iter(iterable)
    return zip(a, a)

#region nested CV
t0 = time.time()

#initialize
df_results = pd.DataFrame()

list_shap_values = list()
list_test_sets = list()
columns = X_all_df.columns

#iterating on epochs-level (for outer CV)

for proband in all_including_probandinformation["Proband"].unique():

    df_proband = all_including_probandinformation[all_including_probandinformation["Proband"]==proband]
    #only use data when minimum of 10 epochs -> otherwise would be less than 5 Fold CV which might be not enough
    if len(df_proband["Epoch"].unique()) < 10:
        continue

    #initialize lists
    outer_results_acc = list()
    outer_results_f1 = list()
    outer_results_precision = list()
    outer_results_recall = list()

    #create DF for this proband (using nexted CV)
    epochs_existing = df_proband["Epoch"].unique()
    for epoch1,epoch2 in pairwise(epochs_existing):
        t0_inner = time.time()
        #compute index for training and test dataset (outer CV)
        test_ix = df_proband[df_proband["Epoch"].isin([epoch1, epoch2])].index

        epochs_existing_train = epochs_existing.tolist()
        epochs_existing_train.remove(epoch1)
        epochs_existing_train.remove(epoch2)
        train_ix = df_proband[df_proband["Epoch"].isin(epochs_existing_train)].index

        #shuffle data (= indices)
        train_ix_shuffled = list(train_ix)
        random.shuffle(train_ix_shuffled)
        test_ix_shuffled = list(test_ix)
        random.shuffle(test_ix_shuffled)

        # split data
        X_train, X_test = X_all.loc[train_ix_shuffled, :], X_all.loc[test_ix_shuffled, :]
        y_train, y_test = y_all.loc[train_ix_shuffled], y_all.loc[test_ix_shuffled]
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
        # configure the cross-validation procedure
        cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
        # define the model
        model = RandomForestClassifier(random_state=1)

        # define search space
        n_estimators = [100, 300, 500, 800, 1200]
        max_depth = [5, 8, 15, 25, 30]
        min_samples_split = [2, 5, 10, 15, 100]
        min_samples_leaf = [1, 2, 5, 10]
        max_features = ["sqrt", "log2", 15]

        space = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                      min_samples_leaf=min_samples_leaf, max_features = max_features)
        # define search
        search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True, n_jobs=-1)
        # execute search
        result = search.fit(X_train, y_train)
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test)
        # evaluate the model
        acc = accuracy_score(y_test, yhat)
        f1 = f1_score(y_true=y_test, y_pred=yhat, pos_label=1, average="binary")
        precision = precision_score(y_true=y_test, y_pred=yhat, pos_label=1, average="binary")
        recall = recall_score(y_true=y_test, y_pred=yhat, pos_label=1, average="binary")
        # store the result
        outer_results_acc.append(acc)
        outer_results_f1.append(f1)
        outer_results_precision.append(precision)
        outer_results_recall.append(recall)
        # report progress
        t1_inner = time.time()
        print("For Proband "+ str(proband) + " results are: >acc=%.3f,f1=%.3f,precision=%.3f,recall=%.3f, est=%.3f, cfg=%s" % (acc,f1, precision, recall, result.best_score_, result.best_params_))
        print("This inner iteration has taken so many seconds: " + str(t1_inner-t0_inner))

        #feature importance: compute SHAP values
        #explainer = shap.Explainer(best_model)
        #shap_values = explainer.shap_values(X_test)
        # for each iteration we save the test_set index and the shap_values
        #list_shap_values.append(shap_values)
        #list_test_sets.append(test_ix)

    #combine all the evaluation data
    df_temp = pd.DataFrame()
    df_temp["Proband"] = [proband]*len(outer_results_acc)
    df_temp["ACC"] = outer_results_acc
    df_temp["F1"] =outer_results_f1
    df_temp["Precision"] = outer_results_precision
    df_temp["Recall"] = outer_results_recall

    df_results = df_results.append(df_temp)
    print("Results until now are: \n" + str(df_results))

t1 = time.time()
print("This process took so many seconds: " +str(t1-t0))
os.system("shutdown /h") #hibernate


#combining SHAP results from all iterations
#test_set = []
#shap_values = []
test_set = list_test_sets[0]
shap_values = np.array(list_shap_values[0])

for i in range(1,len(list_test_sets)):
	test_set = np.concatenate((test_set,list_test_sets[i]),axis=0)
	shap_values = np.concatenate((shap_values,np.array(list_shap_values[i])),axis=1)
#bringing back variable names: X_test is the X_all data reordered due to the 10-test sets in outer CV
test_set = test_set.astype(int)
X_test = pd.DataFrame(X_all[test_set],columns = columns) #this should be used for SHAP plots!

# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results_acc), std(outer_results_acc)))
print('F1: %.3f (%.3f)' % (mean(outer_results_f1), std(outer_results_f1)))
print('Precision: %.3f (%.3f)' % (mean(outer_results_precision), std(outer_results_precision)))
print('Recall: %.3f (%.3f)' % (mean(outer_results_recall), std(outer_results_recall)))

save_obj(shap_values, "normalstress_shap_values_DecisionForest_withoutACC_WITHOUTMEANMEDIANRR")
save_obj(X_test, "normalstress_X_test_DecisionForest_withoutACC_WITHOUTMEANMEDIANRR")
#shap_values = load_obj("shap_values_DecisionForest_withACC.pkl")
#X_test = load_obj("X_test_DecisionForest_withACC.pkl")

#endregion

#region SHAP Feature interpretation
# feature importance plot
shap.summary_plot(shap_values[0], X_test, plot_type="bar")

#SHAP feature importance plot
shap.summary_plot(shap_values[1], X_test)

#Force Plot
#shap.plots.force(explainer.expected_value[0], shap_values[0])

#partial dependence plot: analyze individual features
shap.dependence_plot("HRV_MeanNN", shap_values[0], X_test)

#endregion
#endregion


#endregion
#endregion




##TODO Shuffle dataset for training
