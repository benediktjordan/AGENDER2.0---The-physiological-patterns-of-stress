df_class_allfeatures = load_obj("df_class_allfeatures.pkl")

df_class_allfeatures = load_obj("df_class_allfeatures_noiseIMPROVED.pkl") #temporary

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

features_del = ["HRV_LF", "HRV_LFHF", "HRV_LFn", "HRV_ULF", "HRV_VLF", "HRV_SampEn", "Proband", "Segment", "Epoch"]
df_class_allfeatures_del = df_class_allfeatures.drop(features_del, axis = 1)

#following two lines for LOSOCV (proband information remains included)
features_del_new = ["HRV_LF", "HRV_LFHF", "HRV_LFn", "HRV_ULF", "HRV_VLF", "HRV_SampEn", "Segment", "Epoch"]
df_class_allfeatures_del = df_class_allfeatures.drop(features_del_new, axis = 1)

#endregion

#region drop NaN rows
# Note: only HRV_HFn,HRV_LnHF, HRV_HF contain each 1 NaN value -> only 1 row is dropped
df_class_allfeatures_del_NaN = df_class_allfeatures_del.dropna()
#endregion

#region convert label: Subpipeline 01: convert into binary
df_class_allfeatures_del_NaN_binary = df_class_allfeatures_del_NaN.copy()
df_class_allfeatures_del_NaN_binary.loc[df_class_allfeatures_del_NaN_binary['Label'] > 0, 'Label'] = 1
#save_obj(df_class_allfeatures_del_NaN_binary, "df_class_allfeatures_del_NaN_binary")
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
#save_obj(df_class_allfeatures_del_NaN_binary_shuffled, "df_class_allfeatures_del_NaN_binary_shuffled_includingProband")

#endregion

#region split training & test data

#my splitting function
def split_ensemble(df, split_percentage):
	test_indices = random.sample(range(0, df.shape[0]),
								 int(split_percentage*df.shape[0]))
	df_test = pd.DataFrame()
	df_train = df.copy()

	df_train = df_train.drop(test_indices, axis = 0)
	for i in test_indices:
		df_test = df_test.append(df.iloc[i])

	return df_train, df_test

#Note: better use SKlearn splitting function

#X = df_class_allfeatures_del_NaN_binary_shuffled.drop("Label", axis = 1)
#y = df_class_allfeatures_del_NaN_binary_shuffled["Label"]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=3)

#endregion


#region different datasets from different (sub)pipelines
#Basic Pipelind
df_class_allfeatures_del_NaN_binary_shuffled = load_obj("df_class_allfeatures_del_NaN_binary_shuffled.pkl")

#Basic Pipeline INCLUDING "Proband" column
df_class_allfeatures_del_NaN_binary_shuffled_includingProband = load_obj("df_class_allfeatures_del_NaN_binary_shuffled_includingProband.pkl")

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

# Subpipeline: only physical movement features
df_class_allfeatures_del_NaN_binary_shuffled = load_obj("df_class_allfeatures_del_NaN_binary_shuffled.pkl")
df_class_allfeatures_del_NaN_binary_shuffled_OnlyMovement = df_class_allfeatures_del_NaN_binary_shuffled.copy()
choosecolumns = [ "x1", "x2", "x3", "Label"]
df_class_allfeatures_del_NaN_binary_shuffled_OnlyMovement = df_class_allfeatures_del_NaN_binary_shuffled_OnlyMovement[choosecolumns]
save_obj(df_class_allfeatures_del_NaN_binary_shuffled_OnlyMovement ,"df_class_allfeatures_del_NaN_binary_shuffled_OnlyMovement")

#endregion

#region Trial01: ensemble classification pipeline


t0 = time.time()
##This code is based on: https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/


#region get data
#get data: train & test split
df_class_allfeatures_del_NaN_binary_shuffled = load_obj("df_class_allfeatures_del_NaN_binary_shuffled_EXCEPTPREDICTIVE.pkl")

X_acc = df_class_allfeatures_del_NaN_binary_shuffled.drop(["Label"], axis = 1)
y_acc = df_class_allfeatures_del_NaN_binary_shuffled["Label"]
X_acc_train_array, X_acc_test_array, y_acc_train_array, y_acc_test_array = train_test_split(X_acc, y_acc, test_size=0.1, random_state=3)
X_acc_train_array = X_acc_train_array.to_numpy()
X_acc_test_array = X_acc_test_array.to_numpy()
y_acc_train_array = y_acc_train_array.to_numpy()
y_acc_test_array = y_acc_test_array.to_numpy()

X = df_class_allfeatures_del_NaN_binary_shuffled.drop(["Label", "x1", "x2", "x3"], axis = 1)
y = df_class_allfeatures_del_NaN_binary_shuffled["Label"]
X_train_array, X_test_array, y_train_array, y_test_array = train_test_split(X, y, test_size=0.1, random_state=3)
X_train_array = X_train_array.to_numpy()
X_test_array = X_test_array.to_numpy()
y_train_array = y_train_array.to_numpy()
y_test_array = y_test_array.to_numpy()

X_acc_all_array = df_class_allfeatures_del_NaN_binary_shuffled.drop(["Label"], axis = 1)
X_all_array = df_class_allfeatures_del_NaN_binary_shuffled.drop(["Label", "x1", "x2", "x3"], axis = 1)
y_all_array = df_class_allfeatures_del_NaN_binary_shuffled["Label"]


# scaling
sc_X = StandardScaler()
X_scaled=sc_X.fit_transform(X_all_array)

# define dataset for this iteration
X = X_scaled
Y = y_all_array

X_all = X_scaled
y_all = y_all_array.to_numpy()
X_all_df = X_all_array
#endregion


#region creating stacking functions
t0 = time.time()
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from matplotlib import pyplot

# get a stacking ensemble of models

def get_stacking():
	# define the base models
	level0 = list()
	level0.append(('mlp',MLPClassifier(max_iter=1000, activation= "tanh", alpha= 0.05,
									   hidden_layer_sizes=(256, 128, 64, 32), learning_rate="adaptive",
									   solver="adam",random_state=1)))
	level0.append(('DF', RandomForestClassifier(random_state = 1, max_depth = 25,
												n_estimators = 100, min_samples_split = 2,
												min_samples_leaf = 2, max_features = 15)))
	level0.append(('svm', SVC(kernel='rbf', gamma=0.1, C=100)))
	# define meta learner model
	level1 = LogisticRegression(solver="saga")
	# define the stacking ensemble
	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
	return model

# get a list of models to evaluate
def get_models():
	models = dict()
	models['MLP'] = MLPClassifier(max_iter=1000, activation= "tanh", alpha= 0.0001,
									   hidden_layer_sizes=(256, 128, 64, 32), learning_rate="adaptive",
									   solver="adam",random_state=1)
	models['DF'] = RandomForestClassifier(random_state = 1, max_depth = 30,
												n_estimators = 800, min_samples_split = 2,
												min_samples_leaf = 1, max_features = 15)
	models['svm'] = SVC(kernel='rbf', gamma=0.1, C=100)
	models['stacking'] = get_stacking()
	return models


# evaluate a give model using cross-validation
def evaluate_model(model, X, Y):
	#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, Y, scoring='accuracy', cv=10, n_jobs=-1)
	return scores
#endregion

#region Pipeline 1: "traditional" hyperparameter tuning & training
#do hyperparameter tuning for logistic regression:
t0 = time.time()
stacking = get_stacking()
params = {'final_estimator__C': np.logspace(-3,3,7),
		  'final_estimator__penalty':["l1","l2"]}
grid = GridSearchCV(estimator=stacking, param_grid=params, cv=5,n_jobs = -1)
grid_best = grid.fit(X, Y)
t1 = time.time()
t1-t0

grid_best.best_params_

#Run stacking model
# with CV
#With 10 fold CV
t0 = time.time()
from sklearn.model_selection import cross_val_score
stacking = get_stacking()
accuracy = cross_val_score(stacking, X, Y, scoring= "accuracy", cv=10)
accuracy.mean()
f1 = cross_val_score(stacking, X, Y, scoring= "f1", cv=10)
f1.mean()
precision = cross_val_score(stacking, X, Y, scoring= "precision", cv=10)
precision.mean()
recall = cross_val_score(stacking, X, Y, scoring= "recall", cv=10)
recall.mean()
t1 = time.time()
t1 - t0


# OLD WAY running all models & getting accuracy score
t0 = time.time()
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X, Y)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
t1 = time.time()
t1 - t0


#endregion

#region Pipeline 2: using nested crossvalidation & feature interpretation with SHAP!

#region nested CV
t0 = time.time()
# configure the cross-validation procedure
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# enumerate splits
outer_results_acc = list()
outer_results_f1 = list()
outer_results_precision = list()
outer_results_recall = list()

list_shap_values = list()
list_test_sets = list()
columns = X_all_df.columns
counter = 0
outer_split = cv_outer.split(X_all, y_all)
outer_split = list(outer_split)
save_obj(outer_split, "outer_split")

for train_ix, test_ix in outer_split:
	t0_inner = time.time()
	t0_inner_CV = time.time()
	# split data
	X_train, X_test = X_all[train_ix, :], X_all[test_ix, :]
	y_train, y_test = y_all[train_ix], y_all[test_ix]
	# configure the cross-validation procedure
	cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
	# define the model
	model = get_stacking()

	# define search space
	space = {'final_estimator__C': np.logspace(-3,3,7),
			 'final_estimator__penalty':["l1","l2"]}
	#space = {'final_estimator__C': [1],
	#		 'final_estimator__penalty':["l1"]}

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

	#save intermediate results
	save_obj(outer_results_acc, "outer_results_acc")
	save_obj(outer_results_f1, "outer_results_f1")
	save_obj(outer_results_precision, "outer_results_precision")
	save_obj(outer_results_recall, "outer_results_recall")
	# report progress
	print('>acc=%.3f,f1=%.3f,precision=%.3f,recall=%.3f, est=%.3f, cfg=%s' % (acc,f1, precision, recall, result.best_score_, result.best_params_))
	t1_inner_CV = time.time()
	print("Inner CV has taken so many seconds: " + str(t1_inner_CV-t0_inner_CV))

	#feature importance: compute SHAP values
	#explainer = shap.KernelExplainer(best_model.predict, X_test)
	#shap_values = explainer.shap_values(X_test, nsamples = 100)
	## for each iteration we save the test_set index and the shap_values
	#list_shap_values.append(shap_values)
	#list_test_sets.append(test_ix)
	# save intermediates
	#save_obj(list_shap_values, "list_shap_values")
	#save_obj(list_test_sets, "list_test_sets")
	t1_inner = time.time()
	print("The time of this inner iteration was" + str(t1_inner-t0_inner))
	counter = counter + 1
	save_obj(counter, "counter")
#combining SHAP results from all iterations
#test_set = []
#shap_values = []
t1 = time.time()
print("This process took so many seconds: " +str(t1-t0))

#region to pick up on old progress
#load old progress:
outer_split = load_obj("outer_split.pkl")
outer_results_acc = load_obj("outer_results_acc.pkl")
outer_results_f1 = load_obj("outer_results_acc.pkl")
outer_results_precision = load_obj("outer_results_precision.pkl")
outer_results_recall = load_obj("outer_results_recall.pkl")
list_shap_values = load_obj("list_shap_values.pkl")
list_test_sets = load_obj("list_test_sets.pkl")
counter = load_obj("counter.pkl")

#start again from where I stopped
counter2 = 0
for train_ix, test_ix in outer_split:
	if counter2 <= counter:
		counter2 = counter2+1
		continue
	t0_inner = time.time()
	t0_inner_CV = time.time()
	# split data
	X_train, X_test = X_all[train_ix, :], X_all[test_ix, :]
	y_train, y_test = y_all[train_ix], y_all[test_ix]
	# configure the cross-validation procedure
	cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
	# define the model
	model = SVC()

	# define search space
	space = [{'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
			  'gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001],
			  'C': [0.001, 0.1, 1, 10, 100]}]
	#space = [{'kernel': ['linear'],
	#		  'gamma': [1],
	#		  'C': [0.001]}]
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
	print('>acc=%.3f,f1=%.3f,precision=%.3f,recall=%.3f, est=%.3f, cfg=%s' % (acc,f1, precision, recall, result.best_score_, result.best_params_))
	t1_inner_CV = time.time()
	print("Inner CV has taken so many seconds: " + str(t1_inner_CV-t0_inner_CV))
	#feature importance: compute SHAP values
	explainer = shap.KernelExplainer(best_model.predict, X_test)
	shap_values = explainer.shap_values(X_test, nsamples = 100)
	# for each iteration we save the test_set index and the shap_values
	list_shap_values.append(shap_values)
	list_test_sets.append(test_ix)
	t1_inner = time.time()
	print("The time of this inner iteration was" + str(t1_inner-t0_inner))


#endregion



test_set = list_test_sets[0]
shap_values = np.array(list_shap_values[0])

for i in range(1,len(list_test_sets)):
	test_set = np.concatenate((test_set,list_test_sets[i]),axis=0)
	shap_values = np.concatenate((shap_values,np.array(list_shap_values[i])), axis=0)
#bringing back variable names: X_test is the X_all data reordered due to the 10-test sets in outer CV
test_set = test_set.astype(int)
X_test = pd.DataFrame(X_all[test_set],columns = columns) #this should be used for SHAP plots!


# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results_acc), std(outer_results_acc)))
print('F1: %.3f (%.3f)' % (mean(outer_results_f1), std(outer_results_f1)))
print('Precision: %.3f (%.3f)' % (mean(outer_results_precision), std(outer_results_precision)))
print('Recall: %.3f (%.3f)' % (mean(outer_results_recall), std(outer_results_recall)))


save_obj(shap_values, "normalstress_shap_values_stacking_withoutACC")
save_obj(X_test, "normalstress_X_test_stacking_withoutACC")
#shap_values = load_obj("shap_values_DecisionForest_withACC.pkl")
#X_test = load_obj("X_test_DecisionForest_withACC.pkl")

#endregion

#region SHAP Feature interpretation

shap_values = load_obj("normalstress_shap_values_SVM_withACC.pkl")
X_test = load_obj("normalstress_X_test_SVM_withACC.pkl")

# feature importance plot
shap.summary_plot(shap_values, X_test, plot_type="bar")

#SHAP feature importance plot
shap.summary_plot(shap_values, X_test)


#Force Plot
#shap.plots.force(explainer.expected_value[0], shap_values[0])

#partial dependence plot: analyze individual features
shap.dependence_plot("HRV_MeanNN", shap_values, X_test)

#endregion
#endregion
#endregion



#region Trial02: Random Forest classifier


#get data: train & test split
#df_class_allfeatures_del_NaN_binary_shuffled = load_obj("df_class_allfeatures_del_NaN_binary_shuffled.pkl")
df_class_allfeatures_del_NaN_binary_shuffled = load_obj("df_class_allfeatures_del_NaN_binary_shuffled_includingProband.pkl")

X_acc = df_class_allfeatures_del_NaN_binary_shuffled.drop(["Label"], axis = 1)
y_acc = df_class_allfeatures_del_NaN_binary_shuffled["Label"]
X_acc_train_df, X_acc_test_df, y_acc_train_df, y_acc_test_df = train_test_split(X_acc, y_acc, test_size=0.1, random_state=3)

X = df_class_allfeatures_del_NaN_binary_shuffled.drop(["Label", "x1", "x2", "x3"], axis = 1)
y = df_class_allfeatures_del_NaN_binary_shuffled["Label"]
X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X, y, test_size=0.1, random_state=3)

X_acc_all_df = df_class_allfeatures_del_NaN_binary_shuffled.drop("Label", axis = 1)
X_all_df = df_class_allfeatures_del_NaN_binary_shuffled.drop(["Label", "x1", "x2", "x3"], axis = 1)
y_all_df = df_class_allfeatures_del_NaN_binary_shuffled["Label"]

#Choose data for this iteration
#X_train = X_all_df
#y_train = y_all_df
#X_test = X_test_df
#y_test = y_test_df

X_all = X_acc_all_df.to_numpy()
y_all = y_all_df.to_numpy()
X_all_df = X_all_df
#y_all_df = y_all_df

#region Pipeline 1: "traditional" CV & traditional feature interpretation
# define model
forest = RandomForestClassifier(random_state=1)

# tuning hyperparameter for RF with GridSearch
n_estimators = [100, 300, 500, 800, 1200]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]


hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,min_samples_split = min_samples_split,min_samples_leaf = min_samples_leaf)

gridF = GridSearchCV(forest, hyperF, cv = 10, verbose = 1,n_jobs = -1)
bestF = gridF.fit(X_train, y_train)
##visualizing results
gridF.best_params_

#tuning hyperparameter for RF with Validation Curve
#region n_estimators
param_range = [100, 300, 500, 800, 1200]
train_scores, test_scores = validation_curve(
                                RandomForestClassifier(),
                                X = X_train, y = y_train,
                                param_name = 'n_estimators',
                                param_range = param_range, cv = 3)

# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot mean accuracy scores for training and test sets
plt.plot(param_range, train_mean, label="Training score", color="black")
plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

# Plot accurancy bands for training and test sets
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

# Create plot
plt.title("Validation Curve With Random Forest")
plt.xlabel("Number Of Estimators")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()
#endregion

#region max_depth
param_range1 = [5, 8, 15, 25, 30]
train_scores, test_scores = validation_curve(
                                RandomForestClassifier(),
                                X = X_train, y = y_train,
                                param_name = 'max_depth',
                                param_range = param_range1, cv = 3)

# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot mean accuracy scores for training and test sets
plt.plot(param_range1, train_mean, label="Training score", color="black")
plt.plot(param_range1, test_mean, label="Cross-validation score", color="dimgrey")

# Plot accurancy bands for training and test sets
plt.fill_between(param_range1, train_mean - train_std, train_mean + train_std, color="gray")
plt.fill_between(param_range1, test_mean - test_std, test_mean + test_std, color="gainsboro")

# Create plot
plt.title("Validation Curve With Random Forest")
plt.xlabel("maximum depth")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()
#endregion

#region min_sample_split
param_range2=[2, 5, 10, 15, 100]
train_scores, test_scores = validation_curve(
                                RandomForestClassifier(),
                                X = X_train, y = y_train,
                                param_name = 'min_samples_split',
                                param_range = param_range2, cv = 3)

# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot mean accuracy scores for training and test sets
plt.plot(param_range2, train_mean, label="Training score", color="black")
plt.plot(param_range2, test_mean, label="Cross-validation score", color="dimgrey")

# Plot accurancy bands for training and test sets
plt.fill_between(param_range2, train_mean - train_std, train_mean + train_std, color="gray")
plt.fill_between(param_range2, test_mean - test_std, test_mean + test_std, color="gainsboro")

# Create plot
plt.title("Validation Curve With Random Forest")
plt.xlabel("minimum sample split")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()
#endregion

#region min_samples_leaf
param_range3 = [1, 2, 5, 10]
train_scores, test_scores = validation_curve(
                                RandomForestClassifier(),
                                X = X_train, y = y_train,
                                param_name = 'min_samples_leaf',
                                param_range = param_range3, cv = 3)

# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot mean accuracy scores for training and test sets
plt.plot(param_range3, train_mean, label="Training score", color="black")
plt.plot(param_range3, test_mean, label="Cross-validation score", color="dimgrey")

# Plot accurancy bands for training and test sets
plt.fill_between(param_range3, train_mean - train_std, train_mean + train_std, color="gray")
plt.fill_between(param_range3, test_mean - test_std, test_mean + test_std, color="gainsboro")

# Create plot
plt.title("Validation Curve With Random Forest")
plt.xlabel("minimum sample leaf")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()
#endregion

#running actual model
#from sklearn.ensemble import RandomForestClassifier
#X, y = df.iloc[:,:-1], clusters_tsne_scale['tsne_clusters']
#clf = RandomForestClassifier(n_estimators=100).fit(X_test, y_test)
#print(y)
#Best parameters for ACC
## from Grid search: {'max_depth': 15, 'min_samples_leaf': 10,'min_samples_split': 2, 'n_estimators': 100}
##

# with CV
# 10-Fold Cross validation
from sklearn.metrics import SCORERS
#list of possible scores: sorted(SCORERS.keys())
from sklearn.model_selection import cross_val_score
clf_cv = RandomForestClassifier(random_state = 1, max_depth = 15,
							 n_estimators = 100, min_samples_split = 5,
							 min_samples_leaf = 2)
accuracy = cross_val_score(clf_cv, X_train, y_train, scoring= "accuracy", cv=10)
accuracy.mean()
f1 = cross_val_score(clf_cv, X_train, y_train, scoring= "f1", cv=10)
f1.mean()
precision = cross_val_score(clf_cv, X_train, y_train, scoring= "precision", cv=10)
precision.mean()
recall = cross_val_score(clf_cv, X_train, y_train, scoring= "recall", cv=10)
recall.mean()

# alternative cross validation funciton
from sklearn.model_selection import cross_validate
cv_results = cross_validate(clf, X_train, y_train, cv=10)


#without CV
clf = RandomForestClassifier(random_state = 1, max_depth = 15, n_estimators = 300, min_samples_split = 2, min_samples_leaf = 2).fit(X_train, y_train)
clf.score(X_test, y_test)
y_pred=clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#Visualizing in Confusion matrix
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

#Feature Importance Analysis
#modelname.feature_importance_
data = np.array([clf.feature_importances_, X_train.columns]).T
data = pd.DataFrame(data, columns=['Importance', 'Feature']).sort_values("Importance", ascending=False).head(10)
columns = list(data.Feature.values)
#plot
fig, ax = plt.subplots()
width = 0.4 # the width of the bars
ind = np.arange(data.shape[0]) # the x locations for the groups
ax.barh(ind, data["Importance"], width, color="green")
ax.set_yticks(ind+width/10)
ax.set_yticklabels(columns, minor=False)
plt.title("Feature importance in RandomForest Classifier")
plt.xlabel("Relative importance")
plt.ylabel("feature")
plt.figure(figsize=(5,5))
fig.set_size_inches(6.5, 4.5, forward=True)
plt.show()

#endregion

#region Pipeline 2: using nested crossvalidation & feature interpretation with SHAP!

#region nested CV
t0 = time.time()
# configure the cross-validation procedure
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# enumerate splits
outer_results_acc = list()
outer_results_f1 = list()
outer_results_precision = list()
outer_results_recall = list()

list_shap_values = list()
list_test_sets = list()
columns = X_all_df.columns

for train_ix, test_ix in cv_outer.split(X_all, y_all):
	t0_inner = time.time()
	# split data
	X_train, X_test = X_all[train_ix, :], X_all[test_ix, :]
	y_train, y_test = y_all[train_ix], y_all[test_ix]
	# configure the cross-validation procedure
	cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
	# define the model
	model = RandomForestClassifier(random_state=1)

	# define search space
	n_estimators = [100, 300, 500, 800, 1200]
	max_depth = [5, 8, 15, 25, 30]
	min_samples_split = [2, 5, 10, 15, 100]
	min_samples_leaf = [1, 2, 5, 10]
	max_features = ["sqrt", "log2", 3]

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
	print('>acc=%.3f,f1=%.3f,precision=%.3f,recall=%.3f, est=%.3f, cfg=%s' % (acc,f1, precision, recall, result.best_score_, result.best_params_))
	print("This inner iteration has taken so many seconds: " + str(t1_inner-t0_inner))

	#feature importance: compute SHAP values
	#explainer = shap.Explainer(best_model)
	#shap_values = explainer.shap_values(X_test)
	# for each iteration we save the test_set index and the shap_values
	#list_shap_values.append(shap_values)
	#list_test_sets.append(test_ix)
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

#region Pipeline 3: using LOSOCV
t0 = time.time()
from sklearn.metrics import confusion_matrix

df_class_allfeatures_del_NaN_binary_shuffled = load_obj("df_class_allfeatures_del_NaN_binary_shuffled_includingProband.pkl")
data = df_class_allfeatures_del_NaN_binary_shuffled.copy()

# drop Proband 16 as it is a duplicate of Proband 15
data = data[data["Proband"]!= "AGENDER16"]

#select only probands which have 5 or more stress events

probands_above_5events = ["AGENDER01", "AGENDER02", "AGENDER04", "AGENDER05", "AGENDER06",
						  "AGENDER09", "AGENDER14", "AGENDER15"]

data_new = pd.DataFrame()
for i in probands_above_5events:
	data_new = data_new.append(data[data["Proband"]==i])

data = data_new.reset_index(drop=True)


#which columns to drop (either with ACC or without ACC)
dropcols = []

# Make list of all ID's in idcolumn
IDlist = set(data["Proband"])

#initialize
test_proband = list()
outer_results_acc = list()
outer_results_f1 = list()
outer_results_precision = list()
outer_results_recall = list()

#for loop to iterate through LOSOCV "rounds"


for i in IDlist:
	t0_inner = time.time()
	LOOCV_O = str(i)
	data["Proband"] = data["Proband"].apply(str)
	data_filtered = data[data["Proband"] != LOOCV_O]
	data_cv = data[data["Proband"] == LOOCV_O]

	# define Test data - the person left out of training
	data_test = data_cv.drop(columns=dropcols)
	X_test_df = data_test.drop(columns=["Label", "Proband"])
	X_test = np.array(X_test_df)
	y_test_df = data_test["Label"]  # This is the outcome variable
	y_test = np.array(y_test_df)

	# define Train data - all other people in dataframe
	data_train = data_filtered.drop(columns=dropcols)
	X_train_df = data_train.copy()

	#define the model
	model = RandomForestClassifier(random_state=1)

	#define list of indices for inner CV (use again LOSOCV with the remaining subjects)
	IDlist_inner = list(set(X_train_df["Proband"]))
	inner_idxs = []

	X_train_df = X_train_df.reset_index(drop=True)
	for l in range(0, len(IDlist_inner), 3):
		try:
			IDlist_inner[l+2]
		except:
			continue
		else:
			train = X_train_df[(X_train_df["Proband"] != IDlist_inner[l]) & (X_train_df["Proband"] != IDlist_inner[l+1]) & (X_train_df["Proband"] != IDlist_inner[l+2])]
			test = X_train_df[(X_train_df["Proband"] == IDlist_inner[l]) | (X_train_df["Proband"] ==  IDlist_inner[l+1]) | (X_train_df["Proband"] ==  IDlist_inner[l+2])]
			add = [train.index, test.index]
			inner_idxs.append(add)

	data_train = data_train.drop(columns=["Proband"]) #drop Proband column
	X_train_df = X_train_df.drop(columns=["Label", "Proband"])
	X_train = np.array(X_train_df)
	y_train_df = data_train["Label"]
	y_train_df = y_train_df.reset_index(drop=True)
	y_train = np.array(y_train_df)  # Outcome variable here

	# define search space
	n_estimators = [100, 300, 500, 800, 1200]
	max_depth = [5, 8, 15, 25, 30]
	min_samples_split = [2, 5, 10, 15, 100]
	min_samples_leaf = [1, 2, 5, 10]
	max_features = ["sqrt", "log2", 3]

	space = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
				 min_samples_leaf=min_samples_leaf, max_features=max_features)

	# define search
	search = GridSearchCV(model, space, scoring='accuracy', cv=inner_idxs, refit=True, n_jobs=-1)

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
	test_proband.append(i)
	outer_results_acc.append(acc)
	outer_results_f1.append(f1)
	outer_results_precision.append(precision)
	outer_results_recall.append(recall)
	# report progress
	t1_inner = time.time()
	print('>acc=%.3f,f1=%.3f,precision=%.3f,recall=%.3f, est=%.3f, cfg=%s' % (acc, f1, precision, recall, result.best_score_, result.best_params_))
	print("The proband taken as test-data for this iteration was " + str(i))

	#Visualize Confusion Matrix
	mat = confusion_matrix(y_test, yhat)
	sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
	plt.title('Confusion Matrix where Proband ' + str(i) + " was test-data" )
	plt.xlabel('true label')
	plt.ylabel('predicted label')
	plt.show()

	print("This inner iteration has taken so many seconds: " + str(t1_inner - t0_inner))
t1 = time.time()
print("The whole process has taken so many seconds: " + str(t1 - t0))

# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results_acc), std(outer_results_acc)))
print('F1: %.3f (%.3f)' % (mean(outer_results_f1), std(outer_results_f1)))
print('Precision: %.3f (%.3f)' % (mean(outer_results_precision), std(outer_results_precision)))
print('Recall: %.3f (%.3f)' % (mean(outer_results_recall), std(outer_results_recall)))
os.system("shutdown /h") #hibernate






# Add idcolumn to the dropped columns
drop = [idcolumn]
drop = drop + dropcols

# Initialize empty lists and dataframe
errors = []
rmse = []
mape = []
importances = pd.DataFrame(columns=['value', 'importances', 'id'])









# configure the cross-validation procedure
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# enumerate splits
outer_results_acc = list()
outer_results_f1 = list()
outer_results_precision = list()
outer_results_recall = list()

list_shap_values = list()
list_test_sets = list()
columns = X_all_df.columns

for train_ix, test_ix in cv_outer.split(X_all, y_all):
	t0_inner = time.time()
	# split data
	X_train, X_test = X_all[train_ix, :], X_all[test_ix, :]
	y_train, y_test = y_all[train_ix], y_all[test_ix]
	# configure the cross-validation procedure
	cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
	# define the model
	model = RandomForestClassifier(random_state=1)

	# define search space
	n_estimators = [100, 300, 500, 800, 1200]
	max_depth = [5, 8, 15, 25, 30]
	min_samples_split = [2, 5, 10, 15, 100]
	min_samples_leaf = [1, 2, 5, 10]
	max_features = ["sqrt", "log2", 3]

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
	print('>acc=%.3f,f1=%.3f,precision=%.3f,recall=%.3f, est=%.3f, cfg=%s' % (acc,f1, precision, recall, result.best_score_, result.best_params_))
	print("This inner iteration has taken so many seconds: " + str(t1_inner-t0_inner))

	#feature importance: compute SHAP values
	#explainer = shap.Explainer(best_model)
	#shap_values = explainer.shap_values(X_test)
	# for each iteration we save the test_set index and the shap_values
	#list_shap_values.append(shap_values)
	#list_test_sets.append(test_ix)
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

#region Pipeline 4: using LOSOCV & permutation test
t0 = time.time()

df_class_allfeatures_del_NaN_binary_shuffled = load_obj("df_class_allfeatures_del_NaN_binary_shuffled_includingProband.pkl")
data = df_class_allfeatures_del_NaN_binary_shuffled.copy()

# drop Proband 16 as it is a duplicate of Proband 15
data = data[data["Proband"]!= "AGENDER16"]


#select only probands which have 5 or more stress events
#probands_above_5events = ["AGENDER01", "AGENDER02", "AGENDER04", "AGENDER05", "AGENDER06",
#						  "AGENDER09", "AGENDER14", "AGENDER15"]

data_new = pd.DataFrame()
#for i in probands_above_5events:
#	data_new = data_new.append(data[data["Proband"]==i])



#select only probands which have accuracy less than 50%
probands_below_50= ["AGENDER05", "AGENDER09", "AGENDER14", "AGENDER29"]
for i in probands_below_50:
	data_new = data_new.append(data[data["Proband"]==i])

data = data_new.reset_index(drop=True)
#data = data.reset_index(drop=True)

#which columns to drop (either with ACC or without ACC)
dropcols = []
data = data.drop(columns=dropcols)

# Make list of all ID's in idcolumn
IDlist = set(data["Proband"])

#initialize
test_proband = list()
outer_results_acc = list()
outer_results_f1 = list()
outer_results_precision = list()
outer_results_recall = list()

permutation_pvalue = list()
permutation_modelaccuracy = list()
pvalues_binomial = list()

#for loop to iterate through LOSOCV "rounds"

for i in IDlist:
	t0_inner = time.time()
	LOOCV_O = str(i)
	data["Proband"] = data["Proband"].apply(str)
	data_filtered = data[data["Proband"] != LOOCV_O]
	data_cv = data[data["Proband"] == LOOCV_O]

	# define Test data - the person left out of training
	data_test = data_cv.copy()
	X_test_df = data_test.drop(columns=["Label", "Proband"])
	X_test = np.array(X_test_df)
	y_test_df = data_test["Label"]  # This is the outcome variable
	y_test = np.array(y_test_df)

	# define Train data - all other people in dataframe
	data_train = data_filtered.copy()
	X_train_df = data_train.copy()

	#define the model
	model = RandomForestClassifier(random_state=1)

	#define list of indices for inner CV (use again LOSOCV with the remaining subjects)
	IDlist_inner = list(set(X_train_df["Proband"]))
	inner_idxs = []

	X_train_df = X_train_df.reset_index(drop=True)
	for l in range(0, len(IDlist_inner), 2):
		try:
			IDlist_inner[l+1]
		except:
			continue
		else:
			train = X_train_df[(X_train_df["Proband"] != IDlist_inner[l]) & (X_train_df["Proband"] != IDlist_inner[l+1]) ]
			test = X_train_df[(X_train_df["Proband"] == IDlist_inner[l]) | (X_train_df["Proband"] ==  IDlist_inner[l+1]) ]
			add = [train.index, test.index]
			inner_idxs.append(add)

	data_train = data_train.drop(columns=["Proband"]) #drop Proband column
	X_train_df = X_train_df.drop(columns=["Label", "Proband"])
	X_train = np.array(X_train_df)
	y_train_df = data_train["Label"]
	y_train_df = y_train_df.reset_index(drop=True)
	y_train = np.array(y_train_df)  # Outcome variable here

	# define search space
	n_estimators = [100, 300, 500, 800, 1200]
	max_depth = [5, 8, 15, 25, 30]
	min_samples_split = [2, 5, 10, 15, 100]
	min_samples_leaf = [1, 2, 5, 10]
	max_features = ["sqrt", "log2", 3]

	space = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
				 min_samples_leaf=min_samples_leaf, max_features=max_features)

	# define search
	search = GridSearchCV(model, space, scoring='accuracy', cv=inner_idxs, refit=True, n_jobs=-1)

	# execute search
	result = search.fit(X_train, y_train)

	# get the best performing model fit on the whole training set
	best_model = result.best_estimator_

	#apply permutation test
	## create dataframe which contains all data and delete some stuff
	data_permutation = data.copy()
	data_permutation = data_permutation.reset_index(drop=True)

	## create list which contains indices of train and test samples (differentiate by proband)
	split_permutation = []
	train_permutation = data_permutation[data_permutation["Proband"] != i]
	test_permutation = data_permutation[data_permutation["Proband"] == i]
	add_permutation = [train_permutation.index, test_permutation.index]
	split_permutation.append(add_permutation)

	##Drop some stuff
	#data_permutation = data_permutation.drop(columns=dropcols)

	##Create X and y dataset
	X_permutation = data_permutation.drop(columns=["Label", "Proband"])
	y_permutation = data_permutation["Label"]

	##compute permutation test
	score_model, perm_scores_model, pvalue_model = permutation_test_score(best_model, X_permutation, y_permutation, scoring="accuracy", cv=split_permutation, n_permutations=1000)

	## visualize permutation test restuls
	fig, ax = plt.subplots()
	plt.title("Permutation Test results with Proband " + str(i) + " as Test-Data")
	ax.hist(perm_scores_model, bins=20, density=True)
	ax.axvline(score_model, ls='--', color='r')
	score_label = (f"Score on original\ndata: {score_model:.2f}\n"
				   f"(p-value: {pvalue_model:.3f})")
	ax.text(0.14, 125, score_label, fontsize=12)
	ax.set_xlabel("Accuracy score")
	ax.set_ylabel("Probability")
	plt.savefig('02_Analysis/LOSOCV_statisticaltests/DF_withACC_probandsbelow50%_proband' + str(i) + 'astest_Permutation.png')
	plt.show()

	# evaluate model on the hold out dataset
	yhat = best_model.predict(X_test)

	# evaluate the model
	acc = accuracy_score(y_test, yhat)
	f1 = f1_score(y_true=y_test, y_pred=yhat, pos_label=1, average="binary")
	precision = precision_score(y_true=y_test, y_pred=yhat, pos_label=1, average="binary")
	recall = recall_score(y_true=y_test, y_pred=yhat, pos_label=1, average="binary")

	#Visualize Confusion Matrix
	mat = confusion_matrix(y_test, yhat)
	sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
	plt.title('Confusion Matrix where Proband ' + str(i) + " was test-data" )
	plt.xlabel('true label')
	plt.ylabel('predicted label')
	plt.show()

	#apply binomial test
	pvalue_binom = binom_test(x=mat[0][0]+mat[1][1], n=len(y_test), p=0.5, alternative='greater')

	#feature importance: compute SHAP values
	explainer = shap.Explainer(best_model)
	shap_values = explainer.shap_values(X_test)
	plt.figure()
	plt.title("Feature Importance for iteration with proband " +str(i) + " as test set")
	shap.summary_plot(shap_values[1], X_test_df, plot_type="bar", show=False)
	#plt.savefig('02_Analysis/LOSOCV_statisticaltests/DF_withACC_probandsbelow50%_proband' + str(i) + 'astest_SHAPFeatureImportance_v2.png')
	fig = plt.gcf()
	fig.savefig('02_Analysis/LOSOCV_statisticaltests/DF_withACC_probandsbelow50%_proband' + str(i) + 'astest_SHAPFeatureImportance.png')
	plt.show()

	# store statistical test results (p-value permutation test, accuracy of that permutation iteration, pvalue binomial test) in list
	permutation_pvalue.append(pvalue_model)
	permutation_modelaccuracy.append(score_model)
	pvalues_binomial.append(pvalue_binom)

	# store the result
	test_proband.append(i)
	outer_results_acc.append(acc)
	outer_results_f1.append(f1)
	outer_results_precision.append(precision)
	outer_results_recall.append(recall)

	# Save the results:
	results = pd.DataFrame()
	results["Test-Proband"] = test_proband
	results["Accuracy"] = outer_results_acc
	results["Accuracy by PermutationTest"] = permutation_modelaccuracy
	results["F1"] = outer_results_f1
	results["Precision"] = outer_results_precision
	results["Recall"] = outer_results_recall
	results["P-Value Permutation Test"] = permutation_pvalue
	results["P-Value Binomial Test"] = pvalues_binomial
	results.to_csv("02_Analysis/LOSOCV_statisticaltests/DF_withACC_probandsbelow50%_proband' + str(i) + 'astest_results_cumulated.csv.")

	# report progress
	t1_inner = time.time()
	print('>acc=%.3f,f1=%.3f,precision=%.3f,recall=%.3f, est=%.3f, cfg=%s' % (acc, f1, precision, recall, result.best_score_, result.best_params_))
	print("Permutation-Test p-value was " + str(pvalue_model) + " and Binomial Test p-values was " + str(pvalue_binom))
	print("The proband taken as test-data for this iteration was " + str(i))
	print("This inner iteration has taken so many minutes: " + str((t1_inner - t0_inner)/60))

t1 = time.time()
print("The whole process has taken so many hours: " + str(((t1 - t0)/60/60)))

# summarize the estimated performance of the model
print('Mean Accuracy: %.3f (%.3f)' % (mean(outer_results_acc), std(outer_results_acc)))
print('Mean F1: %.3f (%.3f)' % (mean(outer_results_f1), std(outer_results_f1)))
print('Mean Precision: %.3f (%.3f)' % (mean(outer_results_precision), std(outer_results_precision)))
print('Mean Recall: %.3f (%.3f)' % (mean(outer_results_recall), std(outer_results_recall)))
print("Mean p-value of Permutation Test: %.3f (%.3f)" % (mean(permutation_pvalue), std(permutation_pvalue)))
print("Mean of p-value of Binomial Test: %.3f (%.3f)" % (mean(pvalues_binomial), std(pvalues_binomial)))


os.system("shutdown /h") #hibernate












#endregion



#region Testrun for checking errors

df_check = df_class_allfeatures_del_NaN_binary.sample(frac=1) #shuffle data
df_check1 = df_check.reset_index(drop=True) #reset index

#split data into training and test data
split_percentage = 0.1
df_train, df_test = split_ensemble(df_class_allfeatures_del_NaN_binary_shuffled, split_percentage)

#df_train = df_train.sample(frac=1).reset_index(drop=True)
#df_test = df_test.sample(frac=1)

X_acc_train_df = df_train.drop("Label", axis = 1)
X_train_df = df_train.drop(["Label", "x1", "x2", "x3"], axis = 1)
Y_train_df = df_train["Label"]

X_acc_test_df = df_test.drop("Label", axis = 1)
X_test_df = df_test.drop(["Label", "x1", "x2", "x3"], axis = 1)
Y_test_df = df_test["Label"]
X_train = X_acc_train_df
y_train = Y_train_df
X_test = X_acc_test_df
y_test = Y_test_df

clf = RandomForestClassifier(random_state = 1, max_depth = 15,
							 n_estimators = 100, min_samples_split = 2,
							 min_samples_leaf = 1).fit(X_train, y_train)
y_pred=clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#endregion
#endregion



#region Trial03: SVM
#region testrun
from sklearn import datasets
cancer = datasets.load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109) # 70% training and 30% test
#Create a svm Classifier
clf = SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
#endregion


#region get data
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

#get data
df_class_allfeatures_del_NaN_binary_shuffled = load_obj("df_class_allfeatures_del_NaN_binary_shuffled_ONLYPREDICTIVE.pkl")

X_acc = df_class_allfeatures_del_NaN_binary_shuffled.drop(["Label"], axis = 1)
y_acc = df_class_allfeatures_del_NaN_binary_shuffled["Label"]
X_acc_train_df, X_acc_test_df, y_acc_train_df, y_acc_test_df = train_test_split(X_acc, y_acc, test_size=0.1, random_state=3)

X_acc_train_array = X_acc_train_df.to_numpy()
X_acc_test_array = X_acc_test_df.to_numpy()
y_acc_train_array = y_acc_train_df.to_numpy()
y_acc_test_array = y_acc_test_df.to_numpy()

X = df_class_allfeatures_del_NaN_binary_shuffled.drop(["Label", "x1", "x2", "x3"], axis = 1)
y = df_class_allfeatures_del_NaN_binary_shuffled["Label"]
X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X, y, test_size=0.1, random_state=3)

X_train_array = X_train_df.to_numpy()
X_test_array = X_test_df.to_numpy()
y_train_array = y_train_df.to_numpy()
y_test_array = y_test_df.to_numpy()

X_acc_all_df = df_class_allfeatures_del_NaN_binary_shuffled.drop("Label", axis = 1)
X_all_df = df_class_allfeatures_del_NaN_binary_shuffled.drop(["Label", "x1", "x2", "x3"], axis = 1)
y_all_df = df_class_allfeatures_del_NaN_binary_shuffled["Label"]


#Standardize
#Maybe scale the data (in example data was scaled)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_scaled=sc_X.fit_transform(X_all_df)
#X_scaled_test=sc_X.transform(X_acc_test_array)

#Choose data for this iteration
#X_train = X_scaled
#y_train = y_all_df
#X_test = X_scaled_test
#y_test = y_acc_test_df

X_all = X_scaled
y_all = y_all_df.to_numpy()
X_all_df = X_all_df
#endregion

#region Pipeline 01: normal CV & feature visualization
# Hyperparameter Tuning: GridSearch
params_grid = [{'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
				'gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001],
				'C': [0.001, 0.1, 1, 10, 100]}]

#Linear 1 1000 = 13 minuten
# Linear 1 100 = 1.5 minutes 7
# poly 100 0.001 = 2:20 5
# poly 100 100 = 2 minutes 

grid_svm = GridSearchCV(SVC(), params_grid, cv=5, n_jobs = -1)
grid_svm.fit(X_train, y_train) #fitting model for grid search
### print best parameter after tuning
print(grid_svm.best_params_)

### print how our model looks after hyper-parameter tuning
print(grid_svm.best_estimator_)

#Fit models
#With 10 fold CV
from sklearn.model_selection import cross_val_score
svm_cv = SVC(kernel='rbf', gamma=0.1, C=100)
accuracy = cross_val_score(svm_cv, X_train, y_train, scoring= "accuracy", cv=10)
accuracy.mean()
f1 = cross_val_score(svm_cv, X_train, y_train, scoring= "f1", cv=10)
f1.mean()
precision = cross_val_score(svm_cv, X_train, y_train, scoring= "precision", cv=10)
precision.mean()
recall = cross_val_score(svm_cv, X_train, y_train, scoring= "recall", cv=10)
recall.mean()


##Without CV
svm = SVC(kernel='linear', gamma=100, C=10).fit(X_train, y_train)

### print classification report
from sklearn.metrics import classification_report
predictions = svm.predict(X_test)
print(classification_report(y_test, predictions))

#Predict

# creating a confusion matrix
svm_pred = svm.predict(X_test)
cm = confusion_matrix(y_test, svm_pred)
sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

#Model Analysis/Feature Importance
pd.Series(abs(svm.coef_[0]), index=X_acc_train_df.columns).nlargest(17).plot(kind='barh')
plt.show()

#Feature Importance Analysis
#modelname.feature_importance_
data = np.array([svm.coef_[0], X_train.columns]).T
data = pd.DataFrame(data, columns=['Importance', 'Feature']).sort_values("Importance", ascending=False).head(10)
columns = list(data.Feature.values)
#plot
fig, ax = plt.subplots()
width = 0.4 # the width of the bars
ind = np.arange(data.shape[0]) # the x locations for the groups
ax.barh(ind, data["Importance"], width, color="green")
ax.set_yticks(ind+width/10)
ax.set_yticklabels(columns, minor=False)
plt.title("Feature importance in RandomForest Classifier")
plt.xlabel("Relative importance")
plt.ylabel("feature")
plt.figure(figsize=(5,5))
fig.set_size_inches(6.5, 4.5, forward=True)
plt.show()

#endregion


#region Pipeline 2: using nested crossvalidation & feature interpretation with SHAP!

#region nested CV
t0 = time.time()
# configure the cross-validation procedure
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# enumerate splits
outer_results_acc = list()
outer_results_f1 = list()
outer_results_precision = list()
outer_results_recall = list()

list_shap_values = list()
list_test_sets = list()
columns = X_all_df.columns
counter = 0
outer_split = cv_outer.split(X_all, y_all)
outer_split = list(outer_split)
save_obj(outer_split, "outer_split")

for train_ix, test_ix in outer_split:
	t0_inner = time.time()
	t0_inner_CV = time.time()
	# split data
	X_train, X_test = X_all[train_ix, :], X_all[test_ix, :]
	y_train, y_test = y_all[train_ix], y_all[test_ix]
	# configure the cross-validation procedure
	cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
	# define the model
	model = SVC()

	# define search space
	space = [{'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
			  'gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001],
			  'C': [0.001, 0.1, 1, 10, 100]}]
	#space = [{'kernel': ['linear'],
	#		  'gamma': [1],
	#		  'C': [0.001]}]
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

	#save intermediate results
	save_obj(outer_results_acc, "outer_results_acc")
	save_obj(outer_results_f1, "outer_results_f1")
	save_obj(outer_results_precision, "outer_results_precision")
	save_obj(outer_results_recall, "outer_results_recall")
	# report progress
	print('>acc=%.3f,f1=%.3f,precision=%.3f,recall=%.3f, est=%.3f, cfg=%s' % (acc,f1, precision, recall, result.best_score_, result.best_params_))
	t1_inner_CV = time.time()
	print("Inner CV has taken so many seconds: " + str(t1_inner_CV-t0_inner_CV))
	#feature importance: compute SHAP values
	#explainer = shap.KernelExplainer(best_model.predict, X_test)
	#shap_values = explainer.shap_values(X_test, nsamples = 100)
	# for each iteration we save the test_set index and the shap_values
	#list_shap_values.append(shap_values)
	#list_test_sets.append(test_ix)
	# save intermediates
	#save_obj(list_shap_values, "list_shap_values")
	#save_obj(list_test_sets, "list_test_sets")
	t1_inner = time.time()
	print("The time of this inner iteration was" + str(t1_inner-t0_inner))
	counter = counter + 1
	save_obj(counter, "counter")
#combining SHAP results from all iterations
#test_set = []
#shap_values = []
t1 = time.time()
print("This process took so many seconds: " +str(t1-t0))
os.system("shutdown /h") #hibernate

#region to pick up on old progress
#load old progress:
outer_split = load_obj("outer_split.pkl")
outer_results_acc = load_obj("outer_results_acc.pkl")
outer_results_f1 = load_obj("outer_results_acc.pkl")
outer_results_precision = load_obj("outer_results_precision.pkl")
outer_results_recall = load_obj("outer_results_recall.pkl")
list_shap_values = load_obj("list_shap_values.pkl")
list_test_sets = load_obj("list_test_sets.pkl")
counter = load_obj("counter.pkl")

#start again from where I stopped
counter2 = 0
for train_ix, test_ix in outer_split:
	if counter2 <= counter:
		counter2 = counter2+1
		continue
	t0_inner = time.time()
	t0_inner_CV = time.time()
	# split data
	X_train, X_test = X_all[train_ix, :], X_all[test_ix, :]
	y_train, y_test = y_all[train_ix], y_all[test_ix]
	# configure the cross-validation procedure
	cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
	# define the model
	model = SVC()

	# define search space
	space = [{'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
			  'gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001],
			  'C': [0.001, 0.1, 1, 10, 100]}]
	#space = [{'kernel': ['linear'],
	#		  'gamma': [1],
	#		  'C': [0.001]}]
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
	print('>acc=%.3f,f1=%.3f,precision=%.3f,recall=%.3f, est=%.3f, cfg=%s' % (acc,f1, precision, recall, result.best_score_, result.best_params_))
	t1_inner_CV = time.time()
	print("Inner CV has taken so many seconds: " + str(t1_inner_CV-t0_inner_CV))
	#feature importance: compute SHAP values
	explainer = shap.KernelExplainer(best_model.predict, X_test)
	shap_values = explainer.shap_values(X_test, nsamples = 100)
	# for each iteration we save the test_set index and the shap_values
	list_shap_values.append(shap_values)
	list_test_sets.append(test_ix)
	t1_inner = time.time()
	print("The time of this inner iteration was" + str(t1_inner-t0_inner))


#endregion



test_set = list_test_sets[0]
shap_values = np.array(list_shap_values[0])

for i in range(1,len(list_test_sets)):
	test_set = np.concatenate((test_set,list_test_sets[i]),axis=0)
	shap_values = np.concatenate((shap_values,np.array(list_shap_values[i])), axis=0)
#bringing back variable names: X_test is the X_all data reordered due to the 10-test sets in outer CV
test_set = test_set.astype(int)
X_test = pd.DataFrame(X_all[test_set],columns = columns) #this should be used for SHAP plots!


# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results_acc), std(outer_results_acc)))
print('F1: %.3f (%.3f)' % (mean(outer_results_f1), std(outer_results_f1)))
print('Precision: %.3f (%.3f)' % (mean(outer_results_precision), std(outer_results_precision)))
print('Recall: %.3f (%.3f)' % (mean(outer_results_recall), std(outer_results_recall)))


save_obj(shap_values, "normalstress_shap_values_SVM_withACC")
save_obj(X_test, "normalstress_X_test_SVM_withACC")
#shap_values = load_obj("shap_values_DecisionForest_withACC.pkl")
#X_test = load_obj("X_test_DecisionForest_withACC.pkl")

#endregion

#region SHAP Feature interpretation

shap_values = load_obj("normalstress_shap_values_SVM_withACC.pkl")
X_test = load_obj("normalstress_X_test_SVM_withACC.pkl")

# feature importance plot
shap.summary_plot(shap_values, X_test, plot_type="bar")

#SHAP feature importance plot
shap.summary_plot(shap_values, X_test)


#Force Plot
#shap.plots.force(explainer.expected_value[0], shap_values[0])

#partial dependence plot: analyze individual features
shap.dependence_plot("HRV_MeanNN", shap_values, X_test)

#endregion
#endregion

#region Pipeline 3: using LOSOCV
t0 = time.time()
from sklearn.metrics import confusion_matrix

df_class_allfeatures_del_NaN_binary_shuffled = load_obj("df_class_allfeatures_del_NaN_binary_shuffled_includingProband.pkl")
data = df_class_allfeatures_del_NaN_binary_shuffled.copy()

# drop Proband 16 as it is a duplicate of Proband 15
data = data[data["Proband"]!= "AGENDER16"]

#select only probands which have 5 or more stress events

probands_above_5events = ["AGENDER01", "AGENDER02", "AGENDER04", "AGENDER05", "AGENDER06",
						  "AGENDER09", "AGENDER14", "AGENDER15"]

data_new = pd.DataFrame()
for i in probands_above_5events:
	data_new = data_new.append(data[data["Proband"]==i])

data = data_new.reset_index(drop=True)

#which columns to drop (either with ACC or without ACC)
dropcols = []

# Make list of all ID's in idcolumn
IDlist = set(data["Proband"])

#initialize
outer_results_acc = list()
outer_results_f1 = list()
outer_results_precision = list()
outer_results_recall = list()

#for loop to iterate through LOSOCV "rounds"


for i in IDlist:
	t0_inner = time.time()
	LOOCV_O = str(i)
	data["Proband"] = data["Proband"].apply(str)
	data_filtered = data[data["Proband"] != LOOCV_O]
	data_cv = data[data["Proband"] == LOOCV_O]

	# define Test data - the person left out of training
	data_test = data_cv.drop(columns=dropcols)
	X_test_df = data_test.drop(columns=["Label", "Proband"])
	X_test = np.array(X_test_df)
	y_test_df = data_test["Label"]  # This is the outcome variable
	y_test = np.array(y_test_df)

	# define Train data - all other people in dataframe
	data_train = data_filtered.drop(columns=dropcols)
	X_train_df = data_train.copy()

	#define list of indices for inner CV (use again LOSOCV with the remaining subjects)
	IDlist_inner = list(set(X_train_df["Proband"]))
	inner_idxs = []

	X_train_df = X_train_df.reset_index(drop=True)
	for l in range(0, len(IDlist_inner), 4):
		try:
			IDlist_inner[l+3]
		except:
			continue
		else:
			train = X_train_df[(X_train_df["Proband"] != IDlist_inner[l]) & (X_train_df["Proband"] != IDlist_inner[l+1]) & (X_train_df["Proband"] != IDlist_inner[l+2]) & (X_train_df["Proband"] != IDlist_inner[l+3])]
			test = X_train_df[(X_train_df["Proband"] == IDlist_inner[l]) | (X_train_df["Proband"] ==  IDlist_inner[l+1]) | (X_train_df["Proband"] ==  IDlist_inner[l+2]) | (X_train_df["Proband"] ==  IDlist_inner[l+3])]
			add = [train.index, test.index]
			inner_idxs.append(add)

	data_train = data_train.drop(columns=["Proband"]) #drop Proband column
	X_train_df = X_train_df.drop(columns=["Label", "Proband"])
	X_train = np.array(X_train_df)
	y_train_df = data_train["Label"]
	y_train_df = y_train_df.reset_index(drop=True)
	y_train = np.array(y_train_df)  # Outcome variable here

	# define the model
	model = SVC()

	# define search space
	space = [{'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
			  'gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001],
			  'C': [0.001, 0.1, 1, 10, 100]}]

	# define search
	search = GridSearchCV(model, space, scoring='accuracy', cv=inner_idxs, refit=True, n_jobs=-1)

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
	print('>acc=%.3f,f1=%.3f,precision=%.3f,recall=%.3f, est=%.3f, cfg=%s' % (
	acc, f1, precision, recall, result.best_score_, result.best_params_))
	print("The proband taken as test-data for this iteration was " + str(i))

	# Visualize Confusion Matrix
	mat = confusion_matrix(y_test, yhat)
	sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
	plt.title('Confusion Matrix where Proband ' + str(i) + " was test-data")
	plt.xlabel('true label')
	plt.ylabel('predicted label')
	plt.show()

	print("This inner iteration has taken so many seconds: " + str(t1_inner - t0_inner))
t1 = time.time()
print("The whole process has taken so many seconds: " + str(t1 - t0))

# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results_acc), std(outer_results_acc)))
print('F1: %.3f (%.3f)' % (mean(outer_results_f1), std(outer_results_f1)))
print('Precision: %.3f (%.3f)' % (mean(outer_results_precision), std(outer_results_precision)))
print('Recall: %.3f (%.3f)' % (mean(outer_results_recall), std(outer_results_recall)))
os.system("shutdown /h") #hibernate


#endregion
#endregion


#region Trial04: Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, make_scorer

#get data
df_class_allfeatures_del_NaN_binary_shuffled = load_obj("df_class_allfeatures_del_NaN_binary_shuffled.pkl")

X_acc = df_class_allfeatures_del_NaN_binary_shuffled.drop(["Label"], axis = 1)
y_acc = df_class_allfeatures_del_NaN_binary_shuffled["Label"]
X_acc_train_df, X_acc_test_df, y_acc_train_df, y_acc_test_df = train_test_split(X_acc, y_acc, test_size=0.1, random_state=3)

X = df_class_allfeatures_del_NaN_binary_shuffled.drop(["Label", "x1", "x2", "x3"], axis = 1)
y = df_class_allfeatures_del_NaN_binary_shuffled["Label"]
X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X, y, test_size=0.1, random_state=3)

X_acc_all_df = df_class_allfeatures_del_NaN_binary_shuffled.drop("Label", axis = 1)
X_all_df = df_class_allfeatures_del_NaN_binary_shuffled.drop(["Label", "x1", "x2", "x3"], axis = 1)
y_all_df = df_class_allfeatures_del_NaN_binary_shuffled["Label"]

#Choose data for this iteration
X_train = X_all_df
y_train = y_all_df
X_test = X_acc_test_df
y_test = y_acc_test_df


clfdt = DecisionTreeClassifier()
#clfdt = clfdt.fit(x_train, Y_train)

#fitting the DecisionTreeClassifer to our training data
#y_pred_train = clfdt.predict(X_train)
#We predict the training set

#y_pred_test = clfdt.predict(X_test)
#predict the response for test data

#Hyperparameter Tuning
### parameters={'max_depth': range(2,20,1),'max_features': range(1,96,5) }

# will test max_depth starting from 2 and increasing by 1 until the max_depth is 20
criterion = ["gini", "entropy" ]
max_depth = range(2, 20, 1)
max_features = range(1, 17, 2)
min_samples_leaf = range(1, 10, 1)
min_samples_split = range(1,10,1)

parameters = dict(criterion = criterion ,  max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split = min_samples_split)
#f1 = make_scorer(f1_score, average='weighted')

grid_tree = GridSearchCV(clfdt, parameters, verbose=5, cv=10, scoring="accuracy", n_jobs = -1)
grid_tree.fit(X_train, y_train)

# Single best score achieved across all params (min_samples_split)
print(grid_tree.best_score_)

# Dictionary containing the parameters (min_samples_split) used to generate that score
print(grid_tree.best_params_)

grid_tree.best_estimator_


#Run Decision Tree
#with 10-fold CV
from sklearn.model_selection import cross_val_score
clf_pruned = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=18, max_features=15, min_samples_leaf=1, min_samples_split=2)
accuracy = cross_val_score(clf_pruned, X_train, y_train, scoring= "accuracy", cv=10)
accuracy.mean()
f1 = cross_val_score(clf_pruned, X_train, y_train, scoring= "f1", cv=10)
f1.mean()
precision = cross_val_score(clf_pruned, X_train, y_train, scoring= "precision", cv=10)
precision.mean()
recall = cross_val_score(clf_pruned, X_train, y_train, scoring= "recall", cv=10)
recall.mean()

#without CV
clf_pruned = DecisionTreeClassifier(criterion = "gini", random_state = 100,
									max_depth=18, max_features=15, min_samples_leaf=1,
									min_samples_split=2)
clf_pruned.fit(X_train, y_train)
y_pred=clf_pruned.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


#Visualize Confusion Matrix
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

#Feature Importance: Splitting criteria
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
export_graphviz(clf_pruned, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = X.columns,class_names=['2','3'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('predicted labels outcomes.png')
Image(graph.create_png())

#Feature Importance: Feature Importance
clf_pruned.tree_.compute_feature_importances(normalize=False)
feat_imp_dict = dict(zip(X.columns, clf_pruned.feature_importances_))
feat_imp = pd.DataFrame.from_dict(feat_imp_dict, orient='index')
feat_imp.rename(columns = {0:'Feature_Importance'}, inplace = True)
feat_imp.sort_values(by=['Feature_Importance'], ascending=False)


#endregion





#region Trial05: Deep Neural Multilayer Perceptron (MLP)
#Ressources: https://panjeh.medium.com/scikit-learn-hyperparameter-optimization-for-mlpclassifier-4d670413042b

from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

#region getdata
df_class_allfeatures_del_NaN_binary_shuffled = load_obj("df_class_allfeatures_del_NaN_binary_shuffled_EXCEPTPREDICTIVE.pkl")

X_acc = df_class_allfeatures_del_NaN_binary_shuffled.drop(["Label"], axis = 1)
y_acc = df_class_allfeatures_del_NaN_binary_shuffled["Label"]
X_acc_train_df, X_acc_test_df, y_acc_train_df, y_acc_test_df = train_test_split(X_acc, y_acc, test_size=0.1, random_state=3)
y_acc_train_df = y_acc_train_df.to_numpy()
y_acc_test_df = y_acc_test_df.to_numpy()

X = df_class_allfeatures_del_NaN_binary_shuffled.drop(["Label", "x1", "x2", "x3"], axis = 1)
y = df_class_allfeatures_del_NaN_binary_shuffled["Label"]
X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X, y, test_size=0.1, random_state=3)

X_acc_all_df = df_class_allfeatures_del_NaN_binary_shuffled.drop("Label", axis = 1)
X_all_df = df_class_allfeatures_del_NaN_binary_shuffled.drop(["Label", "x1", "x2", "x3"], axis = 1)
y_all_df = df_class_allfeatures_del_NaN_binary_shuffled["Label"]

#Maybe scale the data (in example data was scaled)
sc_X = StandardScaler()
X_scaled=sc_X.fit_transform(X_all_df)
#X_test_scaled=sc_X.transform(X_acc_test_df)


#Choose data for this iteration
#X_train = X_train_scaled
#y_train = y_all_df
#X_test = X_test_scaled
#y_test = y_acc_test_df
X_all = X_scaled
y_all = y_all_df.to_numpy()
X_all_df = X_all_df

#endregion

#region Pipeline 01: "oldfashion" hyperparameter tuning & feature importance analysis
#Hyperparameter Tuning
mlp_gs = MLPClassifier(max_iter=1000)
parameter_space = {
    'hidden_layer_sizes': [(10,30,10),(20,), (256,128,64,32)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=10)
clf.fit(X_train, y_train) # X is train samples and y is the corresponding labels

print('Best parameters found:\n', clf.best_params_)

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


#Model Training
#with cross validation
from sklearn.model_selection import cross_val_score
mlp_cv = MLPClassifier(max_iter=1000, activation= "tanh", alpha= 0.0001,
					   hidden_layer_sizes=(256, 128, 64, 32), learning_rate="adaptive", solver="adam",random_state=1)

accuracy = cross_val_score(mlp_cv, X_train, y_train, scoring= "accuracy", cv=10)
accuracy.mean()
f1 = cross_val_score(mlp_cv, X_train, y_train, scoring= "f1", cv=10)
f1.mean()
precision = cross_val_score(mlp_cv, X_train, y_train, scoring= "precision", cv=10)
precision.mean()
recall = cross_val_score(mlp_cv, X_train, y_train, scoring= "recall", cv=10)
recall.mean()

#Without crossvalidation
mlp = MLPClassifier(max_iter=1000, activation= "tanh", alpha= 0.0001,
					hidden_layer_sizes=(256, 128, 64, 32), learning_rate="constant", solver="adam",random_state=1).fit(X_train, y_train)

y_true, y_pred = y_test , mlp.predict(X_test)
from sklearn.metrics import classification_report
print('Results on the test set:')
print(classification_report(y_true, y_pred))

#other option: build model again based on results of hyperparameter tuning
mlp = MLPClassifier(hidden_layer_sizes=(256,128,64,32),
					activation="tanh", alpha=0.0001 ,
					random_state=1).fit(X_train, y_train)
y_pred=mlp.predict(X_test)
print(clf.score(X_test, y_test))

#Confusion Matrix
fig=plot_confusion_matrix(mlp, X_test, y_test)
fig.figure_.suptitle("Confusion Matrix")
plt.show()

#endregion

#region Pipeline 2: Nested CV & SHAP feature importance analysis

#region nested CV
t0 = time.time()
# configure the cross-validation procedure
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# enumerate splits
outer_results_acc = list()
outer_results_f1 = list()
outer_results_precision = list()
outer_results_recall = list()

list_shap_values = list()
list_test_sets = list()
columns = X_all_df.columns
counter = 0
outer_split = cv_outer.split(X_all, y_all)
outer_split = list(outer_split)
save_obj(outer_split, "outer_split")

for train_ix, test_ix in outer_split:
	t0_inner = time.time()
	t0_inner_CV = time.time()
	# split data
	X_train, X_test = X_all[train_ix, :], X_all[test_ix, :]
	y_train, y_test = y_all[train_ix], y_all[test_ix]
	# configure the cross-validation procedure
	cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
	# define the model
	model = MLPClassifier(max_iter=1000)

	# define search space
	space = {
		'hidden_layer_sizes': [(10,30,10),(20,), (256,128,64,32)],
    	'activation': ['tanh', 'relu'],
    	'solver': ['sgd', 'adam'],
    	'alpha': [0.0001, 0.05],
    	'learning_rate': ['constant','adaptive'],
	}

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

	#save intermediate results
	save_obj(outer_results_acc, "outer_results_acc")
	save_obj(outer_results_f1, "outer_results_f1")
	save_obj(outer_results_precision, "outer_results_precision")
	save_obj(outer_results_recall, "outer_results_recall")
	# report progress
	print('>acc=%.3f,f1=%.3f,precision=%.3f,recall=%.3f, est=%.3f, cfg=%s' % (acc,f1, precision, recall, result.best_score_, result.best_params_))
	t1_inner_CV = time.time()
	print("Inner CV has taken so many seconds: " + str(t1_inner_CV-t0_inner_CV))
	#feature importance: compute SHAP values
	explainer = shap.KernelExplainer(best_model.predict, X_test)
	shap_values = explainer.shap_values(X_test, nsamples = 100)
	# for each iteration we save the test_set index and the shap_values
	list_shap_values.append(shap_values)
	list_test_sets.append(test_ix)
	# save intermediates
	save_obj(list_shap_values, "list_shap_values")
	save_obj(list_test_sets, "list_test_sets")
	t1_inner = time.time()
	print("The time of this inner iteration was" + str(t1_inner-t0_inner))
	counter = counter + 1
	save_obj(counter, "counter")
#combining SHAP results from all iterations
#test_set = []
#shap_values = []
t1 = time.time()
print("This process took so many seconds: " +str(t1-t0))


#region to pick up on old progress
#load old progress:
outer_split = load_obj("outer_split.pkl")
outer_results_acc = load_obj("outer_results_acc.pkl")
outer_results_f1 = load_obj("outer_results_acc.pkl")
outer_results_precision = load_obj("outer_results_precision.pkl")
outer_results_recall = load_obj("outer_results_recall.pkl")
list_shap_values = load_obj("list_shap_values.pkl")
list_test_sets = load_obj("list_test_sets.pkl")
counter = load_obj("counter.pkl")

#start again from where I stopped
counter2 = 0
for train_ix, test_ix in outer_split:
	if counter2 <= counter:
		counter2 = counter2+1
		continue
	t0_inner = time.time()
	t0_inner_CV = time.time()
	# split data
	X_train, X_test = X_all[train_ix, :], X_all[test_ix, :]
	y_train, y_test = y_all[train_ix], y_all[test_ix]
	# configure the cross-validation procedure
	cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
	# define the model
	model = SVC()

	# define search space
	space = [{'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
			  'gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001],
			  'C': [0.001, 0.1, 1, 10, 100]}]
	#space = [{'kernel': ['linear'],
	#		  'gamma': [1],
	#		  'C': [0.001]}]
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
	print('>acc=%.3f,f1=%.3f,precision=%.3f,recall=%.3f, est=%.3f, cfg=%s' % (acc,f1, precision, recall, result.best_score_, result.best_params_))
	t1_inner_CV = time.time()
	print("Inner CV has taken so many seconds: " + str(t1_inner_CV-t0_inner_CV))
	#feature importance: compute SHAP values
	explainer = shap.KernelExplainer(best_model.predict, X_test)
	shap_values = explainer.shap_values(X_test, nsamples = 100)
	# for each iteration we save the test_set index and the shap_values
	list_shap_values.append(shap_values)
	list_test_sets.append(test_ix)
	t1_inner = time.time()
	print("The time of this inner iteration was" + str(t1_inner-t0_inner))


#endregion



test_set = list_test_sets[0]
shap_values = np.array(list_shap_values[0])

for i in range(1,len(list_test_sets)):
	test_set = np.concatenate((test_set,list_test_sets[i]),axis=0)
	shap_values = np.concatenate((shap_values,np.array(list_shap_values[i])),axis=0)
#bringing back variable names: X_test is the X_all data reordered due to the 10-test sets in outer CV
test_set = test_set.astype(int)
X_test = pd.DataFrame(X_all[test_set],columns = columns) #this should be used for SHAP plots!


# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results_acc), std(outer_results_acc)))
print('F1: %.3f (%.3f)' % (mean(outer_results_f1), std(outer_results_f1)))
print('Precision: %.3f (%.3f)' % (mean(outer_results_precision), std(outer_results_precision)))
print('Recall: %.3f (%.3f)' % (mean(outer_results_recall), std(outer_results_recall)))


save_obj(shap_values, "normal_shap_values_MLP_withoutACC_withoutMeanMedianRR")
save_obj(X_test, "normal_X_test_MLP_withoutACC_withoutMeanMedianRR")
#shap_values = load_obj("shap_values_DecisionForest_withACC.pkl")
#X_test = load_obj("X_test_DecisionForest_withACC.pkl")

#endregion

#region SHAP Feature interpretation
# feature importance plot
shap.summary_plot(shap_values, X_test, plot_type="bar")

#SHAP feature importance plot
shap.summary_plot(shap_values, X_test)

#Force Plot
#shap.plots.force(explainer.expected_value[0], shap_values[0])

#partial dependence plot: analyze individual features
shap.dependence_plot("HRV_MeanNN", shap_values, X_test)

#endregion
#endregion

#region Pipeline 3: using LOSOCV
t0 = time.time()
from sklearn.metrics import confusion_matrix

df_class_allfeatures_del_NaN_binary_shuffled = load_obj("df_class_allfeatures_del_NaN_binary_shuffled_includingProband.pkl")
data = df_class_allfeatures_del_NaN_binary_shuffled.copy()

# drop Proband 16 as it is a duplicate of Proband 15
data = data[data["Proband"]!= "AGENDER16"]

#select only probands which have 5 or more stress events

probands_above_5events = ["AGENDER01", "AGENDER02", "AGENDER04", "AGENDER05", "AGENDER06",
						  "AGENDER09", "AGENDER14", "AGENDER15"]

data_new = pd.DataFrame()
for i in probands_above_5events:
	data_new = data_new.append(data[data["Proband"]==i])

data = data_new.reset_index(drop=True)


#which columns to drop (either with ACC or without ACC)
dropcols = []

# Make list of all ID's in idcolumn
IDlist = set(data["Proband"])

#initialize
outer_results_acc = list()
outer_results_f1 = list()
outer_results_precision = list()
outer_results_recall = list()

#for loop to iterate through LOSOCV "rounds"


for i in IDlist:
	t0_inner = time.time()
	LOOCV_O = str(i)
	data["Proband"] = data["Proband"].apply(str)
	data_filtered = data[data["Proband"] != LOOCV_O]
	data_cv = data[data["Proband"] == LOOCV_O]

	# define Test data - the person left out of training
	data_test = data_cv.drop(columns=dropcols)
	X_test_df = data_test.drop(columns=["Label", "Proband"])
	X_test = np.array(X_test_df)
	y_test_df = data_test["Label"]  # This is the outcome variable
	y_test = np.array(y_test_df)

	# define Train data - all other people in dataframe
	data_train = data_filtered.drop(columns=dropcols)
	X_train_df = data_train.copy()

	#define list of indices for inner CV (use again LOSOCV with the remaining subjects)
	IDlist_inner = list(set(X_train_df["Proband"]))
	inner_idxs = []

	X_train_df = X_train_df.reset_index(drop=True)
	for l in range(0, len(IDlist_inner), 4):
		try:
			IDlist_inner[l+3]
		except:
			continue
		else:
			train = X_train_df[(X_train_df["Proband"] != IDlist_inner[l]) & (X_train_df["Proband"] != IDlist_inner[l+1]) & (X_train_df["Proband"] != IDlist_inner[l+2]) & (X_train_df["Proband"] != IDlist_inner[l+3])]
			test = X_train_df[(X_train_df["Proband"] == IDlist_inner[l]) | (X_train_df["Proband"] ==  IDlist_inner[l+1]) | (X_train_df["Proband"] ==  IDlist_inner[l+2]) | (X_train_df["Proband"] ==  IDlist_inner[l+3])]
			add = [train.index, test.index]
			inner_idxs.append(add)

	data_train = data_train.drop(columns=["Proband"]) #drop Proband column
	X_train_df = X_train_df.drop(columns=["Label", "Proband"])
	X_train = np.array(X_train_df)
	y_train_df = data_train["Label"]
	y_train_df = y_train_df.reset_index(drop=True)
	y_train = np.array(y_train_df)  # Outcome variable here

	# define the model
	model = MLPClassifier(max_iter=1000)

	# define search space
	space = {
		'hidden_layer_sizes': [(10, 30, 10), (20,), (256, 128, 64, 32)],
		'activation': ['tanh', 'relu'],
		'solver': ['sgd', 'adam'],
		'alpha': [0.0001, 0.05],
		'learning_rate': ['constant', 'adaptive'],
	}

	# define search
	search = GridSearchCV(model, space, scoring='accuracy', cv=inner_idxs, refit=True, n_jobs=-1)

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
	print('>acc=%.3f,f1=%.3f,precision=%.3f,recall=%.3f, est=%.3f, cfg=%s' % (
	acc, f1, precision, recall, result.best_score_, result.best_params_))
	print("The proband taken as test-data for this iteration was " + str(i))

	# Visualize Confusion Matrix
	mat = confusion_matrix(y_test, yhat)
	sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
	plt.title('Confusion Matrix where Proband ' + str(i) + " was test-data")
	plt.xlabel('true label')
	plt.ylabel('predicted label')
	plt.show()

	print("This inner iteration has taken so many seconds: " + str(t1_inner - t0_inner))
t1 = time.time()
print("The whole process has taken so many seconds: " + str(t1 - t0))

# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results_acc), std(outer_results_acc)))
print('F1: %.3f (%.3f)' % (mean(outer_results_f1), std(outer_results_f1)))
print('Precision: %.3f (%.3f)' % (mean(outer_results_precision), std(outer_results_precision)))
print('Recall: %.3f (%.3f)' % (mean(outer_results_recall), std(outer_results_recall)))
os.system("shutdown /h") #hibernate


#endregion

#region Pipeline 4: LOSOCV and statistical tests (permutation & binomial)
t0 = time.time()
from sklearn.metrics import confusion_matrix

df_class_allfeatures_del_NaN_binary_shuffled = load_obj("df_class_allfeatures_del_NaN_binary_shuffled_includingProband.pkl")
data = df_class_allfeatures_del_NaN_binary_shuffled.copy()

# drop Proband 16 as it is a duplicate of Proband 15
data = data[data["Proband"]!= "AGENDER16"]

#select only probands which have 5 or more stress events

probands_above_5events = ["AGENDER01", "AGENDER02", "AGENDER04", "AGENDER05", "AGENDER06",
						  "AGENDER09", "AGENDER14", "AGENDER15"]

data_new = pd.DataFrame()
for i in probands_above_5events:
	data_new = data_new.append(data[data["Proband"]==i])

data = data_new.reset_index(drop=True)

#which columns to drop (either with ACC or without ACC)
dropcols = []

# Make list of all ID's in idcolumn
IDlist = set(data["Proband"])

#initialize
test_proband = list()
outer_results_acc = list()
outer_results_f1 = list()
outer_results_precision = list()
outer_results_recall = list()

permutation_pvalue = list()
permutation_modelaccuracy = list()
pvalues_binomial = list()
#for loop to iterate through LOSOCV "rounds"


for i in IDlist:
	t0_inner = time.time()
	LOOCV_O = str(i)
	data["Proband"] = data["Proband"].apply(str)
	data_filtered = data[data["Proband"] != LOOCV_O]
	data_cv = data[data["Proband"] == LOOCV_O]

	# define Test data - the person left out of training
	data_test = data_cv.drop(columns=dropcols)
	X_test_df = data_test.drop(columns=["Label", "Proband"])
	X_test = np.array(X_test_df)
	y_test_df = data_test["Label"]  # This is the outcome variable
	y_test = np.array(y_test_df)

	# define Train data - all other people in dataframe
	data_train = data_filtered.drop(columns=dropcols)
	X_train_df = data_train.copy()

	#define list of indices for inner CV (use again LOSOCV with the remaining subjects)
	IDlist_inner = list(set(X_train_df["Proband"]))
	inner_idxs = []

	X_train_df = X_train_df.reset_index(drop=True)
	for l in range(0, len(IDlist_inner), 3):
		try:
			IDlist_inner[l+2]
		except:
			continue
		else:
			train = X_train_df[(X_train_df["Proband"] != IDlist_inner[l]) & (X_train_df["Proband"] != IDlist_inner[l+1]) & (X_train_df["Proband"] != IDlist_inner[l+2])]
			test = X_train_df[(X_train_df["Proband"] == IDlist_inner[l]) | (X_train_df["Proband"] ==  IDlist_inner[l+1]) | (X_train_df["Proband"] ==  IDlist_inner[l+2])]
			add = [train.index, test.index]
			inner_idxs.append(add)

	data_train = data_train.drop(columns=["Proband"]) #drop Proband column
	X_train_df = X_train_df.drop(columns=["Label", "Proband"])
	X_train = np.array(X_train_df)
	y_train_df = data_train["Label"]
	y_train_df = y_train_df.reset_index(drop=True)
	y_train = np.array(y_train_df)  # Outcome variable here

	# define the model
	model = MLPClassifier(max_iter=1000)

	# define search space
	space = {
		'hidden_layer_sizes': [(10, 30, 10), (20,), (256, 128, 64, 32)],
		'activation': ['tanh', 'relu'],
		'solver': ['sgd', 'adam'],
		'alpha': [0.0001, 0.05],
		'learning_rate': ['constant', 'adaptive'],
	}

	# define search
	search = GridSearchCV(model, space, scoring='accuracy', cv=inner_idxs, refit=True, n_jobs=-1)

	# execute search
	result = search.fit(X_train, y_train)

	# get the best performing model fit on the whole training set
	best_model = result.best_estimator_


	#apply permutation test
	## create dataframe which contains all data and delete some stuff
	data_permutation = data.copy()
	data_permutation = data_permutation.reset_index(drop=True)

	## create list which contains indices of train and test samples (differentiate by proband)
	split_permutation = []
	train_permutation = data_permutation[data_permutation["Proband"] != i]
	test_permutation = data_permutation[data_permutation["Proband"] == i]
	add_permutation = [train_permutation.index, test_permutation.index]
	split_permutation.append(add_permutation)

	##Drop some stuff
	data_permutation = data_permutation.drop(columns=dropcols)

	##Create X and y dataset
	X_permutation = data_permutation.drop(columns=["Label", "Proband"])
	y_permutation = data_permutation["Label"]

	##compute permutation test
	score_model, perm_scores_model, pvalue_model = permutation_test_score(
		best_model, X_permutation, y_permutation, scoring="accuracy", cv=split_permutation, n_permutations=1000)

	## visualize permutation test restuls
	fig, ax = plt.subplots()
	plt.title("Permutation Test results with Proband " + str(i) + " as Test-Data")
	ax.hist(perm_scores_model, bins=20, density=True)
	ax.axvline(score_model, ls='--', color='r')
	score_label = (f"Score on original\ndata: {score_model:.2f}\n"
				   f"(p-value: {pvalue_model:.3f})")
	ax.text(0.14, 125, score_label, fontsize=12)
	ax.set_xlabel("Accuracy score")
	ax.set_ylabel("Probability")
	plt.show()


	# evaluate model on the hold out dataset
	yhat = best_model.predict(X_test)

	# evaluate the model
	acc = accuracy_score(y_test, yhat)
	f1 = f1_score(y_true=y_test, y_pred=yhat, pos_label=1, average="binary")
	precision = precision_score(y_true=y_test, y_pred=yhat, pos_label=1, average="binary")
	recall = recall_score(y_true=y_test, y_pred=yhat, pos_label=1, average="binary")

	# Visualize Confusion Matrix
	mat = confusion_matrix(y_test, yhat)
	sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
	plt.title('Confusion Matrix where Proband ' + str(i) + " was test-data")
	plt.xlabel('true label')
	plt.ylabel('predicted label')
	plt.show()

	#apply binomial test
	pvalue_binom = binom_test(x=mat[0][0]+mat[1][1], n=len(y_test), p=0.5, alternative='greater')

	# store statistical test results (p-value permutation test, accuracy of that permutation iteration, pvalue binomial test) in list
	permutation_pvalue.append(pvalue_model)
	permutation_modelaccuracy.append(score_model)
	pvalues_binomial.append(pvalue_binom)

	# store the result
	test_proband.append(i)
	outer_results_acc.append(acc)
	outer_results_f1.append(f1)
	outer_results_precision.append(precision)
	outer_results_recall.append(recall)

	# report progress
	t1_inner = time.time()
	print('>acc=%.3f,f1=%.3f,precision=%.3f,recall=%.3f, est=%.3f, cfg=%s' % (acc, f1, precision, recall, result.best_score_, result.best_params_))
	print("Permutation-Test p-value was " + str(pvalue_model) + " and Binomial Test p-values was " + str(pvalue_binom))
	print("The proband taken as test-data for this iteration was " + str(i))
	print("This inner iteration has taken so many minutes: " + str((t1_inner - t0_inner)/60))

t1 = time.time()
print("The whole process has taken so many seconds: " + str(t1 - t0))

# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results_acc), std(outer_results_acc)))
print('F1: %.3f (%.3f)' % (mean(outer_results_f1), std(outer_results_f1)))
print('Precision: %.3f (%.3f)' % (mean(outer_results_precision), std(outer_results_precision)))
print('Recall: %.3f (%.3f)' % (mean(outer_results_recall), std(outer_results_recall)))
print("Mean p-value of Permutation Test: %.3f (%.3f)" % (mean(permutation_pvalue), std(permutation_pvalue)))
print("Mean of p-value of Binomial Test: %.3f (%.3f)" % (mean(pvalues_binomial), std(pvalues_binomial)))

# Create Results Dataframe:
results = pd.DataFrame()
results["Test-Proband"] = test_proband
results["Accuracy"] = outer_results_acc
results["Accuracy by PermutationTest"] = permutation_modelaccuracy
results["F1"] = outer_results_f1
results["Precision"] = outer_results_precision
results["Recall"] = outer_results_recall
results["P-Value Permutation Test"] = permutation_pvalue
results["P-Value Binomial Test"] = pvalues_binomial

os.system("shutdown /h") #hibernate

#endregion


#endregion


#region Trial 06: Logistic Regression (focussing on MeanMedian RR Interval only)
#region getdata
df_class_allfeatures_del_NaN_binary_shuffled = load_obj("df_class_allfeatures_del_NaN_binary_shuffled_ONLYPREDICTIVE.pkl")

X_acc = df_class_allfeatures_del_NaN_binary_shuffled.drop(["Label"], axis = 1)
y_acc = df_class_allfeatures_del_NaN_binary_shuffled["Label"]
X_acc_train_array, X_acc_test_array, y_acc_train_array, y_acc_test_array = train_test_split(X_acc, y_acc, test_size=0.1, random_state=3)
X_acc_train_array = X_acc_train_array.to_numpy()
X_acc_test_array = X_acc_test_array.to_numpy()
y_acc_train_array = y_acc_train_array.to_numpy()
y_acc_test_array = y_acc_test_array.to_numpy()

X = df_class_allfeatures_del_NaN_binary_shuffled.drop(["Label", "x1", "x2", "x3"], axis = 1)
y = df_class_allfeatures_del_NaN_binary_shuffled["Label"]
X_train_array, X_test_array, y_train_array, y_test_array = train_test_split(X, y, test_size=0.1, random_state=3)
X_train_array = X_train_array.to_numpy()
X_test_array = X_test_array.to_numpy()
y_train_array = y_train_array.to_numpy()
y_test_array = y_test_array.to_numpy()

X_acc_all_array = df_class_allfeatures_del_NaN_binary_shuffled.drop(["Label"], axis = 1)
X_all_array = df_class_allfeatures_del_NaN_binary_shuffled.drop(["Label", "x1", "x2", "x3"], axis = 1)
y_all_array = df_class_allfeatures_del_NaN_binary_shuffled["Label"]


# scaling
sc_X = StandardScaler()
X_scaled=sc_X.fit_transform(X_all_array)

# define dataset for this iteration
X = X_scaled
Y = y_all_array

X_all = X_scaled
y_all = y_all_array.to_numpy()
X_all_df = X_all_array
#endregion

#region nested CV
t0 = time.time()
# configure the cross-validation procedure
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# enumerate splits
outer_results_acc = list()
outer_results_f1 = list()
outer_results_precision = list()
outer_results_recall = list()

list_shap_values = list()
list_test_sets = list()
columns = X_all_df.columns
counter = 0
outer_split = cv_outer.split(X_all, y_all)
outer_split = list(outer_split)
save_obj(outer_split, "outer_split")

for train_ix, test_ix in outer_split:
	t0_inner = time.time()
	t0_inner_CV = time.time()
	# split data
	X_train, X_test = X_all[train_ix, :], X_all[test_ix, :]
	y_train, y_test = y_all[train_ix], y_all[test_ix]
	# configure the cross-validation procedure
	cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
	# define the model
	model = LogisticRegression(solver="saga")

	# define search space
	space = {'C': np.logspace(-3,3,7),
			 'penalty':["l1","l2"]}
	#space = {'final_estimator__C': [1],
	#		 'final_estimator__penalty':["l1"]}

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

	#save intermediate results
	save_obj(outer_results_acc, "outer_results_acc")
	save_obj(outer_results_f1, "outer_results_f1")
	save_obj(outer_results_precision, "outer_results_precision")
	save_obj(outer_results_recall, "outer_results_recall")
	# report progress
	print('>acc=%.3f,f1=%.3f,precision=%.3f,recall=%.3f, est=%.3f, cfg=%s' % (acc,f1, precision, recall, result.best_score_, result.best_params_))
	t1_inner_CV = time.time()
	print("Inner CV has taken so many seconds: " + str(t1_inner_CV-t0_inner_CV))

	#feature importance: compute SHAP values
	#explainer = shap.KernelExplainer(best_model.predict, X_test)
	#shap_values = explainer.shap_values(X_test, nsamples = 100)
	## for each iteration we save the test_set index and the shap_values
	#list_shap_values.append(shap_values)
	#list_test_sets.append(test_ix)
	# save intermediates
	#save_obj(list_shap_values, "list_shap_values")
	#save_obj(list_test_sets, "list_test_sets")
	t1_inner = time.time()
	print("The time of this inner iteration was" + str(t1_inner-t0_inner))
	counter = counter + 1
	save_obj(counter, "counter")
#combining SHAP results from all iterations
#test_set = []
#shap_values = []
t1 = time.time()
print("This process took so many seconds: " +str(t1-t0))

#region to pick up on old progress
#load old progress:
outer_split = load_obj("outer_split.pkl")
outer_results_acc = load_obj("outer_results_acc.pkl")
outer_results_f1 = load_obj("outer_results_acc.pkl")
outer_results_precision = load_obj("outer_results_precision.pkl")
outer_results_recall = load_obj("outer_results_recall.pkl")
list_shap_values = load_obj("list_shap_values.pkl")
list_test_sets = load_obj("list_test_sets.pkl")
counter = load_obj("counter.pkl")

#start again from where I stopped
counter2 = 0
for train_ix, test_ix in outer_split:
	if counter2 <= counter:
		counter2 = counter2+1
		continue
	t0_inner = time.time()
	t0_inner_CV = time.time()
	# split data
	X_train, X_test = X_all[train_ix, :], X_all[test_ix, :]
	y_train, y_test = y_all[train_ix], y_all[test_ix]
	# configure the cross-validation procedure
	cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
	# define the model
	model = SVC()

	# define search space
	space = [{'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
			  'gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001],
			  'C': [0.001, 0.1, 1, 10, 100]}]
	#space = [{'kernel': ['linear'],
	#		  'gamma': [1],
	#		  'C': [0.001]}]
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
	print('>acc=%.3f,f1=%.3f,precision=%.3f,recall=%.3f, est=%.3f, cfg=%s' % (acc,f1, precision, recall, result.best_score_, result.best_params_))
	t1_inner_CV = time.time()
	print("Inner CV has taken so many seconds: " + str(t1_inner_CV-t0_inner_CV))
	#feature importance: compute SHAP values
	explainer = shap.KernelExplainer(best_model.predict, X_test)
	shap_values = explainer.shap_values(X_test, nsamples = 100)
	# for each iteration we save the test_set index and the shap_values
	list_shap_values.append(shap_values)
	list_test_sets.append(test_ix)
	t1_inner = time.time()
	print("The time of this inner iteration was" + str(t1_inner-t0_inner))


#endregion



test_set = list_test_sets[0]
shap_values = np.array(list_shap_values[0])

for i in range(1,len(list_test_sets)):
	test_set = np.concatenate((test_set,list_test_sets[i]),axis=0)
	shap_values = np.concatenate((shap_values,np.array(list_shap_values[i])), axis=0)
#bringing back variable names: X_test is the X_all data reordered due to the 10-test sets in outer CV
test_set = test_set.astype(int)
X_test = pd.DataFrame(X_all[test_set],columns = columns) #this should be used for SHAP plots!


# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results_acc), std(outer_results_acc)))
print('F1: %.3f (%.3f)' % (mean(outer_results_f1), std(outer_results_f1)))
print('Precision: %.3f (%.3f)' % (mean(outer_results_precision), std(outer_results_precision)))
print('Recall: %.3f (%.3f)' % (mean(outer_results_recall), std(outer_results_recall)))


save_obj(shap_values, "normalstress_shap_values_stacking_withoutACC")
save_obj(X_test, "normalstress_X_test_stacking_withoutACC")
#shap_values = load_obj("shap_values_DecisionForest_withACC.pkl")
#X_test = load_obj("X_test_DecisionForest_withACC.pkl")

#endregion

#endregion


#region Run Example Stacking model
## Code from: https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/

t0 = time.time()
# compare ensemble to each baseline classifier
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from matplotlib import pyplot

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
	return X, y

# get a stacking ensemble of models
def get_stacking():
	# define the base models
	level0 = list()
	level0.append(('lr', LogisticRegression(max_iter=1000)))
	level0.append(('knn', KNeighborsClassifier()))
	level0.append(('cart', DecisionTreeClassifier()))
	level0.append(('svm', SVC()))
	level0.append(('bayes', GaussianNB()))
	# define meta learner model
	level1 = LogisticRegression()
	# define the stacking ensemble
	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
	return model

# get a list of models to evaluate
def get_models():
	models = dict()
	models['lr'] = LogisticRegression()
	models['knn'] = KNeighborsClassifier()
	models['cart'] = DecisionTreeClassifier()
	models['svm'] = SVC()
	models['bayes'] = GaussianNB()
	models['stacking'] = get_stacking()
	return models

# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores

# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X, y)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
t1 = time.time()
t1-t0

#endregion
