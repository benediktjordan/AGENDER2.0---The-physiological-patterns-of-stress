#region Classification Pipeline 3: SVM
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