#region Classification Pipeline 6: Logistic Regression (focussing on MeanMedian RR Interval only)
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

