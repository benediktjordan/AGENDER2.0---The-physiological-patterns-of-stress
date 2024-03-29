
#region Classification Pipeline 1: ensemble classification pipeline


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
