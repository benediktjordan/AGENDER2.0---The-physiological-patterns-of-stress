
#region Pipeline 4: Decision Tree classifier
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