#region SAS2021 presentation

#region getdata
acc_df_withoutACC_normalstress = [0.765, 0.722, 0.743, 0.745, 0.707, 0.777, 0.743, 0.735, 0.707, 0.775]
acc_mlp_withoutACC_normalstress = [0.735, 0.701,0.682, 0.715, 0.671, 0.709, 0.682, 0.696, 0.645, 0.699]
acc_svm_withoutACC_normalstress = [0.716, 0.657, 0.703, 0.718, 0.690, 0.688, 0.669, 0.688, 0.648, 0.669]
acc_stacking_withoutACC_normalstress = [0.765, 0.725, 0.756, 0.743, 0.713, 0.775, 0.730, 0.739, 0.709, 0.760]

acc_df_withACC_normalstress = [0.837, 0.837, 0.854, 0.834, 0.792, 0.803, 0.849, 0.805, 0.790,0.854]
acc_mlp_withACC_normalstress = [0.773, 0.799, 0.758,0.756,0.747,0.760,0.817,0.756,0.754,0.792]
acc_svm_withACC_normalstress = [0.742, 0.714,0.769,0.760,0.728,0.724,0.754,0.745,0.711,0.732]
acc_stacking_withACC_normalstress = [0.839,0.843,0.854,0.830,0.813,0.794,0.875,0.824,0.794,0.856]

acc_df_withACC_highstress = [0.902,0.946,0.880,0.913,0.897,0.929,0.891,0.918,0.918,0.902]
acc_mlp_withACC_highstress = [0.837, 0.902, 0.864, 0.886, 0.870, 0.918, 0.918, 0.918, 0.891, 0.929]
acc_svm_withACC_highstress = [0.859,0.886, 0.826, 0.886, 0.832, 0.875, 0.859, 0.864, 0.891, 0.896]
acc_stacking_withACC_highstress = [0.902,0.940,0.897,0.935,0.908,0.924,0.908,0.929,0.929,0.934]

acc_df_withoutACC_normalstress_onlyMeanMedianRR = [0.712, 0.695, 0.701, 0.715, 0.682, 0.701, 0.665, 0.682, 0.656,0.720]

acc_df_withoutACC_normalstress_exceptMeanMedianRR = [0.699, 0.691, 0.703,0.720,0.677,0.692,0.677,0.667,0.624,0.701]
acc_svm_withoutACC_normalstress_exceptMeanMedianRR = [0.674,0.655,0.688,0.701,0.665,0.684,0.662,0.675,0.633,0.667]
acc_mlp_withoutACC_normalstress_exceptMeanMedianRR = [0.691,0.640,0.699,0.692,0.671,0.705,0.656,0.684,0.639,0.726]
acc_ensemble_withoutACC_normalstress_exceptMeanMedianRR = [0.693,0.686,0.703,0.707,0.696,0.699,0.679,0.690,0.641,0.688]

acc_df_withoutACC_normalstress_only10worstperforming = [0.689,0.629,0.660,0.686,0.656,0.690,0.667,0.652,0.635,0.660]

acc_logisticregression_withoutACC_normalstress_onlyMeanMedianRR = [0.555,0.532,0.537,0.531,0.544,0.533,0.539,0.529,0.552,0.535]

acc_df_withoutACC_normalstress_onlyRMSSDandHF_Power = [0.648,0.623,0.620,0.679,0.618,0.677,0.618,0.633,0.626,0.662]
#endregion

#region Result 1: HRV features play a role in stress

#region Boxplots of Performance metrics


acc_all_withoutACC_normalstress = []
acc_all_withoutACC_normalstress.append(acc_df_withoutACC_normalstress)
#acc_all_withoutACC_normalstress.append(acc_svm_withoutACC_normalstress)

#Seaborn Boxplot
model = ["Decision Forest"]
models = model*10
data = pd.DataFrame({"Accuracy": acc_df_withoutACC_normalstress,
                     "Model": models})


a4_dims = (5.7, 6.27)
fig, ax = pyplot.subplots(figsize=a4_dims)
sns.swarmplot(ax=ax, x="Model", y="Accuracy", data=data, color=".25", size = 10)
sns.boxplot(ax=ax, x="Model", y="Accuracy", data=data,color=cmap(0.05), showmeans=True)
plt.ylabel("Accuracy",fontsize=28)
plt.tick_params(labelsize=25)
plt.tight_layout()

#plt.ylim(0.6, 0.8)
plt.show()

#endregion

#region SHAP Feature Importance for Decision Forest
##SVM
#shap_values_normal_svm_withoutACC = load_obj("normal_shap_values_SVM_withoutACC.pkl")
#X_test_normal_svm_withoutACC = load_obj("normal_X_test_SVM_withoutACC.pkl")

#DF
shap_values_normal_DF_withoutACC = load_obj("normal_shap_values_DecisionForest_withoutACC.pkl")
X_test_normal_DF_withoutACC = load_obj("normal_X_test_DecisionForest_withoutACC.pkl")
#Change Feature names
X_test_normal_DF_withoutACC.rename(columns = {'HRV_MedianNN' : 'Median RR-Interval',
                                              'HRV_MeanNN' : 'Mean RR-Interval',
                                              "HRV_PI": "Porta´s Index",
                                              "HRV_ApEn": "Approximate Entropy",
                                              "HRV_pNN20": "pNN20",
                                              "HRV_PIP": "Inflection Points (%)",
                                              "HRV_HFn": "normalized High-Frequency",
                                              "HRV_PSS": "Short Segments (%)",
                                              "HRV_IALS": "IALS",
                                              "HRV_C1a": "C1a"}, inplace = True)


cmap = mpl.cm.get_cmap('tab20b')
# feature importance plot
shap.summary_plot(shap_values_normal_DF_withoutACC[0], X_test_normal_DF_withoutACC, plot_type="bar", max_display = 52,
                   show=False, color=cmap(0.05))
plt.tick_params(labelsize=18)
#plot_size = (8,3) ,
plt.tight_layout()
plt.xlabel("mean SHAP value", fontsize=19)
plt.show()

#endregion
#endregion

#region Result 2: Comparison of HRV and HRV+movement models
#Idea: create for every model one boxplot with and one without ACC data (next to each other)

#get data

acc_all_withoutACC_normalstress = acc_df_withoutACC_normalstress+acc_mlp_withoutACC_normalstress + acc_svm_withoutACC_normalstress+acc_stacking_withoutACC_normalstress
acc_all_withACC_normalstress = acc_df_withACC_normalstress + acc_mlp_withACC_normalstress + acc_svm_withACC_normalstress+acc_stacking_withACC_normalstress
df = ["Decision Forest"]
mlp = ["Multilayered Perceptron"]
svm = ["Support Vector Machine"]
stacking = ["Stacking Ensemble"]
withoutACC = ["without physical activity"]
withACC = ["with physical activity"]

#acc_all_withoutACC_normalstress.append(acc_svm_withoutACC_normalstress)

#Seaborn Boxplot
models_all = df*10+mlp*10+svm*10+stacking*10 + df*10+mlp*10+svm*10+stacking*10
movement = withoutACC*40 +withACC*40
data = pd.DataFrame({"Accuracy": acc_all_withoutACC_normalstress + acc_all_withACC_normalstress,
                     "Model": models_all,
                     "Movement": movement})

#fig, ax = pyplot.subplots(figsize=a4_dims)
#sns.swarmplot( x="Model", y="Accuracy", data=data, color=".25", size = 10, hue = "Movement")
cmap = mpl.cm.get_cmap('tab20b')
sns.boxplot(x="Model", y="Accuracy", data=data,color=cmap(0.05), showmeans=True, hue = "Movement")
plt.ylabel("Accuracy",fontsize=22)
plt.tick_params(labelsize=18)
plt.xticks([0, 1, 2, 3, ],
           ["Decision Forest", "Multilayered \nPerceptron", "Support Vector \nMachines", "Stacking Ensemble"])
plt.legend(loc = 3,  fontsize = 16, fancybox=True)
plt.tight_layout()
plt.show()

#endregion


#region Result 3: Comparison of normal stress vs. high stress for with ACC


#get data

acc_all_withACC_normalstress = acc_df_withACC_normalstress + acc_mlp_withACC_normalstress + acc_svm_withACC_normalstress+acc_stacking_withACC_normalstress
acc_all_withACC_highstress = acc_df_withACC_highstress+acc_mlp_withACC_highstress+acc_svm_withACC_highstress+acc_stacking_withACC_highstress
df = ["Decision Forest"]
mlp = ["Multilayered Perceptron"]
svm = ["Support Vector Machine"]
stacking = ["Stacking Ensemble"]
allstress = ["all stress levels"]
highstress = ["only high stress"]

#acc_all_withoutACC_normalstress.append(acc_svm_withoutACC_normalstress)

#Seaborn Boxplot
models_all = df*10+mlp*10+svm*10+stacking*10 + df*10+mlp*10+svm*10+stacking*10
stress = allstress*40 +highstress*40
data = pd.DataFrame({"Accuracy": acc_all_withACC_normalstress + acc_all_withACC_highstress,
                     "Model": models_all,
                     "Stress included": stress})

#fig, ax = pyplot.subplots(figsize=a4_dims)
#sns.swarmplot( x="Model", y="Accuracy", data=data, color=".25", size = 10, hue = "Movement")
cmap = mpl.cm.get_cmap('tab20')
sns.boxplot(x="Model", y="Accuracy", data=data,color=cmap(0.05), showmeans=True, hue = "Stress included")
plt.ylabel("Accuracy",fontsize=22)
plt.tick_params(labelsize=18)
plt.xticks([0, 1, 2, 3, ],
           ["Decision Forest", "Multilayered \nPerceptron", "Support Vector \nMachines", "Stacking Ensemble"])
plt.legend(loc = 3,  fontsize = 16, fancybox=True)

plt.tight_layout()
plt.show()


#endregion



#region Discussion: only 1 model

#region Show RMSSD & HF_Power Model

acc_df_withoutACC_normalstress_onlyRMSSDandHF_Power_list = []
acc_df_withoutACC_normalstress_onlyRMSSDandHF_Power_list.append(acc_df_withoutACC_normalstress_onlyRMSSDandHF_Power)
#acc_all_withoutACC_normalstress.append(acc_svm_withoutACC_normalstress)

#Seaborn Boxplot
model = ["Decision Forest"]
models = model*10
data = pd.DataFrame({"Accuracy": acc_df_withoutACC_normalstress_onlyRMSSDandHF_Power,
                     "Model": models})


a4_dims = (5.7, 6.27)
fig, ax = pyplot.subplots(figsize=a4_dims)
sns.swarmplot(ax=ax, x="Model", y="Accuracy", data=data, color=".25", size = 10)
sns.boxplot(ax=ax, x="Model", y="Accuracy", data=data,color=cmap(0.05), showmeans=True)
plt.ylabel("Accuracy",fontsize=28)
plt.tick_params(labelsize=25)
plt.tight_layout()

#plt.ylim(0.6, 0.8)
plt.show()




#get data
DecisionForest = ["Decision Forest"]
LogisticRegression= ["Logistic Regression"]
#Seaborn Boxplot
models_all = DecisionForest*10+LogisticRegression*10
data = pd.DataFrame({"Accuracy": acc_df_withoutACC_normalstress + acc_logisticregression_withoutACC_normalstress_onlyMeanMedianRR,
                     "Model": models_all})

#fig, ax = pyplot.subplots(figsize=a4_dims)
#sns.swarmplot( x="Model", y="Accuracy", data=data, color=".25", size = 10, hue = "Movement")
plt.figure(figsize=(12, 6), dpi=80)

cmap = mpl.cm.get_cmap('tab20b')
sns.boxplot(x="Model", y="Accuracy", data=data,color=cmap(0.05), showmeans=True)
plt.ylabel("Accuracy",fontsize=25)
plt.tick_params(labelsize=22)
plt.xticks([0, 1 ],
           ["Decision Forest", "Logistic Regression"])
plt.legend(loc = 3,  fontsize = 19, fancybox=True)
#plt.title("Decision Forest Performance", fontsize = 25)
plt.tight_layout()

plt.show()
#endregion
#endregion


#region Discussion: 2 Models to compare
#region Comparing Mean/MedianRR vs. all HRV features DF
#get data
all = ["all HRV features"]
only_predictive = ["only Mean & Median RR"]

#Seaborn Boxplot
models_all = all*10+only_predictive*10
data = pd.DataFrame({"Accuracy": acc_df_withoutACC_normalstress + acc_df_withoutACC_normalstress_onlyMeanMedianRR,
                     "Model": models_all})

#fig, ax = pyplot.subplots(figsize=a4_dims)
#sns.swarmplot( x="Model", y="Accuracy", data=data, color=".25", size = 10, hue = "Movement")
plt.figure(figsize=(12, 6), dpi=80)

cmap = mpl.cm.get_cmap('tab20b')
sns.boxplot(x="Model", y="Accuracy", data=data,color=cmap(0.05), showmeans=True)
plt.ylabel("Accuracy",fontsize=25)
plt.tick_params(labelsize=22)
plt.xticks([0, 1 ],
           ["all HRV \n features", "only Mean \n & Median RR"])
plt.legend(loc = 3,  fontsize = 19, fancybox=True)
plt.title("Decision Forest Performance", fontsize = 25)
plt.tight_layout()

plt.show()

#endregion

#region Compare DF 10 worst performing features with all features
#get data
all = ["all HRV features"]
only_worstperforming= ["10 worst performing features"]
#Seaborn Boxplot
models_all = all*10+only_worstperforming*10
data = pd.DataFrame({"Accuracy": acc_df_withoutACC_normalstress + acc_df_withoutACC_normalstress_only10worstperforming,
                     "Model": models_all})

#fig, ax = pyplot.subplots(figsize=a4_dims)
#sns.swarmplot( x="Model", y="Accuracy", data=data, color=".25", size = 10, hue = "Movement")
plt.figure(figsize=(12, 6), dpi=80)

cmap = mpl.cm.get_cmap('tab20b')
sns.boxplot(x="Model", y="Accuracy", data=data,color=cmap(0.05), showmeans=True)
plt.ylabel("Accuracy",fontsize=25)
plt.tick_params(labelsize=22)
plt.xticks([0, 1 ],
           ["all HRV \n features", "10 worst \n performing features"])
plt.legend(loc = 3,  fontsize = 19, fancybox=True)
#plt.title("Decision Forest Performance", fontsize = 25)
plt.tight_layout()

plt.show()
#endregion

#region Compare DF of Mean/MedianRR vs. Logistic Regression of Mean/MedianRR
#get data
DecisionForest = ["Decision Forest"]
LogisticRegression= ["Logistic Regression"]
#Seaborn Boxplot
models_all = DecisionForest*10+LogisticRegression*10
data = pd.DataFrame({"Accuracy": acc_df_withoutACC_normalstress_onlyMeanMedianRR + acc_logisticregression_withoutACC_normalstress_onlyMeanMedianRR,
                     "Model": models_all})

#fig, ax = pyplot.subplots(figsize=a4_dims)
#sns.swarmplot( x="Model", y="Accuracy", data=data, color=".25", size = 10, hue = "Movement")
plt.figure(figsize=(12, 6), dpi=80)

cmap = mpl.cm.get_cmap('tab20b')
sns.boxplot(x="Model", y="Accuracy", data=data,color=cmap(0.05), showmeans=True)
plt.ylabel("Accuracy",fontsize=25)
plt.tick_params(labelsize=22)
plt.xticks([0, 1 ],
           ["Decision Forest", "Logistic Regression"])
plt.legend(loc = 3,  fontsize = 19, fancybox=True)
#plt.title("Decision Forest Performance", fontsize = 25)
plt.tight_layout()

plt.show()
#endregion


#


#endregion

#region Discussion: 8 models to compare
# only REAL HRV features and all HRV features

#get data

acc_all_withACC_normalstress_exceptMeanMedianRR = acc_df_withoutACC_normalstress_exceptMeanMedianRR + acc_mlp_withoutACC_normalstress_exceptMeanMedianRR + acc_svm_withoutACC_normalstress_exceptMeanMedianRR+acc_ensemble_withoutACC_normalstress_exceptMeanMedianRR
acc_all_withACC_normalstress = acc_df_withoutACC_normalstress+acc_mlp_withoutACC_normalstress+acc_svm_withoutACC_normalstress+acc_stacking_withoutACC_normalstress
df = ["Decision Forest"]
mlp = ["Multilayered Perceptron"]
svm = ["Support Vector Machine"]
stacking = ["Stacking Ensemble"]
all = ["all HRV features"]
only_realHRV = ["\"real\" HRV features"]

#acc_all_withoutACC_normalstress.append(acc_svm_withoutACC_normalstress)

#Seaborn Boxplot
models_all = df*10+mlp*10+svm*10+stacking*10 + df*10+mlp*10+svm*10+stacking*10
features = all*40 + only_realHRV*40
data = pd.DataFrame({"Accuracy": acc_all_withACC_normalstress + acc_all_withACC_normalstress_exceptMeanMedianRR,
                     "Model": models_all,
                     "Stress included": features})

#fig, ax = pyplot.subplots(figsize=a4_dims)
#sns.swarmplot( x="Model", y="Accuracy", data=data, color=".25", size = 10, hue = "Movement")
cmap = mpl.cm.get_cmap('tab20b')
sns.boxplot(x="Model", y="Accuracy", data=data,color=cmap(0.05), showmeans=True, hue = "Stress included")
plt.ylabel("Accuracy",fontsize=25)
plt.tick_params(labelsize=22)
plt.xticks([0, 1, 2, 3, ],
           ["Decision Forest", "Multilayered\nPerceptron", "Support Vector\nMachines", "Stacking Ensemble"])
plt.legend(loc = 3,  fontsize = 19, fancybox=True)

plt.tight_layout()
plt.show()

#endregion
#endregion