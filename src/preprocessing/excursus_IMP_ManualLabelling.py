df_class_allfeatures = load_obj("df_class_allfeatures.pkl")

#drop features which are not necessary (low frequency/high NaN values)
features_del = ["HRV_LF", "HRV_LFHF", "HRV_LFn", "HRV_ULF", "HRV_VLF", "HRV_SampEn"]
df_labeling = df_class_allfeatures.drop(features_del, axis = 1)

# Note: only HRV_HFn,HRV_LnHF, HRV_HF contain each 1 NaN value -> only 1 row is dropped
df_labeling = df_labeling.dropna()

#choose y-limits
##y-limits are chosen to be mean+- 2x std
#features_to_visualize = ["HRV_C1a", "HRV_C1d", "HRV_C2a", "HRV_C2d"]
#features_to_visualize = ["HRV_CSI", "HRV_CSI_Modified", "HRV_CVI"]
#features_to_visualize = ["HRV_CVNN", "HRV_CVSD", "HRV_MedianNN"]
#features_to_visualize = ["HRV_RMSSD", "HRV_MeanNN", "HRV_SDNN", "HRV_SDSD"]
#features_to_visualize = ["HRV_MadNN", "HRV_IQRNN", "HRV_TINN", "HRV_HTI"]
#features_to_visualize = ["HRV_pNN50", "HRV_pNN20"]
#features_to_visualize = ["HRV_SD1", "HRV_SD2", "HRV_SD1SD2", "HRV_S"]
features_to_visualize = ["HRV_GI", "HRV_SI", "HRV_AI", "HRV_PI"]
#features_to_visualize = ["HRV_SD1a", "HRV_SD1d", "HRV_SD2a", "HRV_SD2d"]
#features_to_visualize = ["HRV_SDNNa", "HRV_SDNNd", "HRV_Ca", "HRV_Cd"]
#features_to_visualize = ["HRV_PIP", "HRV_IALS", "HRV_PSS", "HRV_PAS"]
#features_to_visualize = ["HRV_ApEn"]
#features_to_visualize = ["HRV_VHF", "HRV_HF", "HRV_HFn", "HRV_PAS"]


y_limits = []
for feature in features_to_visualize:
    mean = df_labeling[feature].describe().loc["mean"]
    std = df_labeling[feature].describe().loc["std"]
    y_limit = [mean-2*std, mean+2*std]
    y_limits.append(y_limit)

# create visualizations

def labeling_features_vis(df, features_to_visualize,y_limits, segmentation_period):
    proband_list = df["Proband"].unique()
    for proband in proband_list:
        epochs_list = df[df["Proband"]==proband]["Epoch"].unique()
        for epoch in epochs_list:
            data = df[(df["Proband"]==proband) & (df["Epoch"]==epoch)]

            # Plot
            nrow = 4;
            ncol = 1;
            fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(15, 8))
            fig.suptitle('Different HRV Features')

            # this loop creates the ECG and peak visualization for peaks, NK2 corrected and Brown corrected
            counter = 0
            for ax, feature in zip(axs, features_to_visualize):
                ax.plot(data["Segment"]/(60/segmentation_period),data[feature])
                ax.set_title(feature)
                ax.set_ylim((y_limits[counter][0], y_limits[counter][1]))
                counter = counter + 1
            plt.show()
            # Adjust plots
            fig.tight_layout()

            #save with label in front
            # Save
            if len(features_to_visualize) == 4:
                fig.savefig(
                    "C:/Users/BJ/PycharmProjects/AGENDER2.0/01_DataCleaning/ManualLabeling/"+ str(features_to_visualize[0]) +
                    "_"+ str(features_to_visualize[1])+"_"+str(features_to_visualize[2])+"_"+str(features_to_visualize[3])+ "_"+str(data["Label"].iloc[0])+
                    "_is the label. Proband_" +str(proband) + "_Epoch_" + str(epoch) + ".png")
            if len(features_to_visualize) == 3:
                fig.savefig(
                    "C:/Users/BJ/PycharmProjects/AGENDER2.0/01_DataCleaning/ManualLabeling/" + str(
                        features_to_visualize[0]) +
                    "_" + str(features_to_visualize[1]) + "_" + str(features_to_visualize[2])  +
                    "_" + str(data["Label"].iloc[0]) +
                    "_is the label. Proband_" + str(proband) + "_Epoch_" + str(epoch) + ".png")
            if len(features_to_visualize) == 2:
                fig.savefig(
                    "C:/Users/BJ/PycharmProjects/AGENDER2.0/01_DataCleaning/ManualLabeling/" + str(
                        features_to_visualize[0]) +
                    "_" + str(features_to_visualize[1]) + "_" + str(data["Label"].iloc[0]) +
                    "_is the label. Proband_" + str(proband) + "_Epoch_" + str(epoch) + ".png")
            if len(features_to_visualize) == 1:
                fig.savefig(
                    "C:/Users/BJ/PycharmProjects/AGENDER2.0/01_DataCleaning/ManualLabeling/" + str(
                        features_to_visualize[0]) +"_" + str(data["Label"].iloc[0]) +
                    "_is the label. Proband_" + str(proband) + "_Epoch_" + str(epoch) + ".png")
            plt.close()  # closes figure

segmentation_period = 30
labeling_features_vis(df_labeling, features_to_visualize, y_limits,  segmentation_period)