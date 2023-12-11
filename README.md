# AGENDER2.0

## Overview

AGENDER 2.0 is a project developed during a master's program in Computational Modeling and Simulation with a focus on Life Sciences at the Technische Universität Dresden (Germany), in cooperation with the Max Planck Institute for Human Cognitive and Brain Sciences Leipzig (Germany). This project aims to identify possible patterns corresponding to acute stress in in heart-activity (Electrocardiogram-derived features like Heart Rate and Heart Rate Variability) and physical activity. The data was collected in a naturalistic setting. 

The project combines advanced machine learning techniques and statistical analysis to analyze ECG and physical activity data, aiming to provide insights into stress patterns in real-world settings. This interdisciplinary work is a blend of computational neuroscience, data science, and psychology, leveraging large-scale data analysis for impactful health-related discoveries

## Repository Content

This repository contains the Python code developed for the AGENDER 2.0 project. The codebase includes various modules for data loading, preprocessing, peak detection, noise detection, transformation, and machine learning classification. The key components are:

- Data Loading and Preprocessing
- Peak Detection Algorithms
- Noise Detection and Filtering
- Feature Engineering and Transformation
- Machine Learning Models for Classification

## Project Results
The AGENDER 2.0 project's investigation into the correlation between heart rate variability (HRV) and stress levels, along with the influence of physical activity, has led to significant findings:

### HR & HRV Relationship to Stress
- Utilized 46 HRV & HR features for stress classification in a binary context.
- Achieved the highest mean accuracy of 74.2% using the Decision Forest model.
- Identified Mean and Median NN Interval as the most contributive features for this model.

  ![alt text](https://github.com/benediktjordan/AGENDER2.0/blob/88c0014916eb5ac2b7d770f6fd7660f3f0a17508/img/Figure_1_Boxplot_of_Decision_Forest_accuracies.png)


### Impact of Physical Activity on Classification Accuracy

- All machine learning models trained with and without physical activity features showed improved performance when including physical activity data.
- The incorporation of physical activity increased the accuracy by 8%, 8%, 5%, and 9% for Decision Forest, MLP, SVM, and Stacking Ensemble models, respectively.

### Emphasis on High-Stress Events

- Focused on two subsets of events: all stress levels and high-stress levels, both binarized.
- Employed 49 HRV, HRV & ACC (physical activity) features for the analysis.
- Models trained to classify only high-stress events consistently outperformed those trained on all stress events, with an increase in accuracy of 9%, 12%, 13%, and 9% for Decision Forest, MLP, SVM, and Stacking Ensemble models, respectively.

These results underscore the efficacy of the AGENDER 2.0's analytical approach, demonstrating the nuanced relationship between physiological markers and perceived stress, and the enhancing effect of accounting for physical activity in stress event classification.


## Data Structure

The AGENDER 2.0 project utilizes a comprehensive dataset encompassing various physiological and activity-based metrics. The dataset was collected by Anne-Christin Loheit in the process of her PhD thesis. 

The key components of the dataset include:
- Electrocardiogram (ECG) Data: t
- Accelerometer Data (ACC): Capturing movement and physical activity, integrated to consider the impact of physical activity on stress levels.
- Experience Sampling Method (ESM) Data: Subjective stress levels reported by participants, providing a ground truth for stress measurement. Each participants received a prompt asking for the stress events since the last prompt (10 minute resolution; 0-100 stress score) via a smartphone app. 

This data was  collected from 25 older adults (11 females; 69±4, 60-76years). ECG and ACC data was collected for for each adult during 5.76 days on average (total of 144 days) with a chest strap (EKG-move3; Movisens, Germany) in the daily live. In total, 168 ESM stress usable stress samples were collected. 

## Data Preprocessing
The AGENDER 2.0 project utilizes a series of sophisticated data preprocessing steps to ensure the reliability and accuracy of the subsequent analysis. These steps are crucial for preparing the physiological data for effective machine learning modeling. Below are the key preprocessing stage:

### Labeling

- For every stress event, a corresponding no-stress event at the same time at another day was extracted (to minimize circadian effects).
- 20-minute time windows of ECG and ACC data around events were used for further analysis.

### Filtering

The first step in preprocessing involves filtering the raw data to remove noise and irrelevant frequency components. This is achieved through:

- Applying a Highpass filter to eliminate lower frequency noise.
- Implementing a 50 Hz powerline filter to remove electrical interference from the data.

### Peak Detection

Accurate detection of R-peaks in the ECG data is critical for HRV analysis. The project employs:

- Advanced algorithms from the Neurokit toolbox to identify peaks accurately.
- Further analysis based solely on peaks detected ensures the precision of subsequent calculations.

### Peak Correction

To ensure the reliability of the detected peaks:

- Erroneous and ectopic heartbeats are identified and corrected.
- A method by Citi et al. (2012) is used for this purpose, improving the accuracy of the heart rate data.

### Noise Exclusion

Noise exclusion is vital to maintain data integrity:

- Manual noise labeling of 15-second segments is conducted.
- Epochs with more than 30% noisy segments are excluded from the analysis.

### Segmentation

Data segmentation is performed to structure the data for analysis:

- The data is segmented into 20-minute epochs.
- Each epoch is further divided into 30-second segments, resulting in a total of 4712 segments for comprehensive analysis.

### Feature Creation

The final step involves the creation of features for the machine learning models:

- 46 HRV features and 3 ACC (movement) features are computed.
- These features are derived with the same time resolution (1 value every 30 seconds), ensuring consistency across all data points.

Each of these steps is meticulously designed and implemented to ensure that the data used in the AGENDER 2.0 project is of the highest quality, enabling robust and reliable analysis in the subsequent stages of the project.

This data preprocessing process resulted in four different datasets: with and without physical activity features as well as all stress events and only high-stress (score above 66) events which were subsequently used in the modeling process. 

## Modeling Process 

Following the comprehensive data preprocessing, the AGENDER 2.0 project advances into an intricate modeling phase, characterized by meticulous model training, hyperparameter tuning, and validation. Key aspects of this phase are outlined below:

### Model Training and Hyperparameter Tuning
- **Model Selection:** Various machine learning models, including Decision Forests, Support Vector Machines, Multilayer Perceptrons, and LSTM networks, are carefully selected based on their suitability for time-series data analysis and stress prediction.
- **Hyperparameter Optimization:** Each model undergoes a rigorous hyperparameter tuning process. This involves experimenting with different configurations to identify the optimal set of parameters that yield the best performance. Techniques like Grid Search and Random Search are employed to systematically explore a wide range of parameter combinations.
- **Customized Training Regimens:** Depending on the model's complexity and the nature of the data, customized training regimens are developed. For instance, LSTM networks require specific considerations for sequence learning, while ensemble methods focus on how individual model predictions are combined.

### Cross-Validation and Model Evaluation

- **Nested Cross-Validation:** To ensure the generalizability of the models and to prevent overfitting, a nested cross-validation approach is utilized. This involves an outer k-fold cross-validation to assess the model's performance and an inner loop for hyperparameter tuning.
  - **Outer Loop:** Divides the dataset into k distinct subsets (or folds). Each fold serves as a test set while the remaining k-1 folds form the training set. This process is repeated k times with each fold used exactly once as a test set.
  - **Inner Loop:** Within each iteration of the outer loop, another layer of k-fold cross-validation is conducted on the training set to tune the hyperparameters. The best parameter set is then used to train the model on the entire training set.
- **Performance Metrics:** The project employs a range of metrics, including accuracy, F1-score, and mean squared error, to evaluate the models' predictive capabilities. These metrics provide a comprehensive view of the models' performance, balancing the aspects of precision, recall, and error rates.

### Interpretation and Visualization

- **Feature Importance Analysis:** Post model training, the project emphasizes understanding the drivers behind the model's predictions. Techniques like SHAP values are utilized to interpret the model's decision-making process, highlighting the most influential features in predicting stress.
- **Result Visualization:** To effectively communicate the findings, the project includes a suite of visualization tools. These tools are used to illustrate the models' performance, feature importance, and any notable patterns in the data, making the results accessible and interpretable.





  
## Installation

To run the AGENDER 2.0 project, you need to have Python installed on your machine. Additionally, certain libraries and dependencies are required, which can be installed via the following command:

```
pip install -r requirements.txt
```

## Usage

To use this project, first clone the repository to your local machine:

```
git clone [repository-link]
```

Navigate to the project directory and execute the main script:

```
cd AGENDER-2.0
python main.py
```

## Contributing

Contributions to the AGENDER 2.0 project are welcome. Please ensure to follow the standard procedures for contributing to open-source projects, including using pull requests for proposing changes.
License

This project is licensed under the MIT License.


## Acknowledgments

Special thanks to the faculty and peers at Technische Universität Dresden, particularly to those who have supervised and provided valuable insights during the development of this project.
