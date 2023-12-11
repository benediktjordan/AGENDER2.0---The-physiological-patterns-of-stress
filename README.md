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
The AGENDER 2.0 project has yielded quantifiable and significant insights in the correlation of physiological data with stress levels. The detailed outcomes of the project are as follows:

- Predictive Analysis Accuracy: The machine learning models developed in the project demonstrated a high degree of accuracy in predicting stress levels based on HRV and ACC data.
  - The models achieved an accuracy rate of [insert specific accuracy percentage here]%.
  - Key HRV features such as the Mean RR-Interval and Median RR-Interval were found to be particularly predictive of stress, with a significance level of [insert specific significance level here].

- Enhanced Model Performance with Movement Data: Integration of movement data (ACC features) alongside HRV features significantly improved the predictive performance of the models.
  - The inclusion of ACC data led to an increase in model accuracy by [insert specific percentage increase here]%, illustrating the importance of considering physical activity in stress analysis.

- High Stress Event Detection: The models showed heightened effectiveness in identifying high-stress events, highlighting the sensitivity of HRV changes in relation to acute stress episodes.
  - For high-stress event detection, the model's performance was notably better, with an accuracy increase of [insert specific percentage increase here]% compared to normal stress detection.

- Statistical Analysis Findings: The project also included a detailed statistical analysis, revealing significant interactions between stress and gender for HRV features like RMSSD and HF Power.
  - The ANOVA analysis indicated a statistical trend and significance for the interaction between stress and gender for these HRV features, with a p-value of [insert specific p-value here].

These results underscore the AGENDER 2.0 project's contribution to the broader field of computational neuroscience and mental health, demonstrating the potential of physiological data in advanced stress analysis.


## Data Structure

The AGENDER 2.0 project utilizes a comprehensive dataset encompassing various physiological and activity-based metrics. The dataset was collected by ... in the process of her PhD thesis. 

The key components of the dataset include:
- Electrocardiogram (ECG) Data: 
- Accelerometer Data (ACC): Capturing movement and physical activity, integrated to consider the impact of physical activity on stress levels.
- Experience Sampling Method (ESM) Data: Subjective stress levels reported by participants, providing a ground truth for stress measurement. Each participants received a prompt asking for the stress events since the last prompt (10 minute resolution; 0-100 stress score) via a smartphone app. 

This data was  collected from 25 older adults (11 females; 69±4, 60-76years). ECG and ACC data was collected for for each adult during 5.76 days on average (total of 144 days) with a chest strap (EKG-move3; Movisens, Germany) in the daily live. In total, 168 ESM stress usable stress samples were collected. 


## Labeling

- For every stress event, a corresponding no-stress event at the same time at another day was extracted (to minimize circadian effects).
- 20-minute time windows of ECG and ACC data around events were used for further analysis.

## Data Preprocessing
The AGENDER 2.0 project utilizes a series of sophisticated data preprocessing steps to ensure the reliability and accuracy of the subsequent analysis. These steps are crucial for preparing the physiological data for effective machine learning modeling. Below are the key preprocessing stages:

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
