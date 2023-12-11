# AGENDER2.0

## Overview

AGENDER 2.0 is a project developed during a master's program in Computational Modeling and Simulation with a focus on Life Sciences at the Technische Universität Dresden. This project aims to explore the intricate relationship between acute stress and Heart Rate Variability (HRV) in daily life, accounting for the impact of physical activity.

The project combines advanced machine learning techniques and statistical analysis to analyze HRV data, aiming to provide insights into stress patterns in real-world settings. This interdisciplinary work is a blend of computational neuroscience, data science, and psychology, leveraging large-scale data analysis for impactful health-related discoveries.
Repository Content

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

The AGENDER 2.0 project utilizes a comprehensive dataset encompassing various physiological and activity-based metrics. The key components of the dataset include:

- Electrocardiogram (ECG) Data: 
- Accelerometer Data (ACC): Capturing movement and physical activity, integrated to consider the impact of physical activity on stress levels.
- Experience Sampling Method (ESM) Data: Subjective stress levels reported by participants, providing a ground truth for stress measurement.

The data is segmented and labeled to align with the stress events, creating a structured format suitable for machine learning models. Each data entry comprises a combination of HRV features, movement data, and corresponding stress labels.

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
