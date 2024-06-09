# Master's Project: Enhancing Antimicrobial Resistance Prediction with Convolutional Neural Networks

GitHub repository for my master's project titled "Enhancing Antimicrobial Resistance Prediction with Convolutional Neural Networks".

## Abstract

Antimicrobial resistance (AMR) poses a critical threat to global health, necessitating rapid and accurate diagnostic tools. Matrix-assisted laser desorption/ionization time-of-flight mass spectrometry (MALDI-TOF MS), combined with advanced machine learning models, offers a promising approach to this challenge. In this study, we evaluated the effectiveness of Ridge regression, random forests, and Convolutional Neural Networks (CNNs) in predicting antibiotic resistance using the DRIAMS dataset. Our results demonstrate that while random forests achieved high accuracy with a mean AUROC of 0.89, CNNs performed slightly better with a mean AUROC of 0.90. Preprocessing steps such as log-transformation, LOWESS smoothing, and rescaling were crucial for enhancing model performance. Shapley values and GradCAM were employed for feature importance mapping, revealing critical spectral regions pivotal for resistance prediction. Both methods largely agreed on important features, though GradCAM was computationally more efficient. This study underscores the significant potential of machine learning in AMR diagnostics, providing a pathway to more accurate and reliable predictive models. Future work should focus on optimizing these models further and exploring advanced techniques such as Transformer models and Kolmogorov-Arnold Networks. Integrating experimental validation of identified features and expanding datasets will likely enhance model generalizability and clinical applicability.

## Repository Contents

- `data/`: Contains the datasets used for this study.
- `scripts/`: Includes scripts for data preprocessing.
- `notebooks/`: Jupyter notebooks used for exploratory data analysis and preliminary modeling.
- `results/`: Output files including performance metrics, plots, and figures used in the thesis.
