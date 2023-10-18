# DNN Manual Tuner GUI

## Overview

This Python script provides a graphical user interface (GUI) for manually tuning hyperparameters of a Deep Neural Network (DNN) using the Tkinter library. The application enables the user to perform various preprocessing steps, feature selection, and model evaluation.

## Features

1. **Input Handling**: Load a dataset in CSV format and visualize its columns.
2. **Data Preprocessing**: Implement data preprocessing steps such as dummy encoding and random observation selection.
3. **Feature Importance Analysis**: Utilize a RandomForestRegressor to compute and display feature importances.
4. **Feature Selection**: Select features based on importance thresholds and visualize the selected features.
5. **SMOTE and Scaling**: Optionally apply Synthetic Minority Over-sampling Technique (SMOTE) and scaling to the selected features.
6. **DNN Hyperparameter Tuning**: Manually set hyperparameters for a DNN, including the number of layers, units per layer, and activation functions.
7. **Model Evaluation**: Train the DNN model and evaluate its performance on both training and test sets.
8. **Results Visualization**: Display the evaluation metrics using Treeview widgets for clear tabular representation.
![image](https://github.com/JeroenKreuk/gui_dnn_manual_tuner/assets/85551796/f71d7f0a-763a-48fb-a77e-86975de54acc)

## Dependencies

- `tkinter`: GUI library for creating the application window.
- `pandas`: Data manipulation and analysis.
- `scikit-learn`: Machine learning tools for preprocessing and evaluation.
- `imbalanced-learn`: Library for dealing with imbalanced datasets.
- `tensorflow.keras`: Deep learning library for building and training neural networks.

## Usage

1. Execute the script.
2. Open a CSV dataset using the "Open database" button.
3. Follow the step-by-step process for data preprocessing, feature selection, and model tuning.
4. Visualize the evaluation metrics for the DNN on both training and test sets.

## Installation

```bash
pip install pandas scikit-learn imbalanced-learn tensorflow
