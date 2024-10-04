# Crop_Recommendation_System
This project provides a machine learning-based solution to recommend the most suitable crops for a given set of environmental conditions. It uses the Support Vector Machine (SVM) algorithm to classify the crops based on soil and climate parameters.

#Features
Predicts suitable crops based on:
Nitrogen (N)
Phosphorus (P)
Potassium (K)
Temperature
Humidity
pH
Rainfall

#Performance evaluation metrics:
Accuracy
Classification Report (Precision, Recall, F1-score)
Confusion Matrix

#Dataset
The model uses a dataset containing various environmental conditions and the corresponding crop labels. The data is pre-processed and split into training and testing sets for model evaluation.

#Dataset file: Crop_recommendation.csv

#Dependencies
Python 3.x
NumPy
Pandas
scikit-learn
Matplotlib
Seaborn

The model will train on the dataset, evaluate its accuracy on the test set, and print performance metrics such as accuracy and a detailed classification report.

#Output
Accuracy: Displays the accuracy of the model on the test dataset.
Classification Report: Provides details about precision, recall, and F1-score for each crop class.
Confusion Matrix: Can be plotted for further analysis of the model's performance.

