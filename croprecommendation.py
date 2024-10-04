import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('/content/Crop_recommendation.csv')

X = data[['Nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = svm.SVC(kernel='linear')

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
df = pd.DataFrame(report).transpose()

# Remove the 'support' column and the 'accuracy' row for plotting
df = df.drop(columns=["support"]).drop(index=["accuracy"])

# Plotting precision, recall, and f1-score for each class
df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(12, 8))
plt.title('Classification Report')
plt.xlabel('Classes')
plt.ylabel('Scores')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.show()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(8, 6))
plt.bar(['Accuracy'], [accuracy], color='skyblue')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.ylim(0, 1)
plt.show()

new_data = pd.DataFrame([[20, 15, 10, 25, 70, 6.5, 100]], columns=X.columns)
predicted_crop_name = clf.predict(new_data)[0]
print("Predicted Crop:", predicted_crop_name)

new_data = pd.DataFrame([[10, 15, 15, 25, 60, 5, 70]], columns=X.columns)
predicted_crop_name = clf.predict(new_data)[0]
print("Predicted Crop:", predicted_crop_name)

new_data = pd.DataFrame([[10, 12, 10, 20, 40, 6.2, 50]], columns=X.columns)
predicted_crop_name = clf.predict(new_data)[0]
print("Predicted Crop:", predicted_crop_name)

new_data = pd.DataFrame([[15, 20, 7.5, 23.768, 65.89, 6.46, 90]], columns=X.columns)
predicted_crop_name = clf.predict(new_data)[0]
print("Predicted Crop:", predicted_crop_name)

new_data = pd.DataFrame([[10, 6, 5, 20, 60, 5.5, 80]], columns=X.columns)
predicted_crop_name = clf.predict(new_data)[0]
print("Predicted Crop:", predicted_crop_name)

new_data = pd.DataFrame([[15, 20, 15, 30, 65, 7.0, 80]], columns=X.columns)
predicted_crop_name = clf.predict(new_data)[0]
print("Predicted Crop:", predicted_crop_name)

import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame containing the dataset
# Extract the crop labels and calculate their frequency
crop_label_counts = data['label'].value_counts()

# Plotting the bar plot
plt.figure(figsize=(8, 6))
crop_label_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Crop Labels')
plt.xlabel('Crop Labels')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Assuming 'y_test' contains the true labels and 'y_pred' contains the predicted labels
# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print(data.label.unique())



