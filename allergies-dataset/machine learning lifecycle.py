from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns   
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

data = pd.read_csv(r"C:\Users\vicky\OneDrive\Documents\Junior Year (11th Grade)\7th Period Machine Learning\food-allergy-analysis-Zenodo.csv")

# Make ASTHMA_START into a binary variable (Our problem is to classify whether someone will develop an ASTHMA allergy based on their demographics)
col = 'ASTHMA_START'
binary_col_name = col.replace("START", "BINARY")
data[binary_col_name] = data[col].notna().astype(int)

# Store originals
data['GENDER_ORI'] = data['GENDER_FACTOR']
data['RACE_ORI'] = data['RACE_FACTOR']
data['ETHNICITY_ORI'] = data['ETHNICITY_FACTOR']

# Need to split '-' to encode
label_encoder = LabelEncoder()
columns = ["GENDER_FACTOR","RACE_FACTOR","ETHNICITY_FACTOR"]
for column in columns:
    data[column] = data[column].str.split(' - ').str[1]
# Encoding
data = pd.get_dummies(data, columns=columns, drop_first=True)
encoded_columns = [col for col in data.columns if col.startswith(tuple(columns))]

# Features
X = data[["BIRTH_YEAR"] + encoded_columns]
y = data[[binary_col_name]]  

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data for better performance in the model
scaler = StandardScaler()
X_train[["BIRTH_YEAR"]] = scaler.fit_transform(X_train[["BIRTH_YEAR"]])
X_test[["BIRTH_YEAR"]] = scaler.transform(X_test[["BIRTH_YEAR"]])

# ---this gets rid of warning errors at end---
y_train = y_train.values.ravel()  # Flatten the target variable to 1D array
y_test = y_test.values.ravel()    # Flatten the target variable for the test set

# Save X_test and y_test for later use in Flask
X_test.to_csv("X_test.csv", index=False)
pd.DataFrame(y_test, columns=["ASTHMA_BINARY"]).to_csv("y_test.csv", index=False)

# Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy=0.4, random_state=42)  # Instead of full balance
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

#-------------------------------------------------LOGISTIC REGRESSION CODE-----------------------------------------------------
# # Train a logistic regression model
# model = LogisticRegression(class_weight={0: 1, 1: 2.5}, random_state=42)
# model.fit(X_train_resampled, y_train_resampled)

# # Make predictions using the trained model
# y_pred_resampled = model.predict(X_test)

# # Evaluate the model's accuracy
# accuracy = accuracy_score(y_test, y_pred_resampled )
# print(f"Model Accuracy: {accuracy:.2f}")
# print("Original Model Evaluation:")
# print(classification_report(y_test, y_pred_resampled , zero_division=0))

# # Train with modified hyperparameters (still logistic regression)
# model_tuned = LogisticRegression(C=0.5, max_iter=200)
# model_tuned.fit(X_train_resampled, y_train_resampled)

# # Make predictions using the tuned model
# y_pred_tuned = model_tuned.predict(X_test)

# # Evaluate the tuned model's accuracy
# accuracy_tuned = accuracy_score(y_test, y_pred_tuned, )

# # Output the accuracy of the tuned model
# print(f"Tuned Model Accuracy: {accuracy_tuned:.2f}")
# print("Tuned Model Evaluation:")
# print(classification_report(y_test, y_pred_tuned, zero_division=0))

#-------------------------------------------------SVM CODE-----------------------------------------------------
# Convert y_train to a Pandas Series (only if necessary)
y_train = pd.Series(y_train, index=X_train.index)  

# Take a smaller sample (20% of training data)
X_train_sample = X_train_resampled.sample(frac=0.2, random_state=42)
y_train_sample = y_train_resampled[X_train_sample.index]  # Select matching y values

# Train SVM on the sample and predict
svm_model = SVC(kernel='linear', class_weight={0: 1, 1: 2.5}, C=1.0)
svm_model.fit(X_train_sample, y_train_sample)
y_pred_svm = svm_model.predict(X_test)

# Evaluate SVM model
print("SVM Model Accuracy:", accuracy_score(y_test, y_pred_svm))
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm, zero_division=1))
  
# Perform k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(svm_model, X_train_resampled, y_train_resampled, cv=kf, scoring='accuracy')

print("Cross-validation accuracy scores:", cv_scores)
print("Mean accuracy:", cv_scores.mean())

#-------------------------------------------------RANDOM FOREST CODE-----------------------------------------------------
# # Train a Random Forest model
# rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight={0: 1, 1: 2.5}, random_state=42)
# rf_model.fit(X_train_resampled, y_train_resampled)
# y_pred_rf = rf_model.predict(X_test)

# # Evaluate Random Forest model
# print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
# print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf, zero_division=1))
#---------------------------------------------ACCURACY AND CONFUSION MATRIX--------------------------------------------

# # Print all accuracies
# print(f"Logistic Regression Accuracy: {accuracy:.2f}")
# print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm):.2f}")
# print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")

# # Confusion matrix
# y_pred_ = model.predict(X_test)
# cm = confusion_matrix(y_test, y_pred_)

# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['No Asthma', 'Asthma'], yticklabels=['No Asthma', 'Asthma'])
# plt.title("Confusion Matrix - Logistic Regression Model")
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()
#-------------------------------------------------VISUALIZATION-----------------------------------------------------
# # Plot for Gender vs Asthma 
# plt.figure(figsize=(8, 6))
# sns.barplot(x='GENDER_ORI', y=binary_col_name, data=data, errorbar=None)
# plt.title('Asthma Likelihood by Gender')
# plt.xlabel('Gender')
# plt.ylabel('Proportion with Asthma')
# plt.show()

# # Plot for Race vs Asthma 
# plt.figure(figsize=(8, 6))
# sns.barplot(x='RACE_ORI', y=binary_col_name, data=data, errorbar=None)
# plt.title('Asthma Likelihood by Race')
# plt.xlabel('Race')
# plt.ylabel('Proportion with Asthma')
# plt.show()

# # Plot for Ethnicity vs Asthma 
# plt.figure(figsize=(8, 6))    
# sns.barplot(x='ETHNICITY_ORI', y=binary_col_name, data=data, errorbar=None)
# plt.title('Asthma Likelihood by Ethnicity')
# plt.xlabel('Ethnicity')
# plt.ylabel('Proportion with Asthma')
# plt.show()
#------------------------------------------------------DEPLOYMENT------------------------------------------------
with open("svm_pipeline-0.1.0.pkl", "rb") as file:
    model = pickle.load(file)

# !zip -r ./trained_pipeline-0.1.0.pkl.zip ./svm_pipeline-0.1.0.pkl

# Make a prediction
new_data = [["S0 - Male", "R0 - White","E0 - Non-Hispanic"]]  # Replace with actual values
prediction = model.predict(new_data)
print("Prediction:", prediction)
