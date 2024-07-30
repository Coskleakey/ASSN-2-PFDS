import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import resample

# Load and inspect dataset
medical_data = pd.read_csv('noshowappointments.csv')
print(medical_data.head())
print(medical_data.info())
print(medical_data.describe())

# Check for Missing Values
print(medical_data.isnull().sum())

# Plot the distribution of the target variable

sns.countplot(x='No-show', data=medical_data)
plt.title('Distribution of No-show Appointments')
plt.xlabel('No-show')
plt.ylabel('Count')
plt.show()

# Data cleaning and preprocessing
medical_data['No-show'] = medical_data['No-show'].apply(lambda x: 1 if x == 'Yes' else 0)
medical_data['Gender'] = medical_data['Gender'].apply(lambda x: 1 if x == 'F' else 0)

# Check class distribution
print(medical_data['No-show'].value_counts())

# Handle class imbalance by upsampling the minority class
df_majority = medical_data[medical_data['No-show'] == 0]
df_minority = medical_data[medical_data['No-show'] == 1]

df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=len(df_majority),    # to match majority class
                                 random_state=42) # reproducible results

medical_data_balanced = pd.concat([df_majority, df_minority_upsampled])

# Plot the balanced distribution of the target variable
sns.countplot(x='No-show', data=medical_data_balanced)
plt.title('Balanced Distribution of No-show Appointments')
plt.xlabel('No-show')
plt.ylabel('Count')
plt.show()
# Convert categorical features to numeric
# Assume features to drop for this example
X_medical = medical_data_balanced.drop(['No-show', 'ScheduledDay', 'AppointmentDay', 'PatientId', 'AppointmentID', 'Neighbourhood'], axis=1)
y_medical = medical_data_balanced['No-show']

# Split data into training and testing sets
X_medical_train, X_medical_test, y_medical_train, y_medical_test = train_test_split(X_medical, y_medical, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_medical_train_scaled = scaler.fit_transform(X_medical_train)
X_medical_test_scaled = scaler.transform(X_medical_test)

# Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_medical_train_scaled, y_medical_train)
y_pred_logistic = logistic_model.predict(X_medical_test_scaled)

# Evaluation
print("Logistic Regression:")
print("Confusion Matrix:\n", confusion_matrix(y_medical_test, y_pred_logistic))
print("Accuracy:", accuracy_score(y_medical_test, y_pred_logistic))
print("Precision:", precision_score(y_medical_test, y_pred_logistic, zero_division=0))
print("Recall:", recall_score(y_medical_test, y_pred_logistic, zero_division=0))
print("F1 Score:", f1_score(y_medical_test, y_pred_logistic, zero_division=0))

# Visualize Confusion Matrix
cm = confusion_matrix(y_medical_test, y_pred_logistic)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
