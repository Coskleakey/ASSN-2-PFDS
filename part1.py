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
