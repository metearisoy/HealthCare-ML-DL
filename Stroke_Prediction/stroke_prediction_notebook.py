# Import Libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline

'''
StandartScaler
DecisionTree
Pipeline(StandartScaler, DecisinTree)
'''

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression DecisionTreeRegressor

import warnings
warnings.filterwarnings('ignore')

# Load and EDA
 
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

df = df.drop(['id'], axis =1)


df.info()

describe = df.describe().T

plt.figure()
sns.countplot(x='stroke', data=df, hue='stroke')
plt.title('Distribution of Stroke Class')
plt.show()

'''
4800 -> 0
250 -> 1

UNBALANCE !!!

Kcy: All results at 0 Accuracy: 
4800 / 5100 = 0.94 Misleading result.

To avoid being misled:

Confusion Matrix (CM)
F1 Score

Imbalanced dataset solution:

We need to increase the count of strokes (1), by collecting more data.
Down-sampling: Reduce the count of (0), but this will lead to data loss.

'''

# Missing Value

df.isnull().sum()


DT_bmi_pipe = Pipeline(steps=[
    ('scale', StandardScaler()),
    ('dtr', DecisionTreeRegressor())
    ])
    
    
X = df[['gender', 'age', 'bmi']].copy()

# male -> 0, female -> 1, other -> -1

X.gender = X.gender.replace({'Male':0, 'Female':1, 'Other': -1}).astype(np.uint8)


missing = X[X.bmi.isna()]

X = X[~X.bmi.isna()]
y = X.pop('bmi')


DT_bmi_pipe.fit(X,y)


predicted_bmi = pd.Series(DT_bmi_pipe.predict(missing[['gender', 'age']]), index=missing.index)


df.loc[missing.index, 'bmi'] = predicted_bmi

'''
To fill in the missing BMI values, we considered their gender and age.
We separated the missing values from the complete ones and used predictions to fill in the missing BMI values.
'''

df.isnull().sum()

# Model Prediction: encoding, training and testing

df['gender'] = df['gender'].replace({'Male':0, 'Female':1, 'Other': -1}).astype(np.uint8)
df['Residence_type'] = df['Residence_type'].replace({'Rural':0, 'Urban':1}).astype(np.uint8)
df['work_type'] = df['work_type'].replace({'Private':0, 'Self-employed':1,
                                           'Govt_job': 2, 'children': -1, 'Never_worked': -2}).astype(np.uint8)


X = df[['gender', 'age', 'hypertension', 'heart_disease','work_type', 'avg_glucose_level', 'bmi']]
y = df['stroke']



X_train , X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=42)
    
logreg_pipe = Pipeline(steps=[('scale', StandardScaler()), ('LR', LogisticRegression())])    


# Model Training

logreg_pipe.fit(X_train, y_train)


# Model Testing

y_pred = logreg_pipe.predict(X_test)
    
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Accuracy: \n', confusion_matrix(y_test, y_pred))
print('Classification report: \n', classification_report(y_test, y_pred))


'''
Accuracy: 
 [[483   0]
 [ 28   0]]
Classification report: 
               precision    recall  f1-score   support

           0       0.95      1.00      0.97       483
           1       0.00      0.00      0.00        28

    accuracy                           0.95       511
   macro avg       0.47      0.50      0.49       511
weighted avg       0.89      0.95      0.92       511
'''


#down-sample

from sklearn.utils import resample

class_0 = df[df['stroke'] == 0]
class_1 = df[df['stroke'] == 1] 

class_0_downsampled = resample(class_0,
                               replace=False,  n_samples=len(class_1),
                               random_state=42)

balanced_df = pd.concat([class_0_downsampled, class_1])

# Suffle

balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)


X = balanced_df[['gender', 'age', 'hypertension', 'heart_disease','work_type', 'avg_glucose_level', 'bmi']]
y = balanced_df['stroke']


X_train , X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=42)
    
logreg_pipe = Pipeline(steps=[('scale', StandardScaler()), ('LR', LogisticRegression())])


logreg_pipe.fit(X_train, y_train)

y_pred = logreg_pipe.predict(X_test)
    
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Accuracy: \n', confusion_matrix(y_test, y_pred))
print('Classification report: \n', classification_report(y_test, y_pred))


'''
Accuracy: 
 [[23  7]
 [ 5 15]]
Classification report: 
               precision    recall  f1-score   support

           0       0.82      0.77      0.79        30
           1       0.68      0.75      0.71        20

    accuracy                           0.76        50
   macro avg       0.75      0.76      0.75        50
weighted avg       0.77      0.76      0.76        50
'''

# Model save and load.


import joblib


joblib.dump(logreg_pipe, 'log_reg_model.pkl')


loaded_log_reg_pipe = joblib.load('log_reg_model.pkl')


new_patient_data = pd.DataFrame({
    'gender':[1],
    'age':[41],
    'hypertension': [0],
    'heart_disease':[0],
    'work_type':[1],
    'avg_glucose_level': [70],
    'bmi': [25]
    })


new_patient_data_result = loaded_log_reg_pipe.predict(new_patient_data)


new_patient_data_result_probalitiy = loaded_log_reg_pipe.predict_proba(new_patient_data)
