# Import Libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

# Load Dataset
df = pd.read_csv('kidney_disease.csv')

df.drop('id', axis= 1, inplace=True)


df.columns = ['age',  'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cell', 
              'pus_cell', 'pus_cell_clumbs', 'bacteria', 'blood_glucose_random', 'blood_urea',
       'serum_creatinin', 'sodium', 'potassium', 'hemoglobin', 'packed_cell_volum', 
       'white_blood__cell_count', 'red_blood_cell_count', 'hypertension', 'diabetes_mellitus',
       'coronary_artery_disease', 'appetite', 'peda_edema', 'aanemia', 'class']


df.info()

describe = df.describe().T

df['packed_cell_volum'] = pd.to_numeric(df['packed_cell_volum'], errors='coerce')
df['white_blood__cell_count'] = pd.to_numeric(df['white_blood__cell_count'], errors='coerce')
df['red_blood_cell_count'] = pd.to_numeric(df['red_blood_cell_count'], errors='coerce')

df.info()



# EDA : KDE

cat_cols = [col for col in df.columns if df[col].dtype == 'object']
num_cols = [col for col in df.columns if df[col].dtype != 'object']

for col in cat_cols:
    print(f'{col}: {df[col].unique()} values')
    
'''
diabetes_mellitus: ['yes' 'no' ' yes' '\tno' '\tyes' nan]
coronary_artery_disease: ['no' 'yes' '\tno' nan]
class: ['ckd' 'ckd\t' 'notckd']
'''

df['diabetes_mellitus'].replace(to_replace ={' yes': 'yes','\tno': 'no', '\tyes': 'yes'}, inplace=True)
df['coronary_artery_disease'].replace(to_replace ={'\tno': 'no'}, inplace=True)
df['class'].replace(to_replace ={'ckd\t': 'ckd'}, inplace=True)

df['class'] = df['class'].map({'ckd':0, 'notckd':1})

plt.figure(figsize=(15,15))
plotnumber = 1

for col in num_cols:
    if plotnumber <= 14:
        az = plt.subplot(3, 5, plotnumber)
        sns.distplot(df[col])
        plt.xlabel(col)
        
    plotnumber +=1
    
plt.tight_layout()
plt.show()


'''plt.figure()
sns.heatmap(df.corr(), annot= True, linecolor='white', linewidths=2) 
plt.show()'''


def kde(col):
    grid = sns.FacetGrid(df, hue='class', height=6, aspect=2)
    grid.map(sns.kdeplot, col)
    grid.add_legend()
    
kde('hemoglobin')
kde('white_blood__cell_count')
kde('packed_cell_volum')
kde('red_blood_cell_count')
kde('albumin')
kde('specific_gravity')
    
    
# Preprocessing: Missin value problem, Feature encoding

df.isna().sum().sort_values(ascending=True)

def solve_mv_random_value(feature):
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    random_sample.index = df[df[feature].isnull()].index
    df.loc[df[feature].isnull(), feature] = random_sample    
    
    
for col in num_cols:
    solve_mv_random_value(col)
    
df[num_cols].isna().sum()


def solve_mv_mode(feature):
    mode = df[feature].mode()[0]
    df[feature] = df[feature].fillna(mode)
    
    
solve_mv_random_value('red_blood_cell')
solve_mv_random_value('pus_cell')
    
for col in cat_cols:
    solve_mv_mode(col)
    

df[cat_cols].isnull().sum()


for col in cat_cols:
    print(f'{col}: {df[col].nunique()}')
    
    
encoder = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])


# Model (DT) training and testing

X = df.drop(['class'], axis= 1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

dtc_acc = accuracy_score(y_test, y_pred)


cm = confusion_matrix(y_test, y_pred)

cr = classification_report(y_test, y_pred)

print('Confusion matrix: \n', cm)
print('classification_report: \n', cr)

# DT Visualization - feature importance

class_names = ['ckd', 'notckd']

plt.figure(figsize=(20,10))
plot_tree(dtc, feature_names= X.columns.tolist(), filled=True, rounded=True, fontsize=7)
plt.show()


X.columns

feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': dtc.feature_importances_})



