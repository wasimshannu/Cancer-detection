#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer prediction using different classifiers
# 
# ### Problem Statement:
# 
# Predict Breast cancer using the following classifiers
# 
# (1) Support Vector classifier
# 
# (2) Logistic Regression
# 
# (3) Random Forest Classifier
# 
# (4) Decision Tree classifier
# 
# (5) K Nearest Neighbours
# 
# (6) Naive Bayes
# 
# Link for the dataset : https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data,
# https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)
# 
# ### About the dataset:
# 
# The description about the data set can be downloaded from https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names
# 
# 

# #### Using the Wisconsin breast cancer diagnostic data set for predictive analysis
# 
# 
# Attribute Information:
# 
# 1) ID number
# 2) Diagnosis (M = malignant, B = benign)
# -3-32.Ten real-valued features are computed for each cell nucleus:
# 
#     a) radius (mean of distances from center to points on the perimeter)
#     
#     b) texture (standard deviation of gray-scale values)
#     
#     c) perimeter
#     
#     d) area
#     
#     e) smoothness (local variation in radius lengths)
#     
#     f) compactness (perimeter^2 / area - 1.0)
#     
#     g). concavity (severity of concave portions of the contour)
#     
#     h). concave points (number of concave portions of the contour)
#     
#     i). symmetry
#     
#     j). fractal dimension ("coastline approximation" - 1)
#     
# The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt # side-stepping mpl backend
import matplotlib.gridspec as gridspec # subplots
import warnings
warnings.filterwarnings("ignore")


# In[2]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


# In[3]:


df = pd.read_csv('data.csv')
pd.set_option('display.max_columns', None)


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


#to get information about the data
df.info()


# In[7]:


df.columns


# ### Descriptive statistis for each column

# In[8]:


df.describe()


# In[9]:


#to check if there is any null values
df.isnull().sum()


# In[10]:


df = df.drop(columns = ['id','Unnamed: 32'], axis = 1)


# In[11]:


df.head()


# In[12]:


df.diagnosis.unique()


# ### Creating two new dataframes based on the diagnosis column

# In[13]:


M = df[df['diagnosis'] == 'M'] # malignant cases
B = df[df['diagnosis'] == 'B'] # benign cases


# In[14]:


# print number of cases based on diagnosis column

print('Number of malignant cases:', len(M))
print('Number of benign cases:', len(B))


# ### Nucleus features vs Diagnosis

# In[15]:


sns.histplot(data=df, x="radius_mean", hue = 'diagnosis')
plt.title('radius_mean v/s diagnosis')
plt.xlabel('radius_mean')
plt.plot()


# In[16]:


sns.histplot(data=df, x="texture_mean", hue = 'diagnosis')
plt.title('texture_mean v/s diagnosis')
plt.xlabel('texture_mean')
plt.plot()


# In[17]:


sns.histplot(data=df, x="perimeter_mean", hue = 'diagnosis')

plt.title('perimeter_mean v/s diagnosis')
plt.xlabel('perimeter_mean')
plt.plot()


# In[18]:


sns.histplot(data=df, x="area_mean", hue = 'diagnosis')

plt.title('area_mean v/s diagnosis')
plt.xlabel('area_mean')
plt.plot()


# In[19]:


sns.histplot(data=df, x="smoothness_mean", hue = 'diagnosis')

plt.title('smoothness_mean v/s diagnosis')
plt.xlabel('smoothness_mean')
plt.plot()


# In[20]:


sns.histplot(data=df, x="compactness_mean", hue = 'diagnosis')

plt.title('compactness_mean')
plt.xlabel('compactness_mean')
plt.plot()


# In[21]:


sns.histplot(data=df, x="concavity_mean", hue = 'diagnosis')

plt.title('concavity_mean')
plt.xlabel('concavity_mean')
plt.plot()


# In[22]:


sns.histplot(data=df, x="concave points_mean", hue = 'diagnosis')

plt.title('concavepoints_mean')
plt.xlabel('concavepoints_mean')
plt.plot()


# In[23]:


sns.histplot(data=df, x="symmetry_mean", hue = 'diagnosis')

plt.title('symmetry_mean')
plt.xlabel('symmetry_mean')
plt.plot()


# In[24]:


sns.histplot(data=df, x = "fractal_dimension_mean", hue = 'diagnosis')

plt.title('fractal_dimension_mean')
plt.xlabel('fractal_dimension_mean')
plt.plot()


# In[25]:


#changing M and B to numerical values 1 and 0 in diagnosis column
#df['diagnosis'] = [1 if i == 'M' else 0 for i in df['diagnosis']]

from sklearn.preprocessing import LabelEncoder
L = LabelEncoder()
L.fit(df['diagnosis'])
L = L.transform(df['diagnosis'])


# ### checking for correlation between the variables

# In[26]:


plt.figure(figsize=(15,12))
sns.heatmap(df.corr(),annot=True)
plt.ioff()


# # Python function that you can use to extract highly correlated columns in a heatmap

# In[36]:


def extract_highly_correlated_columns(df, threshold=0.9):
    """
    This function takes a pandas DataFrame and a correlation threshold as input,
    and returns a list of highly correlated column pairs based on a heatmap.
    """
    corr_matrix = df.corr()
    # Create a mask for the upper triangle of the correlation matrix
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    # Plot the correlation heatmap
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, mask=mask)
    # Find pairs of highly correlated columns
    highly_correlated_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                col_pair = (corr_matrix.columns[i], corr_matrix.columns[j])
                highly_correlated_pairs.append(col_pair)
    return highly_correlated_pairs
highly_correlated_pairs = extract_highly_correlated_columns(df, threshold=0.9)

print(highly_correlated_pairs)


# ## Observations
# 
# mean values of cell radius, perimeter, area, compactness, concavity and concave points can be used in classification of the cancer. Larger values of these parameters tends to show a correlation with malignant tumors.

# In[27]:


def extract_highly_correlated_columns(df, threshold=0.8):
    """
    This function takes a pandas DataFrame and a correlation threshold as input,
    and returns a list of highly correlated column pairs based on a heatmap.
    """
    corr_matrix = df.corr()
    # Create a mask for the upper triangle of the correlation matrix
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    # Plot the correlation heatmap
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, mask=mask)
    # Find pairs of highly correlated columns
    highly_correlated_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                col_pair = (corr_matrix.columns[i], corr_matrix.columns[j])
                highly_correlated_pairs.append(col_pair)
    return highly_correlated_pairs

    return highly_correlated_pairs, common_names

highly_correlated_pairs = extract_highly_correlated_columns(df, threshold=0.9)
print(highly_correlated_pairs)


# ### Creating a test set and a training set
# 

# In[28]:


x = df.drop(['diagnosis'], axis = 1)
y = df['diagnosis']


# ### Splitting the data into test and train data

# In[29]:


train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=0.2, random_state=42)


# ### Standardizing the data

# In[30]:


sc = StandardScaler()
train_X = sc.fit_transform(train_x) # scaling the train data
test_X= sc.transform(test_x)


# # Generic function for making a classification model and accessing the performance. 

# In[31]:


def predict_all_ml_models(df, target_col):
    """
    This function takes a pandas DataFrame and a target column as input,
    and predicts all machine learning models on the data and calculates the accuracy scores.
    """
    # Split the data into train and test sets
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Define the machine learning models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Support Vector Machine": SVC(),
        "K-Nearest Neighbors": KNeighborsClassifier()
    }
    # Fit and predict all machine learning models and calculate the accuracy scores
    accuracy_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores[name] = accuracy
    return accuracy_scores


# # Predict all machine learning models and calculate accuracy scores

# In[32]:


accuracy_scores = predict_all_ml_models(df, target_col="diagnosis")

print(accuracy_scores)


# ### Comparison of all models

# In[33]:


def scores_to_dataframe(scores):
    """
    This function takes a dictionary of model names and accuracy scores as input,
    and converts it into a pandas DataFrame.
    """
    df = pd.DataFrame.from_dict(scores, orient="index", columns=["Accuracy Score"])
    df.index.name = "Model Name"
    return df

# Convert the accuracy scores dictionary into a pandas DataFrame
accuracy_scores = predict_all_ml_models(df, target_col="diagnosis")
df_scores = scores_to_dataframe(accuracy_scores)

print(df_scores)


# ## Conclusion
# 
# The best model to be used for diagnosing breast cancer as found in this analysis is the Random Forest model. It gives a prediction accuracy of ~95%.
# we prefer random forest over logistic regression because it is not effected by outliers.
