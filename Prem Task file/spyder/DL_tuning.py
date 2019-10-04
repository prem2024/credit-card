#!/usr/bin/env python
# coding: utf-8

# # Artificial Neural Network

# # Part 1 - Data Preprocessing
# 

# # STEP 1 : IMPORTING LIBRARIES

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# set the working directory 
import os 
os.getcwd()
os.chdir("F:\\classes\\Python_learning\\Notebooks\\MSR")


# In[3]:


# Load the data

dataset = pd.read_csv('F:\\classes\\Python_learning\\Notebooks\\MSR\\default_of_credit_card_clients.csv')
dataset.head()


# In[4]:


# # Skip  rows at specific index the first row skip because esaliy understand the dataset, python zero base index
dataset = pd.read_csv('F:\\classes\\Python_learning\\Notebooks\\MSR\\default_of_credit_card_clients.csv' , skiprows=[0])
dataset


# In[5]:


# Now lets see how the data looks like
dataset.head()


# In[6]:


# Checking the last few entries of dataset to see the distribution of data
dataset.tail()


# In[7]:


dataset.describe() # statistical view


# # Step 2 : Preprocessing & Cleaning of Data

# In[8]:


dataset.shape


# Means there are 30,000 entries with 25 columns

# In[9]:


# Checking the object type of all the columns to see if there is not a object type mismatch in any column 
print(dataset.dtypes)


# From the above output it is clear that there is no object type mismatch in any column.

# In[10]:


#Checking the number of Null entries in the data columnwise.
dataset.isnull().sum()


# From the above output it is clear that there is no  null values type  in any column.

# In[11]:


dataset.info() # dataset information.


# # STEP 3. Data Visualization & Exploratory Data Analysis

# In[12]:


limit_bal = dataset['LIMIT_BAL'].value_counts() #Amount of given credit (includes individual and family/supplementary credit)
limit_bal


# In[13]:


dataset.LIMIT_BAL.min() # minimum amount.


# In[14]:


dataset.LIMIT_BAL.max() # mximum amount.


# In[15]:


dataset.LIMIT_BAL.mean() # mean of amount or average.


# In[16]:


# Checking the number of counts of defaulters and non defaulters sexwise
g=sns.countplot(x="LIMIT_BAL", data=dataset,hue="default payment next month", palette="muted")


# In[17]:


# Sex meaning is:

# 1 : male
# 2 : female
dataset.SEX.value_counts() #this is fine, more women than men


# In[18]:


# Checking the number of counts of defaulters and non defaulters sexwise
g=sns.countplot(x="SEX", data=dataset,hue="default payment next month", palette="muted")


# It is evident from the above output that females have overall less default payments compare to males

# In[19]:


# Education status meaning is:

# 1 : graduate school
# 2 : university
# 3 : high school
# 4 : others
# 5 : unknown
# 6 : unknow
dataset.EDUCATION.value_counts() # Education cloumn 


# In[20]:


g=sns.countplot(x="EDUCATION", data=dataset,hue="default payment next month", palette="muted")


# It is evident from the above output that university persons have overall less default payments compare to males

# In[21]:


# Marriage status meaning is:

# 0 : unknown (let's consider as others as well)
# 1 : married
# 2 : single
# 3 : others
dataset['MARRIAGE'].value_counts()


# In[22]:


g=sns.countplot(x="MARRIAGE", data=dataset,hue="default payment next month", palette="muted")


# From the above plot it is clear that those people who have marital status single have less default payment wrt married status people.

# In[23]:


sns.boxplot(x='default payment next month',y='AGE',data=dataset,palette='Set2')


# In[24]:


sns.boxplot(x='default payment next month',hue='MARRIAGE', y='AGE',data=dataset,palette="Set3") ## Order to plot the categorical levels by marriage


# In[25]:


sns.pairplot(dataset, hue = 'default payment next month', vars = ['AGE', 'MARRIAGE', 'SEX', 'EDUCATION', 'LIMIT_BAL'] )


# In[26]:


sns.jointplot(x='LIMIT_BAL',y='AGE',data=dataset)


# Distribution of LIMIT BALANCE and AGE 

# In[27]:


g = sns.FacetGrid(data=dataset,col='SEX')
g.map(plt.hist,'AGE')


# Distribution of Male and Female according to their age

# # STEP 4. Finding Correlation

# In[28]:


X = dataset.drop(['default payment next month'],axis=1) #Drop Remove rows or columns by specifying label names and corresponding axis, or by specifying directly index or column names. 
y = dataset['default payment next month']


# In[29]:


X.head()


# In[30]:


y.head()


# In[31]:


X.corrwith(dataset['default payment next month']).plot.bar(
        figsize = (20, 10), title = "Correlation with Default", fontsize = 20,
        rot = 90, grid = True)


# It seems from the above graph is that most negatively correlated feature is LIMIT_BAL but we cannot blindly remove this feature because according to me it is very important feature for prediction. ID is unimportant and it has no role in prediction so we will remove it later.

# In[32]:


dataset2 = dataset.drop(columns = ['default payment next month'])


# In[33]:


dataset2.head()


# In[34]:


sns.set(style="white")

# Compute the correlation matrix
corr = dataset2.corr()


# In[35]:


mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(250, 15, as_cmap=True)
# A heatmap is a two-dimensional graphical representation of data where the individual values
#_____that are contained in a matrix are represented as colors. 
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# # STEP 5 : SPLITTING DATA INTO TRAINING AND TESTING SET

# 
# 
# The training dataset and test dataset must be similar, usually have the same predictors or variables. They differ on the observations and specific values in the variables. If you fit the model on the training dataset, then you implicitly minimize error or find correct responses. The fitted model provides a good prediction on the training dataset. Then you test the model on the test dataset. If the model predicts good also on the test dataset, you have more confidence. You have more confidence since the test dataset is similar to the training dataset, but not the same nor seen by the model. It means the model transfers prediction or learning in real sense.
# 
# So,by splitting dataset into training and testing subset, we can efficiently measure our trained model since it never sees testing data before.Thus it's possible to prevent overfitting.
# 
# I am just splitting dataset into 20% of test data and remaining 80% will used for training the model.

# In[36]:


X = dataset.iloc[:, 1:24].values
y = dataset.iloc[:, 24].values


# In[37]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[38]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


# # # Part 2 - Now let's make the ANN!

# In[39]:


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential # neural network layers
from keras.layers import Dense     # hidden layers


# In[40]:



# Initialising the ANN
classifier = Sequential()


# In[41]:


# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 23))


# In[42]:


# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))


# In[43]:


# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


# In[44]:


classifier.summary()


# In[45]:


# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[46]:


# Fitting the ANN to the Training set
classifier.fit(X_train_scaled, y_train, batch_size = 10, epochs = 100)


# # # # Part 3 - Making predictions and evaluating the model
# 

# In[47]:


# # Predicting the Test set results
y_pred = classifier.predict(X_test_scaled)
y_pred = (y_pred > 0.5)


# In[48]:



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[49]:


from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
roc=roc_auc_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

results = pd.DataFrame([['ANN', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results


#  # Predicting a single new observation
# 

# In[ ]:




# •	LIMIT_BAL: 20000 ,90000
# •	SEX: Gender 2,    2
# •	EDUCATION: 2,     2
# •	MARRIAGE:1     2
# •	AGE: 24,,    34
# •	PAY_0:2 ,  0
# •	PAY_2: 2,   0
# •	PAY_3:-1,   0
# •	PAY_4: -1,  0
# •	PAY_5:-2,   0
# •	PAY_6: -2,  0
# •	BILL_AMT:3,913 , 29239
# •	BILL_AMT2:3,102, 14027
# •	BILL_AMT3: 6,89, 13559
# •	BILL_AMT4: 0, , 14331
# •	BILL_AMT5: 0,,  14948
# •	BILL_AMT6: 0,   15549
# •	PAY_AMT1: 0,    1518 
# •	PAY_AMT2:689,    1500
# •	PAY_AMT3: 0,     1000
# •	PAY_AMT4: 0,     1000
# •	PAY_AMT5: 0,     1000
# •	PAY_AMT6: 0      5000


# •	default.payment.next.month: Default payment (1=yes, 0=no)


# In[51]:


new_prediction = classifier.predict(sc.transform(np.array([[20000, 2,2,1,24,2,2,-1,-1,-2,-2,3913,3102,689,0,0,0,0,689,0,0,0,0]])))
new_prediction = (new_prediction > 0.5)
new_prediction


# In[52]:


new_prediction_1 = classifier.predict(sc.transform(np.array([[90000,2,2,2,34,0,0,0,0,0,0,29239,14027,13559,14331,14948,15549,1518,1500,1000,1000,1000,5000]])))
new_prediction_1= (new_prediction_1 > 0.5)
new_prediction_1


# # conclousion

# # # Predicting a single new observation True = defalut payment ,False= not defalut payment , the above predicted values are correct 
# more accuracy and best parameters , use model tunning like Random serch CV, grid search CV

# # # Part 4 - Evaluating, Improving and Tuning the ANN
# 

# In[ ]:



# Part 4 - Evaluating, Improving and Tuning the ANN

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 23))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train_scaled, y = y_train, cv = 10, n_jobs = -1)


# In[ ]:


mean = accuracies.mean()
variance = accuracies.std()


# In[ ]:



# Improving the ANN
# Dropout Regularization to reduce overfitting if needed

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim =23))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train_scaled, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


# tuning proces in ANN very long time taken process

# In[ ]:




