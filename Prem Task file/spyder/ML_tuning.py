#!/usr/bin/env python
# coding: utf-8

# # STEP 1 : IMPORTING LIBRARIES

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve, GridSearchCV
from time import time
import os
from IPython.display import display


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
# A heatmap is a two-dimensional graphical representation of data where the individual values that are contained in a matrix are represented as colors. 
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


# # STEP 6: Normalizing the data : Feature Scaling

# Feature scaling through standardization can be an important preprocessing step for many machine learning algorithms. Standardization involves rescaling the features such that they have the properties of a standard normal distribution with a mean of zero and a standard deviation of one.
# 
# While many algorithms (such as SVM, K-nearest neighbors, and logistic regression) require features to be normalized,
# 
# 

# In[38]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


# # STEP 7: Applying Machine Learning Models

# 1.LogisticRegression

# In[39]:


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression(random_state = 0)

start =time()
logistic.fit(X_train_scaled, y_train)
end=time()
train_time_logistic =end -start


# In[40]:


# Predicting the Test set results
y_pred = logistic.predict(X_test_scaled)


# In[41]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[42]:


cm


# In[43]:


from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
roc=roc_auc_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

results = pd.DataFrame([['Logistic', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results


# 2.Support Vector Machine (SVM)

# In[44]:


# Fitting Support Vector Machine (SVM) to the Training set
from sklearn.svm import SVC 

svc_model = SVC(kernel='rbf', gamma=0.1,C=100)

start = time()
svc_model.fit(X_train_scaled, y_train)
end=time()
train_time_svc=end-start


# In[45]:



# Predicting the Test set results
y_pred_svc = svc_model.predict(X_test_scaled)


# In[46]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_svc)
cm


# In[47]:



from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
roc=roc_auc_score(y_test, y_pred_svc)
acc = accuracy_score(y_test, y_pred_svc)
prec = precision_score(y_test, y_pred_svc)
rec = recall_score(y_test, y_pred_svc)
f1 = f1_score(y_test, y_pred_svc)

model_results = pd.DataFrame([['SVC ', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results


# 3.KNNeighborsClassifier

# In[48]:


# Fitting KNNeighborsClassifier  to the Training set
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7)

start = time()
knn.fit(X_train_scaled, y_train)
end=time()

train_time_knn=end-start


# In[49]:


# Predicting the Test set results
y_pred_g = knn.predict(X_test_scaled)


# In[50]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_g)
cm


# In[51]:


# Model eveluation 
from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
roc=roc_auc_score(y_test, y_pred_g)
acc = accuracy_score(y_test, y_pred_g)
prec = precision_score(y_test, y_pred_g)
rec = recall_score(y_test, y_pred_g)
f1 = f1_score(y_test, y_pred_g)

model_results = pd.DataFrame([['KNN 7', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results


# # 4.Decision Tree Classification

# In[52]:


# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
Decision_Tree = DecisionTreeClassifier(max_depth = 3,criterion = 'entropy', random_state = 0)

start = time()
Decision_Tree.fit(X_train_scaled, y_train)
end = time()
train_time_DT=end-start


# In[53]:


# Predicting the Test set results
y_pred_DT = Decision_Tree.predict(X_test_scaled)


# In[54]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_DT)
cm


# In[55]:


# Model eveluation 
from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
roc=roc_auc_score(y_test, y_pred_DT)
acc = accuracy_score(y_test, y_pred_DT)
prec = precision_score(y_test, y_pred_DT)
rec = recall_score(y_test, y_pred_DT)
f1 = f1_score(y_test, y_pred_DT)

model_results = pd.DataFrame([['Decision_Tree', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results


# # 5. Random Forest Tree
#     
# 
# Applying Random Forest with 100 trees and criterion entropy
# 
# 

# In[56]:


# train the model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state = 47, 
                                    criterion = 'entropy',n_estimators=100)
start = time()
classifier.fit(X_train_scaled, y_train)
end=time()
train_time_r100=end-start


# In[57]:


# Predicting the Test set results
y_pred_r = classifier.predict(X_test_scaled)


# In[58]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_r)
cm


# In[59]:


#model eveluation
from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
roc=roc_auc_score(y_test, y_pred_r)
acc = accuracy_score(y_test, y_pred_r)
prec = precision_score(y_test, y_pred_r)
rec = recall_score(y_test, y_pred_r)
f1 = f1_score(y_test, y_pred_r)
model_results = pd.DataFrame([['Random_forest_ent100 ', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results


# 6.AdaBoostClassifier

# In[60]:


from sklearn.ensemble  import AdaBoostClassifier
adaboost =AdaBoostClassifier()

start = time()
adaboost.fit(X_train_scaled, y_train)
end = time()
train_time_ada=end-start


# In[61]:


y_pred_ada= adaboost.predict(X_test_scaled)


# In[62]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_ada)
cm


# In[63]:


#model eveluation
from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
roc=roc_auc_score(y_test, y_pred_ada)
acc = accuracy_score(y_test, y_pred_ada)
prec = precision_score(y_test, y_pred_ada)
rec = recall_score(y_test, y_pred_ada)
f1 = f1_score(y_test, y_pred_ada)
model_results = pd.DataFrame([['Adaboost ', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results


# 7. XGBoost 

# In[64]:


from xgboost import XGBClassifier
xgb_classifier = XGBClassifier()
start = time()

xgb_classifier.fit(X_train_scaled, y_train,verbose=True)
end=time()
train_time_xgb=end-start


# In[65]:


y_pred_xgb = xgb_classifier.predict(X_test_scaled)


# In[66]:


# apply confustion matrix
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_xgb)
cm


# In[67]:


from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
roc=roc_auc_score(y_test, y_pred_xgb)
acc = accuracy_score(y_test, y_pred_xgb)
prec = precision_score(y_test, y_pred_xgb)
rec = recall_score(y_test, y_pred_xgb)
f1 = f1_score(y_test, y_pred_xgb)

model_results = pd.DataFrame([['XGboost', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results,sort=True)
results


# # 
# 8.Linear classifiers (SVM, logistic regression, a.o.) with SGD training.

# In[68]:


from sklearn import linear_model
sgd = linear_model.SGDClassifier(max_iter=1000)
start = time()
sgd.fit(X_train_scaled, y_train)
end=time()
train_time_sgd=end-start


# In[69]:


y_pred_sgd = sgd.predict(X_test_scaled)


# In[70]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_sgd)
cm


# In[71]:


from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
roc=roc_auc_score(y_test, y_pred_sgd)
acc = accuracy_score(y_test, y_pred_sgd)
prec = precision_score(y_test, y_pred_sgd)
rec = recall_score(y_test, y_pred_sgd)
f1 = f1_score(y_test, y_pred_sgd)

model_results = pd.DataFrame([['SGD 1000 iter', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results,sort=True)
results


# # 9. GradientBoostingClassifier

# In[72]:


from sklearn  import ensemble
gboost =ensemble.GradientBoostingClassifier()
start = time()
gboost.fit(X_train_scaled, y_train)
end=time()
train_time_g=end-start


# In[73]:


y_pred_gboost = gboost.predict(X_test_scaled)


# In[74]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_gboost)
cm


# In[75]:


from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
roc=roc_auc_score(y_test, y_pred_gboost)
acc = accuracy_score(y_test, y_pred_gboost)
prec = precision_score(y_test, y_pred_gboost)
rec = recall_score(y_test, y_pred_gboost)
f1 = f1_score(y_test, y_pred_gboost)

model_results = pd.DataFrame([['Gboost', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results,sort=True)
results


# In[ ]:





# In[ ]:





# # STEP 8 : ANALYZING AND COMPARING TRAINING TIME OF MACHINE LEARNING MODELS

# In[76]:


import matplotlib.pyplot as plt
import numpy as np
model = ['Adaboost','XGBoost','SGD', 'SVC', 'GBOOST', 'Random forest', 'KNN7','logistic','Decision']
Train_Time = [
    train_time_ada,
    train_time_xgb,
    train_time_sgd,
    train_time_svc,
    train_time_g,
    train_time_r100,
    train_time_DT,
    train_time_logistic,
    train_time_knn
]
index = np.arange(len(model))
plt.bar(index, Train_Time)
plt.xlabel('Machine Learning Models', fontsize=15)
plt.ylabel('Training Time', fontsize=15)
plt.xticks(index, model, fontsize=8, )
plt.title('Comparison of Training Time of all ML models')
plt.show()


# As from the above graph it is evident that Adaboost and XGboost have taken very less time to train in comparison to other models where as SVC has taken maximum time the reason may be we have passed some crucial parameters to SVC.
# 
# 

# # STEP 9. Model Optimization

# Random search outperformed grid search on this dataset across every number of iterations. Also random search seemed to converge to an optimum more quickly than grid search, which means random search with fewer iterations is comparable to grid search with more iterations.
# 
# In highdimensional parameter space, grid search would perform worse with the same iterations because points become more sparse. Also it is common that one of the hyperparameters is unimportant to finding the optimal hyperparameters, in which case grid search wastes a lot of iterations where as random search does not waste any iteration.
# 
# Now we will optimize models accuracy using Randomsearch cv.As shown in above table Adaboost performs best in this dataset. So we will try to further optimize adaboost and SVC by fine tuning its hyperparameters.
# 
# 

# # Parameter Tuning using random Searchcv

# In[77]:


# Adaboost


# In[78]:


from sklearn.model_selection import RandomizedSearchCV, cross_val_score
param_dist = {
      'n_estimators': [10,20,50,100,120,150,200],  
    'random_state':[47],
        'learning_rate':[0.1,0.01,0.001,0.0001]}

# run randomized search
n_iter_search =20
random_search = RandomizedSearchCV(adaboost, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=5)


random_search.fit(X_train_scaled,y_train)


# In[79]:


random_search.best_params_


# In[80]:


# Now, lets see best parameters predction and model evaluation


# In[81]:


y_pred_ada = random_search.predict(X_test_scaled)
from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
roc=roc_auc_score(y_test, y_pred_ada)
acc = accuracy_score(y_test, y_pred_ada)
prec = precision_score(y_test, y_pred_ada)
rec = recall_score(y_test, y_pred_ada)
f1 = f1_score(y_test, y_pred_ada)

results_tuned = pd.DataFrame([['Adaboost Tuned', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results_tuned


# In[82]:


#XGBoost


# In[83]:


from sklearn.model_selection import  RandomizedSearchCV, cross_val_score
param_dist ={'n_estimators': [50,100,150,200], 'max_depth': [3,5,7,10], 'min_child_weight': [2,3,4,5]} 

# run randomized search
n_iter_search =10
random_search = RandomizedSearchCV(xgb_classifier, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=5)


random_search.fit(X_train_scaled,y_train)


# In[84]:


random_search.best_params_


# In[85]:


# Now, lets see best parameters predction and model evaluation


# In[86]:


y_pred_xgb = random_search.predict(X_test_scaled)
from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
roc=roc_auc_score(y_test, y_pred_xgb)
acc = accuracy_score(y_test, y_pred_xgb)
prec = precision_score(y_test, y_pred_xgb)
rec = recall_score(y_test, y_pred_xgb)
f1 = f1_score(y_test, y_pred_xgb)

model =  pd.DataFrame([['XGBoost Tuned', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results_tuned = results_tuned.append(model, ignore_index = True)
results_tuned


# That's great all the metrics parameters accuracy, F1 score Precision, ROC, Recall iof the two models adaboost and XGBoost are optimized now. Further we can also try some other combination of parameters to see if there will be further improvement or not.

# # Plotting of ROC Curve
# 
# 

# In[87]:


from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble  import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

plt.figure()

# Add the models to the list that you want to view on the ROC plot
models = [
{
    'label': 'Adaboost',
    'model': AdaBoostClassifier(random_state=47,n_estimators=120,learning_rate=0.01),
},
{
    'label': 'Gradient Boosting',
    'model': GradientBoostingClassifier(),
},
    {
    'label': 'XGBoost',
    'model': XGBClassifier(),
},
    {
    'label': 'SGD',
    'model': SGDClassifier(max_iter=1000,penalty= 'l2', n_jobs= -1, loss= 'log', alpha=0.0001) ,
},
    
    {
    'label': 'KNN',
    'model': KNeighborsClassifier(n_neighbors = 5),
},
    {
    'label': 'Randomforest',
    'model': RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=3, max_features=10, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=1,
            oob_score=False, random_state=47, verbose=0, warm_start=False),        
    }
]

# Below for loop iterates through your models list
for m in models:
    model = m['model'] # select the model
    model.fit(X_train_scaled, y_train) # train the model
    y_pred=model.predict(X_test_scaled) # predict the test data
# Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test_scaled)[:,1])
# Calculate Area under the curve to display on the plot
    auc = metrics.roc_auc_score(y_test,model.predict(X_test_scaled))
# Now, plot the computed values
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], auc))
# Custom settings for the plot 
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[88]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import validation_curve
# Create range of values for parameter
param_range = np.arange(1, 250, 2)
# Calculate accuracy on training and test set using range of parameter values
train_scores, test_scores = validation_curve(AdaBoostClassifier(), 
                                             X_train_scaled, 
                                             y_train, 
                                             param_name="n_estimators", 
                                             param_range=param_range,
                                             cv=3, 
                                             scoring="accuracy", 
                                             n_jobs=-1)

# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot mean accuracy scores for training and test sets
plt.plot(param_range, train_mean, label="Training score", color="black")
plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

# Plot accurancy bands for training and test sets
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

# Create plot
plt.title("Validation Curve With ADABOOST")
plt.xlabel("Number Of Trees")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()


# # Interpretation of the Validation Curve
# 
# if the number of trees are around 10, then the model suffers from high bias. Two scores are quite close,but both the scores are too far from acceptable level so I think it's a high bias problem.In other words, the model is underfitting.
# 
# At a maximun number of trees of 250, model suffers from high variance since training score is 0.82 but validation score is about 0.81.In other words, a model is overfitting. Again, the data points suggest a sort of graceful curve. However, our model uses a very complex curve to get as close to every data point as possible. Consequently, a model with high variance has very low bias because it makes little to no assumption about the data. In fact, it adapts too much to the data.
# 
# As we see from the curve, max trees of around 30 to 40 best generalizes the unseen data. As max trees increases, bias becomes lower and variance becomes higher. We should keep the balance between the two. Just after 30 to 40 number of trees training score increase upwards and validation score starts to goes down, so I it begins to suffer from overfitting. So that's why any number of trees between 30 and 40 should be a good choice.

# # Conclusion
# 

# # So, we have seen that accuracy of tuned XGboost is around 82.95% and also achieved decent score of all other performance metric such as F1 score, Precision, ROC and Recall.
# 
# Further we can also perform model optimization by using Randomsearch or Gridsearch to find the appropriate parameters to increase the accuracy of the models.
# 
# I think all these three models if properly tuned will perform better.

# In[ ]:




