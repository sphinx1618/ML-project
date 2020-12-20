#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression,LinearRegression
import datetime
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB as MNB, CategoricalNB as CNB
from sklearn import metrics
from sklearn.metrics import accuracy_score as acc, precision_score as prec, recall_score as rec, f1_score as f1, roc_auc_score as roc
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
import xgboost as xgb
import pickle


# In[51]:


file = pd.read_csv('true_data.csv')
select = ['OUTCOME','Gender','intubated','pneumonia','pregnant','Diabetes','asthma','hypertension','another_comorbidity','cardiovascular','obesity','renal_chronicle','smoking','another_case']
file = file[select + ['ICU']]
file.head()


# In[52]:



# print(select)
x = np.array(file[select])
y = file['ICU']

print(x.shape, y.shape)


# In[53]:


import sklearn.model_selection

x_train,x_test, y_train,y_test = sklearn.model_selection.train_test_split(x,y, stratify=y, shuffle=True, test_size=0.2)

print(x_train.shape,x_test.shape, y_train.shape,y_test.shape)


# In[5]:


corrMatrix = file[select + ['ICU']].corr()
print (corrMatrix)


# In[6]:


import seaborn as sns

ax = sns.heatmap(corrMatrix)

plt.savefig('Plots/heatmap.jpg')
plt.show()


# # Naive Bayes 

# In[38]:


mnb = CNB(fit_prior=True)
mnb.fit(x_train,y_train)

y_pred = mnb.predict(x_test)

print('acc',acc(y_test, y_pred))
print('prec',prec(y_test, y_pred))
print('rec',rec(y_test, y_pred))
print('f1',f1(y_test, y_pred))
print('roc',roc(y_test, y_pred))

metrics.plot_roc_curve(mnb,x_test,y_test,color='g')
plt.plot([i/100 for i in range(0,101,2)],[i/100 for i in range(0,101,2)],'r--')
plt.legend(['AUC','Y=X'])
plt.title('ROC - NB')
plt.savefig('Plots/NB_3d.jpg')
plt.show()


# In[39]:


pickle.dump(mnb, open('Best_model_NB.sav','wb'))


# In[8]:


from sklearn.naive_bayes import BernoulliNB as BNB

bnb = BNB(fit_prior=True)
bnb.fit(x_train,y_train)

y_pred = bnb.predict(x_test)

print('acc',acc(y_test, y_pred))
print('prec',prec(y_test, y_pred))
print('rec',rec(y_test, y_pred))
print('f1',f1(y_test, y_pred))
print('roc',roc(y_test, y_pred))

metrics.plot_roc_curve(bnb,x_test,y_test,color='g')
plt.plot([i/100 for i in range(0,101,2)],[i/100 for i in range(0,101,2)],'r--')
plt.legend(['AUC','Y=X'])
plt.title('ROC Curve BNB')
plt.savefig('Plots/NB_BNB.jpg')
plt.show()


# # SVM

# In[9]:


svm_l = SVC(kernel='linear')
svm_l.fit(x_train,y_train)

y_pred = svm_l.predict(x_test)

print('acc',acc(y_test, y_pred))
print('prec',prec(y_test, y_pred))
print('rec',rec(y_test, y_pred))
print('f1',f1(y_test, y_pred))
print('roc',roc(y_test, y_pred))

metrics.plot_roc_curve(svm_l,x_test,y_test,color='g')
plt.plot([i/100 for i in range(0,101,2)],[i/100 for i in range(0,101,2)],'r--')
plt.legend(['AUC','Y=X'])
plt.title('SVM - Linear')
plt.savefig('Plots/SVM_lin.jpg')
plt.show()
# plt.show()


# In[10]:


svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(x_train,y_train)

y_pred = svm_rbf.predict(x_test)

print('acc',acc(y_test, y_pred))
print('prec',prec(y_test, y_pred))
print('rec',rec(y_test, y_pred))
print('f1',f1(y_test, y_pred))
print('roc',roc(y_test, y_pred))

metrics.plot_roc_curve(svm_rbf,x_test,y_test,color='g')
plt.plot([i/100 for i in range(0,101,2)],[i/100 for i in range(0,101,2)],'r--')
plt.legend(['AUC','Y=X'])
plt.title('SVM - rbf')
plt.savefig('Plots/SVM_rbf.jpg')
plt.show()


# In[40]:


pickle.dump(svm_rbf, open('Best_model_SVM.sav','wb'))


# In[11]:


svm_poly = SVC(kernel='poly')
svm_poly.fit(x_train,y_train)

y_pred = svm_poly.predict(x_test)

print('acc',acc(y_test, y_pred))
print('prec',prec(y_test, y_pred))
print('rec',rec(y_test, y_pred))
print('f1',f1(y_test, y_pred))
print('roc',roc(y_test, y_pred))

metrics.plot_roc_curve(svm_poly,x_test,y_test,color='g')
plt.plot([i/100 for i in range(0,101,2)],[i/100 for i in range(0,101,2)],'r--')
plt.legend(['AUC','Y=X'])
plt.title('SVM - poly')
plt.savefig('Plots/SVM_poly.jpg')
plt.show()


# In[13]:


model_xgboost_c = xgb.XGBClassifier(eval_metric='auc')
#reg_alpha=0.55,reg_lambda=1,learning_rate = 0.005,max_depth=8,n_estimators=150, random_state=42, n_jobs=-1, subsample=0.8, 

scores1 = cross_validate(model_xgboost_c, x, y, cv=7, scoring=('f1','recall','precision','roc_auc'))

model_xgboost_c.fit(x_train,y_train)

print("F1-score",np.mean(scores1['test_f1']))
print("Recall",np.mean(scores1['test_recall']))
print("Precision",np.mean(scores1['test_precision']))
print("Roc_Auc",np.mean(scores1['test_roc_auc']))
metrics.plot_roc_curve(model_xgboost_c,x_test,y_test)
plt.plot([i/100 for i in range(0,101,2)],[i/100 for i in range(0,101,2)],'r--')
plt.legend(['AUC','Y=X'])
plt.title('ROC Curve XGB Classifier')
# plt.savefig('Plots/1ROC_XGBC_top' + str(v) + '.jpg')
plt.show()


# In[14]:


model_logistic = LogisticRegression()

scores1 = cross_validate(model_logistic, x, y, cv=5, scoring=('f1','recall','precision','roc_auc'))

model_logistic.fit(x_train,y_train)

print("F1-score",np.mean(scores1['test_f1']))
print("Recall",np.mean(scores1['test_recall']))
print("Precision",np.mean(scores1['test_precision']))
print("Roc_Auc",np.mean(scores1['test_roc_auc']))
metrics.plot_roc_curve(model_logistic,x_test,y_test)
plt.plot([i/100 for i in range(0,101,2)],[i/100 for i in range(0,101,2)],'r--')
plt.legend(['AUC','Y=X'])
plt.title('ROC Curve Logistic')
# plt.savefig('Plots/1ROC_Logi_top' + str(v) + '.jpg')
plt.show()


# # Bagging

# In[19]:


def bagging(mod,name):
    import random
    def generate_sample(X,y):
        #Generating Samples of lenght len(X) by resampling
        X_sample=[]
        y_sample=[]
        for i in range(len(X)):
            index=random.randint(0,len(X)-1)
            X_sample.append(X[index])
            y_sample.append(y[index])
        return [np.array(X_sample),np.array(y_sample)]
    Xtrain=np.array(x_train)
    ytrain=np.array(y_train)
    Xtest=np.array(x_test)
    ytest=np.array(y_test)
    B=100                 #forming 20 models and boot strp samples
    Samples=[]
    Models=[]
    for b in range(B):               #Doing bootstrapping
        print(b,end=" ")
        Samples.append(generate_sample(Xtrain,ytrain))
        model = mod
        model.fit(Samples[-1][0],Samples[-1][1])
        Models.append(model)
    predicted=[]               #predicting ans
    for b in range(B):
        predicted.append(np.array(Models[b].predict(Xtest)))
    predicted=np.array(predicted,dtype=np.float64)
    
    y_pred=np.mean(predicted,axis=0)

    y_pred=np.where(y_pred>0.43,1,0)
    y_pred
    print('Accuracy',acc(ytest, y_pred))
    print('Precision',prec(ytest, y_pred))
    print('Recall',rec(ytest, y_pred))
    print('F1-score',f1(ytest, y_pred))
    print('Roc_Auc',roc(ytest, y_pred))
    metrics.plot_roc_curve(mod,x_test,y_test)
    
    plt.plot([i/100 for i in range(0,101,2)],[i/100 for i in range(0,101,2)],'r--')
    plt.legend(['AUC','Y=X'])
    plt.title('ROC Curve' + ' '+name)
#     plt.savefig('Plots/Boost_ROC_'+name+'_top' + str(v) + '.jpg')
    plt.show()


# # Bagging XGB

# In[20]:


mod = xgb.XGBClassifier(eval_metric='auc')
bagging(mod, 'XGB Classifier')


# # Bagging Logistic

# In[21]:


mod = LogisticRegression()
bagging(mod, 'Logisitic')


# # Bagging NAive Bayes

# In[24]:


mod = CNB(fit_prior=True)
bagging(mod, 'NB')


# # Bagging SVM

# In[26]:


mod = SVC(kernel='rbf')
bagging(mod, 'SVM-poly')


# # Stacking and Bagging

# In[31]:


from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
level0 = list()
level0.append(('cnb', CNB()))
level0.append(('svml', SVC(kernel='linear')))
level0.append(('rf', RandomForestClassifier()))
level0.append(('svmr', SVC(kernel='rbf')))
level0.append(('xg', xgb.XGBClassifier(eval_metric='auc')))
level0.append(('bnb', BNB()))
level1 = LogisticRegression()


# In[32]:


import random
def generate_sample(X,y):
    #Generating Samples of lenght len(X) by resampling
    X_sample=[]
    y_sample=[]
    for i in range(len(X)):
        index=random.randint(0,len(X)-1)
        X_sample.append(X[index])
        y_sample.append(y[index])
    return [np.array(X_sample),np.array(y_sample)]


# In[54]:


Xtrain=np.array(x_train)
ytrain=np.array(y_train)
Xtest=np.array(x_test)
ytest=np.array(y_test)
B= 50                  #forming 20 models and boot strp samples
Samples=[]
Models=[]
for b in range(B):               #Doing bootstrapping
    print(b)
    Samples.append(generate_sample(Xtrain,ytrain))
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    # fit the model on all available data
    model.fit(Xtrain, ytrain)
    Models.append(model)


# # ADA Boosting

# In[36]:


from sklearn.ensemble import AdaBoostClassifier as ADA
mod = ADA(n_estimators=100, random_state=0)

mod.fit(x_train,y_train)

y_pred = mod.predict(x_test)

print('Accuracy',acc(y_test, y_pred))
print('Precision',prec(y_test, y_pred))
print('Recall',rec(y_test, y_pred))
print('"F1-score"',f1(y_test, y_pred))
print('Roc_Auc',roc(y_test, y_pred))

metrics.plot_roc_curve(mod,x_test,y_test)

plt.plot([i/100 for i in range(0,101,2)],[i/100 for i in range(0,101,2)],'r--')
plt.legend(['AUC','Y=X'])
plt.title('ROC Curve ADA boost')
plt.savefig('Plots/ROC_ADA.jpg')
plt.show()


# # TOP Feature Selection

# In[41]:


a = ['OUTCOME','Gender','intubated','pneumonia','pregnant','Diabetes','asthma','hypertension','another_comorbidity','cardiovascular','obesity','renal_chronicle','smoking','another_case']
b = list(corrMatrix['ICU'])
feat = [[b[i],a[i]] for i in range(len(a))]
feat.sort(reverse=True)


# In[42]:


v = 10
sel_feat_top_5 = [feat[i][1] for i in range(v)]
sel_feat_top_5


# In[44]:


x = np.array(file[sel_feat_top_5])
y = file['ICU']

print(x.shape, y.shape)


# In[45]:


import sklearn.model_selection

x_train,x_test, y_train,y_test = sklearn.model_selection.train_test_split(x,y, stratify=y, shuffle=True, test_size=0.2)

print(x_train.shape,x_test.shape, y_train.shape,y_test.shape)


# # SVM

# In[46]:


svm_poly = SVC(kernel='rbf')
svm_poly.fit(x_train,y_train)

y_pred = svm_poly.predict(x_test)

print('acc',acc(y_test, y_pred))
print('prec',prec(y_test, y_pred))
print('rec',rec(y_test, y_pred))
print('f1',f1(y_test, y_pred))
print('roc',roc(y_test, y_pred))

metrics.plot_roc_curve(svm_poly,x_test,y_test,color='g')
plt.plot([i/100 for i in range(0,101,2)],[i/100 for i in range(0,101,2)],'r--')
plt.legend(['AUC','Y=X'])
plt.title('SVM - rbf')
plt.savefig('Plots/1SVM_poly_top' + str(v) + '.jpg')
plt.show()


# # Naive Bayes

# In[49]:


mnb = CNB(fit_prior=True)
mnb.fit(x_train,y_train)

y_pred = mnb.predict(x_test)

print('acc',acc(y_test, y_pred))
print('prec',prec(y_test, y_pred))
print('rec',rec(y_test, y_pred))
print('f1',f1(y_test, y_pred))
print('roc',roc(y_test, y_pred))

metrics.plot_roc_curve(mnb,x_test,y_test,color='g')
plt.plot([i/100 for i in range(0,101,2)],[i/100 for i in range(0,101,2)],'r--')
plt.legend(['AUC','Y=X'])
plt.title('ROC - NB')
plt.savefig('Plots/1NB_top' + str(v) + '.jpg')
plt.show()


# # XGB

# In[48]:


model_xgboost_c = xgb.XGBClassifier(eval_metric='auc')
#reg_alpha=0.55,reg_lambda=1,learning_rate = 0.005,max_depth=8,n_estimators=150, random_state=42, n_jobs=-1, subsample=0.8, 

scores1 = cross_validate(model_xgboost_c, x, y, cv=7, scoring=('f1','recall','precision','roc_auc'))

model_xgboost_c.fit(x_train,y_train)

print("F1-score",np.mean(scores1['test_f1']))
print("Recall",np.mean(scores1['test_recall']))
print("Precision",np.mean(scores1['test_precision']))
print("Roc_Auc",np.mean(scores1['test_roc_auc']))
metrics.plot_roc_curve(model_xgboost_c,x_test,y_test)
plt.plot([i/100 for i in range(0,101,2)],[i/100 for i in range(0,101,2)],'r--')
plt.legend(['AUC','Y=X'])
plt.title('ROC Curve XGB Classifier')
plt.savefig('Plots/1ROC_XGBC_top' + str(v) + '.jpg')
plt.show()


# # Logistic

# In[47]:


model_logistic = LogisticRegression()

scores1 = cross_validate(model_logistic, x, y, cv=5, scoring=('f1','recall','precision','roc_auc'))

model_logistic.fit(x_train,y_train)

print("F1-score",np.mean(scores1['test_f1']))
print("Recall",np.mean(scores1['test_recall']))
print("Precision",np.mean(scores1['test_precision']))
print("Roc_Auc",np.mean(scores1['test_roc_auc']))
metrics.plot_roc_curve(model_logistic,x_test,y_test)
plt.plot([i/100 for i in range(0,101,2)],[i/100 for i in range(0,101,2)],'r--')
plt.legend(['AUC','Y=X'])
plt.title('ROC Curve Logistic')
plt.savefig('Plots/1ROC_Logi_top' + str(v) + '.jpg')
plt.show()

