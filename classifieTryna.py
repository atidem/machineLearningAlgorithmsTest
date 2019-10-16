# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:17:54 2019

@author: atidem
"""

import numpy as np
import pandas as pd 
import statsmodels.api as sm 
import statsmodels.formula.api as smf
import seaborn as sns 
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from warnings import filterwarnings
filterwarnings('ignore')

######## Data Included

diabetes = pd.read_csv("diabetes.csv")
df = diabetes.copy()
df = df.dropna()
df.head()
df.info()
df["Outcome"].value_counts() ## dengesizlik kontrolü
df.describe().T
y = df["Outcome"]
x = df.drop(["Outcome"],axis=1)
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)


####### Logistic Regression

loj = LogisticRegression(solver="liblinear")
loj_model = loj.fit(X_train,y_train)
loj_model
loj_model.intercept_
loj_model.coef_
y_pred = loj_model.predict(X_test)
confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)

y_probs = loj_model.predict_proba(X_test)
y_probs = y_probs[:,1]
y_pred = [1 if i > 0.5 else 0 for i in y_probs] ## treshold 

confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)

logit_roc_auc = roc_auc_score(y_test,loj_model.predict(X_test))
fpr,tpr,tresholds = roc_curve(y_test,loj_model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr,tpr,label='AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel("false pos rate")
plt.ylabel("true pos rate")
plt.title('ROC')
plt.show()

cross_val_score(loj_model,X_test,y_test,cv=10).mean()

###### Naive Bayes

nb = GaussianNB()
nb_model = nb.fit(X_train,y_train)
y_pred = nb_model.predict(X_test)
y_prob = nb_model.predict_proba(X_test)
accuracy_score(y_test,y_pred)

cross_val_score(nb_model,X_test,y_test,cv=10).mean()


##### SVM

svm_model = SVC(kernel="linear").fit(X_train,y_train)
svm_model
y_pred = svm_model.predict(X_test)
accuracy_score(y_test,y_pred)


svc = SVC(kernel="linear")
svc_params = {"C":np.arange(1,10)}

svc_cv_model = GridSearchCV(svc,svc_params,cv=10,n_jobs=-1,verbose=2)
svc_cv_model.fit(X_train,y_train)
svc_cv_model.best_params_

svc_tuned = SVC(kernel="linear", C=5)
svc_tuned.fit(X_train,y_train)
y_pred = svc_tuned.predict(X_test)
accuracy_score(y_test,y_pred)


###### ANN YSA 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier
mlpc = MLPClassifier().fit(X_train_scaled,y_train)
y_pred = mlpc.predict(X_test_scaled)
accuracy_score(y_test,y_pred)

mlpc_params = {"alpha":[0.1,0.01,0.001,0.0001,0.02,0.05],
               "hidden_layer_sizes":[(10,10,2),(100,100,2),(100,2),(20,2)],
               "solver":["lbfgs","adam","sgd"],
               "activation":["relu","logistic"]}
ann = MLPClassifier()
mlpc_cv = GridSearchCV(ann,mlpc_params,cv=10,n_jobs=-1,verbose=2)
mlpc_cv.fit(X_train_scaled,y_train)
mlpc_cv.best_params_

mlpc_tuned = MLPClassifier(activation='relu',alpha=0.1,hidden_layer_sizes=(100,100),solver='sgd')
mlpc_tuned.fit(X_train_scaled,y_train)
y_pred = mlpc_tuned.predict(X_test_scaled)
accuracy_score(y_test,y_pred)


###### CART 

cart = tree.DecisionTreeClassifier()
cart.fit(X_train,y_train)
cart
y_pred = cart.predict(X_test)
accuracy_score(y_test,y_pred)

cart_params = {"max_depth":list(range(1,10)),
               "min_samples_split":list(range(2,50))}
cart = tree.DecisionTreeClassifier()
cart_cv = GridSearchCV(cart,cart_params,cv=10,n_jobs=-1,verbose=2)
cart_cv.fit(X_train,y_train)
cart_cv.best_params_

cart_tuned = tree.DecisionTreeClassifier(max_depth=5,min_samples_split=19)
cart_tuned.fit(X_train,y_train)
y_pred = cart_tuned.predict(X_test)
accuracy_score(y_test,y_pred)


##### Random Forests 

rf = RandomForestClassifier()
rf_model = rf.fit(X_train,y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_test,y_pred)

rf_params = {"max_depth":[2,3,5,8,10],
             "max_features":[2,5,8],
             "n_estimators":[10,20,100,500,1000],
             "min_samples_split":[2,5,10]}
rf = RandomForestClassifier()
rf_cv = GridSearchCV(rf,rf_params,cv=10,n_jobs=-1,verbose=2)
rf_cv.fit(X_train,y_train)
rf_cv.best_params_

rf_tuned = RandomForestClassifier(max_depth=3,max_features=5,min_samples_split=2,n_estimators=10)
rf_tuned.fit(X_train,y_train)
y_pred = rf_tuned.predict(X_test)
accuracy_score(y_test,y_pred)


##### GBM

gbm = GradientBoostingClassifier()
gbm_model = gbm.fit(X_train,y_train)
y_pred = gbm_model.predict(X_test)
accuracy_score(y_test,y_pred)

gbm_params = {"max_depth":[3,5,10],
             "learning_rate":[0.001,0.01,0.1,0.05,0.005],
             "n_estimators":[10,20,100,500,1000],
             "min_samples_split":[2,5,10]}
gbm = GradientBoostingClassifier()
gbm_cv = GridSearchCV(gbm,gbm_params,cv=10,n_jobs=-1,verbose=2)
gbm_cv.fit(X_train,y_train)
gbm_cv.best_params_

gbm_tuned = GradientBoostingClassifier(learning_rate=0.005,max_depth=5,min_samples_split=2,n_estimators=1000)
gbm_tuned.fit(X_train,y_train)
y_pred = gbm_tuned.predict(X_test)
accuracy_score(y_test,y_pred)


##### XGBoost 

xgb = XGBClassifier()
xgb_model = xgb.fit(X_train,y_train)
y_pred = xgb_model.predict(X_test)
accuracy_score(y_test,y_pred)

xgb_params = {"max_depth":[3,5,10],
              "subsample":[0.6,0.8,1],
             "learning_rate":[0.001,0.01,0.1,0.05,0.005],
             "n_estimators":[10,20,100,500,1000,2000],
             "min_samples_split":[2,5,10]}
xgb = XGBClassifier()
xgb_cv = GridSearchCV(xgb,xgb_params,cv=10,n_jobs=-1,verbose=2)
xgb_cv.fit(X_train,y_train)
xgb_cv.best_params_

xgb_tuned = XGBClassifier(learning_rate=0.001,max_depth=10,min_samples_split=2,n_estimators=10,subsample=0.8)
xgb_tuned.fit(X_train,y_train)
y_pred = xgb_tuned.predict(X_test)
accuracy_score(y_test,y_pred)


##### lightGBM

lgbm = LGBMClassifier()
lgbm_model = lgbm.fit(X_train,y_train)
y_pred = lgbm_model.predict(X_test)
accuracy_score(y_test,y_pred)

lgbm_params = {"max_depth":[3,4,5,6],
              "subsample":[0.6,0.8,1],
             "learning_rate":[0.001,0.01,0.1,0.05,0.005],
             "n_estimators":[10,20,100,500,1000,2000],
             "min_child_samples":[5,10,20]}
lgbm = LGBMClassifier()
lgbm_cv = GridSearchCV(lgbm,lgbm_params,cv=10,n_jobs=-1,verbose=2)
lgbm_cv.fit(X_train,y_train)
lgbm_cv.best_params_

lgbm_tuned = LGBMClassifier(learning_rate=0.01,max_depth=3,min_child_samples=20,n_estimators=500,subsample=0.6)
lgbm_tuned.fit(X_train,y_train)
y_pred = lgbm_tuned.predict(X_test)
accuracy_score(y_test,y_pred)


##### CatBoost

cb = CatBoostClassifier()
cb_model = cb.fit(X_train,y_train)
y_pred = cb_model.predict(X_test)
accuracy_score(y_test,y_pred)

catb_params = {"iterations":[200,500],
               "learning_rate":[0.01,0.05,0.1],
               "depth":[3,5,8]}
catb = CatBoostClassifier()
catb_cv = GridSearchCV(catb,catb_params,cv=10,verbose=2)
catb_cv.fit(X_train,y_train)
catb_cv.best_params_

catb_tuned = CatBoostClassifier(depth=8,iterations=200,learning_rate=0.05)
catb_tuned.fit(X_train,y_train)
y_pred = catb_tuned.predict(X_test)
accuracy_score(y_test,y_pred)


























































































