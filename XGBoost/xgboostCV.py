# -*- coding: utf-8 -*-
#インストールガイド
#https://xgboost.readthedocs.io/en/latest/build.html
#パラメータについて良くまとめていただいているサイト
#http://kamonohashiperry.com/archives/209

import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


##regression
boston=load_boston()
dataX=boston.data
datay=boston.target

X_train,X_test,y_train,y_test = train_test_split(dataX,datay,test_size=0.2,random_state=1)
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.3,random_state=1)

#回帰
#パラメータ
#max_depth : 木の深さの最大値
#learning_rate : 0~1 学習率パラメータ
#n_estimators : int Number of trees to fit
#booster : gbtree or gblinear or dart
#nthread : int 並行処理の個数
#gamma : float minimum loss reduction required to make a further partition on a leaf node of the tree
#reg_alpha : float L1 regularization term on weights
#reg_lambda : float L2 regularization term on weights
reg=xgb.XGBRegressor(objective ='reg:squarederror')

#CV
#n_estimatorsをバリデーションするのは愚策
#https://amalog.hateblo.jp/entry/hyper-parameter-search

cv_par = {
    "max_depth" : np.arange(2,6,1),
    "learning_rate" : np.linspace(0.01,1.0,100)
    
}
reg_cv=GridSearchCV(reg,cv_par,verbose=1)
reg_cv.fit(X_val,y_val)
print(reg_cv.best_params_,reg_cv.best_score_)

#最適パラメータを用いた学習
reg=xgb.XGBRegressor(**reg_cv.best_params_)
reg.fit(X_train,y_train,early_stopping_rounds=100,eval_set=[[X_val,y_val]])

pred_test=reg.predict(X_test)
print(mean_squared_error(y_test,pred_test))

#feature importanceがreg.feature_importances_に保存されている。
importances = pd.Series(reg.feature_importances_,index=boston.feature_names)
importances = importances.sort_values()
importances.plot(kind="barh")
plt.title("importance in the xgboost Model")
plt.show()
import graphviz
xgb.to_graphviz(reg)

##classificate
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report

digits=load_digits()
dataX2=digits.data
datay2=digits.target

X_train2,X_test2,y_train2,y_test2 = train_test_split(dataX2,datay2,test_size=0.2,random_state=1)
X_train2,X_val2,y_train2,y_val2 = train_test_split(X_train2,y_train2,test_size=0.2,random_state=1)

clf = xgb.XGBClassifier()

cv_par2 = {
    "max_depth" : np.arange(2,6,1),
    "learning_rate" : np.linspace(0.01,1.0,100)
    
}
clf_cv = GridSearchCV(clf,cv_par2,verbose=1)
clf_cv.fit(X_val2,y_val2)
print(clf_cv.best_params_)

clf = xgb.XGBClassifier(**clf_cv.best_params_)
clf.fit(X_train2,y_train2)

pred=clf.predict(X_test2)
print(confusion_matrix(y_test2,pred))
print(classification_report(y_test2,pred))

xgb.to_graphviz(clf)