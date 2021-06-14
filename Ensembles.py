import pandas as pd
from sklearn.model_selection import train_test_split
train_data=pd.read_csv('FinalData/train_data.csv',index_col=0)
test_data=pd.read_csv('FinalData/test_data.csv',index_col=0)


y=train_data['final_result']
x=train_data.drop(['final_result','home','away'],axis=1)
X_train,X_validation,y_train,y_validation=train_test_split(x,y,stratify=y,test_size=0.2)
#y_train=train_data['final_result'][0:120]
#X_train=train_data.drop(['final_result','home','away'],axis=1)[0:120]
#X_validation=train_data.drop(['final_result','home','away'],axis=1)[120:len(train_data)]
#y_validate=train_data['final_result'][120:len(train_data)]
X_test=test_data.drop(['home','away'],axis=1)
##
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score

"""Try K nearest Neighbors. Grid Search is used to come up with the best possible number of neighbors"""

X_train_scaled=scale(X_train)
knn=KNeighborsClassifier()
neighbors=[i for i in range(1,40)]
parameters={'n_neighbors':neighbors}
gs=GridSearchCV(knn,param_grid=parameters)
gs.fit(X_train_scaled,y=y_train)
pred_knn=gs.predict(X_validation)
print(gs.best_score_)
print(gs.best_params_)
print('Acc:',accuracy_score(y_validation,pred_knn))
##
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()
parameters={'criterion':['gini','entropy'],'max_depth':[None,2,3,4,5,10],'min_samples_leaf':[0.02,0.05,0.1,0.15]}
grid_tree=GridSearchCV(dt,param_grid=parameters)
grid_tree.fit(X_train,y=y_train)
pred_tree=grid_tree.predict(X_validation)
print('Acc:',accuracy_score(y_validation,pred_tree))
print(grid_tree.best_params_)

##

"""Ensemble done by Majority Voting"""

classifiers=[('knn',gs),('DT',grid_tree)]
vc=VotingClassifier(classifiers)
vc.fit(X_train,y_train)
ensemble_prediction=vc.predict(X_validation)
print('Accuracy of the ensemble:',accuracy_score(y_validation,ensemble_prediction))


##


"""Bagging Classifier. Based on the bootstrap aggegation concept. We will perform the aggragation with the best possible tree
taht is the one we previously found using gridsearch"""

from sklearn.ensemble import BaggingClassifier
bc=BaggingClassifier(base_estimator=grid_tree,n_estimators=300,n_jobs=-1,oob_score=True)
bc.fit(X_train,y_train)
pred_bagging=bc.predict(X_validation)
print('Accuracy bagging ensemble:',accuracy_score(y_validation,pred_bagging))
print('Out of bag score:',bc.oob_score_)

##

"""Random Forest"""

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
parameters={'n_estimators':[100,200,300,400],'min_samples_leaf':[0.05,0.1,0.13,0.15,0.2],'min_samples_split':[0.05,0.1,0.2],'max_features':["auto","sqrt","log2"]}
grid_forest=GridSearchCV(rf,param_grid=parameters)
grid_forest.fit(X_train,y_train)
pred_forest=grid_forest.predict(X_validation)
pred_proba_forest=grid_forest.predict_proba(X_validation)

print(grid_forest.best_score_,grid_forest.best_params_)
print(accuracy_score(y_validation,pred_forest))

pred_euro_proba=grid_forest.predict_proba(X_test)
pred_euro=grid_forest.predict(X_test)
##
import matplotlib.pyplot as plt

hyperparameters=grid_forest.best_params_
rf=RandomForestClassifier(max_features='sqrt', min_samples_leaf= 0.1, min_samples_split= 0.2, n_estimators=300)
rf.fit(X_train,y_train)
importances_rf=pd.Series(rf.feature_importances_,index=x.columns)
sorted_importances_rf=importances_rf.sort_values()
sorted_importances_rf.plot(kind='barh')
plt.show()
##

""" Ada Boosting. Create a strong model from weaker ones"""
from sklearn.ensemble import AdaBoostClassifier
dt_ada=DecisionTreeClassifier(max_depth=2)
ada=AdaBoostClassifier(base_estimator=dt_ada,n_estimators=200)
ada.fit(X_train,y_train)
pred_ada=ada.predict(X_validation)
print(accuracy_score(y_validation,pred_ada))
pred_proba_euro_ada=ada.predict_proba(X_test)
pred_ada_euro=ada.predict(X_test)


##
"""Gradient Boosting"""
"""Stochastic Gradeint Boosting - Limit features we use to predict and sample without replacement --> Just include subsample and max_features"""
from sklearn.ensemble import GradientBoostingClassifier
gbt=GradientBoostingClassifier()
params={'n_estimators':[100,200,300],'max_depth':[2,3,None],'learning_rate':[0.05,0.1],'subsample':[0.3,0.5,0.8],'max_features':[0.2,0.3,0.6]}
gb_grid=GridSearchCV(gbt,param_grid=params)
gb_grid.fit(X_train,y_train)
print('Best params for Gradient Boosting:',gb_grid.best_params_)
pred_gbgrid=gb_grid.predict(X_validation)
print(accuracy_score(y_validation,pred_gbgrid))
pred_proba_GB_euro=gb_grid.predict_proba(X_test)
pred_GB_euro=gb_grid.predict(X_test)
##

def train_model_gridsearch(model,parameters_grid,X_training_data,y_training_data,X_validation_data,y_validation_data,X_test_data,perform_pca=False,components=0):
    """Creates the model, and trains it. pca can be applied to the data by using the perform_pca parameter"""
    from sklearn.model_selection import GridSearchCV
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import Normalizer
    from sklearn.metrics import accuracy_score
    m=model
    grid_model=GridSearchCV(m,param_grid=parameters_grid)
    data=[X_training_data,X_validation_data,X_test_data]
    data_transformed=[]
    for i in data:
        new_data = Normalizer().fit_transform(i)
        if perform_pca != False:
            pca=PCA(n_components=components)
            new_data = pca.fit_transform(new_data)

        data_transformed.append(new_data)

    X_train_transformed,X_validation_data_transformed,X_test_transformed=data_transformed[0],data_transformed[1],data_transformed[2]
    grid_model.fit(X_train_transformed,y_training_data)
    pred_grid=grid_model.predict(X_validation_data_transformed)
    print('Accuracy on validation set:',accuracy_score(y_validation_data,pred_grid))
    print('Best params of the model',grid_model.best_params_)
    print('Variance ratio of components',pca.explained_variance_ratio_)
    prediction_euro=grid_model.predict(X_test_transformed)
    prediction_euro_prob=grid_model.predict_proba(X_test_transformed)

    return grid_model,prediction_euro,prediction_euro_prob





parameters_1={'n_estimators':[200],'min_samples_leaf':[0.05,0.1,0.15],'min_samples_split':[0.1],'max_features':["auto","sqrt"]}
from sklearn.ensemble import RandomForestClassifier
model_rf,prediction_rf,prediciton_rf_prob=train_model_gridsearch(RandomForestClassifier(),parameters_1,X_train,y_train,X_validation,y_validation,X_test,True,7)

##
parameters_2={'n_estimators':[200],'max_depth':[2,3],'learning_rate':[0.05,0.1],'subsample':[0.3,0.2,0.5,0.8],'max_features':[0.2,0.3,0.6]}
from sklearn.ensemble import GradientBoostingClassifier
model_gb,prediction_gb,prediciton_gb_prob=train_model_gridsearch(GradientBoostingClassifier(),parameters_2,X_train,y_train,X_validation,y_validation,X_test,True,6)


