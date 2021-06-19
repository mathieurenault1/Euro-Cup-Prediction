import pandas as pd
from sklearn.model_selection import train_test_split
train_data=pd.read_csv('FinalData/train_data.csv',index_col=0)
test_data=pd.read_csv('FinalData/test_data.csv',index_col=0)


y=train_data['final_result']
x=train_data.drop(['final_result','home','away'],axis=1)
X_train,X_validation,y_train,y_validation=train_test_split(x,y,stratify=y,test_size=0.2)
X_test=test_data.drop(['home','away'],axis=1)
teams_boxes21=pd.read_csv('FinalData/team_boxes_21.csv')
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
        else:
            data_transformed.append(i)






    X_train_transformed,X_validation_data_transformed,X_test_transformed=data_transformed[0],data_transformed[1],data_transformed[2]
    grid_model.fit(X_train_transformed,y_training_data)
    pred_grid=grid_model.predict(X_validation_data_transformed)
    print('Accuracy on validation set:',accuracy_score(y_validation_data,pred_grid))
    print('Best params of the model',grid_model.best_params_)

    prediction_euro=grid_model.predict(X_test_transformed)
    prediction_euro_prob=grid_model.predict_proba(X_test_transformed)

    return grid_model,prediction_euro,prediction_euro_prob



##

"""This section produces the models that we will use in the prediciton. The models are trained and the we need several
of them to cope with the restrictions of the PCA"""


parameters_1={'n_estimators':[200],'min_samples_leaf':[0.05,0.1,0.15],'min_samples_split':[0.1],'max_features':["auto","sqrt"]}
from sklearn.ensemble import RandomForestClassifier
model_rf,prediction_rf,prediciton_rf_prob=train_model_gridsearch(RandomForestClassifier(),parameters_1,X_train,y_train,X_validation,y_validation,X_test,True,7)
model_rf_4,prediction_rf_q,prediciton_rf_qprob_final=train_model_gridsearch(RandomForestClassifier(),parameters_1,X_train,y_train,X_validation,y_validation,X_test,True,4)
model_rf_2,prediction_rf_final,prediciton_rf_prob_final=train_model_gridsearch(RandomForestClassifier(),parameters_1,X_train,y_train,X_validation,y_validation,X_test,True,2)
model_rf_1,prediction_rf_final,prediciton_rf_prob_final=train_model_gridsearch(RandomForestClassifier(),parameters_1,X_train,y_train,X_validation,y_validation,X_test,True,1)
##

parameters_2={'n_estimators':[200],'max_depth':[2,3],'learning_rate':[0.05,0.1],'subsample':[0.3,0.2,0.5,0.8],'max_features':[0.2,0.3,0.6]}
from sklearn.ensemble import GradientBoostingClassifier
model_gb,prediction_gb,prediciton_gb_prob=train_model_gridsearch(GradientBoostingClassifier(),parameters_2,X_train,y_train,X_validation,y_validation,X_test,True,4)
model_gb1=train_model_gridsearch(GradientBoostingClassifier(),parameters_2,X_train,y_train,X_validation,y_validation,X_test,True,1)





##

"""This section continues wiht the modelling of the tournament. I use this file instead of creating a new one in order to  be able to continue with"""


def create_countries_and_points_df(data_test):
    countries=[]
    for i in range(len(data_test)):
        if data_test.loc[i,'home'] not in countries:
            countries.append(test_data.loc[i,'home'])
        if data_test.loc[i,'away'] not in countries:
            countries.append(test_data.loc[i,'away'])

    countries_and_points=pd.DataFrame()
    countries_and_points['country']=countries
    countries_and_points['points']=0*len(countries_and_points)
    countries_and_points=countries_and_points.sort_values(by='country').reset_index(drop=True)
    return countries_and_points




def predict_match(i,test,model,probabilistic):
    import random
    """We need to give the function the model that we are going to use. Where do we grab it from ? We grab it from the
    previously defined train_model_gridsearch"""
    if probabilistic==False:
        prediction=model.predict(test[i].reshape(1,-1))
    else:
        prediction = model.predict_proba(test[i].reshape(1, -1))
        r1=prediction[0][0]
        r2=prediction[0][1]
        number=random.uniform(0,1)
        if number <= r1:
            prediction=0
        elif r1 < number <= r1 +r2:
            prediction=1
        else:
            prediction=2

    return prediction



def transform_test_set(data_to_transform,components):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import Normalizer

    #data_to_transform=Normalizer().fit_transform(data_to_transform)
    pca = PCA(n_components=components)
    new_data = pca.fit_transform(data_to_transform)
    return new_data

def group_phase(data_test,test_set,countries_and_points,model,components,probabilistic):
    """data test is the test_data file and test_set is the X_test file"""
    transformed_test_set=transform_test_set(test_set,components)
    global_results=[]
    for i in range(len(data_test)):
        result=predict_match(i,transformed_test_set,model,probabilistic)
        global_results.append(result)
        if result==0:
            winner=data_test.loc[i,'home']
            for c in range(len(countries_and_points)):
                if countries_and_points.loc[c,'country']==winner:
                    countries_and_points.loc[c,'points'] += 3
        elif result==2:
            winner = data_test.loc[i, 'away']
            for c in range(len(countries_and_points)):
                if countries_and_points.loc[c, 'country'] == winner:
                    countries_and_points.loc[c, 'points'] += 3
        else:
            team1=data_test.loc[i,'home']
            team2= data_test.loc[i, 'away']
            for c in range(len(countries_and_points)):
                if countries_and_points.loc[c, 'country'] == team1:
                    countries_and_points.loc[c, 'points'] += 1
                elif countries_and_points.loc[c, 'country']== team2:
                    countries_and_points.loc[c, 'points'] += 1




    return countries_and_points,global_results





def sortgroup(tup):
    import random
    lst = len(tup)
    for i in range(0, lst):

        for j in range(0, lst - i - 1):
            if (tup[j][1] < tup[j + 1][1]):
                temp = tup[j]
                tup[j] = tup[j + 1]
                tup[j + 1] = temp
    for i in range(0, lst - 1):
        if (tup[i][1] == tup[i + 1][1]):
            x = random.uniform(0, 1)
            if x < 0.5:
                temp = tup[i]
                tup[i] = tup[i + 1]
                tup[i + 1] = temp
    return tup



def create_groups_sorted(countries_and_points):
    import copy
    countries_points_groups=copy.copy(countries_and_points)
    teams_a=['Italy','Switzerland','Turkey','Wales']
    teams_b=['Belgium','Denmark','Russia','Finland']
    teams_c=['Netherlands','Ukraine','Austria','North Macedonia']
    teams_d=['England','Croatia','Scotland','Czech Republic']
    teams_e=['Spain','Sweden','Poland','Slovakia']
    teams_f=['Hungary','Portugal','France','Germany']
    GroupA,GroupB,GroupC,GroupD,GroupE,GroupF=[],[],[],[],[],[]

    all_groups=[GroupA,GroupB,GroupC,GroupD,GroupE,GroupF]

    for i in range(len(countries_points_groups)):
        country=countries_points_groups.loc[i, 'country']
        points=countries_points_groups.loc[i, 'points']
        if country in teams_a:
            GroupA.append([country,points])
        elif country in teams_b:
            GroupB.append([country,points])
        elif country in teams_c:
            GroupC.append([country,points])
        elif country in teams_d:
            GroupD.append([country, points])
        elif country in teams_e:
            GroupE.append([country, points])
        else:
            GroupF.append([country, points])

    for group in all_groups:
        group=sortgroup(group)

    thirds = [(GroupA[2][0],GroupA[2][1],'A'), (GroupB[2][0],GroupB[2][1],'B'), (GroupC[2][0],GroupC[2][1],'C'), (GroupD[2][0],GroupD[2][1],'D'), (GroupE[2][0],GroupE[2][1],'E'), (GroupF[2][0],GroupF[2][1],'F')]
    thirds = sortgroup(thirds)
    thirds = thirds[:4]

    return GroupA,GroupB,GroupC,GroupD,GroupE,GroupF,thirds




def phase_16(GroupA,GroupB,GroupC,GroupD,GroupE,GroupF,thirds,teams,model,probabilistic):
    all_matches=create_fixtures_round_of_16(GroupA,GroupB,GroupC,GroupD,GroupE,GroupF,thirds)
    matches_data_df_transformed=create_dataset_any_round(all_matches,teams,7)
    next_phase_matches=play_round(matches_data_df_transformed,all_matches,model,16,probabilistic)
    return next_phase_matches

def phase_8(next_phase_matches,teams,model,components,probabilistic):
    matches_data_8_transformed=create_dataset_any_round(next_phase_matches,teams,components)
    matches_semifinal=play_round(matches_data_8_transformed,next_phase_matches,model,8,probabilistic)
    return matches_semifinal

def phase_4(matches_semifinal,teams,model,components,probabilistic):
    matches_data_4_transformed=create_dataset_any_round(matches_semifinal,teams,components)
    final=play_round(matches_data_4_transformed,matches_semifinal,model,4,probabilistic)
    return final

def play_final(final,teams,model,components,probabilistic):
    match_final_transformed=create_dataset_any_round(final,teams,components)
    glorious_winner=play_round(match_final_transformed,final,model,1,probabilistic)
    return glorious_winner

def draw_a_ball(contenets,allowed_groups,already_drawn):
    import random
    available_rivals=[]
    for team in contenets:
        if team[2] in allowed_groups and team[0] not in already_drawn:
            available_rivals.append(team)
    number=random.randint(0,len(available_rivals)-1)
    chosen_rival=available_rivals[number][0]
    already_drawn.append(chosen_rival)
    print(chosen_rival)
    #while (contenets[number][2] not in allowed_groups and contenets[number][0] in already_drawn):
        #number = random.randint(0, len(contenets) - 1)
    #already_drawn.append(contenets[number][0])
    return chosen_rival




def create_fixtures_round_of_16(group1,group2,group3,group4,group5,group6,thirds):
    drawn_balls=[]
    match1=(group1[1][0],group2[1][0])
    match2 = (group1[0][0], group3[1][0])
    third_team1=draw_a_ball(thirds,['D','E','F'],drawn_balls)
    match3 = (group3[0][0], third_team1)
    third_team2=draw_a_ball(thirds,['A','D','E','F'],drawn_balls)
    match4= (group2[0][0],third_team2)
    match5= (group4[1][0],group5[1][0])
    third_team3=draw_a_ball(thirds,['A','B','C'],drawn_balls)
    match6= (group6[0][0],third_team3)
    match7= (group4[0][0],group6[1][0])
    third_team4= draw_a_ball(thirds,['A','B','C','D'],drawn_balls)
    match8= (group5[0][0],third_team4)
    all_matches=[match1,match2,match3,match4,match5,match6,match7,match8]
    counter=1
    for match in all_matches:
        print('Match {}: {}'.format(counter,match))
        counter+=1
    return all_matches


def create_dataset_any_round(all_matches,teams,components):
    nationalities=list(teams['nationality'])
    nationalities_dict={v:i for i,v in enumerate(nationalities)}
    teams_data=teams.drop(['nationality','country'],axis=1)
    matches_data=[]
    for match in all_matches:
        matches_data.append(teams_data.iloc[nationalities_dict[match[0]]]-teams_data.iloc[nationalities_dict[match[1]]])

    matches_data_df=pd.DataFrame(matches_data)
    matches_data_df_transformed=transform_test_set(matches_data_df,components)
    return matches_data_df_transformed



def play_round(matches_data_df_transformed,all_matches,model,round,probabilistic):
    next_phase_teams=[]
    for i in range(len(matches_data_df_transformed)):
        result=predict_match(i,matches_data_df_transformed,model,probabilistic)
        if result == 0:
            next_phase_teams.append(all_matches[i][0])
        elif result ==2:
            next_phase_teams.append(all_matches[i][1])
        else:
            winner=random.randint(0,1)
            next_phase_teams.append(all_matches[i][winner])

    if round==16:
        phase_8_matches=[(next_phase_teams[5],next_phase_teams[4]),(next_phase_teams[3],next_phase_teams[1]),(next_phase_teams[2],next_phase_teams[0]),(next_phase_teams[7],next_phase_teams[6])]
        return phase_8_matches

    elif round==8:
        phase_4_matches=[(next_phase_teams[1],next_phase_teams[0]),(next_phase_teams[3],next_phase_teams[2])]
        return phase_4_matches

    elif round==4:
        final_match=[(next_phase_teams[0],next_phase_teams[1])]
        return final_match

    elif round==1:
        champion=next_phase_teams[0]
        return champion


def play_EURO2021(test_data,model1,model2,model3,probabilistic):
    """Model 1  must be trained with 7 components
       Model 2 must be trained with 2 components (other numbers can be used but it must be less than 4)
       Model 3 must be trained with 1 component
    """
    countries_and_points = create_countries_and_points_df(test_data)
    countries_and_points, g_results = group_phase(test_data, X_test, countries_and_points, model1, 7,probabilistic)
    GroupA, GroupB, GroupC, GroupD, GroupE, GroupF, thirds = create_groups_sorted(countries_and_points)
    fake_next_phase_matches = phase_16(GroupA, GroupB, GroupC, GroupD, GroupE, GroupF, thirds, teams_boxes21, model1,probabilistic)
    fake_semifinal_matches = phase_8(fake_next_phase_matches, teams_boxes21, model2, 2,probabilistic)
    fake_final_match = phase_4(fake_semifinal_matches, teams_boxes21, model2, 2,probabilistic)
    glorious_champion = play_final(fake_final_match, teams_boxes21, model3, 1,probabilistic)
    return glorious_champion

##
import random
countries_and_points=create_countries_and_points_df(test_data)
countries_and_points,g_results=group_phase(test_data,X_test,countries_and_points,model_rf,7)
GroupA,GroupB,GroupC,GroupD,GroupE,GroupF,thirds=create_groups_sorted(countries_and_points)
fake_next_phase_matches=phase_16(GroupA,GroupB,GroupC,GroupD,GroupE,GroupF,thirds,teams_boxes21,model_rf)
fake_semifinal_matches=phase_8(fake_next_phase_matches,teams_boxes21,model_rf_2,2)
fake_final_match=phase_4(fake_semifinal_matches,teams_boxes21,model_rf_2,2)
fake_champion=play_final(fake_final_match,teams_boxes21,model_rf_1,1)
##

champion_2=play_EURO2021(test_data,model_rf,model_rf_2,model_rf_1,False)
##
champion_3=play_EURO2021(test_data,model_rf,model_rf_2,model_rf_1,True)
##

from sklearn.decomposition import PCA
pca=PCA(n_components=7)
test_transformed=pca.fit_transform(test_data.drop(['home','away'],axis=1))
##
fake_result=model_rf.predict_proba(test_transformed[0].reshape(1,-1))
##
r1=fake_result[0][0]
r2=fake_result[0][1]
for _ in range(10):
    number=random.uniform(0,1)
    print(number)
    print('Result')
    if number <= r1:
        print(0,number,'<',r1)
    elif r1 < number <= r1 +r2:
        print(1,r1,number,r1+r2)

    else:
        print(2)