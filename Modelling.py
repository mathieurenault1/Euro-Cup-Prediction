import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




def initiate_box_approach(teams):
    teams = teams.drop(['nationality'], axis=1)
    team_number = 1
    for i in range(len(teams)):
        teams.loc[i, 'team_number'] = team_number
        team_number += 1

    return teams


def initiate_df(teams,results):
    df = pd.DataFrame()
    df['home'] = 0 * len(results)
    df['phase']= 0*len(results)
    for i in range(len(results)):
        df.loc[i, 'home'] = results.loc[i, 'home']
        df.loc[i, 'phase'] = int(results.loc[i, 'phase'])

    return df


def remove_spaces_and_fix_ireland(results):
    """Make things look similar. Ireland and Northern Ireland interfere"""
    results['home'] = results['home'].str.replace('_', ' ')
    results['away'] = results['away'].str.replace('_', ' ')
    for i in range(len(results)):
        if results.loc[i, 'home'] == 'Ireland':
            results['home'].iloc[i] = 'Republic of Ireland'
        elif results.loc[i, 'away'] == 'Ireland':
            results['away'].iloc[i] = 'Republic of Ireland'

    return results






def create_df_for_the_model(columns,teams,df):
    for i in columns:
        df[i]=0*len(df)
        for j in range(len(df)):
            team=teams[teams['country']==df.loc[j,'home']]
            answer=float(team[i])
            df.loc[j,i]=answer

    df['away'] = 0 * len(results)
    for i in range(len(df)):
        df.loc[i, 'away'] = results.loc[i, 'away']

    for i in columns:
        df[i + '_2'] = 0 * len(df)
        for j in range(len(df)):
            team = teams[teams['country'] == df.loc[j, 'away']]
            answer = float(team[i])
            df.loc[j, i + '_2'] = answer

    return df


def create_results(teams,results,box_approach=True):

    if box_approach==True:
        teams=initiate_box_approach(teams)

    columns=list(teams)

    if 'Unnamed: 0' in columns:
        teams = teams.rename(columns={'Unnamed: 0': 'team_number'})

    results = remove_spaces_and_fix_ireland(results)
    df = initiate_df(teams, results)
    columns.remove('country')
    df = create_df_for_the_model(columns, teams, df)
    df['final_result'] = results['final_result']

    return df



def columns_substraction_method(df):
    """Support function used in the substraction_method"""
    columns=list(df)
    columns.remove('home')
    columns.remove('away')
    columns.remove('phase')
    columns.remove('final_result')
    columns.remove('team_number')
    columns.remove('team_number_2')
    stop=int(len(columns)/2)
    columns=columns[0:stop]
    return columns

def substraction_method(df):
    """Creates the dataframe with the data according to the substraction method. Feed in the final dataframe
    with the matches according to the concatenation method. """
    df=df.fillna(0)
    columns=columns_substraction_method(df)
    new_df=pd.DataFrame()
    new_df['home'] = df['home']
    new_df['away'] = df['away']
    new_df['phase'] = df['phase']
    for i in columns:
        new_df[i]=0*len(df)

    for i in range(len(df)):
        for j in columns:
            new_df.loc[i,j]=int(df.loc[i,j]) - int(df.loc[i,j + '_2'])


    return new_df




"""KEEP THIS AS IT IS. DON'T KNOW WHAT IS WRONG. WHENEVER YOU CALL IT WITH A DIFFERENT NAME THAN TEAMS AND RESULTS IT DOES NOT WORK """



##
teams=pd.read_csv('/Users/david/DataSets/Fifa/FinalData/team_boxes_2012.csv',sep=';')
results=pd.read_csv('/Users/david/DataSets/Fifa/Results/Results2012.csv',sep=';')
df_2012=create_results(teams,results)
df_2012_substraction=substraction_method(df_2012)
teams=pd.read_csv('FinalData/teams_boxes_2016.csv')
results=pd.read_csv('/Users/david/DataSets/Fifa/Results/Results2016_with_phase.csv',sep=';')
df_2016=create_results(teams,results)
df_2016_substraction=substraction_method(df_2016)
teams=pd.read_csv('/Users/david/DataSets/Fifa/team_boxes_WorldCup18.csv',sep=',')
results=pd.read_csv('/Users/david/DataSets/Fifa/Results/ResultsWorldCup18.csv',sep=';')
wc18=create_results(teams,results)
wc18_substraction=substraction_method(wc18)
all_matches=[df_2012,df_2016,wc18]
all_matches_substraction=[df_2012_substraction,df_2016_substraction,wc18_substraction]
df_concatenation=pd.concat(all_matches)
df_substraction=pd.concat(all_matches_substraction)
df_substraction['final_result']=df_concatenation['final_result']

df_substraction=df_substraction.reset_index()
df_substraction=df_substraction.drop(['index'],axis=1)
##

"""Data for 2021"""
teams=pd.read_csv('FinalData/team_boxes_21.csv',sep=',')
teams['nationality']=teams['country']
results=pd.read_csv('FinalData/results2021.csv',sep=';')
results.loc[12,'home']='Finland'
df_2021=create_results(teams,results)
df_2021_substraction=substraction_method(df_2021)






##
"""Modelling starts here"""



x=df_substraction
x=x.fillna(0)
y=x['final_result']
x=x.drop(['home','away','final_result'],axis=1)


X_train,X_test,y_train,y_test=train_test_split(x,y,stratify=y,test_size=0.2,random_state=1)

dt=DecisionTreeClassifier(random_state=1)
dt.fit(X_train,y_train)
y_pred=dt.predict(X_test)
print(accuracy_score(y_test,y_pred))





def intermediate_step(correct, errors):
    """Creates a dataframe contianing the data for all the predicitons independely of if they are wrong"""
    correct['correct'] = 1
    errors['correct'] = 0 * len(errors)
    all_matches = [errors, correct]
    predictions_df = pd.concat(all_matches)

    return predictions_df


def split_results(df,prediction,label_test):
    """Creates two dataframes, one containing the matches we predicted correctly and other containg just the errors"""
    all_indices=label_test.index
    errors=[]
    errors_predicted=[]
    correct=[]
    correct_predicted=[]
    counter=0
    for i in label_test:
        if i == prediction[counter]:
            correct.append(all_indices[counter])
            correct_predicted.append(prediction[counter])



        else:
            errors.append((all_indices[counter]))
            errors_predicted.append(prediction[counter])

        counter +=1

    errors_df = df.iloc[errors]
    correct_df = df.iloc[correct]
    errors_df['predicted']=errors_predicted
    correct_df['predicted']=correct_predicted


    return correct_df, errors_df





correct,errors=split_results(df_substraction,y_pred,y_test)
predictions_df=intermediate_step(correct,errors)




##
import numpy as np
table=predictions_df.pivot_table(columns=['correct'],aggfunc=[np.mean,np.std])
print(table)


##

