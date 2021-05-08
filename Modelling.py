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
##

"""KEEP THIS AS IT IS. DON'T KNOW WHAT IS WRONG. WHENEVER YOU CALL IT WITH A DIFFERENT NAME THAN TEAMS AND RESULTS IT DOES NOT WORK """


teams=pd.read_csv('/Users/david/DataSets/Fifa/FinalData/team_boxes_2012.csv',sep=';')
results=pd.read_csv('/Users/david/DataSets/Fifa/Results/Results2012_with_phase.csv',sep=';')
df_2012=create_results(teams,results)
teams=pd.read_csv('FinalData/teams_boxes_2016.csv')
results=pd.read_csv('/Users/david/DataSets/Fifa/Results/Results2016_with_phase.csv',sep=';')
df_2016=create_results(teams,results)
##

all_matches=[df_2012,df_2016]
all_matches=pd.concat(all_matches)

##

"""Modelling starts here"""



x=all_matches
x=x.fillna(0)
y=x['final_result']
x=x.drop(['home','away','final_result'],axis=1)




X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,stratify=y,random_state=1)
dt=DecisionTreeClassifier(random_state=1)
dt.fit(X_train,y_train)
y_pred=dt.predict(X_test)
print(accuracy_score(y_test,y_pred))




