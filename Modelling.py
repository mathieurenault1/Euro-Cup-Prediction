import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

teams=pd.read_csv('FinalData/national_teams_2016.csv')
results=pd.read_csv('FinalData/FinalResults2016.csv')




def initiate_df(teams,results):
    df = pd.DataFrame()
    df['home'] = 0 * len(results)
    for i in range(len(results)):
        df.loc[i, 'home'] = results.loc[i, 'home']



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



"""Important to keep the order in which the functions are called otherwise won't work. The first must always be
the rename of the unnamed column into categorical. Second the ireland stuff etc"""
teams=teams.rename(columns={'Unnamed: 0': 'team_number'})
results=remove_spaces_and_fix_ireland(results)
df=initiate_df(teams,results)
columns=list(teams)
columns.remove('country')
df=create_df_for_the_model(columns,teams,df)
df['final_result']=results['final_result']      #Creates the final result column



"""Modelling starts here"""

x=df
y=x['final_result']
x=x.drop(['home','away','final_result'],axis=1)





X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=1)
dt=DecisionTreeClassifier(random_state=1)
dt.fit(X_train,y_train)
y_pred=dt.predict(X_test)
print(accuracy_score(y_test,y_pred))




