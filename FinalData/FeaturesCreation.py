import pandas as pd

"""Make sure to change the path. The file should be the one with the filtered players"""
players=pd.read_csv('/Users/david/DataSets/Fifa/FinalData/finaldata16.csv')

nationalities=[]
for i in range(len(players)):
    if players.loc[i,'nationality'] not in nationalities:
        nationalities.append(players.loc[i,'nationality'])


"""Create the dataframe and one row for each pf the teams"""
teams=pd.DataFrame()
teams['country'] = nationalities


def keep_relevant_columns(players):
    """Removes the columns that have no info or are categorical. This should be improved in the future
    INPUT: Dataset with the players
    Out: List with the columns we will use. We will use the list in the create_columns function"""
    columns = list(players)
    columns.remove('id')
    columns.remove('Unnamed: 0')
    columns.remove('Fullname')
    columns.remove('preferred_foot')
    columns.remove('birth_date')
    columns.remove('preferred_positions')
    columns.remove('work_rate')
    columns.remove('weak_foot')
    columns.remove('gk_reflexes')
    columns.remove('nationality')
    columns.remove('value')
    columns.remove('wage')
    return columns

columns=keep_relevant_columns(players)

def create_columns(teams,columns):
    """Creates the average value for each feature for each team and adds it as a column to our previously created dataframe.
    COPY THE STRUCTURE SO THAT YOU CAN CREATE NEW FEATURES THAT APPLY FOR ALL TEAMS"""
    for i in columns:
        teams[i] = 0 * len(teams)
        for j in range(len(teams)):
            nationals=players[players['nationality']==teams.loc[j,'country']]
            nationals[i].fillna(nationals[i].mean())
            teams.loc[j,i]=nationals[i].mean()

    return teams


"""This is the data we will input to the model. We need to add more features that is columns but the structure must be 
the same that is one row is one national team"""

teams=create_columns(teams,columns)








