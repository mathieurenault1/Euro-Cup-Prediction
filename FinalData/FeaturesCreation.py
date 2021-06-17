import pandas as pd
import numpy as np
import math

"""This File is divided into 2 parts. The first one focuses on the creation of simple features having as output the teams dataframe stored in the Final
Data folder as national_teams_2016. The second part focuses on the boxes approach which creates the output teams_boxes_2016"""
"""Make sure to change the path. The file should be the one with the filtered players"""
players=pd.read_csv('/Users/david/DataSets/Fifa/FinalData/finaldata16.csv')
#players=pd.read_csv('\\Users\\Admin\\Documents\\GitHub\\Euro-Cup-Prediction\\FinalData\\finaldata16.csv')


nationalities=[]
for i in range(len(players)):
    if players.loc[i,'nationality'] not in nationalities:
        nationalities.append(players.loc[i,'nationality'])


"""Create the dataframe and one row for each pf the teams"""
teams=pd.DataFrame()
teams['country'] = nationalities
teamsSTD=pd.DataFrame()
teamsSTD['country'] = nationalities
teamsVAR=pd.DataFrame()
teamsVAR['country'] = nationalities

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


def create_columnsV(teamsVAR,columns):
    """ Creates the variance for each feature for each team and adds it as new column"""
    for i in columns:
        teamsVAR[i] = 0 * len(teamsVAR)
        for j in range(len(teamsVAR)):
            nationals=players[players['nationality']==teamsVAR.loc[j,'country']]
            nationals[i].fillna(nationals[i].var())
            teamsVAR.loc[j,i]=nationals[i].var()
    return teamsVAR

def create_columnsS(teamsSTD,columns):
    """ Creates the standard deviation for each feature for each team and adds it as new column"""
    for i in columns:
        teamsSTD[i] = 0 * len(teamsSTD)
        for j in range(len(teamsSTD)):
            nationals=players[players['nationality']==teamsSTD.loc[j,'country']]
            nationals[i].fillna(nationals[i].std())
            teamsSTD.loc[j,i]=nationals[i].std()
    return teamsSTD





"""This is the data we will input to the model. We need to add more features that is columns but the structure must be 
the same that is one row is one national team"""

teams=create_columns(teams,columns)
teamsVAR=create_columnsV(teamsVAR,columns)
teamsSTD=create_columnsS(teamsSTD,columns)

print(teams)
print(teamsVAR)
print(teamsSTD)

"""---------------------------------------------------------------------------------------------------------------------"""

"""Here starts the section to create the box approach.  """




def create_position_column(players):
    """Creates the position for each player. IT MAKES use of the player_position function.
    Args: IN : players , Out: player
        """
    players['player_position']=0*len(players)
    goalkeepers = ['GK']
    defenders = ['RB', 'CB', 'LB', 'RWB', 'LWB']
    midfielders = ['CDM', 'CM', 'RM', 'LM' ]
    attackers = ['CAM', 'RW', 'LW', 'CF', 'ST']
    for i in range(len(players)):


            positions_list = players.loc[i, 'preferred_positions']
            positions_list = positions_list.replace('-', ' ')
            positions_list = positions_list.split()
            try:
                players.loc[i,'player_position']=player_positions(positions_list,goalkeepers,defenders,midfielders,attackers)

            except KeyError:
                players.loc[i, 'player_position']='Error'


    return players




def player_positions(position_list,goalkeepers,defenders,midfielders,attackers):
    """Solves the situation when one player can play in multiple positions. It basically takes into account all positions in
    which a player can play and then based on the area of the field that those positions are decides which one is it most common
    position. Don't use it separately of the create position column function"""
    import operator
    result={}
    for position in position_list:
        if position in defenders:
            if 'defender' in result.keys():
                result['defender'] +=1
            else:
                result.setdefault('defense',1)
        elif position in midfielders:
            if 'midfielder' in result.keys():
                result['midfielder'] += 1
            else:
                result.setdefault('midfielder', 1)
        elif position in attackers:
            if 'attacker' in result.keys():
                result['attacker'] += 1
            else:
                result.setdefault('attacker', 1)
        elif position in goalkeepers:
            if 'goalkeeper' in result.keys():
                result['goalkeeper'] +=1
            else:
                result.setdefault('goalkeeper',1)

    return max(result.items(), key=operator.itemgetter(1))[0]









"""This is kinf of messy nut would work in case we want to change how the features are grouped"""

columns_rating=['current_rating']
columns_weight = ['weight']
columns_height=['height']
columns_ball_control = ['ball_control','vision','crossing','short_pass','long_pass','dribbling']
columns_physical=['acceleration','sprint_speed','agility']
columns_heading=['jumping','balance','heading']
columns_defending=['marking','slide_tackle','stand_tackle','aggression','reactions','interceptions','composure','stamina','strength']
columns_finishing=['att_position','shot_power','finishing','long_shots','curve','fk_acc','penalties']
columns_goalkeeper=['gk_positioning', 'gk_diving', 'gk_handling', 'gk_kicking']
columns_grouped=[(columns_rating,'rating'),(columns_weight,'weight'),(columns_height,'height'),(columns_heading,'heading'),(columns_ball_control,'ball_skills'),(columns_physical,'physical'),(columns_defending,'defending'),(columns_finishing,'finishing')]
positions = ['defense','midfielder','attacker']



def create_boxes(players,std=False):
    """Creates the final dataframe with the information according to the box approach. If std is False we would compute the mean
    if STD is True we would compute the STD dev"""

    df=pd.DataFrame()
    for position in positions:
        selected_players = players[players['player_position'] == position]
        for column in columns_grouped:
            if std==False:
                result = selected_players.pivot_table(values=column[0], index=['nationality'], aggfunc=[np.mean])
                df[str(position)+'_'+str(column[1])] = round(result.sum(axis=1) / result.shape[1]).astype('int')
            else:
                result = selected_players.pivot_table(values=column[0], index=['nationality'], aggfunc=[np.std])
                df[str(position) + '_' + str(column[1])] = round(result.sum(axis=1) / result.shape[1]).astype('int')



    goalkeepers = players[players['player_position'] == 'goalkeeper']
    if std==False:
        result = goalkeepers.pivot_table(values=['current_rating'], index=['nationality'], aggfunc=[np.max])
        df['goalkeeper_rating'] = result.astype('int')
    else:
        result = goalkeepers.pivot_table(values=['current_rating'], index=['nationality'], aggfunc=[np.std])
        df['goalkeeper_rating'] = result.astype('int')

    df['country'] = df.index
    return df


def ratio_dataframe(mean,std):
    mean=mean.drop(['country'],axis=1)
    std = std.drop(['country'], axis=1)
    ratio_df=mean.div(std)
    return ratio_df

##



def confidence_interval_pos_dataframe(mean,std):
    mean=mean.drop(['country'],axis=1)
    std = std.drop(['country'], axis=1)
    confidence_interval_pos=mean + 1.96 * (std/math.sqrt(22))
    return confidence_interval_pos

def confidence_interval_neg_dataframe(mean,std):
    mean=mean.drop(['country'],axis=1)
    std = std.drop(['country'], axis=1)
    confidence_interval_neg=mean - 1.96 * (std/math.sqrt(22))
    return confidence_interval_neg

players=create_position_column(players)
teams_boxes=create_boxes(players,std=False)
teams_boxes_std=create_boxes(players,std=True)
teams_boxes=teams_boxes.fillna(np.mean(teams_boxes))
teams_boxes_std=teams_boxes_std.fillna(np.max(teams_boxes_std))
ratio_df=ratio_dataframe(teams_boxes,teams_boxes_std)
confidence_interval_pos_df=confidence_interval_pos_dataframe(teams_boxes,teams_boxes_std)
confidence_interval_neg_df=confidence_interval_neg_dataframe(teams_boxes,teams_boxes_std)

##
"""Leaf this here as it migh be useful in the future. We will need it to produce the ratio probably for all tournaments """
players_WC18=pd.read_csv('/Users/david/DataSets/Fifa/FinalData/players12.csv')
players_WC18=create_position_column(players_WC18)
teams_boxes_WC=create_boxes(players_WC18,std=False)
teams_boxes_std_WC=create_boxes(players_WC18,std=True)
teams_boxes_std_WC=teams_boxes_std_WC.fillna(np.max(teams_boxes_std_WC))
confidence_interval_neg_df=confidence_interval_neg_dataframe(teams_boxes_WC,teams_boxes_std_WC)
confidence_interval_neg_df.to_csv('FinalData/WC18_intervals.csv')


##

players_21=pd.read_csv('/Users/david/DataSets/Fifa/FinalData/Players_2021.csv',sep=';')
players_21.loc[78,'preferred_positions']='CB'
players_21.loc[308,'preferred_positions']='CB'

##
players_21=create_position_column(players_21)
##
teams_boxes_21=create_boxes(players_21,std=False)
teams_boxes21_std=create_boxes(players_21,std=True)
confidence_interval_neg_df=confidence_interval_neg_dataframe(teams_boxes_21,teams_boxes_std)

##
teams_boxes_21.to_csv('FinalData/team_boxes_21.csv')
teams_boxes21_std.to_csv('FinalData/team_boxes21_std.csv')