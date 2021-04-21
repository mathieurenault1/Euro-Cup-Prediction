import pandas as pd
import copy
import re
participants16=pd.read_csv('/Users/david/DataSets/international-uefa-euro-championship-players-2016-to-2016-stats.csv')
fifa16=pd.read_csv('/Users/david/DataSets/players_16.csv')
print(participants16.columns)
print(fifa16.columns)


def create_dob_column(participants):
    """Creates a new column dob which is the correct date format to match the players birthday
        Args: participants ---> The file with all the players that played EURO for an specific year"""

    import datetime
    participants['dob']= 0 * len(participants)
    for i in range(len(participants)):
        timestamp= datetime.datetime.fromtimestamp(participants['birthday'][i])
        time_str= str(timestamp)
        participants['dob'][i]=time_str[0:10]

    return participants





def get_distinct_nationalities(participants):
    """Creates a list of the International Teams that are in the Tournament
        Args: participants ---> The file with all the players that played EURO for an specific year"""
    nationalities = []
    for i in range(len(participants)):
        nationality=participants.loc[i,'nationality']
        if nationality not in nationalities:
            nationalities.append(nationality)
    return nationalities


def keep_euro_teams(fifa):
    """Filter the fifa dataset so that we only keep the players of the nationalities we are interested about
     Args: fifa ---> The fifa dataset for a specifc year"""
    df=fifa[fifa['nationality'].isin(nationalities)]
    return df



def create_participants_dictionary(participants):
    """Creates a dictionary in which the key is the international team and the values is the list of players that
        compose the International Team
        Args: participants ---> The file with all the players that played EURO for an specific year"""

    participants_dict={}
    for i in range(len(participants)):
        participants_dict.setdefault(participants.loc[i,"nationality"],[]).append(participants.loc[i,'full_name'])
    return participants_dict




def international_team_selection(fifa,participants_copy):
    """Filters the fifa dataset based on the dictionary created in the create participants dictionary. It looks on
    both the shortname of the fifa column and the longname.
    Args: participants ---> The file with all the players that played EURO for an specific year
          fifa ----> fifa dataset for that same year
    Output: It returns a df with the selected players and a dictionary which indicates the players still to match"""

    container=[]
    for i in range(len(fifa)):
        nationality = fifa.iloc[i, 8]

        if fifa.iloc[i, 2] in participants_copy[nationality]:  #Short name search
            container.append(fifa.iloc[i, :])
            try:
                participants_copy[nationality].remove(fifa.iloc[i,2])
            except ValueError:
                pass


        elif fifa.iloc[i, 3] in participants_copy[nationality]:  # Long name search
            container.append(fifa.iloc[i, :])
            try:
                participants_copy[nationality].remove(fifa.iloc[i,3])
            except ValueError:
                pass

    international_players = pd.concat(container, axis=1)
    international_players= international_players.transpose()

    return international_players,participants_copy



def check_players_left(participants):
    result=0
    for k in participants.keys():
        for v in participants[k]:
            result += 1

    print(result)



def second_matching_name_and_surname_combo(fifa, participants):
    """It tries the combination of names and surnames for those players which have long names and more than one surname.
        Args: participants ---> The file with all the players that played EURO for an specific year
        fifa ----> fifa dataset for that same year
        Output: dataframe with the matched players and the dictionary with the players still to match """
    container = []
    for k in participants.keys():
        country = fifa[fifa['nationality'] == k]
        for v in participants[k]:
            name = v.split(' ')
            if len(name) < 3:
                for i in range(len(country)):
                    if re.findall(name[0], country.iloc[i, 3]) and re.findall(name[1], country.iloc[i, 3]):
                        container.append(country.iloc[i, :])
                        try:
                            participants[k].remove(
                                v)  ###remove(country.iloc[i, 3]) probably allowing us to choose more than one
                        except ValueError:
                            pass
            else:
                for i in range(len(country)):
                    if re.findall(name[0], country.iloc[i, 3]) and re.findall(name[2], country.iloc[i, 3]):
                        container.append(country.iloc[i, :])
                        try:
                            participants[k].remove(
                                v)  ###remove(country.iloc[i, 3]) probably allowing us to choose more than one
                        except ValueError:
                            pass

    df = pd.concat(container, axis=1)
    df = df.transpose()
    return df, participants



def create_players_left_df(participants, participants_left):
    """Creates a dataframe with the players that we did not capture. Not really sure if we will need to use it """
    container = []
    for k in participants_left.keys():
        nationals = participants[participants['nationality'] == k]
        for v in participants_left[k]:
            for i in range(len(nationals)):
                if nationals['full_name'].iloc[i] == v:
                    container.append(nationals.iloc[i, :])
                    break

    df2 = pd.concat(container, axis=1)
    df2 = df2.transpose()

    return df2


def match_name_birthday(participants_left, fifa,participants):
    """We try to match by first name or surname and then double check with the birthday(dob column)  so that we do not include wrong players.
     ----
     Arguments: participants ---> The file with all the players that played EURO for an specific year
                fifa ----> fifa dataset for that same year
                participants_left----> the dictionary with the players still missing

    -----
    Out: df--> Players captured
         participants_left --> the dictionary with missing players updated
        """

    container = []
    ghost_players=[]
    for k in participants_left.keys():
        nationals = participants[participants['nationality'] == k]
        nationals_fifa = fifa[fifa['nationality'] == k]
        for v in participants_left[k]:
            for i in range(len(nationals)):
                if v == nationals.iloc[i, 0]:
                    for j in range(len(nationals_fifa)):
                        name = nationals.iloc[i, 0].split(' ')
                        birthday = nationals.iloc[i, -1]
                        if (re.findall(name[0], nationals_fifa.iloc[j, 3]) or re.findall(name[1], nationals_fifa.iloc[
                            j, 3])) and (birthday == nationals_fifa.iloc[j, 5]):
                            result = nationals_fifa.iloc[j, :]
                            container.append(result)
                            participants_left[k].remove(v)



    df = pd.concat(container, axis=1)
    df = df.transpose()
    return df, participants_left





participants16=create_dob_column(participants16)
nationalities=get_distinct_nationalities(participants16)
fifa16=keep_euro_teams(fifa16)
participants=create_participants_dictionary(participants16)
international_players,participants_left=international_team_selection(fifa16,participants)
df,participants_left=second_matching_name_and_surname_combo(fifa16,participants_left)
participants_left_df=create_players_left_df(participants16,participants_left)
df3,participants_left=match_name_birthday(participants_left,fifa16,participants16)
all_players_selected=[international_players,df,df3]
players=pd.concat(all_players_selected)
print('We still miss:', len(participants16)-len(players))
print('The dataset you are looking for is players. That is the one with all matching processes concatenated. The dictionary participants left tell us who is missing')


