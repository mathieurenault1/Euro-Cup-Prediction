import pandas as pd
import copy
import re
from DataAssembly import create_dob_column
from DataAssembly import get_distinct_nationalities
from DataAssembly import create_participants_dictionary
from DataAssembly import keep_euro_teams
from DataAssembly import international_team_selection
from DataAssembly import second_matching_name_and_surname_combo
from DataAssembly import create_players_left_df
from DataAssembly import match_name_birthday


"""Make Sure to Change the Path to the files. This approach would work for modern Fifa datasets"""

participants16=pd.read_csv('/Users/david/DataSets/international-uefa-euro-championship-players-2016-to-2016-stats.csv')
fifa16=pd.read_csv('/Users/david/DataSets/players_16.csv')


def create_dataset(fifa,participants):
    participants=create_dob_column(participants)
    nationalities=get_distinct_nationalities(participants)
    fifa=keep_euro_teams(fifa,nationalities)
    participants_dict=create_participants_dictionary(participants)
    international_players,participants_left=international_team_selection(fifa,participants_dict)
    df,participants_left=second_matching_name_and_surname_combo(fifa16,participants_left)
    participants_left_df=create_players_left_df(participants,participants_left)
    df3,participants_left=match_name_birthday(participants_left,fifa,participants)
    all_players_selected=[international_players,df,df3]
    players=pd.concat(all_players_selected)

    return players,participants_dict,participants_left



players,participants_dict,participants_left=create_dataset(fifa16,participants16)

print('We still miss:', len(participants16)-len(players))
print('The dataset you are looking for is players. That is the one with all matching processes concatenated. The dictionary participants left tell us who is missing')


