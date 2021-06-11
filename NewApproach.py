import pandas as pd
import copy
import re
from DataAssembly import create_dob_column
from DataAssembly import get_distinct_nationalities
from DataAssembly import create_participants_dictionary


"""Make sure to change the path to the files"""

"""Note that this approach scales well and works for ANY year. To change the year simply change the files we are inputing.That is
in the case we want to do for 2012 we would use fifa12 dataset and participants12 dataset """


participants16=pd.read_csv('/Users/david/DataSets/international-uefa-euro-championship-players-2016-to-2016-stats.csv')
fifa16=pd.read_csv('/Users/david/DataSets/Fifa/fifa16.csv',sep=';')



"""Run in case we are working with 2012 file or we need to create birthday column. Obviously 
if you run thsese lines the previous ones have no effect"""
#import pandas as pd
#participants16=pd.read_csv('/Users/david/DataSets/Fifa/players_2012.csv',sep=';')
#participants16['dob'] = 0 * len(participants16)
#for i in range(len(participants16)):
    #participants16.loc[i, 'dob'] = participants16.loc[i, 'Date of birth (age)'][1:11]
#fifa12=pd.read_csv('/Users/david/DataSets/Fifa/fifa12.csv',sep=';')




def modify_dob_participants(participants):
    """Changes the dob of the participants dataframe from year/month/day to day/month/year
    Args:
        input --> participants df
        output --> participants df with new column
    """
    participants['dob_modified']=0*len(participants)
    for i in range(len(participants)):
        year=participants.loc[i,'dob'][0:4]
        month=participants.loc[i,'dob'][5:7]
        day=participants16.loc[i,'dob'][8:11]
        participants.loc[i,'dob_modified']=day+'/'+month+'/'+year

    return participants

def create_dob_participants(participants):
    """Used in case we have the birthdate with mixed numbers and strings. We will have to choose between using the
    previous function modify_dob_participants or this one"""
    participants['dob_modified'] = 0 * len(participants)
    for i in range(len(participants)):
        year = participants.loc[i, 'birthday'][-5:-1]
        day = participants.loc[i, 'birthday'][0:2]
        day_integer = int(day)
        if day_integer < 10:
            day = '0' + day[0]
        month = check_month(i, participants)
        participants.loc[i, 'dob_modified'] = day + '/' + month + '/' + year

    return participants


def check_month(i, participants):
    """Used in the above function create_dob_participants """
    date = participants.loc[i, 'birthday']
    month = re.findall('\D', date)
    final_month = []
    for i in month:
        if i != ' ':
            final_month.append(i)
    month = ''.join(final_month)
    if month == 'January':
        answer = '01'
    elif month == 'February':
        answer = '02'
    elif month == 'March':
        answer = '03'
    elif month == 'April':
        answer = '04'
    elif month == 'May':
        answer = '05'
    elif month == 'June':
        answer = '06'
    elif month == 'July':
        answer = '07'
    elif month == 'August':
        answer = '08'
    elif month == 'September':
        answer = '09'
    elif month == 'October':
        answer = '10'
    elif month == 'November':
        answer = '11'
    elif month == 'December':
        answer = '12'

    return answer




def match(participants,fifa,participants_dict):
    """Matches the players that we want to select and also creates a new column nationality in the fifa dataset
    Args:
        * input --> participants (our participants df), fifa(our fifa df) and the participants_dictionary
        * output --> df (dataframe with selected players),original_dict(the dictionary that we had)"""
    container=[]
    fifa=fifa.sort_values(by='Fullname')
    fifa['nationality']=0*len(fifa)
    for i in range(len(participants)):
        name=participants.loc[i,'full_name'].split()
        nationality=participants.loc[i,'nationality']
        for j in range(len(fifa)):
            if (re.findall(name[0],fifa.loc[j,'Fullname']) or re.findall(name[0],fifa.loc[j,'Fullname'])) and participants.loc[i,'dob_modified']==fifa.loc[j,'birth_date']:
                fifa.loc[j,'nationality']=participants.loc[i,'nationality']
                container.append(fifa.loc[j,:])
                try:
                    participants_dict[nationality].remove(' '.join(name))
                except ValueError:
                    pass

    df = pd.concat(container, axis=1)
    df = df.transpose()
    return df, participants_dict


def create_dataset(participants,fifa):
    """Merges everything together
    Input --> participants dataframe and fifa dataset
    Out---> df (dataframe with selected players),original_dict(the dictionary that we had)
        participants_left(dicitionary with participants still to match)"""
    #participants_df=modify_dob_participants(participants)  #In case we need to change the format of the birthday
    participants = create_dob_participants(participants)
    original_participants=create_participants_dictionary(participants)
    df,participants_left=match(participants,fifa,original_participants)
    return df, original_participants, participants_left








nationalities=get_distinct_nationalities(participants16)
players, original_participants, participants_left=create_dataset(participants16,fifa16)
print(participants_left)


"""IF NAME ERROR REMEMBER THE CHEATING JUST CHANGE THE NAME OF THE DATAFRAME TO participants16"""

"""IT TAKES A WHILE !!!!! and forget about the warning. Players is the dataframe with the players that we matched and participants left is the dictionary with the players
that we still need to include. We matched 458 but there are 95 missing"""





