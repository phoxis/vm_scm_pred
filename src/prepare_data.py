#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ###
# ### Copyright 2024 VistaMilk
# ###
# ### Licensed under the Apache License, Version 2.0 (the "License");
# ### you may not use this file except in compliance with the License.
# ### You may obtain a copy of the License at
# ###
# ###     http://www.apache.org/licenses/LICENSE-2.0
# ###
# ### Unless required by applicable law or agreed to in writing, software
# ### distributed under the License is distributed on an "AS IS" BASIS,
# ### WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# ### See the License for the specific language governing permissions and
# ### limitations under the License.
# ###

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
import logging
import coloredlogs
import random
import os

# Keep False to generate the full dataset. If True, then this script will generate data for only 10 cows.
# This is to test if the data being generated is correct or not.
DEBUG = False

# Setup the logger to log all the messages
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

def _load_weights_milk_farm(data_path):
    '''
    Function which returns the integrated data about milk, farm and weights
    '''
    # Read the source files
    db_milk    = pd.read_csv (data_path / 'db_milk.csv', encoding = 'unicode escape')
    db_weights = pd.read_csv (data_path / 'db_weights.csv', encoding = 'unicode escape')
    db_master  = pd.read_csv (data_path / 'db_master.csv', encoding = 'unicode escape')
    
    
    # Rename columns and fix types
    db_milk.rename (columns = {'tb_num': 'id', 'milk_date': 'date'}, inplace = True)
    db_milk['date'] = pd.to_datetime (db_milk['date'], format = '%d/%m/%Y')
    db_milk['calving'] = pd.to_datetime (db_milk['calving'], format = '%d/%m/%Y')
    
    db_weights.rename(columns = {'tbnum': 'id'}, inplace = True)
    db_weights['date'] = pd.to_datetime (db_weights['date'], format = '%d/%m/%Y')
    db_weights['calving'] = pd.to_datetime (db_weights['calving'], format = '%d/%m/%Y')
    
    db_master.rename (columns = {'tbnum': 'id', ' parity': 'parity', ' exitdate': 'exit_date'}, inplace = True)
    db_master['dob'] = pd.to_datetime (db_master['dob'], format = '%d-%b-%y')
    db_master['exit_date'] = pd.to_datetime (db_master['exit_date'], format = '%d/%m/%Y')
    


    # Merge the milking DB with the master DB to get data about milking events per farm
    weights_milk = pd.merge (db_milk, db_weights, how = 'left', on = ['id', 'date', 'calving', 'parity'])
    weights_milk_farm = pd.merge (weights_milk, db_master, how = 'left', on = ['id', 'parity'])

    # Delete all the columns we don't want to keep
    col_keep = set(['calving', 'parity', 'date', 'milking_time', 'fat',
                'protein', 'lactose', 'urea', 'scc', 'yield', 'max_flow', 'id', 
                'weight','BCS', 'dob', 'farm', 'calvingdate', 'dryoffdate','exit_date'])
    weights_milk_farm.drop(set(weights_milk_farm.columns) - col_keep, axis=1, inplace=True)
    
    # Add column 'days_since_calving'
    weights_milk_farm.loc[:, 'days_since_calving'] = (weights_milk_farm['date'] - weights_milk_farm['calving']).dt.days
    

    #Keeping only those cows born after 2010
    weights_milk_farm = weights_milk_farm[weights_milk_farm['dob'] > '2010-01-01']
    

    if DEBUG:
        # Pick 10 cows at random, if DEBUG is True (see top of this script).
        cows = weights_milk_farm.id.unique()
        selected_cows = random.choices(cows, k=10)
        weights_milk_farm = weights_milk_farm[weights_milk_farm.id.isin(selected_cows)]

    return weights_milk_farm
  
def _generate_clin_subclin (data_path, data_frame):
    # Add an extra column which is set to 1 when a sub-clinical mastitis case is identified
    sub_clin_conditions = [(data_frame['1_scc'] >= 150) & (data_frame['parity'] == 1),
                           (data_frame['1_scc'] <  150) & (data_frame['parity'] == 1),
                           (data_frame['1_scc'] >= 250) & (data_frame['parity'] > 1),
                           (data_frame['1_scc'] <  250) & (data_frame['parity'] > 1)]
    sub_clin_choices = [1,0,1,0]
    
    data_frame.loc[:, 'sub_clin'] = np.select(sub_clin_conditions, sub_clin_choices)


    db_bact = pd.read_csv(data_path / 'db_qtrbact.csv', encoding = 'unicode escape')
    db_bact['id'] = db_bact['tb_num']
    db_bact['mast_date'] = pd.to_datetime(db_bact['mast_date'], format = '%d/%m/%Y')
    db_bact['calving'] = pd.to_datetime(db_bact['calving'], format = '%d/%m/%Y')
    db_bact = db_bact[db_bact['type_bugs1'] != 'negative ']
    db_bact = db_bact.groupby(['id', 'parity', 'mast_date']).first().reset_index()
    db_bact['days_since_calving'] = (db_bact['mast_date'] - db_bact['calving']).dt.days
    columns = ['id', 'parity','days_since_calving', 'mast_date']
    db_bact = db_bact.loc[:, columns]
    
    
    combined = pd.merge(data_frame, db_bact, on = ['id', 'parity', 'days_since_calving'], how = 'left')
    combined['mast_date'] = (combined['mast_date']).dt.dayofyear
    cond = [combined['mast_date'] >= 0]
    choices = [1]
    combined['clin'] = np.select(cond, choices, default = 0)
    
    return combined


def _generate_basic_features(data_path, data_frame):
    '''
    Basic Features:
        id
        parity
        date
        days since calving
        Fat, Protein, Lactose (% and at am and pm)
        Urea (am and pm)
        SCC (am and pm)
        Yield (am and pm)
        Max flow (am and pm)
        Weight, BCS (am and pm)
    '''
    
    # Prepare the feature frame by duplicating the keys for it
    features = data_frame[['id', 'parity', 'date', 'calving', 'farm', 'days_since_calving']].copy(deep = True)

    # Generate features for AM/PM readings
    for milking_time in [1,2]:
        # Extract and rename 
        keep = ['id', 'parity', 'date', 'fat', 'protein',
            'lactose', 'urea', 'scc', 'yield', 'max_flow', 'weight', 'BCS']
        subset = data_frame[data_frame['milking_time'] == milking_time].copy(deep = True)
        subset.drop(set(subset.columns) - set(keep), axis = 1, inplace = True)
        subset.rename(columns={"fat":      f"{milking_time}_fat",     "yield":   f"{milking_time}_yield",
                               "protein":  f"{milking_time}_protein", "weight":  f"{milking_time}_weight",
                               "BCS":      f"{milking_time}_BCS",     "scc":     f"{milking_time}_scc",
                               "urea":     f"{milking_time}_urea",    "lactose": f"{milking_time}_lactose",
                               'max_flow': f'{milking_time}_max_flow' }, inplace=True)
        
        # Take the absolute values of urea
        subset.loc[:, f"{milking_time}_urea"] = subset.loc[:, f"{milking_time}_urea"].abs()
        
        # Merge the features
        features = pd.merge(features, subset, on = ['id', 'parity', 'date'], how = 'outer')

    # Fill na to replace missing values. Consider the first recorded day value to be the value for the next
    # days, unless we get the next value.
    for column in ['1_yield', '2_yield', '1_max_flow', '2_max_flow', 
    '1_fat', '2_fat', '1_weight',  '2_weight', '1_BCS', '2_BCS', 
    '1_lactose', '2_lactose', '1_protein', '2_protein', '1_urea', '2_urea', '1_scc', '2_scc']:
        features[column].fillna(method = 'ffill', inplace = True)

    # Remove duplicated rows, due to outer join
    features.drop_duplicates(inplace = True)

    # Drop records with no yield information (AM or PM)
    features.dropna(subset = ['1_yield', '2_yield'], how = 'any', inplace = True)
    
    return features

def _generate_time_series_features (data_path, data_frame):
    '''
    Extract time series information. Summary statistics for the different attributes (mentioned below), over 15 and 30 days interval.
    This wil return only the summary statistics of the attributes along with id, parity, date attributes.

    15 and 30 day intervals
    Calculated for SCC, Yield, Fat, Protein, Lactose, Urea, Weight, BCS. To reduce the chances that 15 and 30 features are the same in a lot of cases 

    * Skewness
    * Delta (difference from todays observation to x days back or the closest to that point)
    * Mean
    * Median
    * Standard Deviation
    * Minimum
    * Maximum
    '''
    # Prepare the feature frame by duplicating the keys for it
    features = data_frame[['id', 'parity', 'date']].copy(deep=True)
    
    # Create time series features
    for grp_id, grp_data in tqdm(data_frame.groupby(['id', 'parity'])):
        cow_id, parity = grp_id
        # Set the column indexer
        col = (features.id == cow_id) & (features.parity == parity)
        for item in ["1_yield"  , "2_yield"  , "1_max_flow", "2_max_flow","1_fat", "2_fat",
                     "1_lactose", "2_lactose", "1_protein" , "2_protein", "1_urea", "2_urea", 
                     "2_BCS"    , "2_weight" , "1_scc"]:
            for time in [15, 30]:
                # Rolling values on "date" field
                values = grp_data[["date", item]].rolling(window = time, min_periods = 0, on = "date");
                # Compute the max, min, mean, std, skew, median 
                features.loc[col, f'max_{item}_last_{time}_days']   = values.max().loc[:,item];
                features.loc[col, f'min_{item}_last_{time}_days']   = values.min().loc[:,item];
                features.loc[col, f'mean_{item}_last_{time}_days']  = values.mean().loc[:,item];
                features.loc[col, f'std_{item}_last_{time}_days']   = values.std().loc[:,item];
                features.loc[col, f'skew_{item}_last_{time}_days']  = values.skew().loc[:,item];
                features.loc[col, f'med_{item}_last_{time}_days']   = values.median().loc[:,item];
                # Compute the delta between last value and first
                features.loc[col, f'delta_{item}_last_{time}_days'] = values.apply (lambda x: x.iloc[-1] - x.iloc[0], raw = False).loc[:,item];

    return features


def _generate_genetic_features(data_path, data_frame):
    '''
        Genetic Merit data:

            Calculated using the mean value for all parities up to that lactation number for each cow
            I.e, if a cow was first lactation PTA = PTApar1
            If a cow was second lactation PTA = (PTApar2 + PTApar1)/2
            If a cow was third lactation PTA = (PTApar3 + PTApar2 + PTApar1)/3
            This was calculated for each lactation cycle, there were only three PTA values and thus after third lactation it stays the same.
    '''
    #Getting and fixing the tags data
    features = data_frame[['id', 'parity', 'date']].copy(deep=True)
    tags_data = pd.read_csv(data_path / 'tags-1.csv',encoding = 'utf-8')
    tags = tags_data.replace('.', np.NaN)
    tags.loc[:,'techid'] = tags['TECHID']
    columns = ['ANIMAL_TAG', 'techid']
    tags = tags.loc[:,columns]
    tags.loc[:,'techid'] = pd.to_numeric(tags['techid'], downcast = 'integer')
    
    #Getting and fixing the Genetic Merit Data
    PTA = pd.read_csv(data_path / 'PTA_REL_computed_db.txt', sep = " ")
    PTA_id = pd.merge(tags, PTA, on = ['techid'], how = 'inner')
    PTA_id.loc[:,'id'] = PTA_id['ANIMAL_TAG']
    columns = ['id','PTA', 'PTApar1', 'PTApar2', 'PTApar3']
    PTA_cows = PTA_id.loc[:,columns]
    
    #Merging it with the basic data
    genetic_merit = pd.merge(data_frame, PTA_cows, on = ['id'], how = 'left')
    
    #Iterating over each Parity setting
    Parity_1 = genetic_merit[genetic_merit['parity'] == 1]
    Parity_greater_1 = genetic_merit[genetic_merit['parity'] > 1]
    Parity_equal_2 = Parity_greater_1[Parity_greater_1['parity'] == 2]
    Parity_greater_2 = genetic_merit[genetic_merit['parity'] > 2]
    
    Parity_1.loc[:,'PTA_mean'] = Parity_1.loc[:,['PTApar1']]
    Parity_equal_2.loc[:,'PTA_mean'] =  (Parity_equal_2['PTApar2'] + Parity_equal_2['PTApar1']) / 2
    Parity_greater_2.loc[:,'PTA_mean'] = (Parity_greater_2['PTApar2'] + Parity_greater_2['PTApar1'] + Parity_greater_2['PTApar3']) / 3
    Genetic_merit = Parity_1.append(Parity_equal_2)
    Genetic_merit = Genetic_merit.append(Parity_greater_2)
    features.loc[:,'PTA_mean'] = Genetic_merit[['PTA_mean']]
    features.loc[:,'PTA_mean'] = features['PTA_mean'].replace(np.NaN, 0)
    
    return features



# NOTE: Let's not use this right now.
def _generate_farm_year_features(data_path, data_frame):
    '''
    * Creating the columns to showcase how different a cow is on a particular day for that farma dn year compared to the mean value for that farm and year
    * Calculated on:
        -Yield
        -Max flow
        -SCC
        -Fat, Protein, Lactose %
        -Urea
    '''
    # Prepare the feature frame by duplicating the keys for it
    data_frame['calving'] = pd.to_datetime(data_frame['calving'], format = '%Y-%m-%d')
    data_frame['date'] = pd.to_datetime(data_frame['date'], format = '%Y-%m-%d')
    
    data_frame['year'] = data_frame['calving'].dt.year
    data_frame['day_of_year'] = data_frame['date'].dt.dayofyear
    
    features = data_frame[['id', 'parity','date']].copy(deep=True)

    for item in ['yield', 'fat', 'protein', 'lactose', 'scc', 'max_flow']:
        if item != 'scc': 
            for time in [1,2]:
                data_frame[f'mean_{time}_{item}_per_day_in_year'] = data_frame.groupby(['farm','parity','year', 'day_of_year'])[f'{time}_{item}'].transform('mean')

                features.loc[:,f'{time}_{item}_diff_per_day_in_year'] = data_frame[f'{time}_{item}'] - data_frame[f'mean_{time}_{item}_per_day_in_year']
        else:
            data_frame[f'mean_1_{item}_per_day_in_year'] = data_frame.groupby(['farm','parity','year', 'day_of_year'])[f'1_{item}'].transform('mean')
            features.loc[:,f'1_{item}_diff_per_day_in_year'] = data_frame[f'1_{item}'] - data_frame[f'mean_1_{item}_per_day_in_year']
        
    return features


def _generate_infection_features(data_path, data_frame):
    '''
    Extract infection features
    * Time since last infection "time_healthy_sub_clin" (calculated each day to check time since last infection in days)
    * Time till next infection "time_till_inf_sub_clin", "time_till_inf_clin" (calculated each day to check when the next infection will happen)
        - This is computed for both sub-clinical and clinical mastitis. This is essential variable to align the future
          infection target.
    * Infections since calving (calculated each day for each lactation cycle independently)
    

    Input frame must contain: ['id', 'parity', 'date']
    '''
    # Prepare the feature frame by duplicating the keys for it
    features = data_frame[['id', 'parity', 'date', 'calving', 'sub_clin', 'clin']].copy(deep=True)

    # Create a column that counts the number of day until sick
    for grp_id, grp_data in tqdm(data_frame[['id', 'parity', 'days_since_calving', 'sub_clin', 'clin']].groupby(['id', 'parity'])):
        
        time_healthy_sub_clin = []
        days_sub_clin = 0
        infection = False
        infect_per_parity_sub_clin = 0
        infections_per_parity_sub_clin = []
        for row in grp_data[['days_since_calving', 'sub_clin', 'clin']].itertuples():
            #Changed it so that the rows are updated to better show the time since infection as amount of days since last infection instead of just amount of rows as some days arent recorded
            if row.sub_clin == 1:
                # If infection reset the day counter to 0
                time_healthy_sub_clin.append(0)
                days_sub_clin = row.days_since_calving
                infect_per_parity_sub_clin += 1
                infections_per_parity_sub_clin.append(infect_per_parity_sub_clin)
                infection = True
            elif infection == True:
                time_healthy_sub_clin.append(row.days_since_calving - days_sub_clin)
                infections_per_parity_sub_clin.append(infect_per_parity_sub_clin)
            else:
                time_healthy_sub_clin.append(np.nan)
                infections_per_parity_sub_clin.append(infect_per_parity_sub_clin)
        # Update the data frame with the new column
        cow_id, parity = grp_id
        features.loc[(data_frame.id == cow_id) & (data_frame.parity == parity), 'time_healthy_sub_clin'] = time_healthy_sub_clin
        features.loc[(data_frame.id == cow_id) & (data_frame.parity == parity), 'infections_per_parity_sub_clin'] = infections_per_parity_sub_clin

        # Iterate over time_healthy_sub_clin to compute the days until an infection
        time_till_inf_sub_clin = []
        days_sub_clin = 0
        infection_sub_clin = False
        
        for row in grp_data[['days_since_calving', 'sub_clin', 'clin']][::-1].itertuples():
            #Changed it so that the rows are updated to better show the time till infection as amount of days since last infection instead of just amount of rows as some days arent recorded
            if row.sub_clin == 1:
                # If infection reset the day counter to 0
                time_till_inf_sub_clin.append(0)
                days_sub_clin = row.days_since_calving
                infection_sub_clin = True
            elif infection_sub_clin == True:
                time_till_inf_sub_clin.append(days_sub_clin - row.days_since_calving)
            else:
                time_till_inf_sub_clin.append(np.nan)
        time_till_inf_sub_clin = time_till_inf_sub_clin[::-1]
        cow_id, parity = grp_id
        features.loc[(data_frame.id == cow_id) & (data_frame.parity == parity), 'time_till_inf_sub_clin'] = time_till_inf_sub_clin
        
        # Iterate over time_healthy_sub_clin to compute the days until an infection
        time_till_inf_clin = []
        days_clin = 0
        infection_clin = False
      
        for row in grp_data[['days_since_calving', 'sub_clin', 'clin']][::-1].itertuples():
            #Changed it so that the rows are updated to better show the time till infection as amount of days since last infection instead of just amount of rows as some days arent recorded
            if row.clin == 1:
                # If infection reset the day counter to 0
                time_till_inf_clin.append(0)
                days_clin = row.days_since_calving
                infection_clin = True
            elif infection_clin == True:
                time_till_inf_clin.append(days_clin - row.days_since_calving)
            else:
                time_till_inf_clin.append(np.nan)
        time_till_inf_clin = time_till_inf_clin[::-1]
        cow_id, parity = grp_id
        features.loc[(data_frame.id == cow_id) & (data_frame.parity == parity), 'time_till_inf_clin'] = time_till_inf_clin
          
        
    for grp_id, grp_data in tqdm(data_frame[['id', 'parity', 'days_since_calving', 'sub_clin', 'clin']].groupby(['id'])):
        infections_per_cow_sub_clin = []
        infections_sub_clin = 0
        infection_occured_sub_clin = False
        infections = 0
        for row in grp_data[['days_since_calving', 'sub_clin', 'clin']].itertuples():

            #Changed it so that the rows are updated to better show the time till infection as amount of days since last infection instead of just amount of rows as some days arent recorded
            if row.sub_clin == 1:
                # If infection reset the day counter to 0
                infections_sub_clin += 1
                infections_per_cow_sub_clin.append(infections_sub_clin)
                infection_occured_sub_clin = True
            elif infection_occured_sub_clin == True:
                infections_per_cow_sub_clin.append(infections_sub_clin)
            else:
                infections_per_cow_sub_clin.append(infections_sub_clin)
        cow_id = grp_id
        features.loc[(data_frame.id == cow_id), 'infections_per_cow_sub_clin'] = infections_per_cow_sub_clin
        
    return features
  
        
# NOTE: Old _generate_infection_features, Ignore this.
#def _generate_infection_features(data_path, data_frame):
    #'''
    #Extract infection features

    #* Infections since calving (calculated each day for each lactation cycle independently)
    #* Infections since first time milked (calculated each day for each cow seperatly)
    #* Infections since first calving per farm (calculated each day to see how many infections on a farm since the first calving for that year)
    #* Proportion of cows infected (calculated each day to check the overall proportion of cows that 
    #have had at least one subclinical infection since first calving for that year on that farm)

    #Input frame must contain: ['id', 'parity', 'date', 'calving', 'sub_clin', 'days_since_calving']
    #'''
    ## Prepare the feature frame by duplicating the keys for it
    #features = data_frame[['id', 'parity', 'date']].copy(deep=True)
        
    ## Create a column that counts the number of day until sick
    #for grp_id, grp_data in tqdm(data_frame[['id', 'parity', 'days_since_calving', 'sub_clin']].groupby(['id', 'parity'])):
        ## Iterate over all the data
        #time_healthy_sub_clin = []
        #for row in grp_data[['days_since_calving', 'sub_clin']].itertuples():
            #if row.sub_clin == 0:
                ## If no infection use days_since_calving for first entry, otherwise last entry + 1
                #time_healthy_sub_clin.append(row.days_since_calving if len(time_healthy_sub_clin) == 0 else time_healthy_sub_clin[-1] + 1)
            #else:
                ## If infection reset the day counter to 0
                #time_healthy_sub_clin.append(0)
        ## Update the data frame with the new column
        #cow_id, parity = grp_id
        #features.loc[(data_frame.id == cow_id) & (data_frame.parity == parity), 'time_healthy_sub_clin'] = time_healthy_sub_clin

        ## Iterate over time_healthy_sub_clin to compute the days until an infection
        #time_till_inf_sub_clin = []
        #has_been_sick = False
        #for time_value in time_healthy_sub_clin[::-1]:
            #if time_value != 0:
                #time_till_inf_sub_clin.append(time_till_inf_sub_clin[-1] + 1 if has_been_sick else np.nan)
            #else:
                #has_been_sick = True
                #time_till_inf_sub_clin.append(0)
        #features.loc[(data_frame.id == cow_id) & (data_frame.parity == parity), 'time_till_inf_sub_clin'] = time_till_inf_sub_clin[::-1]
        
    #return features
  

def _generate_target_variables (data):
  # Generate sub-clinical labels. If "time_till_inf_sub_clin" is <= 7 days, that means we label that row as having an infection
  # to be occurring in the next seven days for subclinical. The variable "time_till_inf_sub_clin" is generated by function "_generate_infection_features"
  cond = [data['time_till_inf_sub_clin'] <= 7, data['time_till_inf_sub_clin'] > 7]
  choices = [1,0]
  data['early_detect_sub_clinical'] = np.select(cond, choices, default = 0)

  # Generate clinical labels. If "time_till_inf_clin" is <= 7 days, that means we label that row as having an infection
  # to be occurring in the next seven days for subclinical. The variable "time_till_inf_sub_clin" is generated by function "_generate_infection_features"
  cond_train = [data['time_till_inf_clin'] <= 7, data['time_till_inf_clin'] > 7]
  choices = [1,0]
  data['early_detect_clinical'] = np.select(cond_train, choices, default = 0)
  
  cond_after = [data['time_healthy_sub_clin'] <= 7, data['time_healthy_sub_clin'] > 7]
  choices_after = [1,0]
  data['unsure_sub_clin'] = np.select(cond_after, choices_after, default = 0)

  # TODO: save time_till_inf_sub_clin, time_till_inf_clin, 1_scc, sub_clin as the names, *_0days, in the target

  return data[['early_detect_sub_clinical', 'early_detect_clinical', 'unsure_sub_clin']];


def _generate_infrequent_vars_mask (data, infrequent_set = ['1_scc', '1_fat', '1_protein', '1_lactose', '2_lactose', '2_protein', '2_fat'], this_day = 15):

  # This function helps generating lower simulated recording frequencies for the variables given by "infrequent_set"
  # This goes through all the rows and makes a new variable "keep_row_<varname>" for each <varname> in "infrequent_set", for example "keep_row_1_scc".
  # If the value of the corresponding row of "keep_row_1_scc" is 1, that indicates that the corresponding row's "1_scc" value is recorded.
  # If the value of the corresponding row of "keep_row_1_scc" is 0, that indicates that the corresponding row's "1_scc" value should be replaced by the last recorded "1_scc"
  # In this was the recording frequency of the data is generated.
  # "this_day" indicates the simulated recording frequency in days. this_day = 15 indicates that this funciton will generate the "keep_row_<varname>" masks in a way
  # from the given data set, such that the there is atleast 15 days gap between two recorded values.
  # Once these masks are generated, they are returned to the caller, which can then easily replace the values of "<varname>" for the rows where "keep_row_<varname> = 0", with the
  # last row "where keep_row_<varname> = 1".
  
  colnames = [];

  for item in infrequent_set:
    colnames.append (f'keep_row_{item}_{str(this_day)}');
      
  var_masks = pd.DataFrame (0, index = range (data.shape[0]), columns = colnames)
  
  # Create a column that counts the number of day until sick
  keep_row_list = {};
  keep_row_list[str(this_day)] = [];
  
  logger.info ('Infrequencies mask ' + str (this_day))
  for grp_id, grp_data in tqdm(data.groupby(['id', 'parity'])):
    # Iterate over all the data_sub
    for item in infrequent_set:
        keep_row_list[str(this_day)] = []
        infection = False
        for row in grp_data[['days_since_calving', 'id']].itertuples():
            #Changed it so that the rows are updata_subed to better show the time since infection
            #fkas amount of days since last infection instead of just amount of rows as some days arent recorded
            days = row.days_since_calving
            #print (row)
            if infection == False:
                # If infection reset the day counter to 0
                day_first = row.days_since_calving
                infection = True
                keep_row_list[str(this_day)].append(1)
                amount = 0
            elif infection == True and days > (day_first + this_day):
                keep_row_list[str(this_day)].append(1)
                day_first = row.days_since_calving
            else:
                keep_row_list[str(this_day)].append(0)
        cow_id, parity = grp_id
        var_masks.loc[(data.id == cow_id) & (data.parity == parity), f'keep_row_{item}_{str(this_day)}'] = keep_row_list[str(this_day)]
        #data.loc[(data.id == cow_id) & (data.parity == parity), f'keep_row_{item}_{str(this_day)}'] = keep_row_list[str(this_day)]
  return var_masks;


def _generate_infrequent_vars_data (data, var_masks, infrequent_set =  ['1_scc', '1_fat', '1_protein', '1_lactose', '2_lactose', '2_protein', '2_fat'], this_day = 15):

  # This function generates the actual dataset which simulates the given data recording frequency indicated by "this_day".
  # Call the function "_generate_infrequent_vars_mask" before calling this function>
  # "infrequent_set" is the variables of which the recording frequencies are to be simulated/decreased
  # "this_day" is the target frequencies. This value must be identical to what it was passed to "_generate_infrequent_vars_mask"

  # For each attribute in "infrequent_set" where the mask "'keep_row_<variable>" is 0, replace it with NaN
  for item in infrequent_set:
      data[f'{item}'] = data[f'{item}'].mask (var_masks[f'keep_row_{item}_{str(this_day)}'] == 0)
  
  # For each attribute in "infrequent_set" replace the last non nan values for each (id, parity) group with the next nan values.
  # This effectively copies the last recorded value to the next rows, until the next simulated recording (indicated by "keep_row_<variable> = 1, and a non nan <variable> value).
  for column in infrequent_set:
        data[column] = data.groupby(['id', 'parity'])[column].apply(lambda x: x.fillna(method = 'ffill'))
        
  return data;


# This function will prepare and generate the dataset from the database files based on the rules and the given frequencies.
#Files needed in the "data_path" variable in "prepare_dataset" function in "prepare_data.py" function, and the fields required in each function.

#db_master.csv: csv
#cow,tbnum,dob,farm,treatment,subtreatment,year,trtdescription, parity,calvingdate, dryoffdate, exitdate, exittype, exitreason


#db_mastitis.csv: csv
#tb_num,calving,parity,diagnosis_date,serverity,rf,rh,lf,lh,lost_quarter,drug,clinical,sub_clinical,clin_sub,tubes


#db_milk.csv: csv
#tb_num,calving,parity,milk_date,milking_time,fat,protein,lactose,casein,ffa,ts,urea,scc,yield,time,max_flow,conc_fed,row,side,unit


#db_qtrbact.csv: csv
#tb_num,calving,parity,mast_date,milking_time,qtr,num_of_bugs1,type_bugs1


#db_weights.csv: csv
#tbnum,calving,parity,date,weight,BCS

#PTA_REL_computed_db.txt: blank separated
#"ani" "techid" "ndesc" "nrec" "PTApar1" "PTApar2" "PTApar3" "PTA" "relSCC"


def prepare_dataset (data_path, infrequent_set = ['1_scc', '1_fat', '1_protein', '1_lactose', '2_lactose', '2_protein', '2_fat'], less_freq_days_number = [15, 30, 45, 60], reprocess = False):
    '''
    This function loads the source data and generate all the features from it
    The attribute "reprocess" is to reuse intermediate files stored while the data generation process instead of recomputing.
    If "reprocess" is True, then the process will recompute everything and overwrite data files.
    If "reprocess" is False, then first the process will look if an intermediate computed data file is available, if yes, then it will use it instead of reprocessing it.
    '''
    data_path = Path (data_path);
    
    if (reprocess == True):
      logger.info('All datasets will be regenerated and existing ones overwritten')
    else:
      logger.info('If an intermediate dataset is present on disk, it will be used, otherwise it will be generated')
    
    # Load all the source data
    logger.info('Load the source data for Weights, Milk, Farm')
    weights_milk_farm = _load_weights_milk_farm(data_path)

    if (not os.path.isfile (data_path / 'basic_features.csv.gz')) or (reprocess == True):
      # Extract basic features
      logger.info('Extract basic features')
      basic_features = _generate_basic_features(data_path, weights_milk_farm)
      basic_features.to_csv(data_path / 'basic_features.csv.gz', encoding = 'utf-8', index = False, compression='gzip')
    else:
      logger.info('Loading basic features')
      basic_features = pd.read_csv(data_path / 'basic_features.csv.gz')
      basic_features.date = pd.to_datetime(basic_features.date)
    
    for this_day in less_freq_days_number:
      
      # In the case of 0 day, we do not need to do anything
      if this_day > 0:
        # First simulate the infrequencies
        if (not os.path.isfile (data_path / f'infrequent_var_masks_{str (this_day)}.csv.gz')) or (reprocess == True):
          # Generate the rows which we need to keep and which we need to simulate "no data recorded"
          # This only will generate the masks
          logger.info('Simulate infrequent data recording masks')
          infrequent_var_masks = _generate_infrequent_vars_mask (basic_features, infrequent_set = infrequent_set, this_day = this_day)
          infrequent_var_masks.to_csv(data_path / f'infrequent_var_masks_{str (this_day)}.csv.gz', encoding = 'utf-8', index = False, compression = 'gzip')
        else:
          logger.info('Loading infrequent data recording masks')
          infrequent_var_masks = pd.read_csv (data_path / f'infrequent_var_masks_{str (this_day)}.csv.gz')
      
        if (not os.path.isfile (data_path / f'basic_features_{str(this_day)}.csv.gz')) or (reprocess == True):
          logger.info('Generate infrequent recording for ' + str (this_day) + ' days')
          # We need the original basic features file to do simulate the infrequent recordings
          basic_features = pd.read_csv (data_path / 'basic_features.csv.gz', encoding = 'utf-8', compression = 'gzip');
          basic_features.date = pd.to_datetime (basic_features.date);
          
          basic_features = _generate_infrequent_vars_data (basic_features, infrequent_var_masks, infrequent_set, this_day)
          basic_features.to_csv(data_path / f'basic_features_{str(this_day)}.csv.gz', encoding = 'utf-8', index = False, compression = 'gzip')
        else:
          logger.info('Loading infrequent data recording for ' + str (this_day) + ' days')
          basic_features = pd.read_csv (data_path / f'basic_features_{str(this_day)}.csv.gz')
          basic_features.date = pd.to_datetime(basic_features.date)
      else:
        logger.info('Skipping simulatiton of infrequent data, preserving original data for ' + str (this_day) + ' days frequency');
          
      
      # Add the clin and sub_clin fields, which is needed by the infection features function.
      # As sub_clin depends on scc, so it is important to first introduce the frequencies
      save_file_name = f'basic_features_clin_subclin.csv.gz' if this_day == 0 else f'basic_features_clin_subclin_{str(this_day)}.csv.gz'
      if (not os.path.isfile (data_path / save_file_name)) or (reprocess == True):
        logger.info('Generate \'clin\' and \'sub_clin\' field for ' + str (this_day) + ' days')
        basic_features = _generate_clin_subclin (data_path, basic_features)
        basic_features.to_csv(data_path / save_file_name, encoding = 'utf-8', index = False, compression = 'gzip')
      else:
        logger.info('Loading \'clin\' and \'sub_clin\' field for ' + str (this_day) + ' days')
        basic_features = pd.read_csv (data_path / save_file_name);
        basic_features.date = pd.to_datetime(basic_features.date);

      # Then do further processing after simulating the infrequencies, work on 'basic_features' datasets
      # Extract time-series features
      save_file_name = f'time_series_features.csv.gz' if this_day == 0 else f'time_series_features_{str(this_day)}.csv.gz'
      if (not os.path.isfile (data_path / save_file_name)) or (reprocess == True):
        logger.info ("Sorting basic_features based on date");
        # NOTE: This is important as we do a rolling window based on "date" field
        basic_features = basic_features.sort_values (by = "date");
        logger.info ("Extract time-series features for " + str (this_day) + ' days')
        time_series_features = _generate_time_series_features(data_path, basic_features)
        time_series_features.to_csv(data_path / save_file_name, encoding = 'utf-8', index = False, compression = 'gzip')
      else:
        logger.info ('Loading time-series features for ' + str (this_day) + ' days')
        time_series_features = pd.read_csv(data_path / save_file_name)
        time_series_features.date = pd.to_datetime(time_series_features.date)
        
        
      # NOTE: Excluding these features
      ## Extract farm year features
      #save_file_name = f'farm_year_features.csv.gz' if this_day == 0 else f'farm_year_features_{str(this_day)}.csv.gz'
      #if (not os.path.isfile (data_path / save_file_name)) or (reprocess == True):
        #logger.info('Extract farm year features for ' + str (this_day) + ' days')
        #farm_year_features = _generate_farm_year_features (data_path, basic_features)
        #farm_year_features.to_csv (data_path / save_file_name, encoding = 'utf-8', index = False, compression = 'gzip')
      #else:
        #logger.info('Loading farm year features for ' + str (this_day) + ' days')
        #time_series_features = pd.read_csv(data_path / save_file_name)


      # Extract infection features
      save_file_name = f'infection_features.csv.gz' if this_day == 0 else f'infection_features_{str(this_day)}.csv.gz'
      if (not os.path.isfile (data_path / save_file_name)) or (reprocess == True):
        logger.info('Extract infection features for ' + str (this_day) + ' days')
        infection_features = _generate_infection_features(data_path, basic_features)
        infection_features.to_csv(data_path / save_file_name, encoding = 'utf-8', index = False, compression = 'gzip')
      else:
        logger.info('Loading infection features')
        infection_features = pd.read_csv(data_path / save_file_name)
        infection_features.date = pd.to_datetime(infection_features.date)
      
    
    
    # Create genetic features. This does not depend on the synthetic frequencies
    save_file_name = f'genetic_features.csv.gz'
    if (not os.path.isfile (data_path / save_file_name)) or (reprocess == True):
      logger.info('Extract genetic features for ' + str (this_day) + ' days')
      genetic_features = _generate_genetic_features(data_path, basic_features)
      genetic_features.to_csv(data_path / save_file_name, encoding = 'utf-8', index = False, compression = 'gzip')
    else:
      logger.info('Loading genetic features')
      genetic_features = pd.read_csv(data_path / save_file_name)
      genetic_features.date = pd.to_datetime (genetic_features.date)
      

    # Here we will remerge the read datasets, doesnt matter on reprocess = True
    for this_day in less_freq_days_number:
      load_file_name_basic = f'basic_features.csv.gz' if this_day == 0 else f'basic_features_{str(this_day)}.csv.gz'
      load_file_name_time_series = f'time_series_features.csv.gz' if this_day == 0 else f'time_series_features_{str(this_day)}.csv.gz'
      load_file_name_infection_features = f'infection_features.csv.gz' if this_day == 0 else f'infection_features_{str(this_day)}.csv.gz'
      load_file_name_genetic_features = f'genetic_features.csv.gz'
      save_file_name = f'all_features.csv.gz' if this_day == 0 else f'all_features_{str(this_day)}.csv.gz'
      
      basic_features       = pd.read_csv(data_path / load_file_name_basic)
      time_series_features = pd.read_csv(data_path / load_file_name_time_series)
      infection_features   = pd.read_csv (data_path / load_file_name_infection_features)
      genetic_features     = pd.read_csv (data_path / load_file_name_genetic_features)
      
      basic_features.date       = pd.to_datetime (basic_features.date)
      time_series_features.date = pd.to_datetime (time_series_features.date)
      infection_features.date   = pd.to_datetime (infection_features.date)
      genetic_features.date     = pd.to_datetime (genetic_features.date)
      
      # Combine all the features
      logger.info('Combine and save')
      total = basic_features
      total = pd.merge(total, time_series_features, how = 'left', on = ['id', 'parity', 'date']);
      total = pd.merge(total, infection_features, how = 'left', on = ['id', 'parity', 'date']);
      total = pd.merge(total, genetic_features, how = 'left', on = ['id', 'parity', 'date']);
      #total.drop_duplicates(inplace=True)
      total.to_csv(data_path / save_file_name, encoding = 'utf-8', index = False, compression = 'gzip');
    
      if this_day == 0:
        # Build the targets on the original dataset (without any infrequent sets)
        # We need to generate this once.
        logger.info('Generate target variables')
        total = pd.read_csv(data_path / f'all_features.csv.gz')
        targets = _generate_target_variables (total)
        targets.to_csv(data_path / 'targets.csv.gz', encoding = 'utf-8', index = False, compression = 'gzip')


# To generate the dataset, call like below.
if __name__ == '__main__':
    logger.info('Working on training')
    # Give path with the dataset raw files in it. In the below cases we had two sets separately, training and test.
    # To use original recording frequency, use "less_freq_days_number" as 0 in both cases.
    # To simulate different recording frequencies, use the corresponding day value, like 15.
    # Recommended to run each frequency simulated dataset separately in different systems (with low memory) to avoid crashes.
    # Also, it is recommended that to run the "0" day (original recording frequency, dataset first, as the other ones (15, 30, etc. days). This also generates the ground truth.
    # attempts to use the intermediate results from by the original recording frequency dataset generation to make the process faster.
    prepare_dataset(Path('./data/training'), less_freq_days_number = [0])
    logger.info('Working on testing')
    prepare_dataset(Path('./data/testing'), less_freq_days_number = [0])
    
