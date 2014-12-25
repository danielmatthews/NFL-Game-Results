# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 03:16:58 2014

@author: danielmatthews
"""

import pandas as pd
import numpy as np

#Read in nfl_stats.csv
raw = pd.read_csv('nfl_stats.csv')
raw

#Drop unnecessary columns
raw.drop(raw.columns[[0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 21, 22, 23]], axis=1, inplace=True)

#Rename columns for analysis
raw.rename(columns={'Unnamed: 4':'win', '1stD':'dn', 'TotYd':'TY', '1stD.1':'dn_ald', 'TotYd.1':'TY_ald' , 'PassY.1':'PassY_ald','RushY.1':'RushY_ald','TO.1':'TO_df'} , inplace=True)


#Change Wins and Losses to 1, 0
raw.win = np.where(raw.win == 'W', 1, 0)

#Find NaN
pd.isnull(raw).any(1).nonzero()[0]

#Nan -> 0
raw.TO.fillna(0, inplace=True)
raw.TO_df.fillna(0, inplace=True)
raw.PassY_ald.fillna(0, inplace=True)
raw.PassY.fillna(0, inplace=True)
raw.drop(raw.index[[2544, 2545]], inplace=True)

#Write to CSV
raw.to_csv('nflResults.csv', index=False)