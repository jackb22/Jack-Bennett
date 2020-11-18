#importing Data 
#%%
import pandas as pd 
import time 
pbp_df = pd.read_csv('/Users/jackbennett/Desktop/Projects /Learning/NFL Play by Play 2009-2018 (v5).csv', iterator=True, low_memory=False)

#timing for fun 
start =time.time()

df = pd.DataFrame()
for chunk in pbp_df:
    df = pd.concat([df, chunk])

print('Time Elapsed:', time.time()-start)

#%% 
for column in df.columns:
    #looking for relevant rushing columns 
    if 'rush' in column:
        print(column)
    elif 'distance' in column:
        print(column)
    elif 'yardline' in column:
        print(column)
#%%
# only keeping relevant columns 
rushing_df = df[['rush_attempt', 'rush_touchdown', 'yardline_100', 'two_point_attempt']]

rushing_df = rushing_df.loc[(rushing_df['two_point_attempt'] == 0) & (rushing_df['rush_attempt'] == 1)]

rushing_df.shape
#%%
rushing_df.sort_values(by='yardline_100').head(15)
# %%
"""
Here, we are grouping by the yardline from where the play began, and then using value counts
to count the number of times a rushing play was a touchdown (either a 0 or a 1), we can set the
argument normalize = True to be able to calculate the proportion of plays that were touchdowns, instead of the
count.
"""
rushing_df_probs = rushing_df.groupby('yardline_100')['rush_touchdown'].value_counts(normalize=True)

# this gives us back a Series, let's turn it back in to a DataFrame we can use.

rushing_df_probs = pd.DataFrame({
    'probability_of_touchdown': rushing_df_probs.values
}, index=rushing_df_probs.index).reset_index()

rushing_df_probs.head()
# %%
# only keeping rush_touchdown = 1
rushing_df_probs = rushing_df_probs.loc[rushing_df_probs['rush_touchdown'] == 1]
# let's drop the rush_touchdown as well, since that's also redundant.
rushing_df_probs = rushing_df_probs.drop('rush_touchdown', axis=1)
rushing_df_probs.head(15)
# %%
import seaborn as sns; sns.set_style('darkgrid')

rushing_df_probs.plot(x='yardline_100', y='probability_of_touchdown')
# %%
#working with 2019 PBP_Data

PBP_2019_BASE_URL = "https://raw.githubusercontent.com/fantasydatapros/data/master/2019pbp.csv"

pbp_2019_df = pd.read_csv(PBP_2019_BASE_URL, 
                          iterator=True, 
                          low_memory=False, 
                          chunksize=10000,
                          index_col=0)
# %%
pbp_2019_df_final = pd.DataFrame()

for chunk in pbp_2019_df:
    pbp_2019_df_final = pd.concat([pbp_2019_df_final, chunk])
# %%
pbp_2019_df_final.columns
# %%
#appling function later 
def fix_yardline(row):
    yardline = row['YardLineFixed']
    direction = row['YardLineDirection']
    
    if direction == 'OPP':
        return yardline
    else:
        return 100 - yardline

#filtering out the columns we need
pbp_2019_df_final = pbp_2019_df_final[['RushingPlayer', 'OffenseTeam', 'YardLineFixed', 'YardLineDirection']]

#dropping na values to remove rows with RushingPlayer with na. These are non-rushing plays.
pbp_2019_df_final = pbp_2019_df_final.dropna()

# %%
print(pbp_2019_df_final.shape)
print(pbp_2019_df_final.head)
# %%
# now we apply the function we wrote above.
pbp_2019_df_final['yardline_100'] = pbp_2019_df_final.apply(fix_yardline, axis=1)

# rename and drop columns before merging
pbp_2019_df_final = pbp_2019_df_final.rename({
    'RushingPlayer': 'Player',
    'OffenseTeam': 'Tm'
}, axis=1).drop(['YardLineDirection', 'YardLineFixed'], axis=1)

pbp_2019_df_final.head()
# %%
df = pbp_2019_df_final.merge(rushing_df_probs, how='left', on='yardline_100')
df.head()
# %%
import numpy as np

"""
Now, we are grouping by Player and Tm and summing up the results for probability_of_touchdown.
This is our Expected Touchdowns value, and so we rename the column on the same line.
We use agg here to only run the aggregation function on the columns we specify while also
keeping the original DataFrame intact.
More information on pandas.DataFrame.agg
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.agg.html
"""

df = df.groupby(['Player', 'Tm'], as_index=False).agg({
    'probability_of_touchdown': np.sum
}).rename({'probability_of_touchdown': 'Expected Touchdowns'}, axis=1)

df = df.sort_values(by='Expected Touchdowns', ascending=False)
df.head(15)
# %%
df['Expected Touchdowns Rank'] = df['Expected Touchdowns'].rank(ascending=False)
df.head(15)
# %%
YEARLY_BASE_URL = 'https://raw.githubusercontent.com/fantasydatapros/data/master/yearly/2019.csv'

# there is one odd column there which is why we have to use .iloc[:, 1:]. 
stats_df = pd.read_csv(YEARLY_BASE_URL).iloc[:, 1:][['Player', 'Tm','Pos', 'RushingTD']] # these are the only columsn we'll need

stats_df.head()
# %%
#fixing Abbreivaitoins 
print('Differing Team Names:', list(set(stats_df['Tm'].unique()) - set(df['Tm'].unique())))
print('\n')
print('stats_df Team Names:', stats_df['Tm'].unique().tolist())
print('\n')
# %%
# let's replace the differening team names using the replace method
# to find out more about the replace method,
# visit the link below
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html

team_name_map = {
    'TAM': 'TB',
    'KAN': 'KC',
    'LAR': 'LA',
    'NOR': 'NO',
    'GNB': 'GB',
    'NWE': 'NE',
    'SFO': 'SF'
}

stats_df = stats_df.replace({
    'Tm': team_name_map
})
# %%
def fix_player_names(name):
    
    name_split = name.split() # split the name on the whitespace
    
    first_initial = name_split[0][0].upper() # get the first letter of the first element of name_split, and convert it to upper if it wasn't already
    
    last_name = name_split[1].upper() # get the player's last name and convert it to all caps
    
    return '.'.join([first_initial, last_name]) # join the first inital and last name on .

stats_df['Player'] = stats_df['Player'].apply(fix_player_names)

stats_df.head()
# %%
#remove players who are not RB'S
stats_df = stats_df.loc[stats_df['Pos'] == 'RB']

# %%
# merge our tables together on the Player and Tm column
df = stats_df.merge(df, how='left', on=['Player', 'Tm']).dropna()

# drop the position column, redundant
df = df.drop('Pos', axis=1)

# renaming columns 
df = df.rename({'RushingTD': 'Actual Touchdowns'}, axis=1)

# calculate actual TD rank using the rank method
df['Actual Touchdowns Rank'] = df['Actual Touchdowns'].rank(ascending=False)

"""
Regression candidate is the difference between a player's Expected Touchdowns and Actual Touchdowns as 
shown below. More positive numbers mean the player underperformed their season, more negative numbers
means the player overperformed their season. 
"""

df['Regression Candidate'] = df['Expected Touchdowns'] - df['Actual Touchdowns']

"""
The order of taking the difference here for the Regression Rank is reversed here as a 
small actual TD rank is good, while a large actual TD rank is not.
"""

df['Regression Rank Candidate'] = df['Actual Touchdowns Rank'] - df['Expected Touchdowns Rank']

df.head()
# %%
# top negative regression candidates by looking at Expected vs. Actual
df.sort_values(by='Regression Candidate').head(15)
# %%
# top positive regression candidates by looking at Expected vs. Actual
df.sort_values(by='Regression Candidate', ascending=False).head(15)

# %%
# top negative regression candidates by looking at Expected Rank vs. Actual Rank
df.sort_values(by='Regression Rank Candidate', ascending=True).head(15)
# %%
# top positive regression candidates by looking at Expected Rank vs. Actual Rank
df.sort_values(by='Regression Rank Candidate', ascending=False).head(15)

# %%
df.loc[df['Expected Touchdowns'] > 2].sort_values(by='Regression Rank Candidate', ascending=True).head(15)
# %%
from matplotlib import pyplot as plt; sns.set_style('whitegrid');

# using the object-oriented API for greater control over our plots. Setting figsize on the same line.
fig, ax = plt.subplots(figsize=(12, 8))

# creating a new column with either True/False based on if Regression Candidate is a positive number
df['Positive Regression Candidate'] = df['Regression Candidate'] > 0

# simple scatter plot
sns.scatterplot(
    x = 'Expected Touchdowns',
    y = 'Actual Touchdowns',
    hue = 'Positive Regression Candidate',
    data = df,
    palette = ['r', 'g'] # red and green
);

max_act_touchdowns = int(df['Actual Touchdowns'].max()) # max touchdowns
max_exp_touchdowns = int(df['Expected Touchdowns'].max()) # max expected touchdowns

max_tds = max(max_act_touchdowns, max_exp_touchdowns) # max of actual and expected tds

# plotting a line with slope of 1 up to max_tds number. This is the blue line on our viz

 v
sns.lineplot(range(max_tds), range(max_tds)) 

# plotting a line with slope of 1 up to max_tds number. This is the blue line on our viz

# initialize a list of notable player's we'd like to annotate the visualization with.
notable_players = ['L.FOURNETTE', 'A.JONES', 'C.MCCAFFREY', 'R.MOSTERT']

for _, row in df.iterrows():
    if row['Player'] in notable_players: 
        """
        Check our the docs on Axes.text
        
        https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.text.html
        
        Notice we are using the ax object we instantiated above.
        """
        ax.text(
            x = row['Expected Touchdowns']+.1, # add a bit of spacing from the point in the x-direction
            y = row['Actual Touchdowns'] + 0.05, # same but in the y-direction
            s = row['Player'] # annotate with the player's name
        )
# %%
