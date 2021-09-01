# -*- coding: utf-8 -*-
"""
@author: aschu
"""
print('\nBoard Game Reviews Preprocessing & EDA') 
print('======================================================================')

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

path=r'D:\BoardGameReviews\Data'
os.chdir(path)

# Write print statement results to log file
stdoutOrigin=sys.stdout 
sys.stdout = open('boardgame_eda_log.txt', 'w')

# the original csv from https://raw.githubusercontent.com/beefsack/bgg-ranking-historicals/master/

# Games Data
games = pd.read_csv('2020-08-19.csv', index_col = 0)
print('\nGames- Dimensions')
print('\nDimensions of Games', games.shape) 
print('======================================================================')

# To match var format in games data
games.rename(index=str, columns={'Bayes average': 'Geekscore','Name':'name'},
             inplace=True)

print('\nGames- Data Types')
print(games.dtypes)

# Drop Thumbnail and URL 
games = games.drop(['Thumbnail', 'URL'], axis=1)
print('======================================================================')
print('======================================================================')

# Details Data
details = pd.read_csv('games_detailed_info.csv', index_col = 0)
print('\nGame Details - Dimensions')
print(details.shape)
print('======================================================================')

print('\nGame Details - Data Types')
print(details.dtypes)
print('======================================================================')

# To match var format in games & details data
details.rename(index=str, columns={'id': 'ID'}, inplace=True)
details = details.drop(['type', 'thumbnail', 'average', 'usersrated', 
                        'bayesaverage', 'stddev', 'median', 'Board Game Rank'],
                       axis=1)
print('======================================================================')
print('======================================================================')
      
# Reviews Data
reviews = pd.read_csv('bgg-15m-reviews.csv', index_col = 0)
print('\nReviews - Dimensions')
print(reviews.shape)
print('======================================================================')

print('\nReviews - Data Types')
print(reviews.dtypes)
print('======================================================================')

###############################################################################
#################### Create Reviews sample data set  ##########################
###############################################################################
reviews_sample = reviews.sample(n=500000)

# Write sample to csv
reviews_sample.to_csv('BGR_Reviews_sample_5e5.csv', index=False)
del reviews_sample

###############################################################################

###############################################################################
################################ Merge Datasets ###############################
###############################################################################
merge = pd.merge(games, details, on=['ID'])
del games, details

df = pd.merge(merge, reviews, on=['ID', 'name'])
del merge, reviews

print('\nMerge Data - Dimensions')
print(df.shape)
print('======================================================================')

# Examine 'comment' var for missingness 
print('The Comment Variable has ' + str(df['comment'].isna().sum() /
                                        df.shape[0] * 100) + ' % Missing')

# Subset observations without missing comments
df = df[df.comment.notnull()]
print('\nMerge Data - No Missing Comments - Dimensions')
print(df.shape)
print('======================================================================')

# Remove vars with more than 10% missing
df = df.loc[:, df.isnull().mean() < 0.10]

# Examine Variables for Missingness & Quality
print('\nMerged Data - Missing Data Information') 
def missing_data_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        var_type = df.dtypes
        unique_count = df.nunique()
        mis_val_table = pd.concat([mis_val, mis_val_percent,
                                   var_type, unique_count], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values',
                   2 : 'Data Type', 3 : 'Number Unique'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ('The selected dataframe has ' + str(df.shape[1]) + ' columns.\n'      
            'There are ' + str(mis_val_table_ren_columns.shape[0]) +
              ' columns that have missing values.')
        return mis_val_table_ren_columns

print(missing_data_table(df))
print('======================================================================')

# Drop due to high dimensionality of categorical vars
df = df.drop(['boardgameartist', 'boardgamefamily', 'boardgamemechanic', 
              'suggested_language_dependence', 'suggested_playerage', 
              'boardgamecategory', 'boardgamedesigner', 'image', 
              'description'], axis=1)
df = df.drop_duplicates()

print ('The cleaned data for missing & high dimensionality has ' +
       str(df.shape[0]) + ' rows and ' + str(df.shape[1]) + ' columns')
print('======================================================================')

print ('There are ' + str(df['comment'].nunique()) + ' unique comments') #2732276

# Close to create file
sys.stdout.close()
sys.stdout=stdoutOrigin

###############################################################################
############################## EDA of Merged Data #############################
###############################################################################
# Change path to EDA
path=r'D:\BoardGameReviews\EDA'
os.chdir(path)

# Initial automated EDA using Pandas Profiling
profile = ProfileReport(df, title="Board Game Reviews EDA")
profile.to_file(output_file='BGR_EDA_output.html')

###############################################################################
# Drop vars not relevant
df = df.drop(['ID', 'suggested_num_players', 'boardgamepublisher', 'primary',
              'user'], axis=1)

# Data preparation - Outliers
# Set cut off for years since initial is 0 and 3500
df['Year'].values[df['Year'] > 2020] = 2020
df['Year'].values[df['Year'] < 1960] = 1960

# Remove outliers in maxplayers by setting to 12
df['maxplayers'].values[df['maxplayers'] > 12] = 12

# Remove outliers in maxplaytime, minplaytime and playingtime
df['maxplaytime'].values[df['maxplaytime'] > 10000] = 180
df['minplaytime'].values[df['minplaytime'] > 60000] = 120
df['playingtime'].values[df['playingtime'] > 10000] = 180

# Change picture resolution
my_dpi=96

df[['Geekscore']].hist(bins=10)
plt.xlabel('Averge Rating of Game')
plt.ylabel('Number of Games')
plt.savefig('EDA_RatingsNumberGames.png', dpi=my_dpi * 10)

# Find Geekscore average to use for creating new sets for NLP 
print('Mean of Geekscore is %.4f%%' %  df.Geekscore.mean()) #6.5037
print('======================================================================')

# Subset greater than or equal to mean of Geekscore
df1 = df[df.Geekscore >= 6.5037]
print('\nDimensions of Data when >= Mean', df1.shape) #(1481326, 23)

# Create variable for rating group for binary class
df1 = df1.copy()
df1.loc[:,'Rating_Group'] = 'High'
print('======================================================================')

# Subset less than mean of Geekscore
df2 = df[df.Geekscore < 6.5037]
print('\nDimensions of Data when < Mean', df2.shape) #(1512464, 23)

# Create variable for rating group for binary class
df2 = df2.copy()
df2.loc[:,'Rating_Group'] = 'Low'
print('======================================================================')

# Concat dfs back with new var for binary class
df = pd.concat([df1, df2])
del df1, df2
print('Number of observations in each rating group:')
print(df.Rating_Group.value_counts())
print('======================================================================')

###############################################################################
# Examine Quantitative vars
df_num = df.select_dtypes(include = ['float64', 'int64'])
df_num = df_num.drop(['Year', 'yearpublished'], axis=1)

print('The selected dataframe has ' + str(df_num.shape[1]) +
       ' columns that are quantitative variables.')
print('======================================================================')

###############################################################################
# Examine quant vars for correlations
corr = df_num.corr(method="spearman") 

# Correlation matrix
plt.title('Correlation Matrix with Spearman rho')
sns.heatmap(corr, cmap='viridis',  linewidths=0.1, square=True);
plt.savefig('EDA_correlationMatrix_All_spearman.png', dpi=my_dpi * 10,
            bbox_inches='tight')

# Create correlation table with color scale
# Use upper section of matrix
upper_corr_matrix = corr.where(
    np.triu(np.ones(corr.shape), k=1).astype(np.bool))
  
# Convert matrix to 1-D, drop data with null and sort
unique_corr_pairs = upper_corr_matrix.unstack().dropna()
sorted_mat = unique_corr_pairs.sort_values()
sorted_mat = sorted_mat.to_frame()
sorted_mat.columns =['Correlation Coefficient']

# Subset of correlations in with a color scale
plt.title('Features with higher correlations using Spearman rho')
sns.heatmap(sorted_mat[(sorted_mat >= 0.5) | (sorted_mat <= -0.5)], 
            cmap='viridis',  linewidths=0.1, square=True);
plt.savefig('EDA_correlationMatrix_0.5subset_spearman.png', dpi=my_dpi * 10)

# Pairplot of highly correlated features
plt.title('Pairplot of highly correlated features')
sns.pairplot(data=df, vars=['Geekscore', 'Users rated', 'owned', 'numcomments',
                            'numweights', 'wishing', 'wanting', 'trading',
                            'playingtime', 'minplaytime', 'maxplaytime'])
plt.savefig('EDA_pairplot_highlyCorrelatedFeatures.png', dpi=my_dpi * 10)

# Drop due to high correlations with other vars
df = df.drop(['Rank', 'playingtime', 'wishing'], axis=1)
df = df.drop_duplicates()
print('\nDimensions of Final Data:', df.shape) #(2987129, 20)
print('======================================================================')

# Write to csv for NLP
df.to_csv('BGR_NLP.csv', index=False)

###############################################################################
######################## Create sample data set  ##############################
###############################################################################
df_sample = df.sample(n=300000)
# Write sample to csv
df_sample.to_csv('BGR_final_sample_3e5.csv', index=False)
###############################################################################


















