# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:12:32 2023

@author: iokoun
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
import sys
import os

project_dir = Path('.\input')

COMMODITY = 'H2' #Choose from: ['Electricity', 'H2', 'Heat'.'All']. Add to csv-files name (only relevant if filetype_input == 'csv'). If 'Other': go to cell 1.4.0.
VIEW = 'production' #options to load production, storage_vol, capacity, H2_price, etc look at the dir
Do_not_Show_all_years= False # focus on one year accross scenarios or multiple
All_countries = False #does not work for every country yet
Do_not_include_importH2 = 'YES' #yes or no
Year = 2045 #Year to be, plot only one!
#VIEW_Option = 'FFF' 
manual_change_the_order_of_bars = False #HUSK! you need to make a list with the scenario and the order!

#specify the order of the bars and the scenario names
bar_order = ['H2E','GH2E','SSGH2E']

#colour for network
if COMMODITY == 'Electricity' :
    net_colour = 'BrBG'
elif COMMODITY == 'H2' : 
    net_colour = 'RdBu'

#---------------------------------------------------------------------------------------

#load data
df_prod = pd.read_csv(project_dir/'results/InvestmentExcel/PRO_YCRAGF.csv', delimiter=';')



#filter based on attributes
if VIEW == 'production' or VIEW == 'capacity':
    df_prod = df_prod[df_prod['TECH_TYPE'] != 'H2-STORAGE']
    if  COMMODITY == 'Electricity':
        df_prod = df_prod[df_prod['FFF'] != 'ELECTRIC']
        df_prod = df_prod[df_prod['FFF'] != 'HEAT']
    elif COMMODITY == 'H2'and Do_not_include_importH2 == 'YES':
        df_prod = df_prod[df_prod['FFF'] != 'IMPORT_H2']
    
if  COMMODITY == 'Electricity':
    df_prod = df_prod[df_prod['COMMODITY'] == 'ELECTRICITY']
    df_demand = pd.read_csv(project_dir/'results/InvestmentExcel/EL_DEMAND_YCR.csv', delimiter=';')
elif COMMODITY == 'H2':
    df_prod = df_prod[df_prod['COMMODITY'] == 'HYDROGEN']
    df_demand = pd.read_csv(project_dir/'results/InvestmentExcel/H2_DEMAND_YCR.csv', delimiter=';')
elif COMMODITY == 'Heat':
    df_prod = df_prod[df_prod['COMMODITY'] == 'HEAT']

#KEEP the year in discussion
df_demand= df_demand[df_demand['Y'] == Year]
df_prod  = df_prod[df_prod['Y'] == Year]

df_prod_sum = pd.DataFrame(df_prod.groupby(['C','Y', 'Scenario'])['value'].sum().reset_index())
df_demand_sum = pd.DataFrame(df_demand.groupby(['C','Y', 'Scenario'])['value'].sum().reset_index())

# Merge the dataframes on 'C', 'Y', and 'Scenario' columns check for nan and incosistensies
df_merged = pd.merge(df_demand_sum,df_prod_sum, on=['C', 'Y', 'Scenario'], suffixes=( '_demand','_prod'),how ='left')  
#here are the missing rows
nan_rows = df_merged[df_merged['value_prod'].isna()]
#fill those with zeros
df_merged['value_prod']  = df_merged['value_prod'].fillna(0)
# Calculate the difference between demand and production
df_merged['balance'] = df_merged['value_prod'] -df_merged['value_demand']


# put iso code for countries

country_code_dict = {
    'AUSTRIA': 'AT',
    'BELGIUM': 'BE',
    'CZECH_REPUBLIC': 'CZ',
    'DENMARK': 'DK',
    'ESTONIA': 'EE',
    'FINLAND': 'FI',
    'FRANCE': 'FR',
    'GERMANY': 'DE',
    'ITALY': 'IT',
    'LATVIA': 'LV',
    'LITHUANIA': 'LT',
    'NETHERLANDS': 'NL',
    'NORWAY': 'NO',
    'POLAND': 'PL',
    'PORTUGAL': 'PT',
    'SPAIN': 'ES',
    'SWEDEN': 'SE',
    'SWITZERLAND': 'CH',
    'UNITED_KINGDOM': 'UK',
    'SLOVAKIA':'SK',
    'HUNGARY':'HU',
    'SLOVENIA':'SI',
    'CROATIA':'HR',
    'ROMANIA':'RO',
    'BULGARIA':'BG',
    'GREECE':'GR',
    'IRELAND':'IE',
    'LUXEMBOURG':'LU',
    'ALBANIA':'AL',
    'MONTENEGRO':'ME',
    'NORTH_MACEDONIA':'MK',
    'BOSNIA_AND_HERZEGOVINA':'BA',
    'SERBIA':'RS',
    'MALTA':'MT',
    'CYPRUS':'CY',
    'TURKEY':'TR'
    
    
}

'''
country_code_dict = {
    'BOSNIA_AND_HERZEGOVINA':'BOSNIA & HERZEGOVINA ',
    'NORTH_MACEDONIA':'NORTH MACEDONIA',
    'UNITED_KINGDOM': 'UNITED KINGDOM',
    
}
'''


#change the names of the data frames to the code 
df_merged['C'] = df_merged['C'].map(country_code_dict).fillna(df_merged['C'])


# Define a custom color map with white in the center
#in case catecories.
cmap = ListedColormap(['#e74c3c', '#f39c12', '#f1c40f', 'white', '#2ecc71', '#3498db', '#9b59b6'])


#you need to have them as clipboard saved
#df = pd.read_clipboard()            # If you have selected the headers

#unppivot data
matrix = df_merged.pivot(index='C', columns='Scenario', values='balance')

if manual_change_the_order_of_bars:     
    df_pivot_reordered = matrix.reindex(columns=bar_order)
    matrix = df_pivot_reordered


#------------------------------------------------------------------------------------------------------------------
# Create the heatmap
fig, ax = plt.subplots(figsize=(6, 6))
heatmap = ax.imshow(matrix, cmap=net_colour, vmin=-140, vmax=140, extent=[-0.5, len(matrix.columns)-0.5, len(matrix.index)-0.5, -0.5], aspect="auto")

# Add a colorbar
cbar = ax.figure.colorbar(heatmap, ax=ax,ticks=np.arange(-140, 160, 20),extend = 'both' )
cbar.ax.set_ylabel(f'Regional {COMMODITY} Balnace TWh ({Year})')
#f'{COMMODITY} Line utilization ({year}) [%]'
# Add vertical lines to separate scenario blocks
for i in range(len(matrix.columns)):
    ax.axvline(x=i-0.5, color='grey', linewidth=2, linestyle='--')

# Set the tick labels and axis labels
ax.set_xticks(np.arange(len(matrix.columns)))
ax.set_yticks(np.arange(len(matrix.index)))
ax.set_xticklabels(matrix.columns, fontsize=12)
ax.set_yticklabels(matrix.index, fontsize=12)
#ax.set_xlabel('Scenario', fontsize=14)
#ax.set_ylabel('Region', fontsize=14)

# Rotate the tick labels for better readability
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

'''
# Add the value annotations to the heatmap
for i in range(len(matrix.index)):
    for j in range(len(matrix.columns)):
        ax.text(j, i, '{:.2f}'.format(matrix.iloc[i, j]), ha='center', va='center', color='black', fontsize=12)
'''
# Add a title
#plt.title('Scenario Values by Country', fontsize=16)

# Adjust the layout
plt.tight_layout()

# Show the plot
#plt.show()

#Store it

#save plot
map_name = 'Demand_Production_Difference'

# Make Transmission_Map output folder
if not os.path.isdir('output/RandomPlots'):
        os.makedirs('output/RandomPlots')
        
output_dir = 'output/RandomPlots'
plt.savefig(output_dir + '/' +  map_name + str(Year) + COMMODITY + '.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
