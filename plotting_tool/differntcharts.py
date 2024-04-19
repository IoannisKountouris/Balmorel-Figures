# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:36:53 2023

@author: iokoun
"""
#make a plot for imports from third nations

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import os

manual_change_the_order_of_bars = True #HUSK! you need to make a list with the scenario and the order!
show_total = False # if you want to show a dot with the total number per stack 

#specify the order of the bars and the scenario names
bar_order = ['H2E','GH2E']



project_dir = Path('.\input')
df = pd.read_csv(project_dir/'results/InvestmentExcel/F_CONS_YCRA.csv', delimiter=';')

#keep only the imports
df = df[df['FFF'] == 'IMPORT_H2']

#change the country names
C_to_Country = {
'ITALY':'Tunisia & Algeria',
'SPAIN':'Morocco',
'HUNGARY':'Ukraine',
'SLOVAKIA':'Ukraine'
}
df['C'] = df['C'].map(C_to_Country).fillna(df['C'])
df_sum = pd.DataFrame(df.groupby(['Y', 'Scenario','C'])['value'].sum().reset_index())

#change names of coloumns
df_sum = df_sum.rename(columns={'C': 'Country', 'Y': 'Year', 'value':'Value'})

#maybe are some scenarios with zero import and years make them as data
# Set YEAR and SCENARIO as the index
# make combinations for every country and year etc.
# Define the range of years, scenarios, and countries to consider
min_year = 2030
max_year = 2050
step_size=5
scenarios = ['H2E','GH2E']
countries = ['Tunisia & Algeria', 'Morocco', 'Ukraine']
# Generate the discrete years
years = list(range(min_year, max_year + 1, step_size))


# Generate all combinations of years, scenarios, and countries
combinations = [(year, scenario, country) for year in years
                                            for scenario in scenarios
                                            for country in countries]

df_sum.set_index(['Year', 'Scenario', 'Country'], inplace=True)

# Reindex the DataFrame with the desired combinations
df_sum = df_sum.reindex(combinations, fill_value=0).reset_index()

#pass by back to df 
df = df_sum

'''
Try plot
data = {
    'Country': ['Ukraine', 'Ukraine', 'Ukraine', 'Ukraine', 'Ukraine', 'Ukraine', 
                'Tunisia & Algeria', 'Tunisia & Algeria', 'Tunisia & Algeria', 'Tunisia & Algeria', 
                'Tunisia & Algeria', 'Tunisia & Algeria', 'Marocco', 'Marocco', 'Marocco', 'Marocco', 
                'Marocco', 'Marocco'],
    'Year': [2030, 2040, 2050, 2030, 2040, 2050, 2030, 2040, 2050, 2030, 2040, 2050, 2030, 2040, 2050, 2030, 2040, 2050],
    'Scenario': ['BASE', 'BASE', 'BASE', 'GH2E', 'GH2E', 'GH2E', 'BASE', 'BASE', 'BASE', 'GH2E', 'GH2E', 'GH2E', 'BASE', 'BASE', 'BASE', 'GH2E', 'GH2E', 'GH2E'],
    'Value': [0.00, 7.58, 15.62, 0.00, 7.93, 16.35, 0.00, 9.59, 62.45, 0.00, 46.90, 72.79, 0.00, 20.23, 47.05, 0.00, 19.74, 42.84]
}

df = pd.DataFrame(data)
print(df)



#diffrent types of plot based on scenario and year
colors = ['#EA9999', '#CFA7CF', '#9AD8C7']

# Pivot the data frame to have scenarios as columns and countries as rows
df_pivot = df.pivot(index=['Scenario', 'Year'], columns='Country', values='Value')

# Create the stacked bar plot
ax = df_pivot.plot(kind='bar', stacked=True, figsize=(10, 6))

# Set the axis labels and title
ax.set_xlabel('Scenario, Year')
ax.set_ylabel('TWh')
ax.set_title('Stacked Bar Plot of Scenarios by Country and Year')

# Show the plot
plt.show()
'''
#--------------------------------------------------------------------------------------------------
#diffrent types of plot based on year and scenario


# Set Year and Scenario as index
df = df.set_index(['Year', 'Scenario'])

# Pivot the data frame to have countries as columns and scenarios as rows
df_pivot = df.pivot(columns='Country', values='Value')


colors = ['#EA9999', '#CFA7CF', '#9AD8C7']

#make plot
# Create the stacked bar plot

if manual_change_the_order_of_bars:     
    df_pivot_reordered = df_pivot.reindex(bar_order, level=1)
    df_pivot = df_pivot_reordered
    
ax = df_pivot.plot(kind='bar', stacked=True, figsize=(8, 5), color=colors)

# Add black bars on the x-axis to indicate year changes
x_coordinates = [i - 0.5 for i in range(len(df_pivot))]
year_changes = [y[0] for y in df_pivot.index]
for i in range(len(year_changes) - 1):
    if year_changes[i] != year_changes[i + 1]:
        ax.axvline(x_coordinates[i] + 1 , linestyle=(0, (2, 4)),color='#b0b0b0')

# Set the axis labels and title
ax.set_xlabel('Year, Scenario')
ax.set_ylabel('TWh')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.tick_params(bottom=False, left=False)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle='dotted',color='#b0b0b0')
ax.xaxis.grid(False)
pos_tech = (0,0.75)
legend = ax.legend(loc='lower left', ncol = 1,frameon=True,
                              mode='expnad',bbox_to_anchor=pos_tech)
legend.get_frame().set_edgecolor('white')


#ax.set_title('Stacked Bar Plot of Scenarios by Year and Country')
if show_total:
    # Loop through each stacked bar and add the total on top of the bar with a dot
    for i, (scenario, year) in enumerate(df_pivot.index):
        # Get the total of the stacked bar
        total = df_pivot.loc[(scenario, year)].sum()
    
        # Add the text
        ax.text(i, total + 2, str(round(total)), ha='center', fontsize=10)
        # Add a dot on top of the bar
        ax.scatter(i, total, s=20, c='black', marker='o')
    

#save plot
map_name = 'Importing Hydrogen quantities'

# Make Transmission_Map output folder
if not os.path.isdir('output/RandomPlots'):
        os.makedirs('output/RandomPlots')
        
output_dir = 'output/RandomPlots'
plt.savefig(output_dir + '/' +  map_name + '.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
