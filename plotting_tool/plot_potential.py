#plot potential and results in barplots

#!/usr/bin/env Balmorel

import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import xarray as xr

from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import ArrowStyle
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
import seaborn as sns
from sklearn.cluster import KMeans
#from matplotlib.colors import to_rgb


from pathlib import Path
import sys
import os
import glob

#choose correct version of Gams
sys.path.append(r'C:\GAMS\36\apifiles\Python\api_38')
sys.path.append(r'C:\GAMS\36\apifiles\Python\gams')

from gams import GamsWorkspace

manual_change_the_order_of_bars = True #HUSK! you need to make a list with the scenario and the order!

#specify the order of the bars and the scenario names
bar_order = ['H2E','GH2E','SSGH2E']



# In[ ]:


project_dir = Path('.\input')

#Load coordinates files 
df_RenePot = pd.read_csv(project_dir/'input_data/Rene_pot.csv', delimiter=';')
df_Results = pd.read_csv(project_dir/'results/InvestmentExcel/G_CAP_YCRAF.csv', delimiter=';')

#Sum based on tech
display_column = 'TECH_GROUP'

df_RenePot_C_G = pd.DataFrame(df_RenePot.groupby(['CCCRRRAAA', display_column])['Value'].sum().reset_index())

#From MW to GW installation
df_RenePot_C_G['Value'] = df_RenePot_C_G['Value']/1000

#Have consistency of tech_group with tech_type
tech_group_to_tech_type = {
'SOLARPV':'SOLAR-PV' , 
'WINDTURBINE_ONSHORE': 'WIND-ON' ,
'WINDTURBINE_OFFSHORE': 'WIND-OFF'
}

# Apply the change
df_RenePot_C_G[display_column] = df_RenePot_C_G[display_column].map(tech_group_to_tech_type).fillna(df_RenePot_C_G[display_column])

#Change the name of coloumn
df_RenePot_C_G.rename(columns={'TECH_GROUP': 'TECH_TYPE', 'CCCRRRAAA': 'RRR'}, inplace=True)

#Filter to keep only Electricity and rene
df_Results_G_RENE = df_Results[df_Results['COMMODITY'] == 'ELECTRICITY']

#Select rows with 'SOLAR-PV', 'WIND-ON', or 'WIND-OFF' in the 'TECH_TYPE' column
mask = df_Results_G_RENE['TECH_TYPE'].isin(['SOLAR-PV', 'WIND-ON', 'WIND-OFF'])

#filter them
df_Results_G_RENE = df_Results_G_RENE[mask]

#Keep the results from 2050
df_Results_G_RENE = df_Results_G_RENE [df_Results_G_RENE ['Y'] == 2050]

# Create a dictionary to map the elements of column 'C' to lists of corresponding 'RRR' values
mapping_dict = {}
for c, rrr in zip(df_Results_G_RENE['C'], df_Results_G_RENE['RRR']):
    if c not in mapping_dict:
        mapping_dict[c] = {rrr}
    else:
        mapping_dict[c].add(rrr)

# Print the mapping dictionary
print(mapping_dict)

#revert the dictionary
flattened_mapping_dict = {}
for c, rr_set in mapping_dict.items():
    for rr in rr_set:
        flattened_mapping_dict[rr] = c

#create a new 'C' column in 'df_RenePot_C_G' by mapping the 'RRR' values to 'C' values using the 'flattened_mapping_dict'
df_RenePot_C_G['C'] = df_RenePot_C_G['RRR'].map(flattened_mapping_dict)

# group by 'C' and 'TECH_TYPE' columns and sum the 'Value' column
df_RenePot_C_G = df_RenePot_C_G.groupby(['C', 'TECH_TYPE'], as_index=False).agg({'Value': 'sum'})

#Sum all the values from the results
df_Results_G_RENE = df_Results_G_RENE.groupby(['C','Scenario', 'TECH_TYPE'])['value'].sum().reset_index()


#Keep the names
scenario_names = df_Results_G_RENE['Scenario'].unique()
tech_names = df_Results_G_RENE['TECH_TYPE'].unique()
country_names = df_Results_G_RENE['C'].unique()

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

#change the names of the data frames to the code 
df_RenePot_C_G['C'] = df_RenePot_C_G['C'].map(country_code_dict).fillna(df_RenePot_C_G['C'])
df_Results_G_RENE['C'] = df_Results_G_RENE['C'].map(country_code_dict).fillna(df_Results_G_RENE['C'])

missing_rows = df_RenePot_C_G[~df_RenePot_C_G.index.isin(df_Results_G_RENE.index)]



# In[ ]:
    
#Plot    
    
fig, axs = plt.subplots(len(df_Results_G_RENE['Scenario'].unique()), len(df_Results_G_RENE['TECH_TYPE'].unique()), 
                        figsize=(15, 5*len(df_Results_G_RENE['Scenario'].unique())))
plt.xticks(rotation=45)
# Loop through the 'TECH_TYPE' values and plot a bar chart for each one
for i, tech_type in enumerate(df_Results_G_RENE['TECH_TYPE'].unique()):
    # Subset the data for the current 'TECH_TYPE'
    sub_df = df_Results_G_RENE[df_Results_G_RENE['TECH_TYPE'] == tech_type]

    # Loop through the 'Scenario' values and plot a bar for each one
    for j, scenario in enumerate(sub_df['Scenario'].unique()):
        # Subset the data for the current 'Scenario'
        sub_sub_df = sub_df[sub_df['Scenario'] == scenario]

        # Get the x and y values for the bar plot
        x = sub_sub_df['C']
        y = sub_sub_df['value']
        
        # Set the color for the current TECH_TYPE
        if tech_type == 'SOLAR-PV':
           color = '#d2a106'
        elif tech_type == 'WIND-ON':
           color = '#006460'
        elif tech_type == 'WIND-OFF':
           color = '#08bdba'

        # Plot the bar chart
        axs[j,i].bar(x, y, label=tech_type, color=color)
        # Add dots for the maximum capacity per country for the current TECH_TYPE
        for k, country in enumerate(x):
            max_value = df_RenePot_C_G[(df_RenePot_C_G['C'] == country) & (df_RenePot_C_G['TECH_TYPE'] == tech_type)]['Value'].max()
            axs[j,i].scatter(country, max_value, color='black', s=50, zorder=10, marker="_")


        # Set the title and legend for the 
        axs[j,i].tick_params(axis='x', rotation=45)
        axs[j,i].set_title(scenario)
        axs[j,i].legend()
        # Add y-axis label
        axs[j,i].set_ylabel('GW')

# Add a suptitle for the entire figure
#fig.suptitle('Renewable energy Capacities vs Potential')

### 3.3 Save map

map_name = 'RenePot'

# Make Transmission_Map output folder
if not os.path.isdir('output/Potential'):
    os.makedirs('output/Potential')
    
output_dir = 'output/Potential'
plt.savefig(output_dir + '/' +  map_name + '.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()



# In[ ]:
# Set up the subplots with the appropriate dimensions
# Set the figure size and style

# Create a FacetGrid with one row per technology type
g = sns.FacetGrid(df_Results_G_RENE, row="TECH_TYPE", height=6, aspect=2, margin_titles=True)

# Map a bar plot to the FacetGrid with one column per scenario
g.map_dataframe(sns.barplot, x="C", y="value", hue="Scenario", palette="dark", alpha=.6, ci="sd"  )




# In[ ]:

# Define a list of hex codes for the colors matching the rest of the paper
colors = ['#4589ff', '#006460', '#08bdba', '#d1b9b9', '#7f7f7f', 
          '#BA000F', '#8c564b', '#FFC0CB', '#ff7eb6', '#a5e982', 
          '#cd6f00', '#ffbb78', '#d2a106', '#8a3ffc', '#2b1d1d', '#f4eeff']

fig, axs = plt.subplots(len(df_Results_G_RENE['TECH_TYPE'].unique()), 1, figsize=(12, 12), dpi=300)

colors = ['#EA9999', '#CFA7CF', '#9AD8C7']
max_countries = 0 
# Loop through the TECH_TYPE values and plot a bar chart for each one
for i, tech_type in enumerate(df_Results_G_RENE['TECH_TYPE'].unique()):
    
    sub_df = df_Results_G_RENE[df_Results_G_RENE['TECH_TYPE'] == tech_type]
        
    countries = sub_df['C'].unique()
    scenarios = sub_df['Scenario'].unique()
    
    bar_width = len(scenarios) / len(scenarios)
    
    x_offset = bar_width / len(scenarios)
    
    
    # Get unique countries and scenarios
    scenarios_series = pd.Series(bar_order)

    # Create a DataFrame with all combinations of countries and scenarios
    country_scenario_df = pd.DataFrame(columns=['C', 'Scenario'])
    for country in countries:
        df = pd.DataFrame({'C': country, 'Scenario': scenarios_series})
        country_scenario_df = country_scenario_df.append(df, ignore_index=True)

    # Perform a left join between sub_df and country_scenario_df
    merged_df = country_scenario_df.merge(sub_df, how='left', on=['C', 'Scenario'])

    # Replace missing values with zero
    merged_df['value'] = merged_df['value'].fillna(0)

    # Set 'Scenario' column as the index
    merged_df = merged_df.set_index('Scenario')
    
    merged_df['TECH_TYPE'] = merged_df['TECH_TYPE'].fillna(df_Results_G_RENE['TECH_TYPE'].unique()[i])
    
    #pass back the file
    sub_df = merged_df.reset_index()
    
    if manual_change_the_order_of_bars:
        scenarios= bar_order
    
    
    for j, scenario in enumerate(scenarios):
        
        sub_sub_df = sub_df[sub_df['Scenario'] == scenario]
    
        countries_scenario = sub_sub_df['C'].unique()  # Unique countries for the current scenario
        
        y = sub_sub_df['value']
        x = np.arange(len(countries_scenario))  # x array based on unique countries for the scenario
        color = colors[j % len(colors)]
        
        # Add label only to the first subplot
        if i == 0:
            axs[i].bar(x + j / len(scenarios), y, width=1/len(scenarios), color=color, label=scenario)
        else:
            axs[i].bar(x + j / len(scenarios), y, width=1/len(scenarios), color=color)
            
        max_capacities_scenario = []  # Unique max capacities for the current scenario
        for country in countries_scenario:
            max_capacity = df_RenePot_C_G[(df_RenePot_C_G['C'] == country) & (df_RenePot_C_G['TECH_TYPE'] == tech_type)]['Value'].max()
            max_capacities_scenario.append(max_capacity)
        
        # Plot scatter for the current scenario
        axs[i].scatter(x + j / len(scenarios), max_capacities_scenario, color='black', s=20, zorder=10, marker="_")



    axs[i].set_xticks(np.arange(len(countries)) + x_offset)
    axs[i].set_xticklabels(countries)

    axs[i].set_title(tech_type)
    axs[i].tick_params(axis='x', rotation=45)
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['bottom'].set_color('#DDDDDD')
    axs[i].tick_params(bottom=False, left=False)
    axs[i].set_axisbelow(True)
    axs[i].yaxis.grid(True, linestyle='dotted',color='#b0b0b0')
    axs[i].xaxis.grid(False)

    #axs[i].grid(False)

    axs[i].set_ylabel('GW')
# Add legend only to the first subplot
pos_tech = (0,0.75)
legend = axs[0].legend(loc='lower left', ncol = 1,frameon=True,
                              mode='expnad',bbox_to_anchor=pos_tech)
legend.get_frame().set_edgecolor('white')


    
    
map_name = 'RenePot_tech'

# Make Transmission_Map output folder
if not os.path.isdir('output/Potential'):
        os.makedirs('output/Potential')
        
output_dir = 'output/Potential'
plt.savefig(output_dir + '/' +  map_name + '.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


# In[ ]:





