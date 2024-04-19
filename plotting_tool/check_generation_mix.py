# -*- coding: utf-8 -*-
"""
Created on Tue May 16 15:27:35 2023

@author: iokoun

Yearly evolution of one scenario or more. 
"""

#!/usr/bin/env Balmorel
import math
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
from sklearn.cluster import KMeans
#from matplotlib.colors import to_rgb
from matplotlib.patches import Patch


from pathlib import Path
import sys
import os
import glob

#choose correct version of Gams
sys.path.append(r'C:\GAMS\36\apifiles\Python\api_38')
sys.path.append(r'C:\GAMS\36\apifiles\Python\gams')

from gams import GamsWorkspace


project_dir = Path('.\input')

COMMODITY = 'Electricity' #Choose from: ['Electricity', 'H2', 'Heat'.'All']. Add to csv-files name (only relevant if filetype_input == 'csv'). If 'Other': go to cell 1.4.0.
VIEW = 'production' #options to load, production, storage_vol, capacity, H2_price, etc look at the dir
Do_not_Show_all_years= True # focus on one year accross scenarios or multiple
All_countries = False #does not work for every country yet
Year = 2040 #Year to be, plot only one!
Exclude_years =2030 #Exclude years before e.g. 2030
VIEW_Option = 'FFF' #options FFF or TECH_TYPE to aggregate the values
Do_not_include_importH2 = 'YES' #yes or no
Show_labels = False
manual_change_the_order_of_bars = True #HUSK! you need to make a list with the scenario and the order!
 
# In[ ]

#load only from one scenario
df = pd.read_csv(project_dir/'results/InvestmentExcel/checkhydrogenorigin/PRO_YCRAGFST.csv', delimiter=';')
df_demand = pd.read_csv(project_dir/'results/InvestmentExcel/checkhydrogenorigin/H2_DEMAND_YCRST.csv', delimiter=';')
df_hydrogen = pd.read_csv(project_dir/'results/InvestmentExcel/checkhydrogenorigin/PRO_YCRAGFST.csv', delimiter=';')

# if production or capacity take out the storage options 
#also take out the import capacities

if VIEW == 'production' or VIEW == 'capacity':
    df = df[df['TECH_TYPE'] != 'H2-STORAGE']
    df = df[df['TECH_TYPE'] != 'INTERSEASONAL-HEAT-STORAGE']
    df = df[df['TECH_TYPE'] != 'INTRASEASONAL-HEAT-STORAGE']
    df_hydrogen = df_hydrogen[df_hydrogen['FFF'] != 'IMPORT_H2']
    df_hydrogen = df_hydrogen[df_hydrogen['FFF'] != 'NATGAS']
    df_hydrogen = df_hydrogen[df_hydrogen['TECH_TYPE'] != 'H2-STORAGE']
    
    if  COMMODITY == 'Electricity':
        df = df[df['FFF'] != 'ELECTRIC']
        df = df[df['FFF'] != 'HEAT']
    elif COMMODITY == 'Heat':
        df = df[df['FFF'] != 'HEAT']
        #df = df[df['C'] == specific_country]
    elif COMMODITY == 'H2'and Do_not_include_importH2 == 'YES':
        df = df[df['FFF'] != 'IMPORT_H2']
    
if  COMMODITY == 'Electricity':
    df = df[df['COMMODITY'] == 'ELECTRICITY']
    df_hydrogen = df_hydrogen[df_hydrogen['COMMODITY'] == 'HYDROGEN']
elif COMMODITY == 'H2':
    df = df[df['COMMODITY'] == 'HYDROGEN']
    df_hydrogen = df_hydrogen[df_hydrogen['COMMODITY'] == 'HYDROGEN']
elif COMMODITY == 'Heat':
    df = df[df['COMMODITY'] == 'HEAT']
    df_hydrogen = df_hydrogen[df_hydrogen['COMMODITY'] == 'HYDROGEN']

#map technologies and add ones or fuels    
if VIEW_Option == 'TECH_TYPE':
# Create horizontal table with sectors as columns
    display_column = 'TECH_TYPE'
    #Distinguish if has CCS or not for hydrogen
    G_to_tech_type_stor = {
    'GNR_H2S_H2-TNKC_Y-2020':'STEEL-TANK', 
    'GNR_H2S_H2-CAVERN_Y-2030':'SALT-CAVERN',
    'GNR_H2S_H2-TNKC_Y-2030':'STEEL-TANK', 
    'GNR_H2S_H2-CAVERN_Y-2040':'SALT-CAVERN',
    'GNR_H2S_H2-TNKC_Y-2040':'STEEL-TANK', 
    'GNR_H2S_H2-TNKC_Y-2050':'STEEL-TANK',
    'GNR_H2S_H2-CAVERN_Y-2050': 'SALT-CAVERN',
    'GNR_IMPORT_H2':'IMPORT H2'
    }
    df['TECH_TYPE'] = df['G'].map(G_to_tech_type_stor).fillna(df['TECH_TYPE'])
    
    G_to_tech_type = {
    'GNR_STEAM-REFORMING_E-70_Y-2020': 'SMR',
    'GNR_STEAM-REFORMING-CCS_E-70_Y-2020': 'SMR-CCS'
    }
    df['TECH_TYPE'] = df['G'].map(G_to_tech_type).fillna(df['TECH_TYPE'])

if VIEW_Option == 'FFF':
    display_column = 'FFF'
    #If you map fuels to change the fuel type.     
    # Define the dictionary to map old fuel names to new ones
    
    #First split wind to wind on and wind off based on the tech_type
    # create a dictionary to map the values of TECH_TYPE to the corresponding FFF names
    tech_type_to_fff = {"WIND-ON": "WIND-ON", "WIND-OFF": "WIND-OFF"}
    # use the map function to replace the values of FFF based on the values of TECH_TYPE
    df['FFF'] = df['TECH_TYPE'].map(tech_type_to_fff).fillna(df['FFF'])
    # create a dictionary to map the values of FFF to the corresponding fuel types
    fff_to_fuel = {
    'BIOOIL': 'OIL', 
    'LIGHTOIL': 'OIL', 
    'OIL': 'OIL', 
    'FUELOIL': 'OIL',
    'SHALE' : 'OIL',
    'WOODCHIPS': 'BIOMASS', 
    'WOODPELLETS': 'BIOMASS', 
    'WOODWASTE': 'BIOMASS', 
    'WOOD': 'BIOMASS',
    'STRAW': 'BIOMASS',
    'RETORTGAS':'NATGAS',
    'OTHERGAS': 'NATGAS',
    'DUMMY': 'NATGAS',
    'PEAT' : 'NATGAS',
    'WASTEHEAT' :'HEAT',
    'LNG' :'NATGAS',
    'SUN':'SOLAR',
    'WATER':'HYDRO'
    
    }
    # use the map function to replace the values of FFF based on the values of the dictionary
    df['FFF'] = df['FFF'].map(fff_to_fuel).fillna(df['FFF'])
    
    G_to_FFF = {
    'GNR_BO_NGASCCS_E-105_MS-5-MW_Y-2020':'NATGAS-CCS',                   
    'GNR_BO_NGASCCS_E-106_MS-5-MW_Y-2030':'NATGAS-CCS',                   
    'GNR_BO_NGASCCS_E-106_MS-5-MW_Y-2040':'NATGAS-CCS',                   
    'GNR_BO_NGASCCS_E-106_MS-5-MW_Y-2050':'NATGAS-CCS',                   
    'GNR_CC_NGASCCS_BP_E-51_SS-10-MW_Y-2020':'NATGAS-CCS',                   
    'GNR_CC_NGASCCS_BP_E-53_SS-10-MW_Y-2030':'NATGAS-CCS',                   
    'GNR_CC_NGASCCS_BP_E-54_SS-10-MW_Y-2040':'NATGAS-CCS',                   
    'GNR_CC_NGASCCS_BP_E-55_SS-10-MW_Y-2050':'NATGAS-CCS',                   
    'GNR_CC_NGASCCS_CND_E-51_SS-10-MW_Y-2020':'NATGAS-CCS',                   
    'GNR_CC_NGASCCS_CND_E-53_SS-10-MW_Y-2030':'NATGAS-CCS',                   
    'GNR_CC_NGASCCS_CND_E-54_SS-10-MW_Y-2040':'NATGAS-CCS',                   
    'GNR_CC_NGASCCS_CND_E-55_SS-10-MW_Y-2050':'NATGAS-CCS',                   
    'GNR_CC_NGASCCS_CND_E-59_LS-100-MW_Y-2020':'NATGAS-CCS',                   
    'GNR_CC_NGASCCS_CND_E-61_LS-100-MW_Y-2030':'NATGAS-CCS',                   
    'GNR_CC_NGASCCS_CND_E-62_LS-100-MW_Y-2040':'NATGAS-CCS',                   
    'GNR_CC_NGASCCS_CND_E-63_LS-100-MW_Y-2050':'NATGAS-CCS',                   
    'GNR_CC_NGASCCS_EXT_E-59_LS-100-MW_Y-2020':'NATGAS-CCS',                   
    'GNR_CC_NGASCCS_EXT_E-61_LS-100-MW_Y-2030':'NATGAS-CCS',                   
    'GNR_CC_NGASCCS_EXT_E-62_LS-100-MW_Y-2040':'NATGAS-CCS',                   
    'GNR_CC_NGASCCS_EXT_E-63_LS-100-MW_Y-2050':'NATGAS-CCS',                   
    'GNR_ENG_NGASCCS_BP_E-47_Y-2020':'NATGAS-CCS',                   
    'GNR_ENG_NGASCCS_BP_E-48_Y-2030':'NATGAS-CCS',                   
    'GNR_ENG_NGASCCS_BP_E-49_Y-2040':'NATGAS-CCS',                   
    'GNR_ENG_NGASCCS_BP_E-50_Y-2050':'NATGAS-CCS',                   
    'GNR_ENG_NGASCCS_CND_E-47_Y-2020':'NATGAS-CCS',                   
    'GNR_ENG_NGASCCS_CND_E-48_Y-2030':'NATGAS-CCS',                   
    'GNR_ENG_NGASCCS_CND_E-49_Y-2040':'NATGAS-CCS',                   
    'GNR_ENG_NGASCCS_CND_E-50_Y-2050':'NATGAS-CCS',                   
    'GNR_GT_NGASCCS_BP_E-37_SS-5-MW_Y-2020':'NATGAS-CCS',                   
    'GNR_GT_NGASCCS_BP_E-39_SS-5-MW_Y-2030':'NATGAS-CCS',                   
    'GNR_GT_NGASCCS_BP_E-40_SS-5-MW_Y-2040':'NATGAS-CCS',                   
    'GNR_GT_NGASCCS_BP_E-40_SS-5-MW_Y-2050':'NATGAS-CCS',                   
    'GNR_GT_NGASCCS_BP_E-42_LS-40-MW_Y-2020':'NATGAS-CCS',                   
    'GNR_GT_NGASCCS_BP_E-43_LS-40-MW_Y-2030':'NATGAS-CCS',                   
    'GNR_GT_NGASCCS_BP_E-44_LS-40-MW_Y-2040':'NATGAS-CCS',                   
    'GNR_GT_NGASCCS_BP_E-44_LS-40-MW_Y-2050':'NATGAS-CCS',                   
    'GNR_GT_NGASCCS_CND_E-37_SS-5-MW_Y-2020':'NATGAS-CCS',                   
    'GNR_GT_NGASCCS_CND_E-39_SS-5-MW_Y-2030':'NATGAS-CCS',                   
    'GNR_GT_NGASCCS_CND_E-40_SS-5-MW_Y-2040':'NATGAS-CCS',                   
    'GNR_GT_NGASCCS_CND_E-40_SS-5-MW_Y-2050':'NATGAS-CCS',                   
    'GNR_GT_NGASCCS_CND_E-42_LS-40-MW_Y-2020':'NATGAS-CCS',                   
    'GNR_GT_NGASCCS_CND_E-43_LS-40-MW_Y-2030':'NATGAS-CCS',                   
    'GNR_GT_NGASCCS_CND_E-44_LS-40-MW_Y-2040':'NATGAS-CCS',                   
    'GNR_GT_NGASCCS_CND_E-44_LS-40-MW_Y-2050':'NATGAS-CCS',                   
    'GNR_IND-DF_NGASCCS_E-100_MS-3-MW_Y-2020':'NATGAS-CCS',                   
    'GNR_IND-BO_NGASCCS_E-93_MS-20-MW_Y-2020':'NATGAS-CCS',                   
    'GNR_IND-BO_NGASCCS_E-94_MS-20-MW_Y-2030':'NATGAS-CCS',                   
    'GNR_IND-BO_NGASCCS_E-95_MS-20-MW_Y-2040':'NATGAS-CCS',                   
    'GNR_IND-BO_NGASCCS_E-96_MS-20-MW_Y-2050':'NATGAS-CCS',                   
    'GNR_ST_NGASCCS_CND_E-47_LS-400-MW_Y-2020':'NATGAS-CCS',                   
    'GNR_ST_NGASCCS_EXT_E-47_LS-400-MW_Y-2020':'NATGAS-CCS',                   
    'GNR_ST_NGASCCS_BP_E-7_MS-15-MW_Y-2020':'NATGAS-CCS',
    'GNR_STEAM-REFORMING-CCS_E-70_Y-2020':'NATGAS-CCS'
    }
    df['FFF'] = df['G'].map(G_to_FFF).fillna(df['FFF'])   


if VIEW_Option == 'TECH_TYPE':
    df_color_tech = {
    'HYDRO-RESERVOIRS': '#33b1ff',
    'HYDRO-RUN-OF-RIVER': '#4589ff',
    'WIND-ON': '#006460',
    'BOILERS': '#8B008B',
    'ELECT-TO-HEAT': '#FFA500',
    'INTERSEASONAL-HEAT-STORAGE': '#FFD700',
    'CHP-BACK-PRESSURE': '#E5D8D8',
    'SMR-CCS': '#00BFFF',
    'SMR': '#d1b9b9',
    'INTRASEASONAL-HEAT-STORAGE': '#00FFFF',
    'CONDENSING': '#8a3ffc',
    'SOLAR-HEATING': '#FF69B4',
    'CHP-EXTRACTION': '#ff7eb6',
    'SOLAR-PV': '#d2a106',
    'WIND-OFF': '#08bdba',
    'INTRASEASONAL-ELECT-STORAGE': '#ba4e00',
    'ELECTROLYZER': '#ADD8E6',
    'SALT-CAVERN': '#E8C3A8',
    'STEEL-TANK':'#C0C0C0',
    'FUELCELL': '#d4bbff',
    'IMPORT H2':'#cd6f00'
    }
    

if VIEW_Option  == 'FFF':
    df_color_tech = {
    'HYDRO': '#08bdba',
    'WIND-ON': '#5e45ff',
    'WIND-OFF': '#4589ff',
    'BIOGAS': '#23932d',
    'COAL': '#595959',
    'ELECTRIC': '#BA000F',
    'OIL': '#7b4c42',
    'MUNIWASTE': '#757501',
    'BIOMASS': '#006460',
    'HEAT': '#a5e982',
    'NATGAS': '#850017',
    'NATGAS-CCS':'#d35050',
    'OTHER': '#f7f7f7',
    'SOLAR': '#fad254',
    'NUCLEAR': '#cd6f00',
    'LIGNITE': '#2b1d1d',
    'HYDROGEN': '#dbdcec',
}

#get the names
if VIEW_Option == 'TECH_TYPE':
    df_tech_names = df['TECH_TYPE'].unique()
    df_tech_names_sorted = np.sort(df_tech_names)
    df_tech_names = df_tech_names_sorted

if VIEW_Option == 'FFF':   
    df_tech_names = df['FFF'].unique()
    df_tech_names_sorted = np.sort(df_tech_names)
    df_tech_names = df_tech_names_sorted    

#start filtering if you do not show all the years
if Do_not_Show_all_years:
   df= df[df['Y'] == Year]
   
#Exclude years
df= df[df['Y'] >= Exclude_years]


#sum electricity generation
df_sum = pd.DataFrame(df.groupby(['Y', 'Scenario','SSS','TTT',VIEW_Option])['value'].sum().reset_index())

#make index of time steps
seasons = df_sum['SSS'].unique()
time_step = df_sum['TTT'].unique()


# Pivot the DataFrame
df_pivot = df_sum.pivot(index='FFF', columns=['TTT','SSS'], values='value')
df_transposed = df_pivot.T
df_transposed = df_transposed.reset_index(drop=True)
df_transposed.index = df_transposed.index + 1
#Make them GWh
df_transposed = df_transposed/1000
#df_transposed['time'] = range(1, len(df_transposed) + 1)


#-------------------------------------------------------------------------------------------------
#handle hydrogen production

#start filtering if you do not show all the years
if Do_not_Show_all_years:
   df_demand= df_demand[df_demand['Y'] == Year]
   df_hydrogen = df_hydrogen[df_hydrogen ['Y'] == Year]
   
#Exclude years
df_demand= df_demand[df_demand['Y'] >= Exclude_years]
df_hydrogen= df_hydrogen[df_hydrogen['Y'] >= Exclude_years]

#sum electricity generation
df_demand_sum = pd.DataFrame(df_demand.groupby(['Y', 'Scenario','SSS','TTT'])['value'].sum().reset_index())

#make index of time steps

#df_demand_sum['time'] = df_demand_sum.index + 1
df_hydrogen_sum = pd.DataFrame(df_hydrogen.groupby(['Y', 'Scenario','SSS','TTT'])['value'].sum().reset_index())

#make them GWh
df_hydrogen_sum['value'] = df_hydrogen_sum['value']/1000

#-------------------------------------------------------------------------------------------------
#plot

#df_plot= df_sum[['time', 'FFF', 'value']]


# Plot the transposed DataFrame
ax1 = df_transposed.plot(kind='bar', stacked=True, figsize=(14, 6), color=[df_color_tech.get(x, '#f7f7f7') for x in df_transposed.columns])

'''
# Create a twin Axes object
ax2 = ax1.twinx()
# Plot the values from df_demand_sum on the secondary axis
df_demand_sum['value'].plot(ax=ax2, color='red')
# Set the y-axis label for the secondary axis
ax2.set_ylabel('Demand Value', color='red')
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.3))
ax2.legend(loc='upper right')

'''
ax3 = ax1.twinx()
df_hydrogen_sum['value'].plot(ax=ax3, color='black',label='Electrolysis Production')
ax3.legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=2,
            borderaxespad=0, frameon=False)
ax3.set_ylabel('Hydrogen production (GWh)', color='black')


# Move the legend outside of the figure box
ax1.legend(loc='center', bbox_to_anchor=(0.5,-0.20) ,ncol=6, frameon=False)
ax1.set_ylabel('Electricity generation (GWh)', color='black')
ax1.set_xlabel('Simulation time steps', color='black')
#ax1.yaxis.grid(True, linestyle='dotted',color='#b0b0b0')

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_color('#DDDDDD')

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.spines['bottom'].set_color('#DDDDDD')

# Set the number of x-axis tick labels
num_ticks = 12

# Modify the x-axis tick labels
x_labels = df_transposed.index[::len(df_transposed.index)//num_ticks]
ax1.set_xticks(x_labels)

# Show the plot
#plt.show()

#save plot
map_name =  'Generation_mix' + '_' + COMMODITY +  str(Year)
    

# Make Transmission_Map output folder
if not os.path.isdir('output/BarPlots'):
    os.makedirs('output/BarPlots')
        
output_dir = 'output/BarPlots'
plt.savefig(output_dir + '/' +  map_name + '.png', dpi=300, bbox_inches='tight')
# Show the plot
plt.show()
