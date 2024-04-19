#%%
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

COMMODITY = 'H2' #Choose from: ['Electricity', 'H2', 'Heat'.'All']. Add to csv-files name (only relevant if filetype_input == 'csv'). If 'Other': go to cell 1.4.0.
VIEW = 'storage_vol' #options to  production, storage_vol, capacity, network,  etc look at the dir
Take_distances_into_account = True #only for plotting network
Do_not_Show_all_years= False # focus on one year accross scenarios or multiple
All_countries = False #does not work for every country yet
Year = 2050 #Year to be, plot only one! 
Exclude_years =2030 #Exclude years before e.g. 2030
VIEW_Option = 'TECH_TYPE' #options FFF or TECH_TYPE to aggregate the values
#options only for network comparison
if VIEW == 'network': 
    VIEW_Option = 'network'
    #colour for network
    if COMMODITY == 'Electricity' :
        net_flow_colour = '#9AD8C7'
    elif COMMODITY == 'H2' : 
        net_flow_colour = '#EA9999'    
        
Do_not_include_importH2 = 'NO' #yes or no
Show_labels = False
manual_change_the_order_of_bars = True #HUSK! you need to make a list with the scenario and the order!
#specific_country = 'UNITED_KINGDOM'
#specify the order of the bars and the scenario names
#bar_order = ['Myopic','Limited_20','Limited_30','Perfect']
#bar_order = ['H2E','GH2E','SSGH2E']
#bar_order = ['H2E','H2E_75','H2E_50','H2E_25']
bar_order = ['H2E','H2E_60%','H2E_50%','H2E_40%','H2E_30%','H2E_20%','H2E_10%']
#bar_order = ['H2E','H2E_10%','H2E_20%','H2E_30%','H2E_40%','H2E_50%','H2E_60%']
#bar_order = ['H2E','H2E_20%','H2E_40%','H2E_60%','H2E_80%','H2E_100%']
#bar_order = ['GH2E','GH2E_10%','GH2E_20%','GH2E_30%','GH2E_40%','GH2E_50%','GH2E_60%','GH2E_70%','GH2E_80%','GH2E_90%','GH2E_100%']
#bar_order = ['GH2E','GH2E_20%','GH2E_40%','GH2E_60%','GH2E_80%','GH2E_100%']
#bar_order = ['H2E','H2E_10%','H2E_20%','H2E_30%','H2E_40%','H2E_50%','H2E_60%','H2E_70%','H2E_80%','H2E_90%','H2E_100%']
#bar_order = ['H2E_-60%','H2E_-50%','H2E_-40%','H2E_-30%','H2E_-20%','H2E_-10%','H2E','H2E_10%','H2E_20%','H2E_30%']

#bar_order = ['H2E_-30%','H2E_-20%','H2E_-10%','H2E','H2E_10%','H2E_20%','H2E_30%']



#%%
#----------------------------------------------------------------------------------------------------------
#start

if VIEW == "production":
    df = pd.read_csv(project_dir/'results/InvestmentExcel/PRO_YCRAGF.csv', delimiter=';')
elif VIEW == "capacity":
    df = pd.read_csv(project_dir/'results/InvestmentExcel/G_CAP_YCRAF.csv', delimiter=';')
elif VIEW == "storage_vol":
    df = pd.read_csv(project_dir/'results/InvestmentExcel/G_STO_YCRAF.csv', delimiter=';')
    df['value'] = df['value']/1000
elif VIEW == "network":
    if COMMODITY == 'Electricity': 
        df = pd.read_csv(project_dir/'results/InvestmentExcel/X_CAP_YCR.csv', delimiter=';')
    if COMMODITY == 'H2':
        df = pd.read_csv(project_dir/'results/InvestmentExcel/XH2_CAP_YCR.csv', delimiter=';')
    if Take_distances_into_account:
        if COMMODITY == 'Electricity':
            df_grid_length = pd.read_csv(project_dir/'input_data/ElectricityGridDistances.csv', delimiter=',')
        elif COMMODITY =='H2':
            df_grid_length = pd.read_csv(project_dir/'input_data/HydrogenGridDistances.csv', delimiter=',')


# if production or capacity take out the storage options 
#also take out the import capacities
if not  VIEW == "network":
    if VIEW == 'production' or VIEW == 'capacity':
        df = df[df['TECH_TYPE'] != 'H2-STORAGE']
        df = df[df['TECH_TYPE'] != 'INTERSEASONAL-HEAT-STORAGE']
        df = df[df['TECH_TYPE'] != 'INTRASEASONAL-HEAT-STORAGE']
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
    elif COMMODITY == 'H2':
        df = df[df['COMMODITY'] == 'HYDROGEN']
    elif COMMODITY == 'Heat':
        df = df[df['COMMODITY'] == 'HEAT']

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

if VIEW == "network":
    #start filtering if you do not show all the years
    if Do_not_Show_all_years:
        df= df[df['Y'] == Year]
   
    #Exclude years
    df= df[df['Y'] >= Exclude_years]    
    
    #Merge the information of lengths with the original data frame
    if Take_distances_into_account:
        merged_df = df.merge(df_grid_length, on=['IRRRE', 'IRRRI'], how='left')
        #pass back the information
        df = merged_df
        #create a new view
        df['Capacity_length'] = df['value'] * df['length_km']
        #convert to TWkm
        df['Capacity_length'] = df['Capacity_length']/1000 #convert to TWh perhaps devide by 2 double counting

    


#%% Plot
#sum over specific cols
if not  VIEW == "network":
    if All_countries:
        df_sum = pd.DataFrame(df.groupby(['C','Y', 'Scenario',VIEW_Option])['value'].sum().reset_index()) 
        #some times some capacities are close to zero but with a negative make them o
        df_sum.loc[(df_sum['value'] < 0) & (df_sum['value'] > -0.0001), 'value'] = 0
        df_pivot = df_sum.pivot(index=['Y', 'Scenario',VIEW_Option], columns='C', values='value') 
        
        
    else:
        df_sum = pd.DataFrame(df.groupby(['Y', 'Scenario',VIEW_Option])['value'].sum().reset_index())
        #some times some capacities are close to zero but with a negative make them o
        df_sum.loc[(df_sum['value'] < 0) & (df_sum['value'] > -0.0001), 'value'] = 0
        df_pivot = df_sum.pivot(index=['Y', 'Scenario'], columns= VIEW_Option, values='value')
        #in case of nan
        df_pivot = df_pivot.fillna(0)
    
        # Create the stacked bar plot
        #--------------------------------------------------------------------------------------------------
        #diffrent types of plot based on year and scenario
        
        # Set Year and Scenario as index
        #df_pivot = df_pivot.set_index(['Y', 'Scenario'])
        
        # Define colors based on VIEW_Option
        if VIEW_Option == 'TECH_TYPE':
            unique_options = df_sum['TECH_TYPE'].unique()
        elif VIEW_Option == 'FFF':
            unique_options = df_sum['FFF'].unique()
        else:
            unique_options = []

        # Create a dictionary to map unique options to colors
        color_mapping = {option: df_color_tech.get(option, 'gray') for option in unique_options}

        # Apply the colors to each option in df_pivot
        colors_df = [color_mapping.get(option, 'gray') for option in df_pivot.columns]

        if manual_change_the_order_of_bars:     
            df_pivot_reordered = df_pivot.reindex(bar_order, level=1)
            df_pivot = df_pivot_reordered

        
        ax = df_pivot.plot(kind='bar', stacked=True, figsize=(8, 5), color=colors_df)    

        # Set the axis labels and title
        ax.set_xlabel('Year, Scenario')
        if VIEW == 'production':
            ax.set_ylabel('TWh')    
        elif VIEW == 'storage_vol':
            ax.set_ylabel('TWh')
        else:
            ax.set_ylabel('GW')
    
        # Add black bars on the x-axis to indicate year changes
        x_coordinates = [i - 0.5 for i in range(len(df_pivot))]
        year_changes = [y[0] for y in df_pivot.index]
        for i in range(len(year_changes) - 1):
            if year_changes[i] != year_changes[i + 1]:
                ax.axvline(x_coordinates[i] + 1 , linestyle=(0, (2, 4)),color='#b0b0b0')
                
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#DDDDDD')
        ax.tick_params(bottom=False, left=False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle='dotted',color='#b0b0b0')
        ax.xaxis.grid(False)
        pos_tech = (0,1)
        legend = ax.legend(loc='lower left', ncol = 3,frameon=True,
                              mode='expnad',bbox_to_anchor=pos_tech)
        legend.get_frame().set_edgecolor('white')



        #ax.set_title('Stacked Bar Plot of Scenarios by Year and Country')

        # Loop through each stacked bar and add the total on top of the bar with a dot
        if Show_labels:
            for i, (scenario, year) in enumerate(df_pivot.index):
                # Get the total of the stacked bar
                total = df_pivot.loc[(scenario, year)].sum()
                
                # Add the text
                ax.text(i, total + 2, str(round(total)), ha='center', fontsize=10)
                # Add a dot on top of the bar
                ax.scatter(i, total, s=20, c='black', marker='o')
                
        
        
        
        #save plot
        map_name =  COMMODITY + '_' + VIEW + '_' + VIEW_Option
    

        # Make Transmission_Map output folder
        if not os.path.isdir('output/BarPlots'):
            os.makedirs('output/BarPlots')
        
        output_dir = 'output/BarPlots'
        plt.savefig(output_dir + '/' +  map_name + '.png', dpi=300, bbox_inches='tight')
        
        # Show the plot
        plt.show()

#-------------------------------------------------------------------------------------------------------------------
#plot the networks

if VIEW == "network":        
    if All_countries:
        df_sum = pd.DataFrame(df.groupby(['C','Y', 'Scenario'])['Capacity_length'].sum().reset_index()) 
        #some times some capacities are close to zero but with a negative make them o
        df_sum.loc[(df_sum['Capacity_length'] < 0) & (df_sum['Capacity_length'] > -0.0001), 'Capacity_length'] = 0
        #df_pivot = df_sum.pivot(index=['Y', 'Scenario',VIEW_Option], columns='C', values='value') 
        
        
    else:
        df_sum = pd.DataFrame(df.groupby(['Y', 'Scenario'])['Capacity_length'].sum().reset_index())
        #some times some capacities are close to zero but with a negative make them o
        df_sum.loc[(df_sum['Capacity_length'] < 0) & (df_sum['Capacity_length'] > -0.0001), 'Capacity_length'] = 0
        df_pivot = df_sum.pivot_table(index=['Y','Scenario'], values='Capacity_length')
        # Rename the column
        new_column_name = f'Total {COMMODITY} Network'
        df_pivot.rename(columns={'Capacity_length': new_column_name}, inplace=True)
        #in case of nan
        df_pivot = df_pivot.fillna(0)
        
        unique_options = net_flow_colour

        if manual_change_the_order_of_bars:     
            df_pivot_reordered = df_pivot.reindex(bar_order, level=1)
            df_pivot = df_pivot_reordered

        
        ax = df_pivot.plot(kind='bar', stacked=True, figsize=(8, 5), color = unique_options)  
        
        # Set the axis labels and title
        ax.set_xlabel('Year, Scenario')
        if VIEW == 'production':
            ax.set_ylabel('TWh')    
        elif VIEW == 'storage_vol':
            ax.set_ylabel('TWh')
        else:
            ax.set_ylabel('TWkm')
    
        # Add black bars on the x-axis to indicate year changes
        x_coordinates = [i - 0.5 for i in range(len(df_pivot))]
        year_changes = [y[0] for y in df_pivot.index]
        for i in range(len(year_changes) - 1):
            if year_changes[i] != year_changes[i + 1]:
                ax.axvline(x_coordinates[i] + 1 , linestyle=(0, (2, 4)),color='#b0b0b0')
                
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#DDDDDD')
        ax.tick_params(bottom=False, left=False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle='dotted',color='#b0b0b0')
        ax.xaxis.grid(False)
        pos_tech = (0,1)
        legend = ax.legend(loc='lower left', ncol = 3,frameon=True,
                              mode='expnad',bbox_to_anchor=pos_tech)
        legend.get_frame().set_edgecolor('white')



        #ax.set_title('Stacked Bar Plot of Scenarios by Year and Country')

        # Loop through each stacked bar and add the total on top of the bar with a dot
        if Show_labels:
            for i, (scenario, year) in enumerate(df_pivot.index):
                # Get the total of the stacked bar
                total = df_pivot.loc[(scenario, year)].sum()
                
                # Add the text
                ax.text(i, total + 2, str(round(total)), ha='center', fontsize=10)
                # Add a dot on top of the bar
                ax.scatter(i, total, s=20, c='black', marker='o')
                
        
        
        
        #save plot
        map_name =  COMMODITY + '_' + VIEW + '_' + VIEW_Option
    

        # Make Transmission_Map output folder
        if not os.path.isdir('output/BarPlots'):
            os.makedirs('output/BarPlots')
        
        output_dir = 'output/BarPlots'
        plt.savefig(output_dir + '/' +  map_name + '.png', dpi=300, bbox_inches='tight')
        
        # Show the plot
        plt.show()

    
    

    
    


   
