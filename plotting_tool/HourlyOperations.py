#!/usr/bin/env Balmorel
import math
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import xarray as xr
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import ArrowStyle
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
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

# # 1 Preparations

### Set options here.
#Structural options     
filetype_input = 'gdx' #Choose input file type: 'gdx' or 'csv' 
gams_dir = 'C:/GAMS/39' #Only required if filetype_input == 'gdx'
market = 'Investment' #Choose from ['Balancing', 'DayAhead', 'FullYear', 'Investment']
COMMODITY = 'H2' #Choose from: ['Electricity', 'H2', 'Other']. Add to csv-files name (only relevant if filetype_input == 'csv'). If 'Other': go to cell 1.4.0.
SCENARIO = 'endofmodel' #Add scenario to read file name , tje endofmodel to get the storage data
Second_Scenario = 'IMPORT_NOBLUE_CAVERNS' #typical balmorel output
YEAR = 'all' #Add year to read file name (e.g. '2025', '2035', 'full')
SUBSET = 'all' #Add subset to read file name (e.g. 'full')
year = 2050 #Year to be displayed
LINES = 'Capacity' #Choose from: ['Capacity', 'Flow', 'CongestionFlow']. For 'CongestionFlow', exo_end automatically switches to 'Total'.
exo_end = 'Total' # Choose from ['Endogenous', 'Exogenous', 'Total']. For 'CongestionFlow', exo_end automatically switches to 'Total'.


subset_S = False
S_keep = ['S28','S34'] #Season 'S02', 'S08', 'S15', 'S21', 'S28', 'S34', 'S41', 'S47'
T = 'T073' #Hour 


#Choose Country or region 
Region = 'IT'  #'NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'DK1', 'DK2', 'FIN', 'DE4-E', 'DE4-N', 'DE4-S', 'DE4-W', 'NL', 'SE1', 'SE2', 'SE3', 'SE4', 'UK',
              #'EE', 'LV', 'LT', 'PL', 'BE', 'FR', 'IT', 'CH', 'AT', 'CZ', 'ES', 'PT'
#for storage info need area
Area = 'IT_A'   #'DE4-E_A', 'DE4-N_A', 'DE4-S_A', 'DE4-W_A', 'EE_Tallinn', 'LT_Other_DH', 'LV_RigaR', 
                #'NL_A', 'PL_A', 'UK_A', 'FI_large',
               #'NO1_A2', 'NO2_A2', 'NO3_A2', 'NO4_A2', 'NO5_A2', 'SE1_medium',
               #'SE2_medium', 'SE3_large', 'SE4_large', 'DK1_Large', 'DK2_Large',
               #'BE_A', 'FR_A', 'IT_A', 'CH_A', 'AT_A', 'CZ_A', 'ES_A', 'PT_A

# In[ ]:


def read_paramenter_from_gdx(ws,gdx_name,parameter_name,**read_options):
     
     for item in read_options.items():
         if item[0]=="field":
                     field=item[1]

     
     db = ws.add_database_from_gdx(gdx_name)
     
     if "field" in locals() :
         if field=="Level":
             par=dict( (tuple(rec.keys), rec.level) for rec in db[parameter_name] )
         elif field=="Marginal":
             par=dict( (tuple(rec.keys), rec.marginal) for rec in db[parameter_name] )
         elif field=="Lower":
             par=dict( (tuple(rec.keys), rec.lower) for rec in db[parameter_name] )
         elif field=="Upper":
                 par=dict( (tuple(rec.keys), rec.lower) for rec in db[parameter_name] )
         elif field=="Scale":
                     par=dict( (tuple(rec.keys), rec.lower) for rec in db[parameter_name] )
         elif field=="Value":
                     par=dict( (tuple(rec.keys), rec.value) for rec in db[parameter_name] )
     else:
         if "Parameter" in str(type(db[parameter_name])):
             par=dict( (tuple(rec.keys), rec.value) for rec in db[parameter_name] )
         elif "Variable" in str(type(db[parameter_name])):
             par=dict( (tuple(rec.keys), rec.level) for rec in db[parameter_name] )
         elif "Set" in str(type(db[parameter_name])):
             par=dict( (tuple(rec.keys), rec.text) for rec in db[parameter_name] )
         elif "Equation" in str(type(db[parameter_name])):
             par=dict( (tuple(rec.keys), rec.level) for rec in db[parameter_name] )
             
     return par , db[parameter_name].get_domains_as_strings()
 


def dataframe_from_gdx(gdx_name,parameter_name,**read_options):
     
     ws = GamsWorkspace(os.getcwd(),)


     var, cols= read_paramenter_from_gdx(ws,gdx_name,parameter_name,**read_options)
     if "custom_domains" in read_options :
         cols= read_options["custom_domains"]

     
     unzip_var= list(zip(*var))
     
     new_dict=dict()
     i=0
     for col in cols:
         new_dict[col]= list(unzip_var[i])
         i=i+1
         
     
     if "field" in read_options :
         field= read_options.get("field")
         new_dict[field]=[]
         new_dict[field]=list(var.values())
     else:
         new_dict["Value"]=list(var.values())
         
     df=pd.DataFrame.from_dict(new_dict)

     return df
 



# In[ ]:


if filetype_input == 'gdx':
    def df_creation(gdx_file, varname):
        df = pd.DataFrame()
        if '_' in gdx_file:
                # if yes: extract scenario name from gdx filename
            scenario = gdx_file.split('_', 3)[-3]
            year = gdx_file.split('_', 3)[-2]
            subset = gdx_file.split('_', 3)[-1][:-4]
            market = gdx_file.split('\\', 1)[0].split('/',3)[-1]
        else:
               # if no: use nan instead
            scenario = 'nan'

        # create empty temporary dataframe and load the gdx data into it
        temp = pd.DataFrame()
        # temp = gdxpds.to_dataframe(gdx_file, varname, gams_dir=gams_dir,
        #                        old_interface=False)
        
        temp=dataframe_from_gdx(gdx_file,varname)

        # add a scenario column with the scenario name of the current iteration
        temp['Scenario'] = scenario
        temp['Market']  = market
        temp['run'] = scenario + '_' + year + '_' + subset

        # rearrange the columns' order
        cols = list(temp.columns)
        cols = [cols[-1]] + cols[:-1]
        temp = temp[cols]

        # concatenate the temporary dataframe to the preceeding data
        df = pd.concat([df, temp], sort=False)
        return df


# ##### 1.4A.2 - Define var_list

# In[ ]:


if filetype_input == 'gdx':
    if COMMODITY == 'Electricity':
        var_list = []
        var_list = var_list + ['VHYDROGEN_STOVOL_T ', 'VHYDROGEN_STOLOADT', 'VHYDROGEN_GH2_T']
    if COMMODITY == 'H2':
        var_list = []
        var_list = var_list + ['VHYDROGEN_STOVOL_T','VHYDROGEN_STOLOADT', 'VHYDROGEN_GH2_T']



# ##### 1.4A.3 - Use function to read inputs

# In[ ]:


if filetype_input == 'gdx':
    runs = list()
    gdx_file_list = list()

    # directory to the input gdx file(s)
    #gdx_file_list = gdx_file_list + glob.glob('./input/results/'+ market + '/*.gdx')
    
    gdx_file =  glob.glob('./input/results/'+ market + '\\MainResults_' + SCENARIO + '_'  + YEAR + '_' + SUBSET + '.gdx')
    gdx_file = gdx_file[0]

    all_df = {varname: df for varname, df in zip(var_list,var_list)}


    for varname, df in zip(var_list, var_list):
        all_df[varname] = df_creation(gdx_file, varname)
        if all_df[varname]['run'][0] not in runs:
            runs.append(all_df[varname]['run'][0])

    #run_dict = dict(zip(gdx_file_list, runs) )
    #all_df = dict((run_dict[key], value) for (key, value) in all_df.items())
        

# In[ ]:
#load the other gdx   
 
SCENARIO = Second_Scenario  


# In[ ]:


if filetype_input == 'gdx':
    def df_creation(gdx_file, varname):
        df = pd.DataFrame()
        if '_' in gdx_file:
                # if yes: extract scenario name from gdx filename
            scenario = gdx_file.split('_', 3)[-3]
            year = gdx_file.split('_', 3)[-2]
            subset = gdx_file.split('_', 3)[-1][:-4]
            market = gdx_file.split('\\', 1)[0].split('/',3)[-1]
        else:
               # if no: use nan instead
            scenario = 'nan'

        # create empty temporary dataframe and load the gdx data into it
        temp = pd.DataFrame()
        # temp = gdxpds.to_dataframe(gdx_file, varname, gams_dir=gams_dir,
        #                        old_interface=False)
        
        temp=dataframe_from_gdx(gdx_file,varname)

        # add a scenario column with the scenario name of the current iteration
        temp['Scenario'] = scenario
        temp['Market']  = market
        temp['run'] = scenario + '_' + year + '_' + subset

        # rearrange the columns' order
        cols = list(temp.columns)
        cols = [cols[-1]] + cols[:-1]
        temp = temp[cols]

        # concatenate the temporary dataframe to the preceeding data
        df = pd.concat([df, temp], sort=False)
        return df


# ##### 1.4A.2 - Define var_list

# In[ ]:


if filetype_input == 'gdx':
    if COMMODITY == 'Electricity':
        var_list2 = []
        var_list2 = var_list2 + ['VHYDROGEN_STOVOL_T ', 'VHYDROGEN_STOLOADT', 'VHYDROGEN_GH2_T']
    if COMMODITY == 'H2':
        var_list2 = []
        var_list2 = var_list2 + ['PRO_YCRAGFST']



# ##### 1.4A.3 - Use function to read inputs

# In[ ]:


if filetype_input == 'gdx':
    runs = list()
    gdx_file_list = list()

    # directory to the input gdx file(s)
    #gdx_file_list = gdx_file_list + glob.glob('./input/results/'+ market + '/*.gdx')
    
    gdx_file =  glob.glob('./input/results/'+ market + '\\MainResults_' + SCENARIO + '_'  + YEAR + '_' + SUBSET + '.gdx')
    gdx_file = gdx_file[0]

    all_df2 = {varname: df for varname, df in zip(var_list2,var_list2)}


    for varname, df in zip(var_list2, var_list2):
        all_df2[varname] = df_creation(gdx_file, varname)
        if all_df2[varname]['run'][0] not in runs:
            runs.append(all_df2[varname]['run'][0])

    #run_dict = dict(zip(gdx_file_list, runs) )
    #all_df = dict((run_dict[key], value) for (key, value) in all_df.items())


# In[ ]:

#prepare data          
    #Transmission capacity data
    if COMMODITY == 'Electricity':
        df_GNR_capacity = all_df['PRO_YCRAGF']
        df_GNR_capacity = df_GNR_capacity[df_GNR_capacity['COMMODITY'] == 'ELECTRICITY']
    if COMMODITY == 'H2':
        df_prod = all_df2['PRO_YCRAGFST']
        df_prod = df_prod[df_prod['COMMODITY'] == 'HYDROGEN']
        #clear some memory
        all_df2=0
        
#keep data        

#Keep the year and other info    
df_prod.loc[:, 'Y'] = df_prod['Y'].astype(int)
df_prod = df_prod[df_prod['Y'] == year]

#keep RRR
df_prod = df_prod[df_prod['RRR'] == Region]

#take into account the import_H2
F_to_tech_type = {
    'IMPORT_H2': 'IMPORT_H2',
}

df_prod['TECH_TYPE'] = df_prod['FFF'].map(F_to_tech_type).fillna(df_prod['TECH_TYPE'])
  


#Distinguish if has CCS or not for hydrogen
G_to_tech_type = {
    'GNR_STEAM-REFORMING_E-70_Y-2020': 'SMR',
    'GNR_STEAM-REFORMING-CCS_E-70_Y-2020': 'SMR-CCS'
}
df_prod['TECH_TYPE'] = df_prod['G'].map(G_to_tech_type).fillna(df_prod['TECH_TYPE'])
    
#Change the name of H2-STORAGE to H2_STOR_DIS
T_to_tech_type = {
    'H2-STORAGE': 'H2_STOR_DIS',
}
df_prod['TECH_TYPE'] = df_prod['TECH_TYPE'].map(T_to_tech_type).fillna(df_prod['TECH_TYPE'])
   

# for storages information from endoffmodel 
df_hydrogen_storage_charge = all_df['VHYDROGEN_STOLOADT']
df_hydrogen_storage_volume = all_df['VHYDROGEN_STOVOL_T']  

df_hydrogen_storage_volume = df_hydrogen_storage_volume.rename(columns={"G": "GGG","SSS":"S"})

#Keep the year and other info    
df_hydrogen_storage_charge.loc[:, 'Y'] = df_hydrogen_storage_charge['Y'].astype(int)
df_hydrogen_storage_volume.loc[:, 'Y'] = df_hydrogen_storage_volume['Y'].astype(int)
df_hydrogen_storage_charge = df_hydrogen_storage_charge[df_hydrogen_storage_charge['Y'] == year]
df_hydrogen_storage_volume = df_hydrogen_storage_volume[df_hydrogen_storage_volume['Y'] == year]

#keep AAA
df_hydrogen_storage_charge = df_hydrogen_storage_charge[df_hydrogen_storage_charge['AAA'] == Area]
df_hydrogen_storage_volume = df_hydrogen_storage_volume[df_hydrogen_storage_volume['AAA'] == Area]

slack_1 = df_hydrogen_storage_volume[df_hydrogen_storage_volume['GGG'] == 'GNR_H2S_H2-CAVERN_Y-2050']
#try
plt.plot(slack_1['Value'])

#Keep all the time steps
T = df_hydrogen_storage_charge['T'].unique()
S =df_hydrogen_storage_charge['S'].unique()

#Keep necessary columns
df_prod = df_prod.loc[:, ['Y','G','FFF','TECH_TYPE','SSS','TTT','Value']]

df_with_more_elements = df_prod

# Define function to create missing combinations and fill with 0
def fill_missing(group):
    # Create all possible combinations of SSS and TTT
    S = ['S02', 'S08', 'S15', 'S21', 'S28', 'S34', 'S41', 'S47']
    T = ['T073', 'T076', 'T079', 'T082', 'T085', 'T088', 'T091', 'T094',
         'T097', 'T100', 'T103', 'T106', 'T109', 'T112', 'T115', 'T118',
         'T121', 'T124', 'T127', 'T130', 'T133', 'T136', 'T139', 'T142']
    index = pd.MultiIndex.from_product([S, T], names=['SSS', 'TTT'])
    index_df = pd.DataFrame(index=index).reset_index()

    # Merge with group and fill missing values with 0
    merged_df = pd.merge(index_df, group, on=['SSS', 'TTT'], how='outer')
    merged_df['Value'] = merged_df['Value'].fillna(0)

    return merged_df

# Group by G and apply fill_missing function to each group
grouped_df = df_with_more_elements.groupby(['G', 'Y', 'FFF', 'TECH_TYPE']).apply(fill_missing)
grouped_df = grouped_df.drop(['G', 'Y', 'FFF', 'TECH_TYPE'], axis=1)
grouped_df = grouped_df.reset_index()
#

sum_grouped_df = grouped_df.groupby(['Y','SSS','TTT','TECH_TYPE','FFF'])['Value'].sum().reset_index()

df_with_more_elements = sum_grouped_df 
df_with_more_elements  = df_with_more_elements .loc[:, ['TECH_TYPE','SSS','TTT','Value']]


#Sum information for storages
sum_hydrgen_storage_charge = df_hydrogen_storage_charge.groupby(['Y','AAA','S','T'])['Value'].sum().reset_index()
sum_hydrgen_storage_charge  = sum_hydrgen_storage_charge .loc[:, ['S','T','Value']]
sum_hydrgen_storage_charge['TECH_TYPE'] = 'H2_STOR_CH'

sum_hydrogen_storage_volume = df_hydrogen_storage_volume.groupby(['Y','AAA','S','T'])['Value'].sum().reset_index()
sum_hydrogen_storage_volume = sum_hydrogen_storage_volume.loc[:, ['S','T','Value']]
sum_hydrogen_storage_volume['TECH_TYPE'] = 'H2_STOR_VOL'

#append
df_storages = pd.concat([sum_hydrogen_storage_volume, sum_hydrgen_storage_charge])
#adapt name 
df_storages = df_storages.rename(columns={"S": "SSS", "T": "TTT"})

#append with the original dataframe
df_all_h2 =pd.concat([df_with_more_elements,df_storages])

#create an index for time steps
short_df = pd.pivot_table(df_all_h2, values='Value', index=['SSS', 'TTT'], columns=['TECH_TYPE'])

short_df = short_df.reset_index()
short_df['Time'] = short_df.index.get_level_values(0) + 1

#return to a long format
long_df = pd.melt(short_df, id_vars=['Time', 'SSS', 'TTT'], var_name='TECH_TYPE', value_name='Value')

#keeep some information
if subset_S:
    long_df = long_df[long_df['SSS'].isin(S_keep)]
    
    


# In[ ]:
#plot

colors = {'ELECTROLYZER': '#1f77b4',
          'H2_STOR_DIS': '#ff7f0e',
          'IMPORT_H2': '#2ca02c',
          'H2_STOR_CH': '#d62728',
          'H2_STOR_VOL': 'black',
          'SMR-CCS': '#9467bd',
          'SMR': '#8c564b',
          }

fig, ax1 = plt.subplots(figsize=(12, 12))

# plot all TECH_TYPEs except H2_STOR_VOL
for tech_type, group in long_df[long_df['TECH_TYPE'] != 'H2_STOR_VOL'].groupby('TECH_TYPE'):
    ax1.plot(group['Time'], group['Value'], label=tech_type, color=colors[tech_type])

# plot H2_STOR_VOL on a secondary y-axis
ax2 = ax1.twinx()
h2_vol = long_df[long_df['TECH_TYPE'] == 'H2_STOR_VOL']
ax2.plot(h2_vol['Time'], h2_vol['Value'], color='black',label = 'H2_STOR_VOL')

# set labels and legend
ax1.set_xlabel('Time')
ax1.set_ylabel('MWh')
ax2.set_ylabel('HYDROGEN VOLUME MWh')

# Add a legend for the lines
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

#Save map
map_name = COMMODITY
year = str(year)

# Make Transmission_Map output folder
if not os.path.isdir('output/HourlyOperation'):
        os.makedirs('output/HourlyOperation')
        
output_dir = 'output/HourlyOperation'
plt.savefig(output_dir + '/' +  map_name + Region + 'all' + year + '.png', dpi=300, bbox_inches='tight')


plt.show()    




# In[ ]:

'''
#plot

# Get the unique SSS values
unique_sss = df_all_h2['SSS'].unique()

# Create a grid of subplots with two rows and a number of columns equal to the number of unique SSS divided by 2 (rounded up)
num_cols = (len(unique_sss) + 1) // 2
fig, axes = plt.subplots(nrows=2, ncols=num_cols, figsize=(num_cols*6, 12))

# Initialize min and max y-axis values
ymin = float('inf')
ymax = float('-inf')

# Loop through the unique SSS and plot each one in a different subplot
for i, sss in enumerate(unique_sss):
    
    # Get the axis for the current subplot
    row_idx = i % 2
    col_idx = i // 2
    ax1 = axes[row_idx, col_idx]
    
    # Create a secondary axis for H2_STOR_VOL
    ax2 = ax1.twinx()
    
    # Loop through every unique TECH_TYPE
    for tech_type in df_all_h2['TECH_TYPE'].unique():
        
        # Filter the dataframe for the current SSS and TECH_TYPE
        df_plot = df_all_h2[(df_all_h2['SSS'] == sss) & (df_all_h2['TECH_TYPE'] == tech_type)]
        
        # Plot the line for the current TECH_TYPE
        if tech_type == 'H2_STOR_VOL':
            ax2.plot(df_plot['TTT'], df_plot['Value'], color='black', label=tech_type)
        else:
            ax1.plot(df_plot['TTT'], df_plot['Value'], label=tech_type)
        
    
    # Set the title and labels for the plot
    ax1.set_title(f"SSS = {sss}")
    ax1.set_xlabel("TTT")
    ax1.set_ylabel("Value")
    ax2.set_ylabel("Value")
    
    # Add a legend for the lines
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

# Display the plot
plt.tight_layout()
plt.show()

'''
# In[ ]:
#plot

colors = {'ELECTROLYZER': '#1f77b4',
          'H2_STOR_DIS': '#ff7f0e',
          'IMPORT_H2': '#2ca02c',
          'H2_STOR_CH': '#d62728',
          'H2_STOR_VOL': 'black',
          'SMR-CCS': '#9467bd',
          'SMR': '#8c564b',
          }
# Get the unique SSS values
unique_sss = df_all_h2['SSS'].unique()

# Create a grid of subplots with two rows and a number of columns equal to the number of unique SSS divided by 2 (rounded up)
num_cols = (len(unique_sss) + 1) // 2
fig, axes = plt.subplots(nrows=2, ncols=num_cols, figsize=(num_cols*6, 12))

# Initialize min and max y-axis values
ymin1 = float('inf')
ymax1 = float('-inf')
ymin2 = float('inf')
ymax2 = float('-inf')

# Plot the line for the current TECH_TYPE keep sames axis

df_plot = df_all_h2[df_all_h2['TECH_TYPE'] == 'H2_STOR_VOL']
ymin2 = min(ymin2, df_plot['Value'].min())
ymax2 = max(ymax2, df_plot['Value'].max()) + max(ymax2, df_plot['Value'].max())/10

df_plot2 = df_all_h2[df_all_h2['TECH_TYPE'] != 'H2_STOR_VOL']
ymin1 = min(ymin1, df_plot2['Value'].min()) 
ymax1 = max(ymax1, df_plot2['Value'].max()) + max(ymax1, df_plot2['Value'].max())/10


# Loop through the unique SSS and plot each one in a different subplot
for i, sss in enumerate(unique_sss):
    
    # Get the axis for the current subplot
    row_idx = i // num_cols
    col_idx = i % num_cols
    ax1 = axes[row_idx, col_idx]
    
    # Create a secondary axis for H2_STOR_VOL
    ax2 = ax1.twinx()
    
    # Create empty lists to store handles and labels for ax1 and ax2
    handles1, labels1 = [], []
    handles2, labels2 = [], []
    

    # Loop through every unique TECH_TYPE
    for tech_type in df_all_h2['TECH_TYPE'].unique():

        # Filter the dataframe for the current SSS and TECH_TYPE
        df_plot = df_all_h2[(df_all_h2['SSS'] == sss) & (df_all_h2['TECH_TYPE'] == tech_type)]

        # Plot the line for the current TECH_TYPE
        if tech_type == 'H2_STOR_VOL':
           line, = ax2.plot(df_plot['TTT'], df_plot['Value'], color=colors[tech_type], label=tech_type)
           handles2.append(line)
           labels2.append(tech_type)
        else:
            line, = ax1.plot(df_plot['TTT'], df_plot['Value'],color=colors[tech_type], label=tech_type)
            handles1.append(line)
            labels1.append(tech_type)

    # Combine handles and labels for ax1 and ax2
    handles = handles1 + handles2
    labels = labels1 + labels2
    
    # Set the y-axis limits to be the same for all subplots
    ax1.set_ylim([ymin1, ymax1])
    ax2.set_ylim([ymin2, ymax2])
    
    
    # Set the title and labels for the plot
    ax1.set_title(f"SSS = {sss}")
    ax1.set_xlabel("TTT")
    ax1.tick_params(axis='x', labelsize=8, which='major', pad=1, length=2)
    ax1.set_ylabel("MW")
    ax2.set_ylabel("H2_STOR_VOL MWh")
    
    # Set x-axis ticks to show every 5th tick
    ax1.set_xticks(df_plot[::2]['TTT'])
    ax1.tick_params(axis='x', rotation=45)
    
    # Add a legend for the lines
    #ax1.legend(loc='upper left')
    #ax2.legend(loc='upper right')
    
# Create a legend for the lines
plt.subplots_adjust(bottom=0.2)
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.03), ncol=len(labels), fontsize=14)

# Display the plot
plt.tight_layout() 


#Save map
map_name = COMMODITY
year = str(year)

# Make Transmission_Map output folder
if not os.path.isdir('output/HourlyOperation'):
        os.makedirs('output/HourlyOperation')
        
output_dir = 'output/HourlyOperation'
plt.savefig(output_dir + '/' +  map_name + Region + year + '.png', dpi=300, bbox_inches='tight')

plt.show()


