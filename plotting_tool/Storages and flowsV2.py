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
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
#from matplotlib.colors import to_rgb
# import atlite # It worked to import it in console, and then outcomment here
# from csv import reader

# import json
# from descartes import PolygonPatch

#from mpl_toolkits.basemap import Basemap as Basemap

from pathlib import Path
import sys
import os
import glob

#choose correct version of Gams
sys.path.append(r'C:\GAMS\36\apifiles\Python\api_38')
sys.path.append(r'C:\GAMS\36\apifiles\Python\gams')

from gams import GamsWorkspace


# In[ ]:

#recall some manual changes eg two lines max flow 0 (ask iokoun@dtu.dk)
manual_changes= True    
### Set options here.
#Structural options     
filetype_input = 'gdx' #Choose input file type: 'gdx' or 'csv' 
gams_dir = 'C:/GAMS/39' #Only required if filetype_input == 'gdx'
market = 'Investment' #Choose from ['Balancing', 'DayAhead', 'FullYear', 'Investment']
COMMODITY = 'H2' #Choose from: ['Electricity', 'H2', 'Heat']. Add to csv-files name (only relevant if filetype_input == 'csv'). If 'Other': go to cell 1.4.0.
SCENARIO = 'CAVERNS' #Add scenario to read file name
YEAR = 'all' #Add year to read file name (e.g. '2025', '2035', 'full')
SUBSET = 'all' #Add subset to read file name (e.g. 'full')
year =2050 #Year to be displayed
LINES = 'Capacity' #Choose from: ['Capacity', 'Flow', 'CongestionFlow']. For 'CongestionFlow', exo_end automatically switches to 'Total'.
exo_end = 'Total' # Choose from ['Endogenous', 'Exogenous', 'Total']. For 'CongestionFlow', exo_end automatically switches to 'Total'.
S = 'S02' #Season 
T = 'T073' #Hour 
Take_distances_into_account=True #convert the grid from GW to TWkm


#storages options define mix max '
if COMMODITY == 'H2':
    if year <= 2040:
        min_storage=0
        max_storage=5
    elif year > 2040:
        min_storage=0
        max_storage=15
    if year <= 2030:
        min_storage=0
        max_storage=2
    
    

flow_plot = True
plot_variable = 'Congestion' # options Capacity or Congestion

#Visual options transmission lines hydrogen or electricity
label_min = 0 #Minimum transmission capacity (GW) shown on map in text
font_line = 12 #Font size of transmission line labels
font_hub = 12 #Font size of hub labels
font_region = 10 #Font size of region labels
line_decimals = 1 #Number of decimals shown for line capacities
line_width_constant = 3 #Constant related to thickness of lines: the higher the number, the narrower the lines will be
flowline_breaks = [0, 40, 94.999, 100] #Breaks for different congestion categories
legend_values = ['Fully congested', '40-95% congested', '< 50% congested'] #Values displayed in legend
cat = 'linear' # 'linear' = Capacities are scaled linearly, 'cluster' = capacities are clustered in groups, small cluster c!
show_flow_arrows = 'YES' #'YES' or 'NO', net flow arrow, fix the arrows can be quite big
show_label = 'NO'   # 'YES' or 'NO'

if plot_variable == 'Capacity':
    if COMMODITY == 'Electricity' :
        cluster_groups = [1, 5, 10, 15] # The capacity groupings if cat is 'cluster'
        cluster_widths = [1, 5, 10, 15] # The widths for the corresponding capacity group (has to be same size as cluster_groups)  
    elif COMMODITY == 'H2':
        cluster_groups = [1, 5, 10, 20] # The capacity groupings if cat is 'cluster'
        cluster_widths = [1, 5, 10, 20] # The widths for the corresponding capacity group (has to be same size as cluster_groups)

if plot_variable == 'Congestion':
    if COMMODITY == 'Electricity' :
        cluster_groups = [1, 5, 10, 15] # The capacity groupings if cat is 'cluster'
        cluster_widths = [1, 5, 10, 15] # The widths for the corresponding capacity group (has to be same size as cluster_groups)  
    elif COMMODITY == 'H2':
        cluster_groups = [1, 5, 10, 20] # The capacity groupings if cat is 'cluster'
        cluster_widths = [1, 5, 10, 20] # The widths for the corresponding capacity group (has to be same size as cluster_groups)
       

    

#Scale up the width in case of linear the legend
if cat == 'linear':
    cluster_widths = [(1/line_width_constant) * i for i in cluster_widths]
    




#colour for network
if COMMODITY == 'Electricity' :
    net_colour = 'YlGn'
elif COMMODITY == 'H2' : 
    net_colour = 'Blues'
elif COMMODITY == 'Heat' : 
    net_colour = 'YlOrRd'
    
#colour for network
if COMMODITY == 'Electricity' :
    net_flow_colour = '#6fdc7a'
elif COMMODITY == 'H2' : 
    net_flow_colour = '#13EAC9'    
    
    
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
# ### 1.3 Read geographic files

project_dir = Path('.\input')

#Load coordinates files 
df_unique = pd.read_csv(project_dir/'geo_files/coordinates_RRR.csv')
df_region = df_unique.loc[df_unique['Type'] == 'region', ]
df_bypass = pd.read_csv(project_dir/'geo_files/bypass_lines.csv') # coordinates of 'hooks' in indirect lines, to avoid going trespassing third regions

#Define names of geojson and shapefile layers
r_in = list(df_unique.loc[(df_unique['Display'] == 1) & (df_unique['Type'] == 'region'), 'RRR'])
r_out = list(df_unique.loc[(df_unique['Display'] == 0) & (df_unique['Type'] == 'region'), 'RRR'])

layers_in = {region: '' for region in r_in}
layers_out = {region: '' for region in r_out}

#Create dictionaries with layer names for each region; if both a shapefile and geojson file are available for one region, the geojson file is used. 
for region in r_in:
    layers_in[region] = glob.glob(f'{project_dir}/geo_files/geojson_files/'+ region + '.geojson')
    if bool(layers_in[region]) == False:
        layers_in[region] = glob.glob(f'{project_dir}/geo_files/shapefiles/'+ region + '.shp')
for region in r_out:
    layers_out[region] = glob.glob(f'{project_dir}/geo_files/geojson_files/'+ region + '.geojson')
    if bool(layers_out[region]) == False:
        layers_out[region] = glob.glob(f'{project_dir}/geo_files/shapefiles/'+ region + '.shp')

for region in layers_in:
    layers_in[region] = str(layers_in[region])[2:-2] #Remove brackets from file names
for region in layers_out:
    layers_out[region] = str(layers_out[region])[2:-2] #Remove brackets from file names

    
#Convert shapefiles to geojson files  
for region in layers_out:
    if layers_out[region][-4:] == '.shp':
        gpd.read_file(layers_out[region]).to_file(f'{project_dir}/geo_files/geojson_files/'+ region + '.geojson', driver='GeoJSON')
        layers_out[region] = layers_out[region].replace('shapefiles', 'geojson_files').replace('.shp', '.geojson')

if Take_distances_into_account:
    if COMMODITY == 'Electricity':
        df_grid_length = pd.read_csv(project_dir/'input_data/ElectricityGridDistances.csv', delimiter=',')
    elif COMMODITY == 'H2':
        df_grid_length = pd.read_csv(project_dir/'input_data/HydrogenGridDistances.csv', delimiter=',')
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

    

     
# In[ ]:

#you can add more views here and pick up them in the later
if filetype_input == 'gdx':
    if COMMODITY == 'Electricity':
        var_list = []
        var_list = var_list + ['G_STO_YCRAF', 'X_CAP_YCR', 'X_FLOW_YCR']
    if COMMODITY == 'H2':
        var_list = [] 
        var_list = var_list + ['G_STO_YCRAF', 'XH2_CAP_YCR','XH2_FLOW_YCR']
    if COMMODITY == 'Heat':
        var_list = [] 
        var_list = var_list + ['G_STO_YCRAF']



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
    
    #Recover data
        if COMMODITY == 'Electricity':
            df_storages = all_df['G_STO_YCRAF']
            df_storages = df_storages[df_storages['COMMODITY'] == 'ELECTRICITY']
            df_capacity = all_df['X_CAP_YCR']
            df_flow = all_df['X_FLOW_YCR']

        if COMMODITY == 'H2':
            df_storages = all_df['G_STO_YCRAF']
            df_storages = df_storages[df_storages['COMMODITY'] == 'HYDROGEN']
            df_capacity = all_df['XH2_CAP_YCR']
            df_flow = all_df['XH2_FLOW_YCR']
        
        if COMMODITY == 'Heat':
            df_storages =  all_df['G_STO_YCRAF']
            df_storages = df_storages[df_storages['COMMODITY'] == 'HEAT']
         

        

# In[ ]: data for translines and flows
column_dict = {'Val':'Value', 'Y':'Year', 'C':'Country'}
if LINES == 'Capacity' or LINES == 'CongestionFlow':
    df_capacity = df_capacity.rename(columns = column_dict)
    df_flow = df_flow.rename(columns = column_dict)
if LINES == 'Flow' or LINES == 'CongestionFlow':
    df_flow = df_flow.rename(columns = column_dict)


#Replace possible "Eps" with 0
df_capacity.Value=df_capacity.Value.replace('Eps', 0)
df_capacity.Value=pd.to_numeric(df_capacity.Value)
df_flow.Value=df_flow.Value.replace('Eps', 0)
df_flow.Value=pd.to_numeric(df_flow.Value)


# In[ ]:

#Transmission Capacities
if LINES == 'Capacity' or LINES == 'CongestionFlow': #Skip this cell in case LINES == 'Flow'
    df_capacity['Year'] = df_capacity['Year'].astype(int)
    df_flow['Year'] = df_flow['Year'].astype(int)
    df_capacity = df_capacity.loc[df_capacity['Year'] == year, ].reset_index(drop = True) #Keep only data from year of interest
    df_flow = df_flow.loc[df_flow['Year'] == year, ].reset_index(drop = True) #Keep only data from year of interest
    if exo_end == 'Total' or LINES == 'CongestionFlow':
        col_keep = list(np.delete(np.array(df_capacity.columns),np.where((df_capacity.columns == 'VARIABLE_CATEGORY') |                                     (df_capacity.columns == 'Value')) )) #Create list with all columns except 'Variable_Category' and 'Value'
        df_capacity = pd.DataFrame(df_capacity.groupby(col_keep)['Value'].sum().reset_index() )#Sum exogenous and endogenous capacity for each region
    if exo_end == 'Endogenous' and LINES != 'CongestionFlow':
        df_capacity = df_capacity.loc[df_capacity['VARIABLE_CATEGORY'] == 'ENDOGENOUS', ]
    if exo_end == 'Exogenous' and LINES != 'CongestionFlow':
        df_capacity = df_capacity.loc[df_capacity['VARIABLE_CATEGORY'] == 'EXOGENOUS', ]

    for i,row in df_capacity.iterrows():
        for j in range(0,len(df_unique)):
            if df_capacity.loc[i,'IRRRE'] == df_unique.loc[j, 'RRR']:
                df_capacity.loc[i,'LatExp'] = df_unique.loc[j, 'Lat']
                df_flow.loc[i,'LatExp'] = df_unique.loc[j, 'Lat']
                df_capacity.loc[i,'LonExp'] = df_unique.loc[j, 'Lon']
                df_flow.loc[i,'LonExp'] = df_unique.loc[j, 'Lon']
            if df_capacity.loc[i,'IRRRI'] == df_unique.loc[j, 'RRR']:
                df_capacity.loc[i,'LatImp'] = df_unique.loc[j, 'Lat']
                df_flow.loc[i,'LatImp'] = df_unique.loc[j, 'Lat']
                df_capacity.loc[i,'LonImp'] = df_unique.loc[j, 'Lon']
                df_flow.loc[i,'LonImp'] = df_unique.loc[j, 'Lon']
    if len(df_capacity) == 0:
        print("Error: No capacity found. Check year and exo_end.")
        sys.exit()
        
    

# In[ ]:
    
df_try_cap = df_capacity
df_try_flow = df_flow

column_dict = {'Value' : 'Flow'}
#change the colloumn name save flow
df_try_flow = df_flow.rename(columns = column_dict)

#merged the two data frames
merged_df = pd.merge(df_try_cap, df_try_flow[['Flow','IRRRE', 'IRRRI',]], on=['IRRRE', 'IRRRI'], 
                     how='left')
#replace the Nan with zeros meaning there was not flow from some regions to regions should be EPS
merged_df['Flow'] = merged_df['Flow'].fillna(0)


#save the merged_df with a slack
df_tot = merged_df

#create a new coloumn with the groups of flows
df_tot['edge'] = merged_df[['IRRRE', 'IRRRI']].apply(lambda x: ','.join(sorted(x)), axis=1)
#for every group find the max
df_edge_group = df_tot.groupby(by='edge')['Flow'].max()
#keep that in a dictionary
max_flow_dict = df_edge_group.to_dict()

#function checks if the dictionary data correspond with the every row flow
def check_max_flow(edge, flow):
    max_flow = max_flow_dict.get(edge, None)
    if max_flow is None:
        return 'NA'
    if flow == max_flow:
        return 'True'
    else:
        return 'False'

merged_df['Max_flow'] = merged_df.apply(lambda x: check_max_flow(x['edge'], x['Flow']), axis=1)
              
  
#pass it back to the df_capacity frame to continue
df_capacity = merged_df
total_trans_capacity = df_capacity['Value'].sum()



#change the capacities of the pipes with the total flow value in TWh
if plot_variable == 'Congestion':
    #sum the lines with the same edge
    total_flow = df_capacity.groupby(['edge'])['Flow'].sum().reset_index()
    #Change name of col
    total_flow = total_flow.rename(columns={'Flow': 'Total Flow'})
    #merge to a new df
    new_df_capacity = pd.merge( df_capacity,total_flow, on=['edge'])
    #estimate congestion
    new_df_capacity['Congestion'] = new_df_capacity['Total Flow']*1000/(new_df_capacity['Value']*168*52)
    #Pass back elements to df_capacity so to map
    df_capacity = new_df_capacity
    df_capacity['Capacity Pipe'] = new_df_capacity['Value']
    #estimate the sum
    total_trans_capacity = df_capacity['Capacity Pipe'].sum()
    df_capacity['Value'] = new_df_capacity['Congestion']
    df_capacity['Value'] = new_df_capacity['Congestion']*100
    if manual_changes:
        if year > 2040:
            df_capacity.loc[(df_capacity['Country'] == 'BELGIUM') & (df_capacity['IRRRE'] == 'BE') & (df_capacity['IRRRI'] == 'UK'), 'Max_flow'] = False

    
# In[]:

#prepare to estimate GWkm pipes

if Take_distances_into_account:  
    # Merge df_capacity and df_grid_length based on 'IRRRE' and 'IRRRI'
    merged_df_capacity = pd.merge(df_capacity, df_grid_length, on=['IRRRE', 'IRRRI'])
    df_capacity = merged_df_capacity
    df_capacity['Capacity_length'] = df_capacity['Capacity Pipe'] * df_capacity['length_km']
    total_trans_capacity = df_capacity['Capacity_length'].sum()/1000 #convert to TWh perhaps devide by 2 double counting



# In[ ]:
# ### 2.5 Add bypass coordinates for indirect lines



if LINES == 'Capacity':
    df_bypass = pd.merge(df_bypass, df_capacity[['Year', 'Country', 'IRRRE', 'IRRRI', 'UNITS', 'Value','Flow','Max_flow','Capacity Pipe']], on = ['IRRRE', 'IRRRI'], how = 'left')
    #Replace existing row by 2 bypass rows
    keys = list(df_bypass.columns.values)[0:2]
    i1 = df_capacity.set_index(keys).index
    i2 = df_bypass.set_index(keys).index
    df_capacity = df_capacity[~i1.isin(i2)] #Delete existing rows that need bypass
    df_capacity = df_capacity.append(df_bypass, ignore_index = True, sort = True) #Append bypass rows
    

# In[ ]:
# ### 2.7 One direction capacity  lines
#this code does not work btw IK you change the matrix inside the second loop?
#perhaps remove in the future no work


#When capacity is not the same in both directions, display one:
for i,row in df_capacity.iterrows():
    for k,row in df_capacity.iterrows():
        if (df_capacity.loc[k,'IRRRE'] == df_capacity.loc[i,'IRRRI']) & (df_capacity.loc[k,'IRRRI'] == df_capacity.loc[i,'IRRRE']) & (df_capacity.loc[k,'Value'] != df_capacity.loc[i,'Value']):
            df_capacity.loc[i,'Value'] = df_capacity.loc[k,'Value']


# In[ ]:
# ###  2.8 Define line centers



#Define centre of each transmission line
if LINES == 'Flow' or LINES == 'CongestionFlow': #Skip this cell in case LINES == 'Capacity'
    df_flow['LatMid'] = (df_flow['LatImp'] + df_flow['LatExp']) /2
    df_flow['LonMid'] = (df_flow['LonImp'] + df_flow['LonExp']) /2
if LINES == 'Capacity' or LINES == 'CongestionFlow': #Skip this cell in case LINES == 'Flow'
    df_capacity['LatMid'] = (df_capacity['LatImp'] + df_capacity['LatExp']) /2
    df_capacity['LonMid'] = (df_capacity['LonImp'] + df_capacity['LonExp']) /2

    
 

# In[ ]:
#Prepare data 
sum_column = 'VARIABLE_CATEGORY'        
#Keep the year and other info    
df_storages.loc[:, 'Y'] = df_storages['Y'].astype(int)
df_storages = df_storages[df_storages['Y'] == year]     

#sum values
df_storages = pd.DataFrame(df_storages.groupby(['RRR', sum_column])['Value'].sum().reset_index())

#sum values
df_storages = pd.DataFrame(df_storages.groupby(['RRR'])['Value'].sum().reset_index())

#if you do not have a number put zero otherwise the map looks weird.
df_unique_in = df_unique[df_unique['Display']==1]
  
    

# In[ ]:

# Load the geojson files into a GeoDataFrame
geo_df = gpd.GeoDataFrame()
for R in layers_in:
    df = gpd.read_file(layers_in[R])
    df['RRR'] = R
    geo_df = geo_df.append(df)


# Merge the demand values with the GeoDataFrame that takes some time and most likely we can save it and load it again and again
merged_df = geo_df.merge(df_storages, on='RRR', how='left')
#if they do not have a storage investment put zero
merged_df['Value'] = merged_df['Value'].fillna(0)
#turned to TWh
merged_df['Value'] = merged_df['Value'] / 1000





# In[ ]:
#plot

# Create the choropleth map
### 3.1 Plotting the regions

total_storage = merged_df['Value'].sum()

cbar_label = f'{COMMODITY} Storage TWh ({year})'

projection = ccrs.EqualEarth()

fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"projection": projection}, dpi=100)

'''
for R in layers_in:
    geo = gpd.read_file(layers_in[R])
    ax.add_geometries(geo.geometry, crs = projection,
                  facecolor=['#E5E5E5'], edgecolor='grey',
                  linewidth=.2)
'''
# Set limit always after pies because it brakes the graph
ax.set_xlim(-11,32)      
ax.set_ylim(35,72)


    
merged_df.plot(column='Value', cmap=net_colour, vmin=min_storage, 
                   vmax=max_storage, facecolor=[.9, .9,.9], edgecolor='grey',
linewidth=.2, ax=ax)

ax.axis('off')

#define min max check that every time


'''
log_norm = colors.LogNorm(vmin=min_storage, vmax=max_storage)
sm = plt.cm.ScalarMappable(cmap=net_colour, norm=log_norm)
'''


# Add legend
sm = plt.cm.ScalarMappable(cmap=net_colour, 
                           norm=plt.Normalize(vmin=min_storage, 
                                              vmax=max_storage))


cbar = fig.colorbar(sm, ax=ax, location='left', orientation='vertical', fraction=0.03, pad=0.02, extend = 'max' )
cbar.ax.tick_params(labelsize=10)
cbar.set_label(cbar_label,fontsize=12)
ax.text(0.02, 0.92, f'Total {COMMODITY} storage: {total_storage:.1f}TWh', transform=ax.transAxes, fontsize=12)
#ax.set_title('Demand Values by RRR')

#--------------------------------------------------------------------------------------------
#make the flows

### 3.2 Adding transmission lines
# A function for finding the nearest value in an array
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


if Take_distances_into_account:
    ax.text(0.02, 0.88, f'Total {COMMODITY} network: {total_trans_capacity:.1f}TWkm', transform=ax.transAxes, fontsize=12)
else:
    ax.text(0.02, 0.88, f'Total {COMMODITY} network: {total_trans_capacity:.1f}GW', transform=ax.transAxes, fontsize=12)

    

if flow_plot:
    lines = []
    if plot_variable == 'Congestion':
        
        cmap = plt.cm.get_cmap('Reds') # You can choose any other colormap that you like
        normalize = plt.Normalize(vmin=df_capacity['Value'].min(), vmax=df_capacity['Value'].max())

        #Plot tran lines either for H2 or Electricity, options such as linear plot or cluster are available look the begging
        lines = []
        for i,row in df_capacity.iterrows():
            y1 = df_capacity.loc[i,'LatExp']
            x1 =  df_capacity.loc[i,'LonExp']
            y2 = df_capacity.loc[i,'LatImp']
            x2 = df_capacity.loc[i,'LonImp']
            cap = df_capacity.loc[i,'Value']
            # Calculate the color of the line based on investment cost
            color = cmap(normalize(cap))
            cap_line = df_capacity.loc[i,'Capacity Pipe']
            
            if cat == 'cluster':
                nearest = find_nearest(cluster_groups, cap_line) 
                width = np.array(cluster_widths)[cluster_groups == nearest]
            else:
                width = cap_line/line_width_constant
                #some times is a samll line lets clasiffied within the last category
                if width < cluster_widths[0]:
                    width = cluster_widths[0]
            
            #plt plot line
            l, = ax.plot([x1,x2], [y1,y2], color = color, solid_capstyle='round', solid_joinstyle='round', 
                             linewidth = width, zorder=1)
            #save line information
            lines.append(l)
            
            #plt arrows for the direction of flow  if its yes as option and plot only the net direction True/False
            if show_flow_arrows == 'YES':
                if df_capacity.loc[i,'Value'] >= label_min:
                    if df_capacity.loc[i,'Max_flow'] == 'True':
                        head_length = 1.75*1.5/10
                        head_width = 1*1.5/10 
                        #Choose arrow style
                        arrow = "-|>"
                        joinstyle = 'round'
                        style = ''.join([arrow,',head_length =',str(head_length),',head_width =',str(head_width)])
                        #plot the arrow
                        ax.annotate("", xytext=((x1+x2)/2,(y1+y2)/2),xy=((x1+x2)/2+0.01*(x2-x1),(y1+y2)/2+0.01*(y2-y1)), 
                                    arrowprops=dict(arrowstyle= style,color= 'black',joinstyle ='round'))

                    #arrow = FancyArrowPatch((x1+x2)/2,(y1+y2)/2,(x1+x2)/2+0.001*(x2-x1),(y1+y2)/2+0.001*(y2-y1),
                    #                       arrowstyle="->", color=net_colour)
                    #ax.add_patch(arrow)
            if show_label == 'YES':
                if df_capacity.loc[i,'Capacity Pipe'] >= label_min:
                    label = "{:.1f}".format(df_capacity.loc[i,'Capacity Pipe'])
                    plt.annotate(label, # this is the value which we want to label (text)
                        (df_capacity.loc[i,'LonMid'],df_capacity.loc[i,'LatMid']), # x and y is the points location where we have to label
                        textcoords="offset points",
                        xytext=(0,-4), # this for the distance between the points
                        # and the text label
                        ha='center',
                        )
            
            
        ax.axis('off')

        # Plot the colorbar for legend
        '''
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=df_capacity['Value'].min(), 
                                                                 vmax=df_capacity['Value'].max()))
        '''
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, 
                                                                 vmax=100))

        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.02, extend = 'max' )
        cbar.ax.tick_params(labelsize=10)
        cbar_label = f'{COMMODITY} Line utilization ({year}) [%]'
        cbar.set_label(cbar_label,fontsize=12)
        
        ### Making a legend for trans lines H2 or electricity        
        if COMMODITY == 'Electricity':
            subs = 'el'
            if cat == 'cluster' or cat == 'linear':
                # Create lines for legend
                lines = []
                string = []
                for i in range(len(cluster_groups)):
                    # The patch
                    lines.append(Line2D([0], [0], linewidth=cluster_widths[i],
                                        color='dimgrey'))
                    # The text
                    if i == 0:
                        ave = cluster_groups[i]
                        string.append('%0.1f GW$_\mathrm{%s}$'%(ave, subs))
                    elif i == len(cluster_groups)-1:
                        ave = cluster_groups[i]
                        string.append('%0.1f GW$_\mathrm{%s}$'%(ave, subs))
                    else:
                        ave0 = cluster_groups[i]
                        string.append('%0.1f GW$_\mathrm{%s}$'%(ave0, subs))
                
                ax.legend(lines, string,frameon=False, loc='upper left', bbox_to_anchor=(0, 0.85)) 




        if COMMODITY == 'H2':
            subs = 'H2'
            if cat == 'cluster' or cat == 'linear':
                # Create lines for legend
                lines = []
                string = []
                for i in range(len(cluster_groups)):
                    # The patch
                    lines.append(Line2D([0], [0], linewidth=cluster_widths[i],
                                        color='dimgrey'))
                    # The text
                    if i == 0:
                        ave = cluster_groups[i]
                        string.append('%0.1f GW$_\mathrm{%s}$'%(ave, subs))
                    elif i == len(cluster_groups)-1:
                        ave = cluster_groups[i]
                        string.append('%0.1f GW$_\mathrm{%s}$'%(ave, subs))
                    else:
                        ave0 = cluster_groups[i]
                        string.append('%0.1f GW$_\mathrm{%s}$'%(ave0, subs))
                
                ax.legend(lines, string,frameon=False, loc='upper left', bbox_to_anchor=(0, 0.85))




#--------------------------------------------------------------------------------------------
#make the flows if its capacity
        
    if plot_variable == 'Capacity':
        for i,row in df_capacity.iterrows(): 
            y1 = df_capacity.loc[i,'LatExp']
            x1 =  df_capacity.loc[i,'LonExp']
            y2 = df_capacity.loc[i,'LatImp']
            x2 = df_capacity.loc[i,'LonImp']        
            cap = df_capacity.loc[i,'Value']
            #slope of line 
            m = (y2-y1)/(x2-x1)
            
            if cat == 'cluster':
                nearest = find_nearest(cluster_groups, cap) 
                width = np.array(cluster_widths)[cluster_groups == nearest]
            else:
                width = cap/line_width_constant


            # Print an error message, if capacity is a NaN value
            #Plot the lines
            if not(np.isnan(cap)):
                #plt plot line
                l, = ax.plot([x1,x2], [y1,y2], color = net_flow_colour, solid_capstyle='round', solid_joinstyle='round', 
                             linewidth = width, zorder=1)
                #save line information
                lines.append(l)
                
                #plt arrows for the direction of flow  if its yes as option and plot only the net direction True/False
                if show_flow_arrows == 'YES':
                    if df_capacity.loc[i,'Value'] >= label_min:
                        if df_capacity.loc[i,'Max_flow'] == 'True':
                            #be carefull width is a array that why access the element
                            if cat == 'cluster':
                                head_length = 1.25*width[0]/10
                                head_width = 0.75*width[0]/10 
                            else:
                                head_length = 1.25*width/10
                                head_width = 0.75*width/10
                            #Choose arrow style
                            arrow = "-|>"
                            joinstyle = 'round'
                            style = ''.join([arrow,',head_length =',str(head_length),',head_width =',str(head_width)])
                            #plot the arrow
                            ax.annotate("", xytext=((x1+x2)/2,(y1+y2)/2),xy=((x1+x2)/2+0.01*(x2-x1),(y1+y2)/2+0.01*(y2-y1)), 
                                        arrowprops=dict(arrowstyle= style,color= net_flow_colour,joinstyle ='round'))

                        #arrow = FancyArrowPatch((x1+x2)/2,(y1+y2)/2,(x1+x2)/2+0.001*(x2-x1),(y1+y2)/2+0.001*(y2-y1),
                        #                       arrowstyle="->", color=net_colour)
                        #ax.add_patch(arrow)
                    
            else:
                print("There's a NaN value in line\nIRRRE %s\nIRRRI %s"%(df_capacity.loc[i, 'IRRRE'], df_capacity.loc[i, 'IRRRI']))

            # Add labels to  if its activated 
            
            if show_label == 'YES':
                if df_capacity.loc[i,'Value'] >= label_min:
                    label = "{:.1f}".format(df_capacity.loc[i,'Value'])
                    plt.annotate(label, # this is the value which we want to label (text)
                    (df_capacity.loc[i,'LonMid'],df_capacity.loc[i,'LatMid']), # x and y is the points location where we have to label
                    textcoords="offset points",
                    xytext=(0,-4), # this for the distance between the points
                    # and the text label
                     ha='center',
                     )
                    #,arrowprops=dict(arrowstyle="->", color='green'))
    if plot_variable == 'Capacity':
        ### Making a legend for trans lines H2 or electricity        
        if COMMODITY == 'Electricity':
            subs = 'el'
            if cat == 'cluster' or cat == 'linear':
                # Create lines for legend
                lines = []
                string = []
                for i in range(len(cluster_groups)):
                    # The patch
                    lines.append(Line2D([0], [0], linewidth=cluster_widths[i],
                                        color=net_flow_colour))
                    # The text
                    if i == 0:
                        ave = cluster_groups[i]
                        string.append('%0.1f GW$_\mathrm{%s}$'%(ave, subs))
                    elif i == len(cluster_groups)-1:
                        ave = cluster_groups[i]
                        string.append('%0.1f GW$_\mathrm{%s}$'%(ave, subs))
                    else:
                        ave0 = cluster_groups[i]
                        string.append('%0.1f GW$_\mathrm{%s}$'%(ave0, subs))
                
                ax.legend(lines, string,frameon=False, loc='upper left', bbox_to_anchor=(0, 0.77)) 




        if COMMODITY == 'H2':
            subs = 'H2'
            if cat == 'cluster' or cat == 'linear':
                # Create lines for legend
                lines = []
                string = []
                for i in range(len(cluster_groups)):
                    # The patch
                    lines.append(Line2D([0], [0], linewidth=cluster_widths[i],
                                        color=net_flow_colour))
                    # The text
                    if i == 0:
                        ave = cluster_groups[i]
                        string.append('%0.1f GW$_\mathrm{%s}$'%(ave, subs))
                    elif i == len(cluster_groups)-1:
                        ave = cluster_groups[i]
                        string.append('%0.1f GW$_\mathrm{%s}$'%(ave, subs))
                    else:
                        ave0 = cluster_groups[i]
                        string.append('%0.1f GW$_\mathrm{%s}$'%(ave0, subs))
                
                ax.legend(lines, string,frameon=False, loc='upper left', bbox_to_anchor=(0, 0.85))



            

#--------------------------------------------------------------------------------------------
#Save map
map_name = COMMODITY
year = str(year)
if not os.path.isdir('output/Storages_flows/'  + SCENARIO + '/' + market):
    os.makedirs('output/Storages_flows/'  + SCENARIO + '/' + market)


output_dir = 'output/Storages_flows/'  + SCENARIO + '/' + market

plt.savefig(output_dir + '/' +  map_name + year + '.png', dpi=300, bbox_inches='tight')
                  
plt.show()               

            

'''
#--------------------------------------------------------------------------------------------
#Save map
map_name = COMMODITY
year = str(year)

# Make Transmission_Map output folder
if not os.path.isdir('output/Storages_flows/'):
        os.makedirs('output/Storages_flows')
        
output_dir = 'output/Storages_flows'
plt.savefig(output_dir + '/' +  map_name + year + SCENARIO + '.png', dpi=300, bbox_inches='tight')

plt.show()
'''
