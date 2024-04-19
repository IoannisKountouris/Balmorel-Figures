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
    
### Set options here.
#Structural options     
filetype_input = 'gdx' #Choose input file type: 'gdx' or 'csv' 
gams_dir = 'C:/GAMS/39' #Only required if filetype_input == 'gdx'
market = 'Investment' #Choose from ['Balancing', 'DayAhead', 'FullYear', 'Investment']
COMMODITY = 'H2' #Choose from: ['Electricity', 'H2', 'Heat']. Add to csv-files name (only relevant if filetype_input == 'csv'). If 'Other': go to cell 1.4.0.
SCENARIO = 'IMPORT_BLUE_CAVERNS' #Add scenario to read file name
YEAR = 'all' #Add year to read file name (e.g. '2025', '2035', 'full')
SUBSET = 'all' #Add subset to read file name (e.g. 'full')
year = 2050 #Year to be displayed
LINES = 'Capacity' #Choose from: ['Capacity', 'Flow', 'CongestionFlow']. For 'CongestionFlow', exo_end automatically switches to 'Total'.


flow_plot = True

#colour for network
if COMMODITY == 'Electricity' :
    net_colour = 'YlGn'
elif COMMODITY == 'H2' : 
    net_colour = 'RdPu'
elif COMMODITY == 'Heat' : 
    net_colour = 'YlOrRd'
    
    
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
        var_list = var_list + ['G_STO_YCRAF']
    if COMMODITY == 'H2':
        var_list = [] 
        var_list = var_list + ['G_STO_YCRAF']
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

        if COMMODITY == 'H2':
            df_storages = all_df['G_STO_YCRAF']
            df_storages = df_storages[df_storages['COMMODITY'] == 'HYDROGEN']
        
        if COMMODITY == 'Heat':
            df_storages =  all_df['G_STO_YCRAF']
            df_storages = df_storages[df_storages['COMMODITY'] == 'HEAT']
         
     

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

if year < 2040:
    min_storage=0
    max_storage=10
elif year >= 2040:
    min_storage=0
    max_storage=27
    
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


cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.02, extend = 'max' )
cbar.ax.tick_params(labelsize=10)
cbar.set_label(cbar_label,fontsize=12)
ax.text(0.02, 0.92, f'Total {COMMODITY} Storage: {total_storage:.0f} TWh', transform=ax.transAxes, fontsize=12)
#ax.set_title('Demand Values by RRR')

#--------------------------------------------------------------------------------------------
#make the flows


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

