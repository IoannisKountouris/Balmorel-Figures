

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


from pathlib import Path
import sys
import os
import glob

#choose correct version of Gams
sys.path.append(r'C:\GAMS\36\apifiles\Python\api_38')
sys.path.append(r'C:\GAMS\36\apifiles\Python\gams')

from gams import GamsWorkspace

# # 1 Preparations

# ### 1.1 Set Options

### Set options here.
#Structural options     
filetype_input = 'gdx' #Choose input file type: 'gdx' or 'csv' 
gams_dir = 'C:/GAMS/39' #Only required if filetype_input == 'gdx'
market = 'Investment' #Choose from ['Balancing', 'DayAhead', 'FullYear', 'Investment']
COMMODITY = 'H2' #Choose from: ['Electricity', 'H2', 'Heat']. Add to csv-files name (only relevant if filetype_input == 'csv'). If 'Other': go to cell 1.4.0.
YEAR = 'all' #Add year to read file name (e.g. '2025', '2035', 'full')
SUBSET = 'all' #Add subset to read file name (e.g. 'full')
year = 2050 #Year to be displayed
LINES = 'Capacity' #Choose from: ['Capacity', 'Flow', 'CongestionFlow']. For 'CongestionFlow', exo_end automatically switches to 'Total'.

Manual_scale = True #use manual scale or not, based min max

#colour for network
if COMMODITY == 'Electricity' :
    net_colour = 'YlGn'
elif COMMODITY == 'H2' : 
    net_colour = 'RdPu'


#storages options define mix max ' million euros max
if Manual_scale:
    if COMMODITY == 'Electricity':
        if year <= 2040:
            vmin=0
            vmax=1.4
        elif year > 2040:
            vmin=0
            vmax=1.4
        if year <= 2030:
            vmin=0
            vmax=1.4
    elif COMMODITY == 'H2': 
        if year <= 2040:
            vmin=0
            vmax=0.65
        elif year > 2040:
            vmin=0
            vmax=0.65
        if year <= 2030:
            vmin=0
            vmax=0.65
    
    
# In[ ]

project_dir = Path('.\input')

#Load coordinates files 

if COMMODITY == "H2":
    df_Trans_cost = pd.read_csv(project_dir/'input_data/HydrogenTransInvestmentsCostExtended.csv', delimiter=';')
elif COMMODITY == "Electricity":
    df_Trans_cost = pd.read_csv(project_dir/'input_data/ElectricityTransInvestmentsCostExtended.csv', delimiter=';')
    


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
#Prepare data 
sum_column = 'VARIABLE_CATEGORY'        
#Keep the year and other info    
df_Trans_cost.loc[:, 'YYY'] = df_Trans_cost['YYY'].astype(int)
df_Trans_cost = df_Trans_cost[df_Trans_cost['YYY'] == year]     





for i,row in df_Trans_cost.iterrows():
    for j in range(0,len(df_unique)):
        if df_Trans_cost.loc[i,'IRRRE'] == df_unique.loc[j, 'RRR']:
            df_Trans_cost.loc[i,'LatExp'] = df_unique.loc[j, 'Lat']
            df_Trans_cost.loc[i,'LonExp'] = df_unique.loc[j, 'Lon']
        if df_Trans_cost.loc[i,'IRRRI'] == df_unique.loc[j, 'RRR']:
            df_Trans_cost.loc[i,'LatImp'] = df_unique.loc[j, 'Lat']
            df_Trans_cost.loc[i,'LonImp'] = df_unique.loc[j, 'Lon']

if len(df_Trans_cost) == 0:
    print("Error: No capacity found. Check year and exo_end.")
    sys.exit()



# In[ ]:
# ### 2.5 Add bypass coordinates for indirect lines


df_bypass = pd.merge(df_bypass, df_Trans_cost[['YYY', 'IRRRE', 'IRRRI', 'Value']], on = ['IRRRE', 'IRRRI'], how = 'left')
#Replace existing row by 2 bypass rows
keys = list(df_bypass.columns.values)[0:2]
i1 = df_Trans_cost.set_index(keys).index
i2 = df_bypass.set_index(keys).index
df_Trans_cost = df_Trans_cost[~i1.isin(i2)] #Delete existing rows that need bypass
df_Trans_cost = df_Trans_cost.append(df_bypass, ignore_index = True, sort = True) #Append bypass rows

#filter turkey out
df_Trans_cost = df_Trans_cost[df_Trans_cost['IRRRE'] != 'TR']
df_Trans_cost = df_Trans_cost[df_Trans_cost['IRRRI'] != 'TR']

#make it million EUR
df_Trans_cost['Value'] = df_Trans_cost['Value']/1000000
# In[]



# Define the colormap
#cmap = plt.cm.get_cmap('cool')

### 3.1 Plotting the regions
projection = ccrs.EqualEarth()

fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"projection": projection}, dpi=100)

for R in layers_in:
    geo = gpd.read_file(layers_in[R])
    ax.add_geometries(geo.geometry, crs = projection,
                  facecolor=['#d3d3d3'], edgecolor='#46585d',
                  linewidth=.2)



for R in layers_out:
    geo = gpd.read_file(layers_out[R])
    ax.add_geometries(geo.geometry, crs = projection,
                  facecolor=['#d3d3d3'], edgecolor='#46585d',
                  linewidth=.2)
    
# Set limit always  the graph
ax.set_xlim(-11,35)      
ax.set_ylim(34,72)


#in case no manual take min max
if not Manual_scale:
    vmin=df_Trans_cost['Value'].min() 
    vmax=df_Trans_cost['Value'].max()
    


cmap = plt.cm.get_cmap(net_colour) # You can choose any other colormap that you like
normalize = plt.Normalize(vmin, vmax)


#Plot tran lines either for H2 or Electricity, options such as linear plot or cluster are available look the begging
lines = []
for i,row in df_Trans_cost.iterrows(): 
    y1 = df_Trans_cost.loc[i,'LatExp']
    x1 =  df_Trans_cost.loc[i,'LonExp']
    y2 = df_Trans_cost.loc[i,'LatImp']
    x2 = df_Trans_cost.loc[i,'LonImp']
    cap = df_Trans_cost.loc[i,'Value']
    # Calculate the color of the line based on investment cost
    color = cmap(normalize(cap))


    #plt plot line
    l, = ax.plot([x1,x2], [y1,y2], color = color, solid_capstyle='round', solid_joinstyle='round', 
                     linewidth = 2, zorder=1)
    #save line information
    lines.append(l)


ax.axis('off')


# Plot the colorbar for legend
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, 
                                                         vmax))

cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.02, extend = 'max' )
cbar.ax.tick_params(labelsize=10)
cbar_label = f'{COMMODITY} Transmission Investment cost ({year}) mEUR/MW'
cbar.set_label(cbar_label,fontsize=12)

#--------------------------------------------------------------------------------------------
#Save map
map_name = COMMODITY
year = str(year)

# Make Transmission_Map output folder
if not os.path.isdir('output/GridCost'):
        os.makedirs('output/GridCost')
        
output_dir = 'output/GridCost'
plt.savefig(output_dir + '/' +  map_name + year + '.png', dpi=300, bbox_inches='tight')

plt.show()






