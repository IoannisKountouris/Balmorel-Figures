#!/usr/bin/env python
# coding: utf-8

# # 1 Preparations

# ### 1.1 Set Options

# In[ ]:


### Set options here.
#Structural options
filetype_input = 'gdx' #Choose input file type: 'gdx' or 'csv' 
gams_dir = 'C:/GAMS/36' #Only required if filetype_input == 'gdx'
market = 'Investment' #Choose from ['Balancing', 'DayAhead', 'FullYear', 'Investment']
COMMODITY = 'H2' #Choose from: ['Electricity', 'H2', 'Other']. Add to csv-files name (only relevant if filetype_input == 'csv'). If 'Other': go to cell 1.4.0.
SCENARIO = 'V12_IMPORT_BLUEH2' #Add scenario to read file name
YEAR = 'all' #Add year to read file name (e.g. '2025', '2035', 'full')
SUBSET = 'all' #Add subset to read file name (e.g. 'full')
year = 2030 #Year to be displayed
LINES = 'Capacity' #Choose from: ['Capacity', 'Flow', 'CongestionFlow']. For 'CongestionFlow', exo_end automatically switches to 'Total'.
exo_end = 'Total' # Choose from ['Endogenous', 'Exogenous', 'Total']. For 'CongestionFlow', exo_end automatically switches to 'Total'.
S = 'S02' #Season 
T = 'T073' #Hour    

# hubs
hub_display = True
hub_size = 10
hub_decimals = 10 #Number of decimals shown for hub capacities
background_hubsize = True #Displaying the true size of the hub as a circle on the map.
hub_area = 7 #MW / km^2, background hub size on map. 
hub_area_opacity = 10.7 #Opacity of background hub size. 


#Visual options
label_min = 0.1 #Minimum transmission capacity (GW) shown on map in text
font_line = 12 #Font size of transmission line labels
font_hub = 12 #Font size of hub labels
font_region = 10 #Font size of region labels
line_decimals = 1 #Number of decimals shown for line capacities
line_width_constant = 1.5 #Constant related to thickness of lines: the higher the number, the narrower the lines will be
flowline_breaks = [0, 40, 94.999, 100] #Breaks for different congestion categories
legend_values = ['Fully congested', '40-95% congested', '< 50% congested'] #Values displayed in legend

#colors
background_color = 'white'
regions_ext_color = 'lightgrey'
regions_model_color = 'grey'
region_text = 'black'
capline_color = 'orange' #you can use orange or others green
flowline_color = ['#3D9200', '#feb24c','#960028']
line_text = 'black'
hub_color = 'lightblue'
hub_background_color = 'lightblue'
hub_text = 'black'


# ### 1.2 Import Packages

# In[ ]:


from pathlib import Path
import sys
import os
import glob
# import gdxpds
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium import plugins
from IPython.display import HTML, display
import json
from folium.features import DivIcon #For text labels on hubs
from IPython.display import display, HTML
from csv import reader

sys.path.append(r'C:\GAMS\36\apifiles\Python\api_38')
sys.path.append(r'C:\GAMS\36\apifiles\Python\gams')

from gams import GamsWorkspace


display(HTML(data="""
<style>
    div#notebook-container    { width: 95%; }
    div#menubar-container     { width: 65%; }
    div#maintoolbar-container { width: 99%; }
</style>
"""))



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


# # 1.4 Read run-specific files

# ##### 1.4.0 If COMMODITY == 'Other': define variables or file names

# In[ ]:


if COMMODITY == 'Other':
    if filetype_input == 'gdx':
        var_list = ['G_CAP_YCRAF', 'XH2_CAP_YCR', 'XH2_FLOW_YCRST', 'PRO_YCRAGFST'] #Fill in variables to read, e.g. ['G_CAP_YCRAF', 'X{COMMODITY}_CAP_YCR', 'X{COMMODITY}_FLOW_YCRST', 'PRO_YCRAGST']
    if filetype_input == 'csv':
        flow_file = 'FlowH2Hourly_'+ SCENARIO + '_' + YEAR + '_' + SUBSET + '.csv' #Fill in flow file name if applicable, e.g. 'Flow{COMMODITY}Hourly_'+ SCENARIO + '_' + YEAR + '_' + SUBSET + '.csv'
        transcap_file = 'CapacityH2Transmission_' + SCENARIO + '_' + YEAR + '_'+ SUBSET + '.csv' #Fill in transmission capacity file name, e.g. 'Capacity{COMMODITY}Transmission_'+ SCENARIO + '_' + YEAR + '_'+ SUBSET + '.csv' 


# ### 1.4A - GDX Inputs

# ##### 1.4A.1 Function: reading gdx-files

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
        if LINES == 'Capacity' or LINES == 'CongestionFlow' or LINES == 'Flow': 
            var_list = var_list + ['G_CAP_YCRAF', 'X_CAP_YCR']
        if LINES == 'Flow' or LINES == 'CongestionFlow':
            var_list = var_list + ['X_FLOW_YCRST']
        if hub_display == True:
            var_list = var_list + ['PRO_YCRAGFST']
    if COMMODITY == 'H2':
        var_list = []
        if LINES == 'Capacity' or LINES == 'CongestionFlow' or LINES == 'Flow': 
            var_list = var_list + ['G_CAP_YCRAF', 'XH2_CAP_YCR']
        if LINES == 'Flow' or LINES == 'CongestionFlow':
            var_list = var_list + ['XH2_FLOW_YCRST']
        if hub_display == True:
            var_list = var_list + ['PRO_YCRAGFST']


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
    
    #Transmission capacity data
    if LINES == 'Capacity' or LINES == 'CongestionFlow'  or LINES == 'Flow':
        if COMMODITY == 'Electricity':
            df_capacity = all_df['X_CAP_YCR']
        if COMMODITY == 'H2':
            df_capacity = all_df['XH2_CAP_YCR']
        if COMMODITY == 'Other':
            df_capacity = all_df[var_list[1]]

    #Transmission flow data
    if LINES == 'Flow' or LINES == 'CongestionFlow' : 
        if COMMODITY == 'Electricity':
            df_flow = all_df['X_FLOW_YCRST']
        if COMMODITY == 'H2':
            df_flow = all_df['XH2_FLOW_YCRST']
    if COMMODITY == 'Other':
        if LINES == 'Flow':
            df_flow = all_df[var_list[1]]
        if LINES == 'CongestionFlow':
            df_flow = all_df[var_list[2]]


# ##### 1.4A.4 - Hub data

# In[ ]:


if filetype_input == 'gdx' and hub_display == True:
    hub_windgen = (pd.read_csv(project_dir/'geo_files/hub_technologies.csv', sep = ',', quotechar = '"').hub_name) 
    df_capgen = all_df['G_CAP_YCRAF']
    if LINES == 'Flow' or LINES == 'CongestionFlow':
        df_hubprod = all_df['PRO_YCRAGFST']
        df_hubprod['Y'] = df_hubprod['Y'].astype(int)
        df_hubprod = df_hubprod.loc[(df_hubprod['G'].isin(hub_windgen)) & (df_hubprod['TECH_TYPE'] == 'WIND-OFF') &                                     (df_hubprod['Y']==year) & (df_hubprod['SSS'] == S) & (df_hubprod['TTT']==T), ]


# ### 1.4B1 - Read CSV files

# In[ ]:


map_name = 'Transmission' + COMMODITY + '_' + LINES + '_' + str(year) + '_Map.html'
if filetype_input == 'csv':
    generation_file = 'CapacityGeneration_'+  SCENARIO + '_' + YEAR + '_' + SUBSET + '.csv'
    if COMMODITY == 'Electricity':
        flow_file = 'FlowElectricityHourly_'+ SCENARIO + '_' + YEAR + '_' + SUBSET + '.csv'
        transcap_file = 'CapacityElectricityTransmission_'+ SCENARIO + '_' + YEAR + '_'+ SUBSET + '.csv'
    if COMMODITY == 'H2':
        flow_file = 'FlowH2Hourly_'+ SCENARIO + '_' + YEAR + '_' + SUBSET + '.csv'
        transcap_file = 'CapacityH2Transmission_'+ SCENARIO + '_' + YEAR + '_'+ SUBSET + '.csv'
     
    #Transmission capacity data
    df_capacity = pd.read_csv(str(project_dir) + '/results/' + str(market) + '/' + str(transcap_file), sep = ',', quotechar = '"') 
    #Transmission flow data
    if LINES == 'Flow' or LINES == 'CongestionFlow':
        df_flow = pd.read_csv(str(project_dir) + '/results/' + str(market) + '/' + str(flow_file), sep = ',', quotechar = '"')

    if hub_display == True:
        prod_file = 'ProductionHourly_'+ SCENARIO + '_' + YEAR + '_' + SUBSET + '.csv'
        hub_windgen = (pd.read_csv(project_dir/'geo_files/hub_technologies.csv', sep = ',', quotechar = '"').hub_name) 
        #Generation capacity data
        df_capgen = pd.read_csv(str(project_dir) + '/results/' + str(market) + '/' + str(generation_file), sep = ',', quotechar = '"') 
        if LINES == 'Flow' or LINES == 'CongestionFlow':
        #Hub production data
            df_hubprod = pd.read_csv(str(project_dir) + '/results/' + str(market) + '/' + str(prod_file), sep = ',', quotechar = '"') 
            df_hubprod = df_hubprod.loc[(df_hubprod['G'].isin(hub_windgen)) & (df_hubprod['TECH_TYPE'] == 'WIND-OFF') &                                         (df_hubprod['Y']==year) & (df_hubprod['SSS'] == S) & (df_hubprod['TTT']==T), ]


# ### 1.4B2 - Calibrate column names

# In[ ]:


column_dict = {'Val':'Value', 'Y':'Year', 'C':'Country'}
if LINES == 'Capacity' or LINES == 'CongestionFlow':
    df_capacity = df_capacity.rename(columns = column_dict)
if LINES == 'Flow' or LINES == 'CongestionFlow':
    df_flow = df_flow.rename(columns = column_dict)
if hub_display == True:
    df_capgen = df_capgen.rename(columns = column_dict)
    if LINES == 'Flow' or LINES == 'CongestionFlow': 
            df_hubprod = df_hubprod.rename(columns = column_dict)


# # 2 Processing of dataframes

# ### 2.1 Replace "EPS" with 0

# In[ ]:


#Replace possible "Eps" with 0
df_capacity.Value=df_capacity.Value.replace('Eps', 0)
df_capacity.Value=pd.to_numeric(df_capacity.Value)
if LINES == 'Flow' or LINES == 'CongestionFlow': #Skip this cell in case LINES == 'Capacity'
    df_flow.Value=df_flow.Value.replace('Eps', 0)
    df_flow.Value=pd.to_numeric(df_flow.Value)
if hub_display == True:
    df_capgen.Value=df_capgen.Value.replace('Eps', 0)
    df_capgen.Value=pd.to_numeric(df_capgen.Value)
    if LINES == 'Flow' or LINES == 'CongestionFlow':
        df_hubprod.Value=df_hubprod.Value.replace('Eps', 0)
        df_hubprod.Value=pd.to_numeric(df_hubprod.Value)


# ### 2.2 Add Coordinates + Select Time + Convert Units

# In[ ]:


#Flows
if LINES == 'Flow' or LINES == 'CongestionFlow': #Skip this cell in case LINES == 'Capacity'
    df_flow['Year'] = df_flow['Year'].astype(int)
    #Keep only data from moment of interest
    df_flow = df_flow.loc[df_flow['Year'] == year] 
    df_flow = df_flow.loc[df_flow['SSS'] == S,]
    df_flow = df_flow.loc[df_flow['TTT'] == T, ]
    for i,row in df_flow.iterrows():
        for j in range(0,len(df_unique)):
            if df_flow.loc[i,'IRRRE'] == df_unique.loc[j, 'RRR']:
                df_flow.loc[i,'LatExp'] = df_unique.loc[j, 'Lat']
                df_flow.loc[i,'LonExp'] = df_unique.loc[j, 'Lon']
            if df_flow.loc[i,'IRRRI'] == df_unique.loc[j, 'RRR']:
                df_flow.loc[i,'LatImp'] = df_unique.loc[j, 'Lat']
                df_flow.loc[i,'LonImp'] = df_unique.loc[j, 'Lon']

    #Convert flow from MWh to GWh
    df_flow['Value'] = df_flow['Value'] / 1000
    df_flow = df_flow.reset_index(drop = True)
    if len(df_flow) == 0:
        print("Error: Timestep not in data; check year, S and T.")
        sys.exit()


# ### 2.3 Group hub data

# In[ ]:


#Generation Capacities
if hub_display == True:
    df_capgen['Year'] = df_capgen['Year'].astype(int)
    # df_capgen = df_capgen.merge(df_unique, on = 'RRR', how = 'left', left_index = True).reset_index(drop = True) #Add coordinates of each region
    #poly
    df_capgen = df_capgen.merge(df_unique, on = 'RRR', how = 'left' ).reset_index(drop = True) #Add coordinates of each region
    df_capgen = df_capgen.loc[df_capgen['Year'] == year] #Keep only data from year of interest
    df_hubcap = df_capgen.loc[df_capgen['G'].isin(hub_windgen),] #Keep only hub data 
    df_hubcap_agg = pd.DataFrame(df_hubcap.groupby(['Year', 'Country', 'RRR', 'Lat', 'Lon'])['Value'].sum().reset_index()) #Sum all capacities (of different wind turbines) at each location
    df_hubcap_agg['Radius'] = np.sqrt(df_hubcap_agg['Value'] * 1000 / hub_area / np.pi) # Create column of hub radius (in kilometres)

    if LINES == 'Flow' or LINES == 'CongestionFlow':
        #Merge all relevant hub info into one dataframe
        df_hubprod = pd.DataFrame(df_hubprod.groupby(['Year', 'Country', 'RRR'])['Value'].sum().reset_index()) #Sum all production (of different wind turbines) at each location
        df_hubprod.Value = df_hubprod.Value/1000
        df_hubprod.rename(columns = {'Value': 'prod_GWh'}, inplace = True)
        df_hub = pd.merge(df_hubcap_agg, df_hubprod[['RRR', 'prod_GWh']], on = 'RRR', how = 'left', left_index = True).reset_index(drop = True) 
        #Display a zero instead of NaN values (i.e. if there is no production in that hour, so df_hubprod row does not exist)
        df_hub.loc[df_hub.prod_GWh.isna() == True, 'prod_GWh'] = 0
    else: 
        df_hub = df_hubcap_agg.copy()
        


# ### 2.4 Prepare capacity dataframe

# In[ ]:


#Transmission Capacities
if LINES == 'Capacity' or LINES == 'CongestionFlow': #Skip this cell in case LINES == 'Flow'
    df_capacity['Year'] = df_capacity['Year'].astype(int)
    df_capacity = df_capacity.loc[df_capacity['Year'] == year, ].reset_index(drop = True) #Keep only data from year of interest
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
                df_capacity.loc[i,'LonExp'] = df_unique.loc[j, 'Lon']
            if df_capacity.loc[i,'IRRRI'] == df_unique.loc[j, 'RRR']:
                df_capacity.loc[i,'LatImp'] = df_unique.loc[j, 'Lat']
                df_capacity.loc[i,'LonImp'] = df_unique.loc[j, 'Lon']
    if len(df_capacity) == 0:
        print("Error: No capacity found. Check year and exo_end.")
        sys.exit()


# ### 2.5 Add bypass coordinates for indirect lines

# In[ ]:


if LINES == 'Capacity':
    df_bypass = pd.merge(df_bypass, df_capacity[['Year', 'Country', 'IRRRE', 'IRRRI', 'UNITS', 'Value']], on = ['IRRRE', 'IRRRI'], how = 'left')
    #Replace existing row by 2 bypass rows
    keys = list(df_bypass.columns.values)[0:2]
    i1 = df_capacity.set_index(keys).index
    i2 = df_bypass.set_index(keys).index
    df_capacity = df_capacity[~i1.isin(i2)] #Delete existing rows that need bypass
    df_capacity = df_capacity.append(df_bypass, ignore_index = True, sort = True) #Append bypass rows
    
if LINES == 'Flow' or LINES == 'CongestionFlow': #Skip this cell in case LINES == 'Capacity'
    df_bypass = pd.merge(df_bypass, df_flow[['Year', 'Country', 'IRRRE', 'IRRRI', 'SSS', 'TTT', 'UNITS', 'Value']], on = ['IRRRE', 'IRRRI'], how = 'left').dropna()
    #Replace existing row by 2 bypass rows
    keys = list(df_bypass.columns.values)[0:2]
    i1 = df_flow.set_index(keys).index
    i2 = df_bypass.set_index(keys).index
    df_flow = df_flow[~i1.isin(i2)]#Delete existing rows that need bypass
    df_flow = df_flow.append(df_bypass, ignore_index = True, sort = True)#Append bypass rows


# ### 2.6 Calculate Congestion

# In[ ]:


if LINES == 'CongestionFlow': #Skip this cell in case LINES != 'CongestionFlow'
    df_flow = pd.merge(df_flow, df_capacity[['Year', 'Country', 'IRRRE', 'IRRRI', 'Value']], on = ['Year', 'Country', 'IRRRE', 'IRRRI'], how = 'left')
    df_flow.rename(columns={'Value_x': 'Value', 'Value_y' : 'Capacity'}, inplace = True)
    df_flow['Congestion'] = df_flow['Value'] / df_flow['Capacity'] * 100

    #Create color codes for congestion of lines
    df_flow['color'] = pd.cut(df_flow['Congestion'], bins = flowline_breaks, labels = flowline_color )



# ### 2.7 One direction capacity  lines

# In[ ]:


#When capacity is not the same in both directions, display one:
for i,row in df_capacity.iterrows():
    for k,row in df_capacity.iterrows():
        if (df_capacity.loc[k,'IRRRE'] == df_capacity.loc[i,'IRRRI']) & (df_capacity.loc[k,'IRRRI'] == df_capacity.loc[i,'IRRRE']) & (df_capacity.loc[k,'Value'] != df_capacity.loc[i,'Value']):
            df_capacity.loc[i,'Value'] = df_capacity.loc[k,'Value']


# ###  2.8 Define line centers

# In[ ]:


#Define centre of each transmission line
if LINES == 'Flow' or LINES == 'CongestionFlow': #Skip this cell in case LINES == 'Capacity'
    df_flow['LatMid'] = (df_flow['LatImp'] + df_flow['LatExp']) /2
    df_flow['LonMid'] = (df_flow['LonImp'] + df_flow['LonExp']) /2
if LINES == 'Capacity' or LINES == 'CongestionFlow': #Skip this cell in case LINES == 'Flow'
    df_capacity['LatMid'] = (df_capacity['LatImp'] + df_capacity['LatExp']) /2
    df_capacity['LonMid'] = (df_capacity['LonImp'] + df_capacity['LonExp']) /2


# # 3 Create Map Features

# ### 3.1 Create map

# In[ ]:


#Create map 
map_center = [55.220228, 10.419778]
m = folium.Map(location= map_center, zoom_start=5, tiles='')
#Add background layers (sea, regions in model, countries outside of model)
folium.Polygon(locations = [[-90,-180], [90,-180], [90,180], [-90,180]], color = background_color, fill_color = background_color, opacity = 1, fill_opacity = 1 ).add_to(m) #Background

for region in layers_in: 
    folium.GeoJson(data = layers_in[region], name = 'regions_in',                style_function = lambda x:{'fillColor': regions_model_color, 'fillOpacity': 0.5, 'color': regions_model_color, 'weight':1}).add_to(m) #Regions within model
for region in layers_out: 
    folium.GeoJson(data = layers_out[region], name = 'regions_out',                    style_function = lambda x:{'fillColor': regions_ext_color, 'fillOpacity': 0.5, 'color': regions_ext_color, 'weight':1}).add_to(m) #Neighbouring countries


# ### 3.2 Create background hub size

# In[ ]:


if hub_display == True:
    if background_hubsize == True:
        for i,row in df_hub.iterrows():
                folium.Circle(
                  location=[df_hub.loc[i,'Lat'], df_hub.loc[i,'Lon']],
                  popup=df_hub.loc[i,'RRR'],
                  radius = df_hub.loc[i,'Radius']*1000,
                  color = hub_background_color,
                  opacity = 0,
                  fill=True,
                  fill_color = hub_background_color,
                  fill_opacity = hub_area_opacity
               ).add_to(m)


# ### 3.3 Add lines

# In[ ]:


#Add capacity lines
if LINES == 'Capacity':
    for i,row in df_capacity.iterrows():
        folium.PolyLine(([df_capacity.loc[i,'LatExp'], df_capacity.loc[i,'LonExp']],[df_capacity.loc[i,'LatImp'],df_capacity.loc[i,'LonImp']]),                             color=capline_color, line_cap = 'butt', weight=df_capacity.loc[i,'Value']/line_width_constant, opacity=1).add_to(m)  
        if df_capacity.loc[i,'Value'] > label_min:
            if line_decimals == 0:
                folium.Marker(location=[df_capacity.loc[i,'LatMid'], df_capacity.loc[i,'LonMid']],
                          icon=DivIcon(
                              icon_size=(150,36), 
                                       icon_anchor=(11,7),
                     html='<div style="font-size: {}pt; color : {}">{}</div>'.format(font_line, line_text, \
                            df_capacity.loc[i,'Value'].round(line_decimals).astype(int)))).add_to(m)
            else: 
                folium.Marker(location=[df_capacity.loc[i,'LatMid'], df_capacity.loc[i,'LonMid']],
                          icon=DivIcon(
                              icon_size=(150,36), 
                                       icon_anchor=(11,7),
                     html='<div style="font-size: {}pt; color : {}">{}</div>'.format(font_line, line_text, \
                            round(df_capacity.loc[i,'Value'],line_decimals)))).add_to(m)
#Add flows (single color)                
if LINES == 'Flow':
    attr = {'font-weight': 'bold', 'font-size': '24'}
    for i,row in df_flow.iterrows():
        flow = folium.PolyLine(([df_flow.loc[i,'LatExp'], df_flow.loc[i,'LonExp']],                              [df_flow.loc[i,'LatImp'],df_flow.loc[i,'LonImp']]),                             color=capline_color, line_cap = 'butt', weight=df_flow.loc[i,'Value']/line_width_constant, opacity=1).add_to(m)   
        plugins.PolyLineTextPath(flow, '\u2192', repeat=False ,center = True, offset=10, orientation = -360,                                  attributes=attr).add_to(m)  #Arrow
        if df_flow.loc[i,'Value'] > label_min:
            if line_decimals == 0:
                folium.Marker(location=[df_flow.loc[i,'LatMid'], df_flow.loc[i,'LonMid']],
                          icon=DivIcon(
                              icon_size=(150,36), 
                                       icon_anchor=(11,7),
                     html='<div style="font-size: {}pt; color : {}">{}</div>'.format(font_line, line_text, \
                            df_flow.loc[i,'Value'].round(line_decimals).astype(int)))).add_to(m)
            else: 
                folium.Marker(location=[df_flow.loc[i,'LatMid'], df_flow.loc[i,'LonMid']],
                          icon=DivIcon(
                              icon_size=(150,36), 
                                       icon_anchor=(11,7),
                     html='<div style="font-size: {}pt; color : {}">{}</div>'.format(font_line, line_text, \
                            round(df_flow.loc[i,'Value'],line_decimals)))).add_to(m)    
#Add flows (color based on congestion)
if LINES == 'CongestionFlow':
    attr = {'font-weight': 'bold', 'font-size': '24'}
    for i,row in df_flow.iterrows():
        flow = folium.PolyLine(([df_flow.loc[i,'LatExp'], df_flow.loc[i,'LonExp']],                              [df_flow.loc[i,'LatImp'],df_flow.loc[i,'LonImp']]),                             color=df_flow.loc[i,'color'], line_cap = 'butt', weight=df_flow.loc[i,'Value']/line_width_constant, opacity=1).add_to(m)   
        plugins.PolyLineTextPath(flow, '\u2192', repeat=False ,center = True, offset=10, orientation = -180,                                  attributes=attr).add_to(m)  #Arrow
        if df_flow.loc[i,'Value'] > label_min:
            if line_decimals == 0:
                folium.Marker(location=[df_flow.loc[i,'LatMid'], df_flow.loc[i,'LonMid']],
                          icon=DivIcon(
                              icon_size=(150,36), 
                                       icon_anchor=(11,7),
                     html='<div style="font-size: {}pt; color : {}">{}</div>'.format(font_line, line_text, \
                            df_flow.loc[i,'Value'].round(line_decimals).astype(int)))).add_to(m)
            else: 
                folium.Marker(location=[df_flow.loc[i,'LatMid'], df_flow.loc[i,'LonMid']],
                          icon=DivIcon(
                              icon_size=(150,36), 
                                       icon_anchor=(11,7),
                     html='<div style="font-size: {}pt; color : {}">{}</div>'.format(font_line, line_text, \
                                round(df_flow.loc[i,'Value'],line_decimals)))).add_to(m)    


# ### 3.4 Add region names

# In[ ]:


#Add region names
for i,row in df_region.loc[df_region['Display']==1, ].iterrows():
    folium.Marker(location=[df_region.loc[i,'Lat'], df_region.loc[i,'Lon']],
                  icon=DivIcon(
                      icon_size=(150,36), 
                               icon_anchor=(7,7),
     html='<div style="font-size: {}pt; color : {}">{}</div>'.format(font_region, region_text, df_region.loc[i,'RRR']))).add_to(m)  


# ### 3.5 Add hubs

# In[ ]:


#Add hub capacities as bubbles
if hub_display == True:
    if LINES == 'Capacity': 
        for i,row in df_hub.iterrows():
            folium.CircleMarker(
              location=[df_hub.loc[i,'Lat'], df_hub.loc[i,'Lon']],
              popup=df_hub.loc[i,'RRR'],
              radius = hub_size,
              color= hub_color,
              opacity = 0,
              fill=True,
              fill_color= hub_color,
              fill_opacity = 1
           ).add_to(m)

            if hub_decimals == 0:
                folium.Marker(location=[df_hub.loc[i,'Lat'], df_hub.loc[i,'Lon']],
                              icon=DivIcon(
                                  icon_size=(150,36), 
                                           icon_anchor=(7,9),
                 html='<div style="font-size: {}pt; color : {}">{}</div>'.format(font_hub, hub_text, df_hub.loc[i,'Value'].round(hub_decimals).astype(int)))).add_to(m)
            else:
                folium.Marker(location=[df_hub.loc[i,'Lat'], df_hub.loc[i,'Lon']],
                              icon=DivIcon(
                                  icon_size=(150,36), 
                                           icon_anchor=(7,9),
                 html='<div style="font-size: {}pt; color : {}">{}</div>'.format(font_hub, hub_text, round(df_hub.loc[i,'Value'], hub_decimals)))).add_to(m)    

    if LINES == 'Flow' or LINES == 'CongestionFlow':
        for i,row in df_hub.iterrows():
            folium.CircleMarker(
              location=[df_hub.loc[i,'Lat'], df_hub.loc[i,'Lon']],
              popup=df_hub.loc[i,'RRR'],
              radius = hub_size,
              color= hub_color,
              opacity = 0,
              fill=True,
              fill_color= hub_color,
              fill_opacity = 1
           ).add_to(m)

            if hub_decimals == 0:
                folium.Marker(location=[df_hub.loc[i,'Lat'], df_hub.loc[i,'Lon']],
                              icon=DivIcon(
                                  icon_size=(150,36), 
                                           icon_anchor=(7,9),
                 html='<div style="font-size: {}pt; color : {}">{}</div>'.format(font_hub, hub_text, df_hub.loc[i,'prod_GWh'].round(hub_decimals).astype(int)))).add_to(m)
            else:
                folium.Marker(location=[df_hub.loc[i,'Lat'], df_hub.loc[i,'Lon']],
                              icon=DivIcon(
                                  icon_size=(150,36), 
                                           icon_anchor=(7,9),
                 html='<div style="font-size: {}pt; color : {}">{}</div>'.format(font_hub, hub_text, round(df_hub.loc[i,'prod_GWh'], hub_decimals)))).add_to(m)    


# ### 3.6 Add Legend

# In[ ]:


color_keys = ['color1', 'color2', 'color3']
color_dict = dict(zip(color_keys, flowline_color))
legend_keys = ['item1', 'item2', 'item3']
legend_dict = dict(zip(legend_keys, legend_values))

if LINES == 'CongestionFlow':
    from branca.element import Template, MacroElement

    template = """
    {% macro html(this, kwargs) %}

    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>jQuery UI Draggable - Default functionality</title>
      <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">

      <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
      <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>

      <script>
      $( function() {
        $( "#maplegend" ).draggable({
                        start: function (event, ui) {
                            $(this).css({
                                right: "auto",
                                top: "auto",
                                bottom: "auto"
                            });
                        }
                    });
    });

      </script>
    </head>
    <body>


    <div id='maplegend' class='maplegend' 
        style='position: absolute; z-index:9999; border:2px solid grey; background-color:rgba(255, 255, 255, 1);
         border-radius:6px; padding: 10px; font-size:14px; right: 20px; bottom: 20px;'>

    <div class='legend-title'>Congestion rate</div>
    <div class='legend-scale'>
      <ul class='legend-labels'>
        <li><span style='background:color3;opacity:1;'></span>item1</li>
        <li><span style='background:color2;opacity:1;'></span> item2 </li>
        <li><span style='background:color1;opacity:1;'></span> item3 </li>

      </ul>
    </div>
    </div>

    </body>
    </html>

    <style type='text/css'>
      .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
      .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
      .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
      .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 1px solid #999;
        }
      .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
      .maplegend a {
        color: #777;
        }
    </style>
    {% endmacro %}"""
    for key in color_dict.keys():
        template = template.replace(key, color_dict[key])
    for key in legend_dict.keys():
        template = template.replace(key, legend_dict[key])
    
    macro = MacroElement()
    macro._template = Template(template)

    m.get_root().add_child(macro)


# ### 4 Save Output

# In[ ]:


# Make Transmission_Map output folder
if not os.path.isdir('output/Transmission_Map/' + LINES + '/' + SCENARIO + '/' + market):
    os.makedirs('output/Transmission_Map/' + LINES + '/' + SCENARIO + '/' + market)


# In[ ]:


output_dir = 'output/Transmission_Map/' + LINES + '/' + SCENARIO + '/' + market
m.save(output_dir + '/' +  map_name)


# ### 5 Display Map

# In[ ]:


m

#take a screenshot 

#import os
#import time
#from selenium import webdriver
 
#delay=7

#tmpurl = output_dir + '/' +  map_name
#tmpurl='file://{path}/{output}/{mapfile}'.format(path=os.getcwd(),output=output_dir,mapfile=map_name)


#Open a browser window...
#browser = webdriver.Firefox()
#..that displays the map...
#browser.get(tmpurl)
#Give the map tiles some time to load
#time.sleep(delay)
#Grab the screenshot
#browser.save_screenshot('name.png')
#Close the browser
#browser.quit()



