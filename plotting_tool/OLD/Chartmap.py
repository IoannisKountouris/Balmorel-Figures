#!/usr/bin/env python
# coding: utf-8

# # 1 Preparations

# ### 1.1 Set Options

# In[17]:


### Set options here
#Structural options
input_folder = 'Investment'
variable = 'my_project_G_CAP_YCRAF_220803-1546' #To define input file
display_column = 'Technology'
display_variable = 'ELECTROLYZER' #NB: Only relevant for choropleth maps
year = 2050 #Year to be displayed
scenario = ['Base'] #Scenario(s) to be displayed 

Commodity = ['HYDROGEN'] #you can add HEAT or H2 ['ELECTRICITY','HEAT','HYDROGEN']

#Visual options
hub_display = False
region_label = True #Label of region's acronym
font_size_region = 15 #Font size of the region display


#Colors
background_color = 'white'
regions_ext_color = 'lightgrey'
regions_model_color = 'grey'
region_text = 'black'
hub_color = 'lightblue'
hub_background_color = 'lightblue'
hub_text = 'black'
choropleth_colorscheme = 'YlGn' #Colour scheme of the choropleth map. Choose from: [BuGn, BuPu, GnBu, OrRd, PuBu, PuBuGn, PuRd, RdPu, YlGn, YlGnBu, YlOrBr, YlOrRd]
manual_colors = False #True if colours are added to values manually in csv-file, False if not. 

#Chart options
chart_type = 'pie' #Choose from 'doughnut', 'pie', 'bar', 'choropleth'
doughnut_size = 1 #Size of doughnut / pie chart
bar_size = 0.8
doughnut_location = [40,30] #Distance of bottom-right corner of the chart from the region's coordinates. Higher values shift the chart to the West (first value) and North (second value). 
bar_location = [30,80] #Distance of bottom-right corner of the chart from the region's coordinates. Higher values shift the chart to the West (first value) and North (second value). 
choropleth_location = [7,7]
title_distance = -40 #Distance between chart title (country label) and chart
display_decimals = 1 #Number of decimals in pop-up table or on choropleth map
manual_bins = '' #Bins for the choropleth map, e.g. [0,0.5,1,2,5], if no manual bins: ''


# ### 1.2 Import Packages

# In[18]:


from pathlib import Path
from selenium import webdriver
import sys
import os
import glob
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium import plugins
import json
from folium.features import DivIcon #For text labels on hubs
from IPython.display import display, HTML
import io
import pandas.io.formats.style
from csv import reader
#for charts
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
display(HTML(data="""
<style>
    div#notebook-container    { width: 95%; }
    div#menubar-container     { width: 65%; }
    div#maintoolbar-container { width: 99%; }
</style>
"""))




# ### 1.3 Load Files

# In[19]:


#Set file names
project_dir = Path('./input')
map_name = chart_type + 'Chartmap_' + variable + '_' + display_column + '_'+ str(year) + '.html'

variable_file = input_folder + '\\' + variable + '.csv'
df_var = pd.read_csv(project_dir/'results/'/variable_file , sep =';')

#keep only the commodity needed

df_var = df_var.loc[df_var['Commodity'].isin(Commodity)]

#If hydrogen do not plot the hydrogen storage.

df_var = df_var[df_var['Technology'] != 'H2-STORAGE']

#Load coordinates files 
df_region = pd.read_csv(project_dir/'geo_files/coordinates_RRR.csv')

#Define names of geojson and shapefile layers
r_in = list(df_region.loc[(df_region['Display'] == 1) & (df_region['Type'] == 'region'), 'RRR'])
r_out = list(df_region.loc[(df_region['Display'] == 0) & (df_region['Type'] == 'region'), 'RRR'])

layers_in = {region: '' for region in r_in}
layers_out = {region: '' for region in r_out}

#Create dictionaries with layer names for each region; 
    #if both a shapefile and geojson file are available for one region, the geojson file is used. 
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

#Create dataframe with only the regions that need to be displayed
if hub_display == True:
    df_region = df_region.loc[df_region['Display']==1, ]
else:
    df_region = df_region.loc[(df_region['Type'] == 'region')  & (df_region['Display']==1), ]


# # 2 Data processing

# ### 2.1 Select relevant year

# In[20]:


#Select only releavant year and scenario
df_var = df_var.loc[df_var['Year']==year, ]
if 'Scenario' in list(df_var.columns): 
    df_var = df_var.loc[df_var['Scenario'].isin(scenario), ]
unit = df_var.Unit.unique()[0] #Store unit value to display in table on map


# ### 2.2 Create horizontal table with sectors as columns

# In[21]:


# Create horizontal table with sectors as columns
df_var = pd.DataFrame(df_var.groupby(['Region', display_column])['Value'].sum().reset_index())
df_var = df_var.pivot(index = 'Region', columns = display_column, values = 'Value')

#Add coordinates
df_region = df_region.rename(columns = {'RRR':'Region'})
df_var = pd.merge(df_var, df_region[['Lat', 'Lon', 'Region']], on = ['Region'], how = 'right')
df_var = df_var.fillna(0)


# ### 2.3 Create color dictionaries for charts

# In[22]:


if chart_type == 'pie' or chart_type == 'doughnut' or chart_type == 'bar': 

    insert_keys = ['insert0','insert1', 'insert2', 'insert3', 'insert4', 'insert5', 'insert6', 'insert7', 'insert8', 'insert9', 'insert_ten', 'insert_eleven', 'insert_twelve', 'insert_thirteen', 'insert_fourteen', 'insert_fifteen',                  'insert_sixteen', 'insert_seventeen', 'insert_eightteen', 'insert_nineteen']
    item_keys = ['item0','item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'item7', 'item8', 'item9', 'item_ten', 'item_eleven', 'item_twelve', 'item_thirteen', 'item_fourteen', 'item_fifteen',                  'item_sixteen', 'item_seventeen', 'item_eightteen', 'item_nineteen']
    color_keys = ['color0', 'color1', 'color2', 'color3', 'color4', 'color5', 'color6', 'color7', 'color8', 'color9',                   'color_ten', 'color_eleven', 'color_twelve', 'color_thirteen', 'color_fourteen', 'color_fifteen',                  'color_sixteen', 'color_seventeen', 'color_eightteen', 'color_nineteen']
    textcouleur_keys = ['textcouleur0', 'textcouleur1', 'textcouleur2', 'textcouleur3', 'textcouleur4', 'textcouleur5', 'textcouleur6', 'textcouleur7', 'textcouleur8', 'textcouleur9', 'textcouleurten']

    if manual_colors == False:
        color_list = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c',                       '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075',                      '#808080', '#ffffff', '#000000'] #colors to be used on chartcolor_keys = ['color0', 'color1', 'color2', 'color3', 'color4', 'color5', 'color6', 'color7', 'color8', 'color9', 'color_ten', etc]
        color_dict = dict(zip(color_keys, color_list))
        textcouleur_list = ['black']*len(color_list)
        textcouleur_dict = dict(zip(textcouleur_keys, textcouleur_list))

    if manual_colors == True:
        df_color = pd.read_csv(project_dir/'geo_files/manual_colors_input.csv', sep=',')
        #Sort dataframe of colors to contain right sequence
        lst = list(df_var.columns)[1:-2]
        df_color['Value_cat'] = pd.Categorical(df_color['Value'], categories = lst, ordered = True )
        df_color = df_color.sort_values('Value_cat')
        del df_color['Value_cat']
        #Create dictionary
        color_list = list(df_color.Color)    
        hatch_list = list(df_color.Hatch)
        textcouleur_list = list(df_color.Textcolor) 
        color_dict = dict(zip(color_keys, color_list))
        textcouleur_dict = dict(zip(textcouleur_keys, textcouleur_list))

    #Check if enough colours are defined 
    if manual_colors == True:
        if len(df_color) != len(list(df_var.columns)[1:-2]):
            sys.exit('Warning: the number of pre-defined colors does not match the number of variables. \n Update the input color sheet or set "manual_colors" to False')

#color_list
# ### 2.4 Prepare list of lists for chart data

# In[23]:


if chart_type == 'pie' or chart_type == 'doughnut' or chart_type == 'bar': 
    chart_data = list(range(len(df_var)))
    for i in range(len(df_var)):
        chart_data[i] = list(range(1,len(list(df_var.columns))-2))
    for i in range(len(chart_data)):
        for j in range(1,len(list(df_var.columns))-2):
            chart_data[i][j-1] = df_var.iloc[i,j]


# ### 2.5 Create charts for map

# In[24]:


if chart_type == 'doughnut':
    doughnut_pie = 0.5 #Thickness of the inside ring of the doughnut: 1 = pie chart, 0.5 = doughnut chart 
else:
    doughnut_pie = 1

if chart_type == 'pie' or chart_type == 'doughnut': 
    fig = plt.figure(figsize=(doughnut_size, 1.3*doughnut_size))
    fig.patch.set_alpha(0)
    ax = fig.add_subplot(1,1,1)
    plots = []
    for i,row in df_var.iterrows():
        pie  = ax.pie(chart_data[i], wedgeprops=dict(width = doughnut_pie),                   colors = tuple(color_list), startangle = 90 )
        if region_label == True: 
            ax.set_title(df_var.loc[i, 'Region'], pad = title_distance, fontsize = font_size_region)
        buff = io.StringIO()
        plt.savefig(buff, format='SVG', transparent = True)
        buff.seek(0)
        svg = buff.read()
        svg = svg.replace('\n', '')
        plots.append(svg)
        
if chart_type == 'bar':
    fig = plt.figure(figsize=(bar_size, 1.3*bar_size))
    fig.patch.set_alpha(0)
    ax = fig.add_subplot(1,1,1)
    plots = []
    for i,row in df_var.iterrows():
        ax.axis("off")
        bars = plt.bar(list(range(1,len(chart_data[i])+1)), height = chart_data[i], color = color_list)
        buff = io.StringIO()
        plt.savefig(buff, format='svg', transparent = True)
        buff.seek(0)
        svg = buff.read()
        svg = svg.replace('\n', '')
        plots.append(svg)
        plt.cla()
    plt.clf()
    plt.close()
plt.show()


# # 3 Map creation

# ### 3.1 Create map and add layers

# In[25]:


#Create map 
map_center = [55.220228, 10.419778]
m = folium.Map(location= map_center, zoom_start=5, tiles='')
#Add background layers (sea, regions in model, countries outside of model)
folium.Polygon(locations = [[-90,-180], [90,-180], [90,180], [-90,180]], color = background_color, fill_color = background_color, opacity = 1, fill_opacity = 1 ).add_to(m) #Background
for region in layers_in: 
    folium.GeoJson(data = layers_in[region], name = 'regions_in',                style_function = lambda x:{'fillColor': regions_model_color, 'fillOpacity': 0.5, 'color': regions_model_color, 'weight':1}).add_to(m) #Regions within model
for region in layers_out: 
    folium.GeoJson(data = layers_out[region], name = 'regions_out',                    style_function = lambda x:{'fillColor': regions_ext_color, 'fillOpacity': 0.5, 'color': regions_ext_color, 'weight':1}).add_to(m) #Neighbouring countries


# ### 3.2 Function: add charts and popup tables

# In[26]:


def write_to_html_file(df, title='', filename='out.html'):

    template = '''
<html>
<head>
<style>

    h2 {
        text-align: center;
        font-family: Helvetica, Arial, sans-serif;
        font-size: 130%;
    }
    table { 
        margin-left: auto;
        margin-right: auto;
    }
    table, th, td {
        border: 1px solid black;
        border-collapse: collapse;
    }
    th, td {
        padding: 5px;
        text-align: center;
        font-family: Helvetica, Arial, sans-serif;
        font-size: 90%;
    }

    table tr:nth-of-type(1) {background: color0; color: textcouleur0}
    table tr:nth-of-type(2) {background: color1; color: textcouleur1}
    table tr:nth-of-type(3) {background: color2; color: textcouleur2}
    table tr:nth-of-type(4) {background: color3; color: textcouleur3}
    table tr:nth-of-type(5) {background: color4; color: textcouleur4}
    table tr:nth-of-type(6) {background: color5; color: textcouleur5}
    table tr:nth-of-type(7) {background: color6; color: textcouleur6}
    table tr:nth-of-type(8) {background: color7; color: textcouleur7}
    table tr:nth-of-type(9) {background: color8; color: textcouleur8}
    table tr:nth-of-type(10) {background: color9; color: textcouleur9}
    table tr:nth-of-type(11) {background: color_ten; color: textcouleurten}
    table tr:nth-of-type(12) {background: color_eleven; color: textcouleur_eleven}
    table tr:nth-of-type(13) {background: color_twelve; color: textcouleur_twelve}
    table tr:nth-of-type(14) {background: color_thirteen; color: textcouleur_thirteen}
    table tr:nth-of-type(15) {background: color_fourteen; color: textcouleur_fourteen}
    table tr:nth-of-type(16) {background: color_fifteen; color: textcouleur_fifteen}
    table tr:nth-of-type(17) {background: color_sixteen; color: textcouleur_sixteen}
    table tr:nth-of-type(18) {background: color_seventeen; color: textcouleur_seventeen}
    table tr:nth-of-type(19) {background: color_eightteen; color: textcouleur_eightteen}
    table tr:nth-of-type(20) {background: color_nineteen; color: textcouleur_nineteen}
    .wide {
        width: 90%; 
    }

</style>
</head>
<body>
    '''
    for key in color_dict.keys():
        template = template.replace(key, color_dict[key])
    if manual_colors == False:
        for key in textcouleur_dict.keys():
            template = template.replace(key, textcouleur_dict[key])
    if manual_colors == True:
        for key in textcouleur_dict.keys():
            template = template.replace(key, textcouleur_dict[key])
    template += '<h2> %s </h2>\n' % title
    if type(df) == pd.io.formats.style.Styler:
        template += df.render()
    else:
        template += df.to_html(classes='wide', header = False, escape=False)
    template += '''
</body>
</html>
'''
    with open(filename, 'w') as f:
        f.write(template)


# ### 3.3 Create map with display table

# In[27]:


if chart_type != 'choropleth':
    #Prepare dataframe to display
    df_display = df_var.copy()
    del df_display['Lat']
    del(df_display['Lon'])
    df_display = df_display.set_index('Region')

    #Add everything to the map
    for i, row in df_var.iterrows():
        temp = round(pd.DataFrame(df_display.loc[df_var.loc[i,'Region']]),display_decimals)
        html = write_to_html_file(temp, title = df_var.loc[i, 'Region'] + ' ' + variable + ": " + display_column + ' ('  + unit + ')') 
        marker = folium.Marker(location=[df_var.loc[i,'Lat'], df_var.loc[i,'Lon']], popup = folium.Popup(open("out.html", "r").read()))
        if chart_type == 'doughnut' or chart_type == 'pie':
            icon = folium.DivIcon(html=plots[i], icon_anchor=(doughnut_location[0],doughnut_location[1]))
        if chart_type == 'bar':
            icon = folium.DivIcon(html=plots[i], icon_anchor=(bar_location[0],bar_location[1]))

        marker.add_child(icon)
        m.add_child(marker)


# ### 3.4 Choropleth

# ##### 3.4.1 Merge region layers for choropleth

# In[28]:


if chart_type == 'choropleth':
    merged_layer = gpd.read_file('.\input\\geo_files\\geojson_files\\'+ r_in[0] + '.geojson')
    for region in r_in[1:]:
        add_region = gpd.read_file('.\input\\geo_files\\geojson_files\\'+ region + '.geojson')
        merged_layer = merged_layer.append(add_region)


# ##### 3.4.2 Add colors to choropleth

# In[29]:


if chart_type == 'choropleth':
    if len(str(manual_bins)) > 0:
        bins = manual_bins
    else:
        bins = list(df_var[display_variable].quantile([0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.9, 1]))
    
    folium.Choropleth(
        geo_data= merged_layer,
        name='choropleth',
        data=df_var,
        columns=['Region', display_variable],
        key_on='feature.properties.id',
        fill_color=choropleth_colorscheme,
        fill_opacity=0.7,
        line_opacity=0.2,
#        scheme="Quantiles",
        legend_name= display_variable,
        bins = bins
    ).add_to(m)
#Display values on map
    for i,row in df_var.iterrows():
        folium.Marker(location=[df_var.loc[i,'Lat'], df_var.loc[i,'Lon']],
                      icon=DivIcon(
                          icon_size=(150,36), 
                                   icon_anchor=tuple(choropleth_location),
         html='<div style="font-size: {}pt; color : {}">{}</div>'.format(font_size_region, \
                                                                         region_text, \
                                                                         round(df_var.loc[i,display_variable], \
                                                                               display_decimals)))\
                     ).add_to(m)  


# ### 3.5 Add legend and display map

# In[30]:


if chart_type == 'pie' or chart_type == 'doughnut' or chart_type == 'bar': 
    empty_values = ['']*20
    legend_layout = ["""<span style='background:color0;opacity:1;'></span>item0""",                    """<span style='background:color1;opacity:1;'></span>item1""",                    """<span style='background:color2;opacity:1;'></span>item2""",                    """<span style='background:color3;opacity:1;'></span>item3""",                    """<span style='background:color4;opacity:1;'></span>item4""",                    """<span style='background:color5;opacity:1;'></span>item5""",                    """<span style='background:color6;opacity:1;'></span>item6""",                    """<span style='background:color7;opacity:1;'></span>item7""",                    """<span style='background:color8;opacity:1;'></span>item8""",                    """<span style='background:color9;opacity:1;'></span>item9""",                    """<span style='background:color_ten;opacity:1;'></span>item_ten""",                    """<span style='background:color_eleven;opacity:1;'></span>item_eleven""",                    """<span style='background:color_twelve;opacity:1;'></span>item_twelve""",                    """<span style='background:color_thirteen;opacity:1;'></span>item_thirteen""",                    """<span style='background:color_fourteen;opacity:1;'></span>item_fourteen""",                    """<span style='background:color_fifteen;opacity:1;'></span>item_fifteen""",                    """<span style='background:color_sixteen;opacity:1;'></span>item_sixteen""",                    """<span style='background:color_seventeen;opacity:1;'></span>item_seventeen""",                    """<span style='background:color_eightteen;opacity:1;'></span>item_eightteen""",                    """<span style='background:color_nineteen;opacity:1;'></span>item_nineteen"""]
    legend_layout = legend_layout[0:len(df_display.columns)]
    empty_values = empty_values[len(df_display.columns):]
    layout_list = legend_layout + empty_values
    layout_dict = dict(zip(insert_keys, layout_list))
    if manual_colors == False:
        legend_values = list(df_display.columns)
    if manual_colors == True:
        legend_values = list(df_color.Value.unique())
    item_dict = dict(zip(item_keys, legend_values))



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

    <div class='legend-title'>Legend</div>
    <div class='legend-scale'>
      <ul class='legend-labels'>
        <li>insert0</li>
        <li>insert1</li>
        <li>insert2</li>
        <li>insert3</li>
        <li>insert4</li>
        <li>insert5</li>
        <li>insert6</li>
        <li>insert7</li>
        <li>insert8</li>
        <li>insert9</li>
        <li>insert_ten</li>
        <li>insert_eleven</li>
        <li>insert_twelve</li>
        <li>insert_thirteen</li>
        <li>insert_fourteen</li>
        <li>insert_fifteen</li>
        <li>insert_sixteen</li>
        <li>insert_seventeen</li>
        <li>insert_eightteen</li>
        <li>insert_nineteen</li>
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
    for key in layout_dict.keys():
        template = template.replace(key, layout_dict[key])
    for key in color_dict.keys():
        template = template.replace(key, color_dict[key])
    for key in item_dict.keys():
        template = template.replace(key, item_dict[key])

    macro = MacroElement()
    macro._template = Template(template)

    #Add region names for bar charts
    if chart_type == 'bar' and region_label == True:
        for i,row in df_region.loc[df_region['Display']==1, ].iterrows():
            folium.Marker(location=[df_region.loc[i,'Lat'], df_region.loc[i,'Lon']],
                      icon=DivIcon(
                          icon_size=(150,36), 
                                   icon_anchor=(7,-7),
         html='<div style="font-size: {}pt; color : {}">{}</div>'.format(font_size_region, region_text, df_region.loc[i,'Region']))).add_to(m)  

    m.get_root().add_child(macro)


# In[31]:


m




# # 4 Save map

# In[32]:


#Make Chartmap folder
if not os.path.isdir('output/Chartmap/'):
    os.makedirs('output/Chartmap/')
    
#Save map 
m.save('output/Chartmap/' + map_name)

#Delete temporary html table
if os.path.isfile('out.html'):
    os.remove('out.html')
    
#mapUrl =  'output/Chartmap/' + map_name   

#driver = webdriver.Firefox()
#driver.get(mapUrl)
#time.sleep(5)

#driver.save_screenshot('output.jpg')
#driver.quit()