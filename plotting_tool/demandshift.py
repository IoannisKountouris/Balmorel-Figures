# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 13:50:49 2023

@author: iokoun
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import os


df = pd.read_clipboard()
df = df.reset_index()
#data=df

colors = {
    'Endoshifting GH2E': 'red',  # Red color for Endoshifting GH2E
    'Endoshifting H2E': 'blue',  # Blue color for Endoshifting H2E
    'Exogenous Demand': 'green',  # Green color for Exogenous Demand
    'Max': 'black',              # Black color for Max
    'Min': 'yellow'              # Yellow color for Min
}

# Set up the matplotlib figure and axes, based on the number of regions
fig, ax = plt.subplots(figsize=(12, 7))

# Width of a bar
bar_width = 0.3

# Positions of the left bar-boundaries
bar_l = np.arange(len(df['Regions']))

# Positions of the x-axis ticks (center of the bars as bar labels)
tick_pos = bar_l + bar_width

# Create the bars for each category
bars_gh2e = ax.bar(bar_l, df['Endoshifting GH2E'], width=bar_width, label='Endoshifting GH2E', color=colors['Endoshifting GH2E'], alpha=0.7)
bars_demand = ax.bar(bar_l + bar_width*2, df['Exogenous Demand'], width=bar_width, label='Exogenous Demand', color=colors['Exogenous Demand'], alpha=0.7)
bars_h2e = ax.bar(bar_l + bar_width, df['Endoshifting H2E'], width=bar_width, label='Endoshifting H2E', color=colors['Endoshifting H2E'], alpha=0.7)

# Calculate the top of the demand bars to place the min/max dots
demand_tops = df['Endoshifting GH2E'] + df['Endoshifting H2E'] + df['Exogenous Demand']

# Add the min and max data points for each region
min_points = df['Min']
max_points = df['Max']

# Place the dots on top of the middle of the bars
ax.scatter(bar_l + bar_width, max_points, color=colors['Max'], zorder=5, label='Max', marker='x')
ax.scatter(bar_l + bar_width, min_points, color=colors['Min'], zorder=5, label='Min', marker='x')


# Place the dots on top of the Exogenous Demand bars
#ax.scatter(bar_l + bar_width*2, max_points, color='black', zorder=5, label='Max')
#ax.scatter(bar_l + bar_width*2, min_points, color='yellow', zorder=5, label='Min')

ax.set_ylabel('TWh')

# Set the x ticks with names
plt.xticks(tick_pos, df['Regions'], rotation=45)

# Improve the layout of the plot
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.tick_params(bottom=False, left=False)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle='dotted',color='#b0b0b0')
ax.xaxis.grid(False)
pos_tech = (0,1)
#legend = ax.legend(loc='lower left', ncol = 3,frameon=True,
#                      mode='expnad',bbox_to_anchor=pos_tech)
legend = plt.legend(loc='upper right')
legend.get_frame().set_edgecolor('white')


# Set a buffer around the edge
plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])

#plt.show()

#save plot
map_name = 'DemandShiftPlot_2050'

# Make Transmission_Map output folder
if not os.path.isdir('output/DemandSyNShift'):
        os.makedirs('output/DemandSyNShift')
        
output_dir = 'output/DemandSyNShift'
plt.savefig(output_dir + '/' +  map_name + '.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()