# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 09:59:52 2023

@author: iokoun
"""

from mpl_toolkits.basemap import Basemap
import networkx as nx
import matplotlib.pyplot as plt

# Create a new Basemap instance for Europe
m = Basemap(llcrnrlon=-20, llcrnrlat=35, urcrnrlon=50, urcrnrlat=70)

# Draw coastlines, countries and fill continents
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color='coral')

# Create a new NetworkX graph
G = nx.Graph()

# Add the transmission lines as edges to the graph
# the edges will be represented by the longitude and latitude of the nodes
G.add_edge((-10, 40), (0, 50))
G.add_edge((0, 50), (10, 60))
G.add_edge((10, 60), (20, 55))

# Plot the graph on the Basemap
nx.draw_networkx_edges(G, m.plot_points(), m, edgelist=G.edges(), edge_color='b', width=2)

# Show the plot
plt.show()