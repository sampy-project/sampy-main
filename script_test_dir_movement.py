import numpy as np

from sampy.agent.builtin_agent import BasicMammal
from sampy.graph.builtin_graph import OrientedHexagonalLattice
from sampy.addons.GIS_interface.geographic_grid import HexGrid
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import matplotlib as mpl

# graph = OrientedHexagonalLattice(nb_hex_x_axis=100, nb_hex_y_axis=100)
# agents = BasicMammal(graph=graph)

# agents.add_agents({'age': [52, 52],
#                    'gender': [0, 1],
#                    'territory': [3150, 3150],
#                    'position': [3150, 3150]})

# arr_nb_steps = np.array([10, 7])

# list_dict = []
# for _ in range(4):
#     list_dict.append(agents.directional_dispersion_from_arr_nb_steps(arr_nb_steps, np.array([0., 0., 0.1, 0.8, 0.1, 0.]), return_path=True))

# for u in list_dict:
#     print(u[0])



np.random.seed(1789) # bonne seed pour dessin avec prob_deviation [0., 0., 0.01, 0.98, 0.01, 0.]


# we create the raw graph (NOTE THAT THESE ARE THE **NEW** COORDINATES!)
bottom_left = (54.131269, -100.687966)
bottom_right = (53.900542, -86.940449)
top_left = (62.232449, -101.888057)
top_right = (61.932146, -84.614660)

map_60 = HexGrid.azimuthal_from_corners(bottom_left, bottom_right, top_left, top_right, 
                                        cell_area=60)

# We crop the graph and add the info from Emily's csv
# print(param.outputfile_af, '\n', param.outputfile_rf, '\n', param.path_to_landscape)
map_60.modify_from_csv(r"C:\Users\remal\Documents\projet_churchill_for_Em\2025_04_25_apr_landscape.csv", 
                        "cell_id",to_keep="to_include", sep=',',
                        dict_attribute_to_type={"init_inf_cell" : bool,
                                                "k_red": float,
                                                "k_arctic": float,
                                                "cells_for_spread": bool
                                                })
agents = BasicMammal(graph=map_60)

agents.add_agents({'age': [52, 52, 52, 52],
                   'gender': [0, 1, 1, 0],
                   'territory': [6330, 6440, 6441, 6550],
                   'position': [6330, 6440, 6441, 6550]})

arr_nb_steps = np.array([20, 15, 21, 27])

list_dict = []
for _ in range(4):
    list_dict.append(agents.directional_dispersion_from_arr_nb_steps(arr_nb_steps, np.array([0., 0., 0., 1., 0., 0.]), return_path=True))

fig, ax = plt.subplots()
ax.set_aspect('equal')


# cm = plt.get_cmap('afmhot')


# sm = plt.cm.ScalarMappable(norm = mpl.colors.Normalize(vmin=np.min(data),vmax=np.max(data)),cmap=cm) #
# sm.set_array([])

for u in list_dict:
    print(u[3])

# counter = 0
for counter, (x,y) in enumerate(zip(map_60.df_attributes['coord_x'], map_60.df_attributes['coord_y'])):

    if counter in list_dict[0][3]:
        hexagon = RegularPolygon((x, y), numVertices=6, radius=5, orientation=np.pi/2, facecolor='black',
                            alpha=1.)  
    elif counter in list_dict[1][3]:
        hexagon = RegularPolygon((x, y), numVertices=6, radius=5, orientation=np.pi/2, facecolor='white',
                            alpha=1.)  
    elif counter in list_dict[2][3]:
        hexagon = RegularPolygon((x, y), numVertices=6, radius=5, orientation=np.pi/2, facecolor='green',
                            alpha=1.)  
    elif counter in list_dict[3][3]:
        hexagon = RegularPolygon((x, y), numVertices=6, radius=5, orientation=np.pi/2, facecolor='red',
                            alpha=1.)  
    else:
        hexagon = RegularPolygon((x, y), numVertices=6, radius=5, orientation=np.pi/2, facecolor='lightgray',
                            alpha=1.)  
            
    # if data[counter] > 0:
    #     hexagon = RegularPolygon((x, y), numVertices=6, radius=5, orientation=np.pi/2, facecolor=cm(float(data[counter]/np.max(data))),
    #                         alpha=1.)

    ax.add_patch(hexagon)
    counter += 1

ax.tick_params(bottom=False,left=False)
# plt.colorbar(sm, ax=ax, label='k mean by cell', shrink=0.6)

plt.autoscale(enable=True)

# plt.savefig(r"C:/Users/remal/Documents/projet_churchill_for_Em/test_recup_moba/with_corr_tr/k" +str(val_of_coef_times_10)+"/result_script1/img_kmean_rf2.png", dpi=300)

plt.show()
plt.close()
plt.clf()