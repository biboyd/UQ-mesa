#!/usr/bin/python

# Quadratic Response Surface Modelling Algorithm
#
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from quadratic_analysis import read_coeff, obj_func_init

def load_data(dataDir):
    B = np.load(f"{dataDir}/Block.npy")
    R = np.load(f"{dataDir}/Reims.npy")
    M = np.load(f"{dataDir}/StarM.npy")
    return B, R, M

def plot3d(dataDir, x, y, z):
    B, R, M, = load_data(dataDir)

    # setup plotly data
    mydata = []
    mydata.append(go.Scatter3d(x=B, y=R, z=M,
                               mode='markers', name='Succeed', opacity=0.8)
              )
    mydata.append(go.Surface(x=x, y=y, z=z, opacity=0.8))

    #plot
    fig = go.Figure(data=mydata)

    #update labels
    fig.update_layout(scene=dict(xaxis_title='Block', yaxis_title='Reims', zaxis_title='Solar Mass'),
                 width=700, height=700,
                 margin=dict(r=20, l=20, b=20, t=20))

    return fig


dataDir = "./"

x = [0.1, 0.5]
y = [0.4, 0.9]

dx = x[1] - x[0]
dy = y[1] - y[0]

xc = (x[0] + x[1]) / 2.0
yc = (y[0] + y[1]) / 2.0

# Plot the function over the range.

plot_npts = 1000
x_arr1 = np.linspace(x[0], x[1], plot_npts)
y_arr1 = np.linspace(y[0], y[1], plot_npts)
x_arr, y_arr = np.meshgrid(x_arr1, y_arr1)
z_arr = obj_func_init(x_arr, y_arr)

# plotly plot
plotly_fig = plot3d(dataDir, x_arr1, y_arr1, z_arr)
plotly_fig.write_html("quad_points.html")
plotly_fig.write_image("quad_points.png")