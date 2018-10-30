#!/usr/bin/env python3

import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Some NaN values naturally occur, so ignore them
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

r  = 2              # Radius of initial wavefront
dr = 0.1            # Difference between red and green circle radii
d  = np.sqrt(2)*r   # Offset of slits from origin
n  = 8              # Number of waves to plot

class Circle:
    r = None        # Radius of Circle
    d = None        # Offset of Circle
    c = None        # Color of Circle

    X = None        # Array of x values defining Circle
    Y = None        # Array of y values defining Circle

    # Initialize Circle
    def __init__(self, r, d, c):
        self.r = r
        self.d = d
        self.c = c

        # Create arrays for x and y coordinates defining Circle
        self.X = np.linspace(-self.r+self.d, +self.r+self.d, 500)
        self.Y = self.func(self.X)

        # Endpoints of circle may not lie on x axis, so correct
        self.Y[ 0] = 0.
        self.Y[-1] = 0.

    # Function defining positive half of a circle in Cartesian coordinates
    def func(self, x):
        with np.errstate(invalid='ignore'):
            return np.sqrt(self.r**2 - (x - self.d)**2)

    # Allow the Circle to plot itself
    def plot(self):
        plt.plot(self.X, self.Y, self.c, linewidth=1.)

    # Return intersection point of current Circle with another Circle
    def intersect(self, circle):
        return fsolve(lambda x : self.func(x) - circle.func(x), (circle.r**2 - self.r**2)/(4.*abs(self.d)))

# Create figure with aspect ratio of unity and scale axes such that all waves fit nicely
plt.figure(figsize=(1920*9/1080.0, 9.))
plt.axes().set_aspect('equal')
plt.title("Diffraction Grating Visualization")
plt.xlim([-n*r-d, n*r+d])
plt.ylim([ 0,     n*r+d])
plt.axis('off')

# Create data structure to hold Circle objects
circles = {'grn': {'top': [], 'bot': []},
           'red': {'top': [], 'bot': []}}

# Generate iteratively larger concentric circles at each slit
for i in range(n):
    circles['grn']['top'].append(Circle(i*r,      +d, 'g'))
    circles['grn']['bot'].append(Circle(i*r,      -d, 'g'))
    circles['red']['top'].append(Circle(i*(r+dr), +d, 'r'))
    circles['red']['bot'].append(Circle(i*(r+dr), -d, 'r'))

# Create data structure to hold interference intersections
bands = [{ 'grn': {'x': [], 'y': []},
           'red': {'x': [], 'y': []} } for _ in range(2*n-1)]

# Iterate through all generated Circles, plot them, detect intersections of
# wavefronts, and store in data structure. Note: order values are shifted up by
# n-1 so that all indices are positive for storage in the array
for col in circles:
    for loc in circles[col]:
        for circle in circles[col][loc]:
            circle.plot()

    # Sometimes imaginary numbers will be returned, so use isnan to check
    for j, topCircle in enumerate(circles[col]['top']):
        for i, botCircle in enumerate(circles[col]['bot']):
            k = j - i + n - 1
            x = topCircle.intersect(botCircle)[0]
            if not np.isnan(x):
                y = topCircle.func(x)
                if not np.isnan(y):
                    bands[k][col]['x'].append(x)
                    bands[k][col]['y'].append(y)

# Take all interference bands and plot if arrays are not empty. Alpha channel
# is used here to show decreasing strength of subsequent orders
for k, band in enumerate(bands):
    k -= n - 1
    a = 1 - abs(k)/n*2.
    for col, data in band.items():
        if len(data['x']) == 0: continue
        if   col == 'grn': c = 'g'
        elif col == 'red': c = 'r'
        plt.plot(data['x'], data['y'], 'o-', color=c, alpha=a, label=("Order %d" % abs(k)))

# Plot the dual slit plate along the X axis
plt.plot([-n*r-d,  -d-r/2.], [0, 0], 'k-', linewidth=3.)
plt.plot([-d+r/2., +d-r/2.], [0, 0], 'k-', linewidth=3.)
plt.plot([+d+r/2., +n*r+d ], [0, 0], 'k-', linewidth=3.)

# Show the legend and then save and display the figure
plt.legend()
plt.tight_layout()
plt.savefig('diffraction.png')
plt.show()
