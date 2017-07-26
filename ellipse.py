#!/usr/bin/python

# This code generates a set of random, uniformly distributed numbers on an
# ellipse in two different ways
import numpy as np
import matplotlib.pyplot as plt
import random 
import matplotlib.patches as patches

# set number of points and ellipse size, based on interval

N = 200
ymax = 0.9
ymin = 0.3
xmax = 0.1
xmin = 0.01

a = ( xmax - xmin )/2.
b = ( ymax - ymin )/2.

print("X interval is: [{},{}]".format(xmax, xmin))
print("Y interval is: [{},{}]".format(ymax, ymin))

#--------------------------------#
#           METHOD 1             #
#--------------------------------#
# Generate random points within a unit circle
# then stretch it out into ellipse

r = np.sqrt( np.random.rand(N,1) )
theta = 2. * np.pi * np.random.rand(N,1)

x = (xmin + a) + ( a * r * np.cos(theta) )
y = (ymin + b) + ( b * r * np.sin(theta) )


#---------------------------------------#
#              METHOD 2                 #
#---------------------------------------#
# Just pick points on the rectangular interval,
# then trim off the edges not in the ellipse

xx = (a + xmin) + (a * (2. * np.random.rand(2*N,1) - 1.))
yy = (b + ymin) + (b * (2. * np.random.rand(2*N,1) - 1.))

elsp = ( (xx - (xmin+a)) **2 ) / (a * a) + ( (yy - (ymin+b)) **2 ) / (b * b)

x2 = []
y2 = []

for i in range(len(elsp)):
    if elsp[i] < 1.:
        x2.append(xx[i])
        y2.append(yy[i])

#-------------------------------------#
#            METHOD 3                 #
#-------------------------------------#
# The two previous methods were ellipses inscribed
# in our rectangular interval
# This method has the rectangular interval
# inscribed in the ellipse. So we sample points
# outside of our original interval

x3 = (xmin + a) + (np.sqrt(2.) * a * r * np.cos(theta))
y3 = (ymin + b) + (np.sqrt(2.) * b * r * np.sin(theta))

# Plots 

fig, ax = plt.subplots(figsize=(8,6))

#ax.add_patch(patches.Ellipse((xmin+a,ymin+b),2. * a,2. * b,alpha=0.2, label='Inscribed Ellipse'))
#ax.add_patch(patches.Rectangle((xmin,ymin),2. *a,2. *b,color='g',alpha=0.1, label='Square Interval' ))
#ax.add_patch(patches.Ellipse((xmin+a,ymin+b),2. * np.sqrt(2.) * a,2. * np.sqrt(2.) * b,alpha=0.1, label='Circumscribed Ellipse'))

ax.add_patch(patches.Ellipse((xmin+a,ymin+b),2. * np.sqrt(2.) * a,2. * np.sqrt(2.) * b, facecolor='#e8d7ef',edgecolor='#ece2f0', label='Circumscribed Ellipse'))

ax.add_patch(patches.Rectangle((xmin,ymin),2. *a,2. *b, facecolor='#4aa6ad', edgecolor='#1c9099', label='Square Interval' ))

ax.add_patch(patches.Ellipse((xmin+a,ymin+b),2. * a,2. * b, facecolor='#a6bddb', edgecolor='#92aed3', label='Inscribed Ellipse'))

#ax.scatter(x,y,c='r')
#ax.scatter(x2,y2,c='b')
#ax.scatter(x3,y3,c='g')
ax.legend(fontsize=18)

ax.text(0.05, 0.6, r'$\frac{x_1^2}{\Delta_1^2} + \frac{x_2^2}{\Delta_2^2} = 1$',horizontalalignment='center',verticalalignment='center',fontsize=24)
ax.text(0.05, 0.95, r'$\frac{x_1^2}{(\sqrt{2}\Delta_1)^2} + \frac{x_2^2}{(\sqrt{2}\Delta_2)^2} = 1$',horizontalalignment='center',verticalalignment='center',fontsize=24)
ax.text(0.01, 0.3, r'$[2 \Delta_1,\; 2 \Delta_2]$',horizontalalignment='left',verticalalignment='bottom',fontsize=18)

plt.ylim([0.1,1.4])
plt.xlim([-0.02,0.13])
plt.xlabel('$x_1 = \eta_B$')
plt.ylabel('$x_2 = \eta_R$')
#plt.title('Elliptical Constraints and Square Interval')
plt.savefig('intervalshapes_v2.eps', format='eps', dpi=1200)
plt.show()
