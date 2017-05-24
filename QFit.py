
# coding: utf-8

# In[1]:

"""
Fits a quadratic function to the data. 

Reports fit parameter covariance, etc.

Follows example from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
"""


# In[2]:

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from matplotlib.mlab import griddata


# In[3]:

# Read Data
dfname = 'output2.csv'
raw_data = np.genfromtxt(dfname, delimiter=',', skip_header=1)
# Each element of data is a row from the csv file, so convert to columns
data = np.transpose(raw_data)
# data[0] = Blocker factors
# data[1] = Reimers factors
# data[2] = CO WD Mass


# In[4]:

ivars = data[0:2] # independent variables
dvars = data[2]   # dependent variable


# In[5]:

pnames = ['cx2', 'cy2', 'cxy', 'cx', 'cy', 'c0']


# In[6]:

# Define fitting function
def quadratic_2d(z, cx2, cy2, cxy, cx, cy, c0):
    x = z[0]
    y = z[1]
    f = c0 + cx * x + cy * y + cxy * x * y + cx2 * x**2 + cy2 * y**2
    return f


# In[7]:

# Fit quadratic_2d to data
popt, pcov = curve_fit(quadratic_2d, ivars, dvars)


# In[8]:

popt


# In[9]:

pcov


# For data with unknown uncertainties (e.g. CO WD Mass with no uncertainty),
# the covariance matrix is computed by using a variance calculated as the variance 
# of the residuals between the data and the best fit. It contains information
# about the statistical error of the fit due to scatter in the supplied data.

# In[10]:

# Get standard deviation on fit parameters
pstd = np.sqrt(np.diag(pcov))


# In[11]:

pstd


# In[12]:

# Print the fit parameters with 1 std errors
for n, v, s in zip(pnames, popt, pstd):
    print('{} = {} +/- {}'.format(n, v, s))


# In[13]:

# Plot the covariance matrix to see correlations between the fit parameters
labels = ['','cx2', 'cy2', 'cxy', 'cx', 'cy', 'c0']

fig = plt.figure()

plt.rc('text',usetex=True)
plt.rc('font',family='serif')

ax1 = fig.add_subplot(121)
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes('right', size='5%', pad=0.05)
cmap1 = mpl.cm.get_cmap('viridis')
bounds1 = [-5,-4,-3,-2,-1,0]
norm1 = mpl.colors.Normalize(vmin=np.amin(bounds1),vmax=np.amax(bounds1))
img1 = ax1.imshow(np.log10(np.absolute(pcov)), interpolation='nearest', cmap=cmap1, norm=norm1)
ax1.invert_xaxis()
plt.colorbar(img1, cmap=cmap1, cax=cax1, ticks=bounds1, norm=norm1)
ax1.set_title('$\mathrm{Log_{10} |COVARIANCE|}$')
ax1.set_yticklabels(labels)
ax1.set_xticklabels(labels)

ax2 = fig.add_subplot(122)
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes('right', size='5%', pad=0.05)
cmap2 = mpl.colors.ListedColormap(['red','blue'])
bounds2=[-1,0,1]
norm2 = mpl.colors.BoundaryNorm(bounds2, cmap2.N)
img2 = ax2.imshow(np.sign(pcov), interpolation='nearest', cmap=cmap2, norm=norm2)
ax2.invert_xaxis()
plt.colorbar(img2, cmap=cmap2, norm=norm2, boundaries=bounds2, ticks=bounds2, cax=cax2)
ax2.set_title('$\mathrm{Sign(COVARIANCE)}$')
ax2.set_yticklabels(labels)
ax2.set_xticklabels(labels)

plt.tight_layout(pad=3)

fig.savefig('covariance.eps',format='eps',dpi=1000)


# In[14]:

# Plot data
plt.clf()
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
splot = ax1.scatter(data[0], data[1], data[2], c=data[2], cmap='viridis')
ax1.view_init(45, 30)
ax1.set_xlabel('$\\eta_B$')
ax1.set_ylabel('$\\eta_R$')
ax1.set_zlabel('$M_{WD}$')
plt.tight_layout()

fig.savefig('scat.eps',format='eps',dpi=1000)


# In[15]:

# Plot data
xi = np.linspace(min(data[0]),max(data[0]))
yi = np.linspace(min(data[1]),max(data[1]))
X, Y = np.meshgrid(xi,yi)
Z = griddata(data[0],data[1],data[2],xi,yi,interp='linear')
plt.clf()
fig = plt.figure()
ax1 = fig.gca(projection='3d')
suplot = ax1.plot_surface(X, Y, Z,rstride=1,cstride=1, cmap=plt.cm.viridis,vmax=max(data[2]),vmin=min(data[2]),linewidth=0.4)
ax1.view_init(45, 30)
ax1.set_xlabel('$\\eta_B$')
ax1.set_ylabel('$\\eta_R$')
ax1.set_zlabel('$M_{WD}$')
plt.tight_layout()

fig.savefig('surf3.eps',format='eps',dpi=1000)


# In[16]:

# Plot fit
plt.clf()
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
xvec = np.linspace(np.amin(data[0]), np.amax(data[0]), num=2*len(data[0]))
yvec = np.linspace(np.amin(data[1]), np.amax(data[1]), num=2*len(data[1]))

x, y = np.meshgrid(xvec, yvec)
zfit = quadratic_2d([x,y], *popt)
vals = []
for i in zfit:
    for j in i:
        vals.append(j)
surf = ax1.plot_surface(x, y, zfit, cmap=mpl.cm.viridis, vmax=round(max(vals),1), linewidth=0.4, alpha=1.0)
ax1.view_init(45, 30)
fig.colorbar(surf)
ax1.set_xlabel('$\\eta_B$')
ax1.set_ylabel('$\\eta_R$')
ax1.set_zlabel('$M_{WD}$')
plt.tight_layout()

fig.savefig('surf.eps',format='eps',dpi=1000)


# In[17]:

# Get fit residuals
mfit = quadratic_2d(data[0:2], *popt)
resd = data[2] - mfit

print('min resd: {}\n'.format(np.amin(resd)))
print('max resd: {}\n'.format(np.amax(resd)))

# Sum of squares 
avgm = np.mean(data[2])
sqrs = data[2] - avgm
tssq = np.sum(sqrs**2)
print('total sum of squares is: {}\n'.format(tssq))

# Sum of squared residuals
ssrd = np.sum(resd**2)
print('sum of squared residuals: {}\n'.format(ssrd))
print('sqrt sum of squared residuals: {}\n'.format(np.sqrt(ssrd)))

# Sum of squared (residual/data)
ssrn = np.sum((resd/data[2])**2)
print('sum of squared (residual/data): {}\n'.format(ssrn))
print('sqrt sum of squared (residual/data): {}\n'.format(np.sqrt(ssrn)))

# Calculate R^2 = 1 - ssrd/tssq
FUV = ssrd/tssq
R2 = 1. - FUV
print('R-squared value is: {}\n'.format(R2))


# In[18]:

# Plot Residuals on the eta domain
fig = plt.figure()
ax1 = fig.add_subplot(111)
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes('right', size='5%', pad=0.05)
cmap1 = mpl.cm.get_cmap('seismic_r')
bounds1 = [-0.025,-0.015,0,0.015,0.025]
norm1 = mpl.colors.Normalize(vmin=np.amin(bounds1),vmax=np.amax(bounds1))
img1 = ax1.scatter(data[0], data[1], c=resd, cmap=cmap1, norm=norm1)
#ax1.invert_xaxis()
plt.colorbar(img1, cmap=cmap1, cax=cax1, ticks=bounds1, norm=norm1, label='$M_{res}~(M_{\odot})$')
#ax1.set_title('$\mathrm{Log_{10} |COVARIANCE|}$')
ax1.set_ylabel('$\eta_R$')
ax1.set_xlabel('$\eta_B$')
plt.tight_layout()


# In[19]:

# Plot projected residuals
fig = plt.figure()

ax1 = fig.add_subplot(121)
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes('right', size='5%', pad=0.05)
cmap1 = mpl.cm.get_cmap('seismic_r')
bounds1 = [-0.025,-0.015,0,0.015,0.025]
norm1 = mpl.colors.Normalize(vmin=np.amin(bounds1),vmax=np.amax(bounds1))
img1 = ax1.scatter(data[0], resd, c=resd, cmap=cmap1, norm=norm1)
ax1.set_xlim([-0.05, 0.15])
#ax1.invert_xaxis()
plt.colorbar(img1, cmap=cmap1, cax=cax1, ticks=bounds1, norm=norm1, label='$M_{res}~(M_{\odot})$')
#ax1.set_title('$\mathrm{Log_{10} |COVARIANCE|}$')
ax1.set_xlabel('$\eta_B$')

ax2 = fig.add_subplot(122)
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes('right', size='5%', pad=0.05)
cmap2 = mpl.cm.get_cmap('seismic_r')
bounds2 = [-0.025,-0.015,0,0.015,0.025]
norm2 = mpl.colors.Normalize(vmin=np.amin(bounds2),vmax=np.amax(bounds2))
img2 = ax2.scatter(data[1], resd, c=resd, cmap=cmap2, norm=norm2)
ax2.set_xlim([0.25, 0.95])
#ax2.invert_xaxis()
plt.colorbar(img2, cmap=cmap2, cax=cax2, ticks=bounds2, norm=norm2, label='$M_{res}~(M_{\odot})$')
#ax2.set_title('$\mathrm{Log_{10} |COVARIANCE|}$')
ax2.set_xlabel('$\eta_R$')

plt.tight_layout()


# In[20]:

# Plot projected residuals
fig, (ax1,ax2) = plt.subplots(1,2, sharey=True)

cmap1 = mpl.cm.get_cmap('seismic_r')
bounds1 = [-0.025,-0.015,0,0.015,0.025]
norm1 = mpl.colors.Normalize(vmin=np.amin(bounds1),vmax=np.amax(bounds1))
img1 = ax1.scatter(data[0], resd, c=resd, cmap=cmap1, norm=norm1)
ax1.set_xlim([-0.05, 0.15])
#plt.colorbar(img1, cmap=cmap1, cax=cax1, ticks=bounds1, norm=norm1, label='$M_{res}~(M_{\odot})$')
ax1.set_xlabel('$\eta_B$')

#ax2 = fig.add_subplot(122)
cmap2 = mpl.cm.get_cmap('seismic_r')
bounds2 = [-0.025,-0.015,0,0.015,0.025]
norm2 = mpl.colors.Normalize(vmin=np.amin(bounds2),vmax=np.amax(bounds2))
img2 = ax2.scatter(data[1], resd, c=resd, cmap=cmap2, norm=norm2)
ax2.set_xlim([0.25, 0.95])
ax2.set_xlabel('$\eta_R$')
plt.subplots_adjust(wspace=0)
yticks = ax1.yaxis.get_major_ticks()
yticks2 = ax2.yaxis.get_major_ticks()
yticks2[7].set_visible(False)
yticks[7].set_visible(False)
yticks[0].set_visible(False)

ax1.set_xlim([-0.005,0.105])
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(img2, cmap=cmap2, cax=cax2, ticks=bounds2, norm=norm2, label='$M_{res}~(M_{\odot})$')

fig.savefig('residuals.eps',format='eps',dpi=1000)



# In[21]:

# Plot data
plt.clf()
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
splot = ax1.scatter(data[0], data[1], data[2], c='blue')

xvec = np.linspace(np.amin(data[0]), np.amax(data[0]), num=10*len(data[0]))
yvec = np.linspace(np.amin(data[1]), np.amax(data[1]), num=10*len(data[1]))
x, y = np.meshgrid(xvec, yvec)
zfit = quadratic_2d([x,y], *popt)
surf = ax1.plot_surface(x, y, zfit, cmap=mpl.cm.viridis, linewidth=0, alpha=0.5)
fig.colorbar(surf)

ax1.view_init(25, 60)
ax1.set_xlabel('$\\eta_B$')
ax1.set_ylabel('$\\eta_R$')
ax1.set_zlabel('$M_{WD}$')
plt.tight_layout()

fig.savefig('planeandscatter.eps',format='eps',dpi=1000)


# In[ ]:



