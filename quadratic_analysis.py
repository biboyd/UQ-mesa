#!/usr/bin/python

# Quadratic Response Surface Modelling Algorithm
#
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def obj_func_init(x, y):
    yint = 0.559567248037217
    fx = -1.1322751540276104
    fy = 0.13778613804801929
    fxx = 5.768756688622785
    fyy = -0.20601276511812017
    fxy = 0.4645735171653338/2.0
    fyx = 0.4645735171653338/2.0
    
    # yint = 0.5596    
    # fx = -1.1323     
    # fy = 0.1378      
    # fxx = 5.7688     
    # fyy = -0.2060     
    # fxy = 0.4646/2.0 
    # fyx = 0.4646/2.0 

    # yint = 0.5490
    # fx = -0.6414
    # fy = 0.1296
    # fxx = 2.0293
    # fyy = -0.1975
    # fxy = 0.4279/2.0
    # fyx = 0.4279/2.0           

    obj = yint + fx * x + fy * y + \
                 fxx * x**2 + fyy * y**2 + \
                 fxy * x * y + fyx * y * x

    return obj

def obj_func_final(f0, hp, mu, tp):
    obj = f0

    for i in range(2):
        obj += hp[i] * tp[i]

    for i in range(2):
        obj += mu[i] * tp[i]**2

    return obj

N = 2

x = [0.01, 0.1]
y = [0.3, 0.9]

dx = x[1] - x[0]
dy = y[1] - y[0]

xc = (x[0] + x[1]) / 2.0
yc = (y[0] + y[1]) / 2.0

# Plot the function over the range.

plot_npts = 1000
x_arr = np.linspace(x[0], x[1], plot_npts)
y_arr = np.linspace(y[0], y[1], plot_npts)
x_arr, y_arr = np.meshgrid(x_arr, y_arr)
z_arr = obj_func_init(x_arr, y_arr)
fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(x_arr, y_arr, z_arr)

ax.view_init(elev = 45, azim = 35)
plt.savefig('function.png')

# Get a rough estimate of the bounds of the inner elliptical region through sampling.

mask = np.where(( (x_arr - xc) / (dx / 2.0) )**2 + ( (y_arr - yc) / (dy / 2.0) )**2 <= 1.0)

print("Minimum in inner elliptical region is: ", np.min(z_arr[mask]))
print("Maximum in inner elliptical region is: ", np.max(z_arr[mask]))

mask = np.where(( (x_arr - xc) / (dx / np.sqrt(2.0)) )**2 + ( (y_arr - yc) / (dy / np.sqrt(2.0)) )**2 <= 1.0)

print("Minimum in outer elliptical region is: ", np.min(z_arr[mask]))
print("Maximum in outer elliptical region is: ", np.max(z_arr[mask]))

#Inner
#a_ik = np.matrix ( [[1.0/(dx/2.)**2, 0.0], [0.0, 1.0/(dy/2.)**2]] )

#Outer
a_ik = np.matrix ( [[1.0/(dx/ np.sqrt(2.0))**2, 0.0], [0.0, 1.0/(dy/ np.sqrt(2.0))**2]] )

# Z is the inverse of a_ik
Z = np.linalg.inv(a_ik)

print('The inverse matrix is: ')
print(Z)

# Compute eigenvectors/values

w, v = np.linalg.eig(Z)

print('Eigenvalues of Z are: ')
print(w)
print('Eigenvectors of Z are: ')
print(v)

# Coefficients from quadratic fits

# yint = 0.5490
# fx = -0.6414
# fy = 0.1296
# fxx = 2.0293
# fyy = -0.1975
# fxy = 0.4279/2.0
# fyx = 0.4279/2.0

yint = 0.559567248037217
fx = -1.1322751540276104
fy = 0.13778613804801929
fxx = 5.768756688622785
fyy = -0.20601276511812017
fxy = 0.4645735171653338/2.0
fyx = 0.4645735171653338/2.0

f_i = [fx, fy]
f_ij = np.matrix( [[fxx, fxy], [fyx, fyy]] )
x_i = [xc, yc]

# Construct f'_0

f_0 = yint
for i in range(N):
    f_0 += f_i[i] * x_i[i]
    for j in range(N):
        f_0 += f_ij[i,j] * x_i[i] * x_i[j]

# Construct f'_i

for i in range(N):
    for j in range(N):
        f_i[i] += 2.0 * f_ij[i,j] * x_i[j]

print('f\'_0 is: ')
print(f_0)
print('f\'_i is: ')
print(f_i)
print('f_ij is: ')
print(f_ij)

lambda_k = np.zeros(N)

for k in range(N):
    lambda_k[k] = 1.0 / w[k]

print('lambda_k: ')
print(lambda_k)

g_k = np.zeros(N)

for k in range(N):
    for i in range(N):
        g_k[k] += f_i[i] * v[i,k]

g_kl = np.zeros( (N, N) )

for k in range(N):
    for l in range(N):
        for j in range(N):
            for i in range(N):
                g_kl[k,l] += f_ij[i,j] * v[i,k] * v[j,l]

print('g_k is: ')
print(g_k)
print('g_kl is: ')
print(g_kl)

g_pk = np.zeros(N)
for k in range(N): 
    g_pk[k] = g_k[k] / np.sqrt(lambda_k[k])

g_pkl = np.zeros( (N,N) )
for k in range(N):
    for l in range(N):
        g_pkl[k,l] = g_kl[k,l] / (np.sqrt(lambda_k[k]) * np.sqrt(lambda_k[l]))

print('g_pk is: ')
print(g_pk)
print('g_pkl is: ')
print(g_pkl)

mu, u = np.linalg.eig(g_pkl)

print('Eigenvalues of g\'_kl are: ')
print(mu)
print('Eigenvectors of g\'_kl are: ')
print(u)

h_p = np.zeros(N)

for p in range(N):
    for k in range(N):
        h_p[p] += u[k,p] * g_pk[k]

t_p = np.zeros(N)

for k in range(N):
    t_p[k] = -h_p[k] / (2.0 * mu[k])

print("t_p is:")
print(t_p)

print("sum of t_p**2 is:")
print(np.sum(t_p**2))

if (np.sum(t_p**2)) <= 1.0:
    print('Extremum found in the interior of the domain.')
else:
    print('Extremum not found in the interior of the domain.')

print("objective function:")
print(obj_func_final(f_0, h_p, mu, t_p))

def Cond42(a,b,c):
    result = 0.0
    for i in range( len(a) ):
        result += a[i]**2 / ( 4. * (b[i] + c)**2 )
    result -= 1.
    return result

N_iter_max = 1000000

N_iter = 1

# Print Values of Cond42 at -100, 0, 100
print("Cond42 at -100: ", Cond42(h_p, mu, -100.0))
print("Cond42 at 0: ", Cond42(h_p, mu, 0.0))
print("Cond42 at 100: ", Cond42(h_p, mu, 100.0))

# Change this range to [-100, 0] (say) to get the other bound

dlo = 0.0
dhi = 100.0

tol = 1.0e-8

while (N_iter <= N_iter_max):
        dmid = (dlo + dhi) / 2.0

        func_lo = Cond42(h_p, mu, dlo)
        func_mid = Cond42(h_p, mu, dmid)

        if ((dhi - dlo) / 2.0 < tol):
            break
        if (func_lo * func_mid < 0.0):
            dhi = dmid
        else:
            dlo = dmid

        N_iter = N_iter + 1

        if (N_iter == N_iter_max):
            print("Result not converged; stopping.")
            exit()

print("Final value of lambda:")
lam = dmid
print(lam)

for k in range(N):
    t_p[k] = -h_p[k] / (2.0 * (mu[k] + lam))

print("t_p:")
print(t_p)

print("sum of t_p**2:")
print(np.sum(t_p**2))

print("objective function:")
print(obj_func_final(f_0, h_p, mu, t_p))



# Get the bounds of the final objective function; this should be
# identical to the bounds of the original objective function if
# we did everything correctly.

for i in range(plot_npts):
    for j in range(plot_npts):
        x_arr[i,j] = (x_arr[i,j] - xc) * lambda_k[0]

for i in range(plot_npts):
    for j in range(plot_npts):
        y_arr[i,j] = (y_arr[i,j] - yc) * lambda_k[1]

for i in range(plot_npts):
    for j in range(plot_npts):
        z_arr[i,j] = obj_func_final(f_0, h_p, mu, [x_arr[i,j], y_arr[i,j]])

mask = np.where(x_arr**2 + y_arr**2 <= 1.0)

print("Minimum in elliptical region is: ", np.min(z_arr[mask]))
print("Maximum in elliptical region is: ", np.max(z_arr[mask]))

#plt.show()
