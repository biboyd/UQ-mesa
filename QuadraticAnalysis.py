from math import factorial, floor
from itertools import combinations
import numpy as np
import sys
from scipy.optimize import curve_fit

class Point(object):
    def __init__(self, r=[], v=None):
        # Position in n-D space
        self.r = np.array(r, copy=True)
        # Value of point (Scalar)
        self.v = v
        # Dimensionality of point
        self.dm = len(r)

class Grid(object):
    def __init__(self, points):
        self.points = points
        self.dm     = points[0].dm
        self.coords = []
        self.values = []
        self.getCoords()
        self.getValues()

    def getCoords(self):
        self.coords = [[] for n in range(self.dm)]
        for p in self.points:
            for i, ri in enumerate(p.r):
                self.coords[i].append(ri)
        self.coords = np.array(self.coords)

    def getValues(self):
        self.values = np.array([p.v for p in self.points])
        
class QuadraticFit(object):
    def __init__(self, grid):
        # Takes a grid of Point objects and fits them
        # using an N-dimensional quadratic function.
        self.grid   = grid
        self.dm     = grid.dm
        self.ncoefs = 1+2*self.dm+floor(factorial(self.dm)/2)
        self.coefficients  = np.zeros(self.ncoefs)
        self.covariance    = []
        self.std_error     = []
        self.do_quad_fit()

    def combinations(self, x, n):
        # Yield all unique combinations of elements of x
        # picked n-at-a-time without replacement.
        # n <= len(x) must be satisfied.
        if n > len(x):
            sys.exit('ERROR: asked for combinations of x picking more elements than are in x!')
        #TODO: implement unsorted combinations function
            

    def get_coefs_first(self, coefficients=self.coefficients):
        # Given a vector of coefficients,
        # return the vector of coefficients of
        # first-order variables: e.g. ci*xi
        return coefficients[1:self.dm+1]

    def get_coefs_cross(self, coefficients=self.coefficients):
        # Given a vector of coefficients,
        # return the vector of coefficients of
        # cross-variable products: e.g. cij*xi*xj
        ncross = floor(factorial(self.dm)/2)
        return coefficients[self.dm+1:self.dm+ncross+2]

    def get_coefs_second(self, coefficients=self.coefficients):
        # Given a vector of coefficients,
        # return the vector of coefficients of
        # second-order variables: e.g. ci*xi**2
        return coefficients[self.dm+ncross+2:]
    
    def quadratic_nd(self, z, *coefs):
        f = coefs[0]
        cfirst = self.get_coefs_first(coefs)
        ccross = self.get_coefs_cross(coefs)
        csecond = self.get_coefs_second(coefs)
        if len(csecond) != len(z):
            sys.exit('ERROR: number of second order coefficients does not match dimensionality.')
        for ci, zi in zip(cfirst, z):
            f += ci * zi
        #TODO: use unsorted combinations function
        for k, zi, zj in enumerate(combinations(z, 2)):
            f += ccross[k] * zi * zj
        for ci, zi in zip(csecond, z):
            f += ci * zi**2
        return f

    def do_quad_fit(self):
        self.coefficients, self.covariance = curve_fit(self.quadratic_nd,
                                                       self.grid.coords, self.grid.values)
        self.std_error = np.sqrt(np.diag(self.covariance))

class EllipticOptimize(object):
    def __init__(self, quadfit, lo, hi):
        # Given a QuadraticFit object, optimize the quadratic function
        # over the outer and inner ellipses defined by the
        # rectangular region [lo, hi].
        self.quadfit = quadfit
        self.dm      = quadfit.dm
        self.lo      = np.array(lo)
        self.hi      = np.array(hi)
        self.dr      = self.hi - self.lo
        self.center  = 0.5 * (self.hi + self.lo)
        self.amat_inner = []
        self.amat_outer = []
        self.zmat_inner = []
        self.zmat_outer = []

    def quad_transform_nd(self, f0, hp, mu, tp):
        f = f0
        for hi, mi, ti in zip(hp, mu, tp):
            f += hi * ti + mi * ti**2
        return f

    def set_amat(self):
        # Set the A matrix entries for the inner (inscribed)
        # and outer (circumscribed) ellipses.
        # Note the circumscribed ellipse accounts for dimensionality.
        self.amat_inner = np.diag(np.diag([4.0/(dri**2) for dri in self.dr]))
        self.amat_outer = np.diag(np.diag([(4.0/self.dm)/(dri**2) for dri in self.dr]))

    def set_zmat(self):
        self.zmat_inner = np.linalg.inv(self.amat_inner)
        self.zmat_outer = np.linalg.inv(self.amat_outer)

    def get_extrema(self, zmat):
        w, v = np.linalg.eig(zmat)

        f1 = self.quadfit.get_coefs_first()
        #TODO: this is where you left off
        f2 = np.matrix( [[fxx, fxy], [fyx, fyy]] )

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
        print lambda_k

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

        print ('g_k is: ')
        print g_k
        print ('g_kl is: ')
        print g_kl

        g_pk = np.zeros(N)
        for k in range(N): 
            g_pk[k] = g_k[k] / np.sqrt(lambda_k[k])

        g_pkl = np.zeros( (N,N) )
        for k in range(N):
            for l in range(N):
                g_pkl[k,l] = g_kl[k,l] / (np.sqrt(lambda_k[k]) * np.sqrt(lambda_k[l]))

        print('g_pk is: ')
        print g_pk
        print('g_pkl is: ')
        print g_pkl

        mu, u = np.linalg.eig(g_pkl)

        print('Eigenvalues of g\'_kl are: ')
        print mu
        print ('Eigenvectors of g\'_kl are: ')
        print u

        h_p = np.zeros(N)

        for p in range(N):
            for k in range(N):
                h_p[p] += u[k,p] * g_pk[k]

        t_p = np.zeros(N)

        for k in range(N):
            t_p[k] = -h_p[k] / (2.0 * mu[k])

        print "t_p is:"
        print t_p

        print "sum of t_p**2 is:"
        print np.sum(t_p**2)

        if (np.sum(t_p**2)) <= 1.0:
            print ('Extremum found in the interior of the domain.')
        else:
            print('Extremum not found in the interior of the domain.')

        print "objective function:"
        print(obj_func_final(f_0, h_p, mu, t_p))

        def Cond42(a,b,c):
            result = 0.0
            for i in range( len(a) ):
                result += a[i]**2 / ( 4. * (b[i] + c)**2 )
            result -= 1.
            return result

        N_iter_max = 1000000

        N_iter = 1

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
                    print "Result not converged; stopping."
                    exit

        print "Final value of lambda:"
        lam = dmid
        print lam

        for k in range(N):
            t_p[k] = -h_p[k] / (2.0 * (mu[k] + lam))

        print "t_p:"
        print t_p

        print "sum of t_p**2:"
        print np.sum(t_p**2)

        print "objective function:"
        print obj_func_final(f_0, h_p, mu, t_p)



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

print "Minimum in elliptical region is: ", np.min(z_arr[mask])
print "Maximum in elliptical region is: ", np.max(z_arr[mask])

plt.show()
