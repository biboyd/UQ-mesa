import sys
import operator
import itertools
import numpy as np
from math import factorial, floor
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize, brentq
import random

class Point(object):
    def __init__(self, r=[], v=None):
        # Position in n-D space
        self.r = np.array(r, copy=True)
        # Value of point (Scalar)
        self.v = v
        # Dimensionality of point
        self.dm = len(r)

class Grid(object):
    def __init__(self, points=None):
        self.coords = []
        self.values = []
        if points:
            self.points = points
            self.dm     = points[0].dm
            self.getCoords()
            self.getValues()

    def initFromCSV(self, fname, delimiter=',', skip_header=1, dim=2):
        # effectively an alias for initFromTXT but delimited by commas
        # instead of spaces.
        self.initFromTXT(fname, delimiter=delimiter, skip_header=skip_header, dim=dim)

    def initFromTXT(self, fname, delimiter=' ', skip_header=1, dim=2):
        # Given the name (fname) of a txt file
        # containing a series of entries x1, x2, ..., xn, v
        # with skip_header lines of header,
        # extract the points (1 per line) from the file
        # and store in the grid.

        # The argument dim specifies the dimensionality of the
        # independent variables x1, x2, ..., xn. Thus dim=n.

        # Additional columns may follow v, but they will be ignored.
        # That is:
        # x1, x2, ..., xn, v, [ignored columns]

        # Clear the grid first
        self.dm = dim
        self.points = []
        self.coords = []
        self.values = []

        # Open file and read points
        raw_data = np.genfromtxt(fname, delimiter=delimiter, skip_header=skip_header)
        if len(raw_data)==0:
            sys.exit('ERROR: tried to open file with no data.')
        
        # Each element of data is a row from the file file: create points
        self.points = []
        for row in raw_data:
            self.points.append(Point(r=row[0:dim], v=row[dim]))

        # Check dimensionality, set coordinates and values
        assert(self.dm == self.points[0].dm)
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
        self.cross_indices = []
        self.init_cross_indices()
        self.do_quad_fit()
        self.sum_squared_residuals = None
        self.coefficient_determination = None
        self.compute_fit_statistics()

    def __str__(self):
        # Returns a string with the quadratic fit with coefficients
        # and 1-sigma errors in a pretty format.
        pretty = "f(x) = ({} +/- {}) + \n".format(self.get_coefs_const(),
                                                  self.get_error_const())
        for i, (ci, dci) in enumerate(zip(self.get_coefs_first(),
                                          self.get_error_first())):
            pretty += "       ({} +/- {}) x[{}] + \n".format(ci, dci, i)
        for k, (ck, dck) in enumerate(zip(self.get_coefs_cross(),
                                          self.get_error_cross())):
            i, j = self.get_cross_indices(k)
            pretty += "       ({} +/- {}) x[{}] * x[{}] + \n".format(ck, dck, i, j)
        for i, (ci, dci) in enumerate(zip(self.get_coefs_second(),
                                          self.get_error_second())):
            pretty += "       ({} +/- {}) x[{}]**2".format(ci, dci, i)
            if i < self.dm - 1:
                pretty += " + \n"
        pretty += "\n\n"
        pretty += "Sum of squares of residuals: {}\n".format(self.sum_squared_residuals)
        pretty += "Coefficient of determination (R^2): {}".format(self.coefficient_determination)
        return pretty

    def writelog(self, file_handle):
        # Write a log of the quadratic fit to the file_handle
        file_handle.write('# QUADRATIC FIT LOG\n')
        file_handle.write('# DIMENSIONALITY = {}\n'.format(self.dm))
        file_handle.write('# NUMBER OF POINTS = {}\n'.format(len(self.grid.points)))
        file_handle.write('# INTERCEPT:\n')
        file_handle.write('{}\n'.format(self.get_coefs_const()))
        file_handle.write('# FIRST ORDER COEFFICIENTS:\n')
        for ci, dci in zip(self.get_coefs_first(),
                           self.get_coefs_first(self.std_error)):
            file_handle.write('{} +/- {}\n'.format(ci, dci))
        file_handle.write('# CROSS COEFFICIENTS:\n')
        ccross = self.get_coefs_cross()
        dccross = self.get_coefs_cross(self.std_error)
        for k, ck in enumerate(ccross):
            dck = dccross[k]
            i, j = self.get_cross_indices(k)
            file_handle.write('({},{}) : {} +/- {}\n'.format(i,j,ck,dck))
        file_handle.write('# SECOND ORDER COEFFICIENTS:\n')
        for ci, dci in zip(self.get_coefs_second(),
                           self.get_coefs_second(self.std_error)):
            file_handle.write('{} +/- {}\n'.format(ci, dci))
        
    def init_cross_indices(self):
        # Set up a matrix of indices
        # into the coefs_cross vector
        # (returned by self.get_coefs_cross)
        # where the (i,j) entry of the matrix
        # contains the index k into coefs_cross
        # storing the coefficient c[k] for
        # the cross term c[k]*z[i]*z[j]
        self.cross_indices = np.zeros((self.dm, self.dm), dtype=int)
        iupper = 0
        ilower = 0
        for i in range(self.dm):
            for j in range(self.dm):
                if j > i:
                    self.cross_indices[i,j] = iupper
                    iupper += 1
                elif j < i:
                    self.cross_indices[i,j] = ilower
                    ilower += 1

    def get_cross_indices(self, k):
        # Given the index k into the coefs_cross vector,
        # return the indices i, j into z such that the
        # coefficient c[k] corresponds to the
        # term c[k]*z[i]*z[j] with i != j.
        if k < 0 or k > self.dm * (self.dm - 1)/2:
            sys.exit('ERROR: k not in range [0,N*(N-1)/2]')
        else:
            for i in range(self.dm):
                for j in range(self.dm):
                    if j > i and self.cross_indices[i,j]==k:
                        return i,j

    def get_coefs_const(self, coefficients=[]):
        # Given a vector of coefficients,
        # return the intercept coefficient
        if len(coefficients)==0:
            coefficients = self.coefficients
        return coefficients[0]

    def get_error_const(self):
        return self.get_coefs_const(self.std_error)

    def get_coefs_first(self, coefficients=[]):
        # Given a vector of coefficients,
        # return the vector of coefficients of
        # first-order variables: e.g. ci*xi
        if len(coefficients)==0:            
            coefficients = self.coefficients
        return coefficients[1:self.dm+1]

    def get_error_first(self):
        return self.get_coefs_first(self.std_error)

    def get_coefs_cross(self, coefficients=[]):
        # Given a vector of coefficients,
        # return the vector of coefficients of
        # cross-variable products: e.g. cij*xi*xj
        if len(coefficients)==0:                        
            coefficients = self.coefficients
        ncross = int(floor(factorial(self.dm)/2))
        return coefficients[self.dm+1:self.dm+1+ncross]

    def get_error_cross(self):
        return self.get_coefs_cross(self.std_error)    

    def get_coefs_second(self, coefficients=[]):
        # Given a vector of coefficients,
        # return the vector of coefficients of
        # second-order variables: e.g. ci*xi**2
        if len(coefficients)==0:                        
            coefficients = self.coefficients
        ncross = int(floor(factorial(self.dm)/2))
        return coefficients[self.dm+1+ncross:]

    def get_error_second(self):
        return self.get_coefs_second(self.std_error)    

    def quadratic_nd(self, z, *coefs):
        f = self.get_coefs_const(coefs)
        cfirst = self.get_coefs_first(coefs)
        ccross = self.get_coefs_cross(coefs)
        csecond = self.get_coefs_second(coefs)
        if len(csecond) != len(z):
            sys.exit('ERROR: number of second order coefficients does not match dimensionality.')
        for ci, zi in zip(cfirst, z):
            f += ci * zi
        for k, ck in enumerate(ccross):
            i, j = self.get_cross_indices(k)
            f += ck * z[i] * z[j]
        for ci, zi in zip(csecond, z):
            f += ci * zi**2
        return f

    def compute_fit_statistics(self):
        # Obtain fitting statistics like the sum of squares of residuals
        # and the coefficient of determination
        residuals = [p.v - self.quadratic_nd(p.r, *self.coefficients) for p in self.grid.points]
        residuals = np.array(residuals)
        data_mean  = np.mean(self.grid.values)
        sum_squared_data = np.sum((self.grid.values - data_mean)**2)
        self.sum_squared_residuals = np.sum(residuals**2)
        self.coefficient_determination = 1.0 - self.sum_squared_residuals/sum_squared_data

    def do_quad_fit(self):
        self.coefficients, self.covariance = curve_fit(self.quadratic_nd,
                                                       self.grid.coords,
                                                       self.grid.values,
                                                       p0=np.ones(self.ncoefs))
        self.std_error = np.sqrt(np.diag(self.covariance))

class RectangularOptimize(object):
    def __init__(self, quadfit, lo, hi, npts=20, fail_threshold=10, verbose=False):
        # Given a QuadraticFit object, optimize the quadratic function
        # over the rectangular region [lo, hi].
        # This is inspired by sample code from Doug Swesty.
        
        self.quadfit = quadfit
        self.dm      = quadfit.dm
        self.lo      = np.array(lo)
        self.hi      = np.array(hi)
        self.dr      = self.hi - self.lo
        self.center  = 0.5 * (self.hi + self.lo)
        self.verbose = verbose
        self.min_function, self.min_location, self.max_function, self.max_location = self.analyze(npts=npts,
                                                                                                  fail_threshold=fail_threshold)

    def __str__(self):
        pretty = 'Rectangular Bounds: [{}, {}]\n'.format(self.min_function, self.max_function)
        return pretty

    def do_optimize(self, start_guess=[]):
        # Optimize the quadratic function using scipy.optimize
        # and the 'L-BFGS-B' method
        method = 'L-BFGS-B'
        ztol = 1.0e-9

        # Use the array passed as the starting guess
        # for the optimization. If no array is passed
        # then generate a random sample in the rectangular domain.
        if list(start_guess):
            guess = np.array(start_guess)
        else:
            guess = np.array([random.uniform(xlo, xhi) for xlo, xhi in zip(self.lo, self.hi)])

        # Rectangular domain bounds for optimization
        bounds = [[xlo, xhi] for xlo, xhi in zip(self.lo, self.hi)]

        # Get minimum of quadratic function
        minopt = minimize(lambda x: self.quadfit.quadratic_nd(x, *self.quadfit.coefficients),
                          guess, method=method, bounds=bounds, tol=ztol)

        fmin = minopt['fun']
        xmin = minopt['x']
        success_min = minopt['success']
        assert(success_min)
        
        # Get maximum of quadratic function
        maxopt = minimize(lambda x: -self.quadfit.quadratic_nd(x, *self.quadfit.coefficients),
                          guess, method=method, bounds=bounds, tol=ztol)

        fmax = -maxopt['fun']
        xmax = maxopt['x']
        success_max = maxopt['success']
        assert(success_max)

        return fmin, xmin, fmax, xmax

    def analyze(self, npts=20, fail_threshold=10):
        # Sample the rectangle multiple times to get starting
        # values for optimization. This avoids the optimization
        # "getting stuck" especially at edges (thanks to Doug).
        #
        # npts is the number of initial points to sample for do_optimize().

        min_function = None
        min_location = None
        max_function = None
        max_location = None

        nfailed = 0
        nsuccess = 0

        while(nsuccess < npts and nfailed < fail_threshold):
            try:
                fmin, xmin, fmax, xmax = self.do_optimize()
            except AssertionError:
                nfailed += 1
                pass
            else:
                if nsuccess == 0:
                    min_function = fmin
                    min_location = xmin
                    max_function = fmax
                    max_location = xmax
                else:
                    if fmin < min_function:
                        min_function = fmin
                        min_location = xmin
                    if fmax > max_function:
                        max_function = fmax
                        max_location = xmax
                nsuccess += 1

        if nsuccess == npts:
            self.success = True
            if self.verbose:
                print('Rectangular optimization succeeded with {} successful samples and {} failures'.format(nsuccess, nfailed))
        else:
            self.success = False
            print('Rectangular Optimization failed - you may try increasing fail_threshold')
        return min_function, min_location, max_function, max_location

class EllipticOptimize(object):
    def __init__(self, quadfit, lo, hi, nmesh=False, verbose=False):
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
        if nmesh:
            self.domesh  = True
            self.nmeshpts = nmesh
        else:
            self.domesh = False
            self.nmeshpts = 0
        self.verbose = verbose
        self.minmax_inner = None
        self.minmax_outer = None

        self.set_amat()
        self.set_zmat()

        if self.verbose:
            print('\n------------ INNER ELLIPSE OPTIMIZATION')
        self.inner_min, self.inner_max, self.inner_xmin, self.inner_xmax, isuccess = self.get_extrema(self.amat_inner, 'inner')

        if self.verbose:
            print('\n------------ INNER ELLIPSE SUMMARY')
            print('solved inner min = {}'.format(self.inner_min))
            print('inner min at x = {}'.format(self.inner_xmin))
            print('solved inner max = {}'.format(self.inner_max))
            print('inner max at x = {}'.format(self.inner_xmax))

        # This doesn't work
        # inner_min, inner_max, isuccess2 = self.get_extrema_slsqp(self.amat_inner)
        # if self.verbose:
        #     print('SLSQP inner min = {}'.format(inner_min))
        #     print('SLSQP inner max = {}'.format(inner_max))

        isuccess2 = True

        if self.domesh:
            self.mesh_inner_min, self.mesh_inner_max, self.mesh_inner_xmin, self.mesh_inner_xmax, isuccess3 = self.get_extrema_mesh(self.amat_inner, self.nmeshpts)
            if self.verbose:
                print('Mesh Sampling inner min = {}'.format(self.mesh_inner_min))
                print('inner min at x = {}'.format(self.mesh_inner_xmin))            
                print('Mesh Sampling inner max = {}'.format(self.mesh_inner_max))
                print('inner max at x = {}'.format(self.mesh_inner_xmax))
        else:
            self.mesh_inner_min = None
            self.mesh_inner_max = None
            self.mesh_inner_xmin = None
            self.mesh_inner_xmax = None
            isuccess3 = True
            
        isuccess = isuccess and isuccess2 and isuccess3
            
        if self.verbose:
            print('\n------------ OUTER ELLIPSE OPTIMIZATION')
        self.outer_min, self.outer_max, self.outer_xmin, self.outer_xmax, osuccess = self.get_extrema(self.amat_outer, 'outer')
        
        if self.verbose:
            print('\n------------ OUTER ELLIPSE SUMMARY')            
            print('solved outer min = {}'.format(self.outer_min))
            print('outer min at x = {}'.format(self.outer_xmin))            
            print('solved outer max = {}'.format(self.outer_max))
            print('outer max at x = {}'.format(self.outer_xmax))                        

        # This doesn't work
        # outer_min, outer_max, osuccess2 = self.get_extrema_slsqp(self.amat_outer)
        # if self.verbose:
        #     print('SLSQP outer min = {}'.format(outer_min))
        #     print('SLSQP outer max = {}'.format(outer_max))

        osuccess2 = True
        
        if self.domesh:            
            self.mesh_outer_min, self.mesh_outer_max, self.mesh_outer_xmin, self.mesh_outer_xmax, osuccess3 = self.get_extrema_mesh(self.amat_outer, self.nmeshpts)
            if self.verbose:
                print('Mesh Sampling outer min = {}'.format(self.mesh_outer_min))
                print('outer min at x = {}'.format(self.mesh_outer_xmin))
                print('Mesh Sampling outer max = {}'.format(self.mesh_outer_max))
                print('outer max at x = {}'.format(self.mesh_outer_xmax))            
        else:
            self.mesh_outer_min = None
            self.mesh_outer_max = None
            self.mesh_outer_xmin = None
            self.mesh_outer_xmax = None
            osuccess3 = True
            
        osuccess = osuccess and osuccess2 and osuccess3
            
        self.success = isuccess and osuccess

    def __str__(self):
        pretty = ""
        pretty += 'Inner Bounds: [{}, {}]\n'.format(self.inner_min, self.inner_max)
        pretty += 'Inner Bounds: [{}, {}] (Mesh)\n'.format(self.mesh_inner_min, self.mesh_inner_max)
        pretty += 'Outer Bounds: [{}, {}]\n'.format(self.outer_min, self.outer_max)
        pretty += 'Outer Bounds: [{}, {}] (Mesh)'.format(self.mesh_outer_min, self.mesh_outer_max)
        return pretty

    def writelog(self, file_handle):
        # Given a file_handle, write a log of the Elliptic Optimization
        file_handle.write('\n# ELLIPTIC OPTIMIZATION LOG\n')
        file_handle.write('# INNER MINIMUM, INNER MAXIMUM:\n')
        file_handle.write('{}, {}\n'.format(self.inner_min, self.inner_max))
        file_handle.write('# LOCATION OF INNER MINIMUM:\n')
        file_handle.write('{}\n'.format(self.inner_xmin))        
        file_handle.write('# LOCATION OF INNER MAXIMUM:\n')
        file_handle.write('{}\n'.format(self.inner_xmax))                
        file_handle.write('# OUTER MINIMUM, OUTER MAXIMUM:\n')
        file_handle.write('{}, {}\n'.format(self.outer_min, self.outer_max))        
        file_handle.write('# LOCATION OF OUTER MINIMUM:\n')
        file_handle.write('{}\n'.format(self.outer_xmin))                
        file_handle.write('# LOCATION OF OUTER MAXIMUM:\n')
        file_handle.write('{}\n'.format(self.outer_xmax))                        

        # Write a log of the Mesh Optimization
        file_handle.write('\n# ELLIPTIC MESH SAMPLING LOG\n')
        file_handle.write('# NUMBER OF POINTS PER DIMENSION:\n')
        file_handle.write('{}\n'.format(self.nmeshpts))
        file_handle.write('# INNER MINIMUM, INNER MAXIMUM:\n')
        file_handle.write('{}, {}\n'.format(self.mesh_inner_min, self.mesh_inner_max))
        file_handle.write('# LOCATION OF INNER MINIMUM:\n')
        file_handle.write('{}\n'.format(self.mesh_inner_xmin))        
        file_handle.write('# LOCATION OF INNER MAXIMUM:\n')
        file_handle.write('{}\n'.format(self.mesh_inner_xmax))                
        file_handle.write('# OUTER MINIMUM, OUTER MAXIMUM:\n')
        file_handle.write('{}, {}\n'.format(self.mesh_outer_min, self.mesh_outer_max))        
        file_handle.write('# LOCATION OF OUTER MINIMUM:\n')
        file_handle.write('{}\n'.format(self.mesh_outer_xmin))                
        file_handle.write('# LOCATION OF OUTER MAXIMUM:\n')
        file_handle.write('{}\n'.format(self.mesh_outer_xmax))                        
        
    def quad_transform_nd(self, fp0, hp, mu, tp):
        f = fp0
        for hi, mi, ti in zip(hp, mu, tp):
            f += hi * ti + mi * ti**2
        return f

    def set_amat(self):
        # Set the A matrix entries for the inner (inscribed)
        # and outer (circumscribed) ellipses.
        # Note the circumscribed ellipse accounts for dimensionality.
        self.amat_inner = np.zeros((self.dm, self.dm))
        self.amat_outer = np.zeros((self.dm, self.dm))        
        for i, dri in enumerate(self.dr):
            self.amat_inner[i,i] = 4.0/(dri**2)
            self.amat_outer[i,i] = (4.0/self.dm)/(dri**2)

    def set_zmat(self):
        self.zmat_inner = np.linalg.inv(self.amat_inner)
        self.zmat_outer = np.linalg.inv(self.amat_outer)

    def chi(self, x, h, mu):
        f = 0.0
        for hp, mup in zip(h,mu):
            f += hp**2 / (4. * (mup + x)**2)
        f -= 1.0
        return f

    def dchidx(self, x, h, mu):
        f = 0.0
        for hp, mup in zip(h,mu):
            f += -0.5*hp**2 / ((mup + x)**3)
        return f
    
    def get_lambda_bounds(self, h, mu):
        # Return llo, lhi where chi < 0.
        # llo < all poles and lhi > all poles
        # (Since chi -> -1 as lambda -> +/- Infinity)
        poles = np.sort(-mu)
        llo = np.amin(poles)
        lhi = np.amax(poles)
        clo = 1.0
        chi = 1.0
        dllo = abs(0.001*llo)
        dlhi = abs(0.001*lhi)
        while clo > 0:
            llo = llo - dllo
            clo = self.chi(llo, h, mu)
        while chi > 0:
            lhi = lhi + dlhi
            chi = self.chi(lhi, h, mu)
        return llo, lhi

    def get_tp_from_lambda(self, x, hp, mu):
        # Returns the vector tp
        # corresponding to hp, mu, x=lambda
        tp = np.zeros(self.dm)
        for k in range(self.dm):
            tp[k] = -hp[k] / (2.0 * (mu[k] + x))
        return tp

    def get_x_from_tp(self, tp, u, lambdas, v):
        # Returns the vector x from the
        # transformed vector tp
        z = np.zeros(self.dm)
        for m in range(self.dm):
            for p in range(self.dm):
                z[m] += tp[p] * u[m,p]
        y = z/np.sqrt(lambdas)
        dx = np.zeros(self.dm)
        for i in range(self.dm):
            for k in range(self.dm):
                dx[i] += y[k] * v[i,k]
        x = self.center + dx
        return x

    def elliptic_constraint_fun(self, z, amat):
        # Elliptic constraint function which should be
        # non-negative if the constraint is satisfied
        f = 1.0
        for i in range(self.dm):
            f -= amat[i,i] * (z[i] - self.center[i])**2
        return f

    def get_inscribed_rectangle(self, amat):
        # Returns the arrays lo, hi specifying the low and high
        # boundaries of the rectangle inscribed by the ellipse
        # specified by the coefficients amat.
        lo = np.zeros(self.dm)
        hi = np.zeros(self.dm)
        halfwidth = np.zeros(self.dm)
        for i in range(self.dm):
            halfwidth[i] = 1.0/np.sqrt(amat[i,i])
        lo = self.center - halfwidth
        hi = self.center + halfwidth
        return lo, hi

    def get_extrema_mesh(self, amat, npts=1000):
        # Create a mesh of points over which to evaluate the
        # quadratic surface within the ellipse described by the coefficients in amat.
        # The domain of the mesh should be the rectangle inscribed by the ellipse.
        alo, ahi = self.get_inscribed_rectangle(amat)
        x_arr = [np.linspace(ilo, ihi, npts) for ilo, ihi in zip(alo, ahi)]
        x_mesh = np.meshgrid(*x_arr)
        z_arr = np.copy(x_mesh[0])
        z_arr = self.quadfit.quadratic_nd(x_mesh, *self.quadfit.coefficients)

        x_flat = [xm.flatten() for xm in x_mesh]
        z_flat = z_arr.flatten()

        fmin = np.amax(z_flat)
        fmax = np.amin(z_flat)
        xmin = np.zeros(self.dm)
        xmax = np.zeros(self.dm)
        
        for i, zi in enumerate(z_flat):
            xvec = np.array([xf[i] for xf in x_flat])
            if self.elliptic_constraint_fun(xvec, amat) >= 0.0:
                if zi < fmin:
                    fmin = zi
                    xmin = xvec
                if zi > fmax:
                    fmax = zi
                    xmax = xvec

        return fmin, fmax, xmin, xmax, True
            
    def get_extrema_slsqp(self, amat):
        # Get extrema of fit function within the ellipse
        # defined by amat using scipy nonlinear
        # optimization with constraints
        ztol = 1.0e-9

        # Get minimum of quadratic function
        res = minimize(lambda x: self.quadfit.quadratic_nd(x, *self.quadfit.coefficients),
                       self.center, method='SLSQP',
                       constraints={'type': 'ineq',
                                    'fun' : self.elliptic_constraint_fun,
                                    'args': [amat]},
                       tol=ztol)
        
        # Check to make sure the minimum satisfies the elliptic constraint
        if res.success and self.elliptic_constraint_fun(res.x, amat) >= 0.0:
            # Construct fmin
            fmin = self.quadfit.quadratic_nd(res.x, *self.quadfit.coefficients)
        else:
            return None, None, False
        
        # Get maximum of quadratic function
        res = minimize(lambda x: -self.quadfit.quadratic_nd(x, *self.quadfit.coefficients),
                       self.center, method='SLSQP',
                       constraints={'type': 'ineq',
                                    'fun' : self.elliptic_constraint_fun,
                                    'args': [amat]},
                       tol=ztol)

        # Check to make sure the maximum satisfies the elliptic constraint
        if res.success and self.elliptic_constraint_fun(res.x, amat) >= 0.0:
            # Construct fmax
            fmax = self.quadfit.quadratic_nd(res.x, *self.quadfit.coefficients)
        else:
            return None, None, False
        
        return fmin, fmax, True
    
    def get_extrema(self, amat, ellipse_type=None):
        lambdas, v = np.linalg.eig(amat)

        # Construct f1 (f_i)
        f1 = self.quadfit.get_coefs_first()
        csecond = self.quadfit.get_coefs_second()
        ccross  = self.quadfit.get_coefs_cross()

        # Construct f2 (f_ij)
        f2 = np.zeros((self.dm, self.dm))
        for i in range(self.dm):
            for j in range(self.dm):
                if i==j:
                    f2[i,j] = csecond[i]
                else:
                    k = self.quadfit.cross_indices[i,j]
                    # Halve the cross coefficient
                    # since it is the sum of f2[i,j]+f2[j,i]
                    f2[i,j] = 0.5 * ccross[k]

        # Construct fp0 (f'_0)
        fp0 = self.quadfit.get_coefs_const()
        for i in range(self.dm):
            fp0 += f1[i] * self.center[i]
            for j in range(self.dm):
                fp0 += f2[i,j] * self.center[i] * self.center[j]

        # Construct fp1 (f'_i)
        fp1 = np.copy(f1)
        for i in range(self.dm):
            for j in range(self.dm):
                fp1[i] += 2.0 * f2[i,j] * self.center[j]

        if self.verbose:
            print('fp0 is: ')
            print(fp0)
            print('fp1 is: ')
            print(fp1)
            print('f2 is: ')
            print(f2)

        if self.verbose:
            print('lambdas: ')
            print(lambdas)

        g1 = np.zeros(self.dm)
        for k in range(self.dm):
            for i in range(self.dm):
                g1[k] += fp1[i] * v[i,k]

        g2 = np.zeros((self.dm, self.dm))
        for k in range(self.dm):
            for l in range(self.dm):
                for j in range(self.dm):
                    for i in range(self.dm):
                        g2[k,l] += f2[i,j] * v[i,k] * v[j,l]

        if self.verbose:
            print('g1 is: ')
            print(g1)
            print('g2 is: ')
            print(g2)

        gp1 = g1/np.sqrt(lambdas)

        gp2 = np.zeros((self.dm,self.dm))
        for k in range(self.dm):
            for l in range(self.dm):
                gp2[k,l] = g2[k,l]/(np.sqrt(lambdas[k]) * np.sqrt(lambdas[l]))

        if self.verbose:
            print('gp1 is: ')
            print(gp1)
            print('gp2 is: ')
            print(gp2)

        mu, u = np.linalg.eig(gp2)

        if self.verbose:
            print('Eigenvalues of gp2 are: ')
            print(mu)
            print('Eigenvectors of gp2 are: ')
            print(u)

        hp = np.zeros(self.dm)

        for p in range(self.dm):
            for k in range(self.dm):
                hp[p] += u[k,p] * gp1[k]

        # Prepare to find extrema of the objective function on the ellipse boundary
        
        # Get the sorted poles of chi(lambda) where chi -> Infinity
        poles = np.sort(-mu)
                
        # Get lo and hi bounds of the roots lambda
        llo, lhi = self.get_lambda_bounds(hp, mu)

        if self.verbose:
            print('found bounds on lambda:')
            print('LO: chi({}) = {}'.format(llo, self.chi(llo, hp, mu)))
            print('HI: chi({}) = {}'.format(lhi, self.chi(lhi, hp, mu)))
            xval_vec = np.linspace(llo, lhi, 10000)
            fval_vec = np.array([self.chi(xvvi, hp, mu) for xvvi in xval_vec])
            plt.plot(xval_vec, fval_vec)
            for pi in poles:
                plt.axvline(x=pi, linestyle='--', color='gray')
            plt.legend(loc='upper center')
            plt.xlabel('Lagrange Multiplier')
            plt.ylabel('chi(x) on {} ellipse boundary'.format(ellipse_type))
            plt.ylim([-1, 10])
            plt.savefig('chix_lagrange_multiplier_{}.eps'.format(ellipse_type))
            plt.clf()
            
        # lo and hi bounds of intervals to look for roots
        intlo = [llo, poles[-1]]
        inthi = [poles[0], lhi]

        # List of roots for lambda
        lroots = []
        
        ztol = 1.0e-9 # Find roots to a tolerance of 1e-9
        # Find the minima of chi between each of the poles and check sign
        for i in range(len(poles)-1):
            lwb = poles[i]
            upb = poles[i+1]
            res = minimize(self.chi, 0.5*(lwb+upb), args=(hp, mu),
                           method='SLSQP', jac=self.dchidx,
                           bounds=[(lwb, upb)], tol=ztol)
            if res.success:
                xmin = res.x[0]
                # Check sign of minima
                minchi = self.chi(xmin, hp, mu)
                if minchi < 0 and abs(minchi) > ztol:
                    # Add these intervals to the search intervals
                    intlo.append(lwb)
                    inthi.append(xmin)
                    intlo.append(xmin)
                    inthi.append(upb)
                elif abs(minchi) <= ztol:
                    # Add the solution to the roots
                    lroots.append(xmin)

        # Search between all the intervals to locate roots
        maxiter = 10000
        for ilo, ihi in zip(intlo, inthi):
            if self.verbose:
                print('Locating chi roots between {} and {}'.format(ilo, ihi))
            xr, res = brentq(self.chi, ilo, ihi, args=(hp, mu),
                             maxiter=maxiter, full_output=True)
            if res.converged:
                # Add the solution to the roots
                lroots.append(xr)
            else:
                # Warn that brentq did not converge
                print('WARNING: brentq did not converge on chi between {} and {} with {} max iterations! (Ignoring interval)'.format(ilo, ihi, maxiter))

        if self.verbose:
            print('Found roots of chi(x): {}'.format(lroots))
                
        # Find the maximum and minimum of f(t) given the roots of chi
        fextrema = []
        tplist   = []

        # Find the extrema of f(t) inside the ellipse
        # Find where the objective function is extremized within the ellipse
        tp = -hp/(2.0 * mu)

        if self.verbose:
            print("tp is:")
            print(tp)

            print("sum of tp**2 is:")
            print(np.sum(tp**2))

            if (np.sum(tp**2)) <= 1.0:
                print ('Unconstrained extremum found in the interior of the domain.')
            else:
                print('Unconstrained extremum not found in the interior of the domain.')

            print("objective function:")
            print(self.quad_transform_nd(fp0, hp, mu, tp))
        
        # Check tp found above to see if it is really within the ellipse
        if np.sum(tp**2) <= 1.0:
            ftpi = self.quad_transform_nd(fp0, hp, mu, tp)
            fextrema.append(ftpi)
            tplist.append(tp)
            if self.verbose:
                print('Value of f at internal extrema: {}'.format(ftpi))
            
        for i, xr in enumerate(lroots):
            tpi = self.get_tp_from_lambda(xr, hp, mu)
            ftpi = self.quad_transform_nd(fp0, hp, mu, tpi)
            if self.verbose:
                print('sum tpi: {}'.format(np.sum(tpi**2)))
                print('f(tpi): {}'.format(ftpi))                
            fextrema.append(ftpi)
            tplist.append(tpi)            

        if self.verbose:
            xval_test = np.linspace(llo, lhi, 10000)
            xval_vec  = []
            fval_vec  = []
            for xvt in xval_test:
                tpi = self.get_tp_from_lambda(xvt, hp, mu)
                if np.sum(tpi**2) <= 1:
                    fvi = self.quad_transform_nd(fp0, hp, mu, tpi)
                    xval_vec.append(xvt)
                    fval_vec.append(fvi)

            imax, fmax = max(enumerate(fval_vec), key=operator.itemgetter(1))
            imin, fmin = min(enumerate(fval_vec), key=operator.itemgetter(1))
            print('TESTING: found max(fval_vec) = {}'.format(fmax))
            print('TESTING: found min(fval_vec) = {}'.format(fmin))
            
            plt.plot(xval_vec, fval_vec)
            plt.legend(loc='upper center')
            plt.xlabel('Lagrange Multiplier')
            plt.ylabel('f(x) on {} ellipse boundary'.format(ellipse_type))
            plt.savefig('fx_lagrange_multiplier_{}.eps'.format(ellipse_type))
            plt.clf()
            
        if len(fextrema) < 2:
            if self.verbose:
                print('ERROR: insufficient function extrema found!')
            fmax = None
            fmin = None
            xmin = None
            xmax = None
            success = False
        else:
            imax, fmax = max(enumerate(fextrema), key=operator.itemgetter(1))
            imin, fmin = min(enumerate(fextrema), key=operator.itemgetter(1))
            tpmax = tplist[imax]
            tpmin = tplist[imin]
            xmin = self.get_x_from_tp(tpmin, u, lambdas, v)
            xmax = self.get_x_from_tp(tpmax, u, lambdas, v)            
            success = True

            min_on_boundary = False
            max_on_boundary = False
            
            if imin != 0:
                min_on_boundary = True
                lambda_min = lroots[imin-1]
            if imax != 0:
                max_on_boundary = True
                lambda_max = lroots[imax-1]          
            
            if self.verbose:
                print("At Function Maximum :")
                print("tp :")
                print(tpmax)

                print("sum of tp**2 :")
                print(np.sum(tpmax**2))

                print('x : {}'.format(xmin))

                print("function maximum :")
                print(fmax)

                print("function maximum on boundary?: {}".format(max_on_boundary))

                print("")

                print("At Function Minimum :")
                print("tp :")
                print(tpmin)

                print("sum of tp**2 :")
                print(np.sum(tpmin**2))

                print('x : {}'.format(xmax))
                
                print("function minimum :")
                print(fmin)

                print("function minimum on boundary?: {}".format(min_on_boundary))                

        return fmin, fmax, xmin, xmax, success

class QuadraticAnalysis(object):    
    def __init__(self, grid, lo, hi, method='elliptical',
                 nmesh=None, npts=20, fail_threshold=10,
                 ofile=None, verbose=False):

        # Common parameters
        self.grid = grid
        self.lo = lo
        self.hi = hi
        self.verbose = verbose
        self.outputfile = ofile
        self.qfit = None
        self.method = method
        self.success = False

        # Ellipse optimization
        self.eopt = None
        self.nmesh = nmesh

        # Scipy optimization on rectangular domain
        self.ropt = None 
        self.npts = npts
        self.fail_threshold = fail_threshold

        # Do the analysis
        self.success = self.analyze(self.method)
        self.write_results()
        
    def analyze(self, method='elliptical'):
        # Do the quadratic fit and either elliptical or rectangular optimization
        self.qfit = QuadraticFit(self.grid)
        if method.lower().strip() == 'elliptical':
            self.eopt = EllipticOptimize(self.qfit, self.lo, self.hi,
                                         nmesh=self.nmesh,
                                         verbose=self.verbose)
            return self.eopt.success
        elif method.lower().strip() == 'rectangular':
            self.ropt = RectangularOptimize(self.qfit, self.lo, self.hi,
                                            npts=self.npts, fail_threshold=self.fail_threshold,
                                            verbose=self.verbose)
            return self.ropt.success
        else:
            sys.exit('Received unknown analysis method: {}'.format(method))

    def write_results(self):
        if self.outputfile:
            fout = open(self.outputfile, 'w')
            self.qfit.writelog(fout)
            self.eopt.writelog(fout)
            fout.close()
        
class EnsembleAnalysis(object):
    def __init__(self, grid, lo, hi, nensemble, max_size_ensemble=500,
                 ofile=None, verbose=False):
        self.grid = grid
        self.lo = lo
        self.hi = hi
        self.success = True

        self.inner_fminlist = []
        self.inner_fmaxlist = []
        self.outer_fminlist = []
        self.outer_fmaxlist = []
        
        self.inner_fmin_ave = []
        self.inner_fmin_std = [0]
        self.inner_fmax_ave = []
        self.inner_fmax_std = [0]
        
        self.outer_fmin_ave = []
        self.outer_fmin_std = [0]
        self.outer_fmax_ave = []
        self.outer_fmax_std = [0]
        
        self.nensemble = nensemble
        self.verbose = verbose
        self.outputfile = ofile
        self.analyze(max_size_ensemble)
        self.write_results()
        
    def analyze(self, max_size_ensemble=500):
        # Do sampling of the points
        for i, samplepts in enumerate(itertools.combinations(self.grid.points, self.nensemble)):
            if i == max_size_ensemble:
                break
            print('running combination {} with {} points.'.format(i, len(samplepts)))
            g = Grid(samplepts)
            qa = QuadraticAnalysis(g, self.lo, self.hi)
            self.success = self.success and qa.eopt.success
            if not self.success:
                sys.exit('ERROR: ensemble analyis failed at combination {}!'.format(i))
            self.inner_fminlist.append(qa.eopt.inner_min)
            self.inner_fmaxlist.append(qa.eopt.inner_max)
            self.outer_fminlist.append(qa.eopt.outer_min)
            self.outer_fmaxlist.append(qa.eopt.outer_max)

        # Do progressive averaging to evaluate convergence
        self.inner_fmin_ave.append(self.inner_fminlist[0])
        self.inner_fmax_ave.append(self.inner_fmaxlist[0])
        self.outer_fmin_ave.append(self.outer_fminlist[0])
        self.outer_fmax_ave.append(self.outer_fmaxlist[0])
        if len(self.inner_fminlist) > 1:
            for i in range(2, len(self.inner_fminlist)+1):
                self.inner_fmin_ave.append(np.mean(self.inner_fminlist[:i]))
                self.inner_fmin_std.append(np.std(self.inner_fminlist[:i]))
                self.outer_fmin_ave.append(np.mean(self.outer_fminlist[:i]))
                self.outer_fmin_std.append(np.std(self.outer_fminlist[:i]))
                self.inner_fmax_ave.append(np.mean(self.inner_fmaxlist[:i]))
                self.inner_fmax_std.append(np.std(self.inner_fmaxlist[:i]))
                self.outer_fmax_ave.append(np.mean(self.outer_fmaxlist[:i]))
                self.outer_fmax_std.append(np.std(self.outer_fmaxlist[:i]))

    def write_results(self):
        if self.outputfile:
            fout = open(self.outputfile, 'w')
            fout.write('# ENSEMBLE SAMPLING ELLIPTIC OPTIMIZATION LOG\n')
            if self.success:
                fout.write('# SUCCESS!\n')
            else:
                fout.write('# FAILURE!\n')
            fout.write('# NUMBER OF POINTS PER SAMPLE = {}\n'.format(self.nensemble))
            fout.write('number samples, '+
                       'inner min ave, '+
                       'inner min std, '+
                       'inner max ave, '+
                       'inner max std, '+
                       'outer min ave, '+
                       'outer min std, '+
                       'outer max ave, '+
                       'outer max std\n')                       
            i = 1
            for imina, imins, imaxa, imaxs, omina, omins, omaxa, omaxs in zip(self.inner_fmin_ave, self.inner_fmin_std,
                                                                              self.inner_fmax_ave, self.inner_fmax_std,
                                                                              self.outer_fmin_ave, self.outer_fmin_std,
                                                                              self.outer_fmax_ave, self.outer_fmax_std):
                fout.write('{}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(i, imina, imins, imaxa, imaxs, omina, omins, omaxa, omaxs))
                i += 1
            fout.close()

class Histogram(object):
    def __init__(self, bin_values = [], bin_edges = [], normalize=False):
        self.bin_values = np.array(bin_values)
        if any(bin_edges):
            self.bin_centers = np.array(self.bin_edges_to_centers(bin_edges))
        else:
            self.bin_centers = []
        if any(self.bin_centers):
            self.center = self.get_center()
        if any(bin_values) and normalize:
            self.bin_values = self.bin_values/np.sum(self.bin_values)

    def bin_edges_to_centers(self, bin_edges):
        binc = []
        for i, be in enumerate(bin_edges):
            if i < len(bin_edges) - 1:
                bc = bin_edges[i] + 0.5*(bin_edges[i+1] - bin_edges[i])
                binc.append(bc)
        return np.array(binc)

    def get_center(self):
        bc = np.sum(self.bin_values * self.bin_centers)/np.sum(self.bin_values)
        return bc
            
        
