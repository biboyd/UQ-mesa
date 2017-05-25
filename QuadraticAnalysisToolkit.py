import sys
import operator
import itertools
import numpy as np
from math import factorial, floor
from scipy.optimize import curve_fit, minimize, brentq

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

    def initFromCSV(self, fname, delimiter=',', skip_header=1):
        # Given the name (fname) of a csv file
        # containing a series of entries x1, x2, ..., xn, v
        # with skip_header lines of header,
        # extract the points (1 per line) from the csv
        # and store in the grid.

        # Clear the grid first
        self.dm = None
        self.points = []
        self.coords = []
        self.values = []

        # Open csv and read points
        raw_data = np.genfromtxt(fname, delimiter=delimiter, skip_header=skip_header)
        if len(raw_data)==0:
            sys.exit('ERROR: tried to open csv with no data.')
        
        # Each element of data is a row from the csv file: create points
        self.points = []
        for row in raw_data:
            self.points.append(Point(r=row[0:-1], v=row[-1]))

        # Set dimensionality, coordinates, and values
        self.dm = self.points[0].dm
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

    def writelog(self, file_handle):
        # Write a log of the quadratic fit to the file_handle
        file_handle.write('# QUADRATIC FIT LOG\n')
        file_handle.write('# DIMENSIONALITY = {}\n'.format(self.dm))
        file_handle.write('# NUMBER OF POINTS = {}\n'.format(len(self.grid.points)))
        file_handle.write('# INTERCEPT:\n')
        file_handle.write('{}\n'.format(self.get_coefs_const()))
        file_handle.write('# FIRST ORDER COEFFICIENTS:\n')
        for ci, dci in zip(self.get_coefs_first(), self.get_coefs_first(self.std_error)):
            file_handle.write('{} +/- {}\n'.format(ci, dci))
        file_handle.write('# CROSS COEFFICIENTS:\n')
        ccross = self.get_coefs_cross()
        dccross = self.get_coefs_cross(self.std_error)
        for k, ck in enumerate(ccross):
            dck = dccross[k]
            i, j = self.get_cross_indices(k)
            file_handle.write('({},{}) : {} +/- {}\n'.format(i,j,ck,dck))
        file_handle.write('# SECOND ORDER COEFFICIENTS:\n')
        for ci, dci in zip(self.get_coefs_second(), self.get_coefs_second(self.std_error)):
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

    def get_coefs_first(self, coefficients=[]):
        # Given a vector of coefficients,
        # return the vector of coefficients of
        # first-order variables: e.g. ci*xi
        if len(coefficients)==0:            
            coefficients = self.coefficients
        return coefficients[1:self.dm+1]

    def get_coefs_cross(self, coefficients=[]):
        # Given a vector of coefficients,
        # return the vector of coefficients of
        # cross-variable products: e.g. cij*xi*xj
        if len(coefficients)==0:                        
            coefficients = self.coefficients
        ncross = int(floor(factorial(self.dm)/2))
        return coefficients[self.dm+1:self.dm+1+ncross]

    def get_coefs_second(self, coefficients=[]):
        # Given a vector of coefficients,
        # return the vector of coefficients of
        # second-order variables: e.g. ci*xi**2
        if len(coefficients)==0:                        
            coefficients = self.coefficients
        ncross = int(floor(factorial(self.dm)/2))
        return coefficients[self.dm+1+ncross:]

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

    def do_quad_fit(self):
        self.coefficients, self.covariance = curve_fit(self.quadratic_nd,
                                                       self.grid.coords, self.grid.values,
                                                       p0=np.ones(self.ncoefs))
        self.std_error = np.sqrt(np.diag(self.covariance))

class EllipticOptimize(object):
    def __init__(self, quadfit, lo, hi, verbose=False):
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
        self.verbose = verbose
        self.minmax_inner = None
        self.minmax_outer = None

        self.set_amat()
        self.set_zmat()
        if self.verbose:
            print('------------ INNER ELLIPSE OPTIMIZATION')
        self.inner_min, self.inner_max, isuccess = self.get_extrema(self.amat_inner,
                                                                    self.zmat_inner)
        if self.verbose:
            print('------------ OUTER ELLIPSE OPTIMIZATION')
        self.outer_min, self.outer_max, osuccess = self.get_extrema(self.amat_outer,
                                                                    self.zmat_outer)
        self.success = isuccess and osuccess

    def writelog(self, file_handle):
        # Given a file_handle, write a log of the Elliptic Optimization
        file_handle.write('# ELLIPTIC OPTIMIZATION LOG\n')
        file_handle.write('# INNER MINIMUM, INNER MAXIMUM:\n')
        file_handle.write('{}, {}\n'.format(self.inner_min, self.inner_max))
        file_handle.write('# OUTER MINIMUM, OUTER MAXIMUM:\n')
        file_handle.write('{}, {}\n'.format(self.outer_min, self.outer_max))
        
    def quad_transform_nd(self, f0, hp, mu, tp):
        f = f0
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
        f -= 1.
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
        while clo > 0:
            llo = llo - 100.0 * abs(llo)
            clo = self.chi(llo, h, mu)
        while chi > 0:
            lhi = lhi + 100.0 * abs(lhi)
            chi = self.chi(lhi, h, mu)
        return llo, lhi

    def get_tp_from_lambda(self, x, hp, mu):
        # Returns the vector tp
        # corresponding to hp, mu, x=lambda
        tp = np.zeros(self.dm)
        for k in range(self.dm):
            tp[k] = -hp[k] / (2.0 * (mu[k] + x))
        return tp
    
    def get_extrema(self, amat, zmat):
        w, v = np.linalg.eig(zmat)

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
                    f2[i,j] = ccross[k]

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

        lambdas = np.zeros(self.dm)

        for k in range(self.dm):
            lambdas[k] = 1.0 / w[k]

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

        gp1 = np.zeros(self.dm)
        for k in range(self.dm): 
            gp1[k] = g1[k] / np.sqrt(lambdas[k])

        gp2 = np.zeros((self.dm,self.dm))
        for k in range(self.dm):
            for l in range(self.dm):
                gp2[k,l] = g2[k,l] / (np.sqrt(lambdas[k]) * np.sqrt(lambdas[l]))

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

        tp = np.zeros(self.dm)

        # Find where the objective function is extremized within the ellipse
        for k in range(self.dm):
            tp[k] = -hp[k] / (2.0 * mu[k])

        if self.verbose:
            print("tp is:")
            print(tp)

            print("sum of tp**2 is:")
            print(np.sum(tp**2))

            if (np.sum(tp**2)) <= 1.0:
                print ('Extremum found in the interior of the domain.')
            else:
                print('Extremum not found in the interior of the domain.')

            print("objective function:")
            print(self.quad_transform_nd(fp0, hp, mu, tp))

        # Now find extrema of the objective function on the ellipse boundary
        
        # Get the sorted poles of chi(lambda) where chi -> Infinity
        poles = np.sort(-mu)
                
        # Get lo and hi bounds of the roots lambda
        llo, lhi = self.get_lambda_bounds(hp, mu)

        # lo and hi bounds of intervals to look for roots
        intlo = [llo, lhi]
        inthi = [poles[0], poles[-1]]

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
        maxiter = 1000000
        for ilo, ihi in zip(intlo, inthi):
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
        fextrema = np.zeros(len(lroots)+1)
        fextrema[0] = self.quad_transform_nd(fp0, hp, mu, tp)
        tplist = [tp] # Check f at the extremum found within the ellipse
        for i, xr in enumerate(lroots):
            tpi = self.get_tp_from_lambda(xr, hp, mu)
            if self.verbose:
                print('sum tpi: {}'.format(np.sum(tpi**2)))
            tplist.append(tpi)
            fextrema[i+1] = self.quad_transform_nd(fp0, hp, mu, tpi)

        if self.verbose:
            print('Values of f at roots of chi(x): {}'.format(fextrema[1:]))

            # 2D sanity checking
            npts = 1000

            x_arr = np.linspace(self.lo[0], self.hi[1], npts)
            y_arr = np.linspace(self.lo[0], self.hi[1], npts)
            x_arr, y_arr = np.meshgrid(x_arr, y_arr)
            z_arr = np.copy(x_arr)

            for i in range(npts):
                for j in range(npts):
                    x_arr[i,j] = x_arr[i,j]

            for i in range(npts):
                for j in range(npts):
                    y_arr[i,j] = y_arr[i,j]

            for i in range(npts):
                for j in range(npts):
                    z_arr[i,j] = self.quadfit.quadratic_nd([x_arr[i,j], y_arr[i,j]],
                                                           *self.quadfit.coefficients)

            mask = np.where(amat[0,0]*(x_arr-self.center[0])**2 + amat[1,1]*(y_arr-self.center[1])**2 <= 1.0)

            print("Sampled Minimum in elliptical region is: ", np.min(z_arr[mask]))
            print("Sampled Maximum in elliptical region is: ", np.max(z_arr[mask]))
            
        if len(fextrema) < 2:
            if self.verbose:
                print('ERROR: insufficient function extrema found!')
            fmax = None
            fmin = None
            success = False
        else:
            imax, fmax = max(enumerate(fextrema), key=operator.itemgetter(1))
            imin, fmin = min(enumerate(fextrema), key=operator.itemgetter(1))
            tpmax = tplist[imax]
            lambda_max = lroots[imax]
            tpmin = tplist[imin]
            lambda_min = lroots[imin]
            success = True
            if self.verbose:
                print("At Function Maximum :")
                print("tp :")
                print(tpmax)

                print("sum of tp**2 :")
                print(np.sum(tpmax**2))

                print("function maximum :")
                print(fmax)

                print("")

                print("At Function Minimum :")
                print("tp :")
                print(tpmin)

                print("sum of tp**2 :")
                print(np.sum(tpmin**2))

                print("function minimum :")
                print(fmin)

        return fmin, fmax, success

class QuadraticAnalysis(object):    
    def __init__(self, grid, lo, hi, ofile=None, verbose=False):
        self.grid = grid
        self.lo = lo
        self.hi = hi
        self.verbose = verbose
        self.outputfile = ofile
        self.qfit = None
        self.eopt = None
        self.analyze()
        self.write_results()
        
    def analyze(self):
        # Do the quadratic fit and elliptic optimization
        self.qfit = QuadraticFit(self.grid)
        self.eopt = EllipticOptimize(self.qfit, self.lo, self.hi,
                                     verbose=self.verbose)

    def write_results(self):
        if self.outputfile:
            fout = open(self.outputfile, 'w')
            self.qfit.writelog(fout)
            self.eopt.writelog(fout)
            fout.close()
        
class EnsembleAnalysis(object):
    def __init__(self, grid, lo, hi, nensemble, ofile=None, verbose=False):
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
        self.analyze()
        self.write_results()
        
    def analyze(self):
        # Do sampling of the points
        for i, samplepts in enumerate(itertools.combination(self.grid.points, self.nensemble)):
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

    def write_ave_std(self, file_handle, average, stdev):
        # Given a file_handle, write the ave and std with a counter
        i = 1
        for ave, std in zip(average, stdev):
            file_handle.write('{}, {}, {}'.format(i, ave, std))
            i += 1
    
    def write_results(self):
        if self.outputfile:
            fout = open(self.outputfile, 'w')
            fout.write('# ENSEMBLE SAMPLING ELLIPTIC OPTIMIZATION LOG\n')
            if self.success:
                fout.write('# SUCCESS!\n')
            else:
                fout.write('# FAILURE!\n')
            fout.write('# NUMBER OF POINTS PER SAMPLE = {}\n'.format(self.nensemble))
            fout.write('# BEGIN INNER ELLIPSE MINIMUM:\n')
            fout.write('number samples, '+
                       'inner min ave, '+
                       'inner min std\n')
            self.write_ave_std(fout, self.inner_fmin_ave, self.inner_fmin_std)
            fout.write('# END INNER ELLIPSE MINIMUM:\n')            
            fout.write('# BEGIN INNER ELLIPSE MAXIMUM:\n')            
            fout.write('number samples, '+
                       'inner max ave, '+
                       'inner max std\n')
            self.write_ave_std(fout, self.inner_fmax_ave, self.inner_fmax_std)            
            fout.write('# END INNER ELLIPSE MAXIMUM:\n')                        
            fout.write('# BEGIN OUTER ELLIPSE MINIMUM:\n')            
            fout.write('number samples, '+
                       'outer min ave, '+
                       'outer min std\n')
            self.write_ave_std(fout, self.outer_fmin_ave, self.outer_fmin_std)            
            fout.write('# END OUTER ELLIPSE MINIMUM:\n')                        
            fout.write('# BEGIN OUTER ELLIPSE MAXIMUM:\n')            
            fout.write('number samples, '+
                       'outer max ave, '+
                       'outer max std\n')
            self.write_ave_std(fout, self.outer_fmax_ave, self.outer_fmax_std)            
            fout.write('# END OUTER ELLIPSE MAXIMUM:\n')                        
            fout.close()
