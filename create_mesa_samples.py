#!/usr/bin/python
#
# June 2016
# Melissa Hoffman
#
# Updated December 2019
# Don Willcox
#
# Create MESA samples for UQ

import numpy as np
import random
import os, shutil, re
import math
import argparse
from itertools import combinations
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--suite_name", type=str, required=True, help="Name of this suite of runs.")
parser.add_argument("-tdir", "--template_directory", type=str, required=True, help="Name of template directory for running MESA.")
parser.add_argument("-tinlist", "--template_inlist", type=str, required=True, help="Name of template inlist to use for MESA.")
parser.add_argument("-n", "--number_samples", type=int, required=True, help="Number of samples to create.")
parser.add_argument("-u", "--uniform_random", action="store_true", help="Do uniform random sampling.")
parser.add_argument("-c", "--cauchy_random", action="store_true", help="Do Cauchy random sampling.")
parser.add_argument("-e", "--evenly_spaced", action="store_true", help="Do evenly spaced grid sampling.")
parser.add_argument("-p", "--parameters", type=str, nargs=2, required=True, help="Names of inlist parameters corresponding to each dimension.")
parser.add_argument("-lo", "--domain_lo", type=float, nargs=2, required=True, help="Domain lower bounds in each dimension: [xlo_0, xlo_1]")
parser.add_argument("-hi", "--domain_hi", type=float, nargs=2, required=True, help="Domain upper bounds in each dimension: [xhi_0, xhi_1]")
args = parser.parse_args()


def ChangeValue(inlist,param,newval):
    # Changes parameter param value to newval
    # in-place in the file inlist.
    lines = []
    fin = open(inlist, "r")
    for line in fin:
        lines.append(line)
    fin.close()

    fout = open(inlist, "w")
    for line in lines:
        if param in line:
            fout.write('      '+param+' = '+str(newval)+'\n')
        else:
            fout.write(line)
    fout.close()

def get_evenly_spaced_grid(Npar, lo, hi):
    # For variables x_0, x_1, ... the arguments are lo and hi vectors
    # lo = xlo_0, xlo_1, ...
    # hi = xhi_0, xhi_1, ...

    NperDim = int(np.power(args.number_samples, 1.0/Npar))
    print("For evenly spaced grid, using {} samples per dimension.".format(NperDim))
    axes_samples = []
    for i in range(Npar):
        x = np.linspace(args.domain_lo[i], args.domain_hi[i], num=NperDim)
        axes_samples.append(x)

    x_ik = np.array(np.meshgrid(*axes_samples)).reshape(Npar, -1)
    return x_ik

def get_uniform_random_samples(Npar, lo, hi):
    # For variables x_0, x_1, ... the arguments are lo and hi vectors
    # lo = xlo_0, xlo_1, ...
    # hi = xhi_0, xhi_1, ...

    x_ik = np.zeros( (Npar, args.number_samples) )
    for i in range(Npar):
        x_ik[i,:] = np.random.uniform(args.domain_lo[i], args.domain_hi[i], args.number_samples)
    return x_ik


def get_cauchy_samples(Npar, lo, hi):
    # For variables x_0, x_1, ... the arguments are lo and hi vectors
    # lo = xlo_0, xlo_1, ...
    # hi = xhi_0, xhi_1, ...

    # Calculate Cauchy distributed variables, see FK 2004 for algorithm
    # Number of experiments is args.number_samples

    x_lim = np.zeros( (Npar, 2) )
    for ip in range(Npar):
        x_lim[ip,:] = [lo[ip], hi[ip]]

    # Initialize arrays
    r_ik = np.zeros( (Npar, args.number_samples) )
    c_ik = np.zeros( (Npar, args.number_samples) )
    d_ik = np.zeros( (Npar, args.number_samples) )
    x_ik = np.zeros( (Npar, args.number_samples) )

    delta = np.zeros( (Npar) )     # this is the interval half-width
    xtilde_ik = np.zeros( (Npar) ) # this is the interval midpoint

    for ip in range(Npar):
        delta[ip] = (hi[ip] - lo[ip]) / 2.0
        xtilde_ik[ip] = (hi[ip] + lo[ip]) / 2.0

    Kvals = []

    # Calculate Cauchy samples
    for k in range(args.number_samples):
        for i in range(Npar):
            r_ik[i,k] = random.uniform(0.0,1.0)

        c_ik[:,k] = np.tan( math.pi * (r_ik[:,k] - 0.5) )

        K = max( abs( c_ik[:,k] ))

        Kvals.append(K)

        d_ik[:,k] = ( delta[:] * c_ik[:,k] ) / K

        x_ik[:,k] = xtilde_ik[:] + d_ik[:,k]

    return x_ik


def write_samples(Npar, x, label):
    # given random samples in x, create the run directories
    # with x a numpy array of shape (Npar, args.number_samples)
    # Npar is the number of parameters

    root_dir = os.getcwd()
    fpath = os.path.join(root_dir, args.suite_name, label)

    # These are the template files that will be copied and made into the series of directories
    template = args.template_directory # 'prems_to_wd_template'
    main_list = args.template_inlist # 'inlist_template'

    summary_file = 'sample_summary_{}.txt'.format(label)
    with open(summary_file,'w') as b:
        b.write("{} Samples:\n".format(label))
        b.write('-----------------------------------------\n')

    for i in range(args.number_samples):
        name = 'c'+str(i)

        # Copy templates and create directory
        working_dir = os.path.join(fpath, name)
        shutil.rmtree(working_dir, ignore_errors=True)
        shutil.copytree(template, working_dir)
        shutil.copy(main_list, working_dir)
        os.chdir(working_dir)

        # Change parameter values in inlist
        for ip in range(Npar):
            ChangeValue(main_list, args.parameters[ip], x[ip,i])

        os.chdir(root_dir)

        with open('sample_summary_{}.txt'.format(label),'a') as b:
            b.write('Folder name: '+name+'\n')
            for ip in range(Npar):
                b.write('{}: '.format(args.parameters[ip])+str(x[ip,i])+'\n')
            b.write('-----------------------------------------\n')


def sanity_check_inputs():
    assert(len(args.parameters) == len(args.domain_lo))
    assert(len(args.domain_lo) == len(args.domain_hi))


def plot_sampling(Npar, x, label):
    parameter_indexes = range(Npar)
    pairs_to_plot = [x for x in combinations(parameter_indexes, 2)]
    for pairs in pairs_to_plot:
        ip0 = pairs[0]
        ip1 = pairs[1]
        area = 100
        color = 'green'
        plt.clf()
        plt.scatter(x[ip0,:], x[ip1,:], s=area, c=color, alpha=0.5)
        plt.xlabel(args.parameters[ip0])
        plt.ylabel(args.parameters[ip1])
        plt.tight_layout()
        plotname = "samples_{}_{}_{}.eps".format(label, args.parameters[ip0], args.parameters[ip1])
        plt.savefig(plotname)


if __name__=="__main__":
    sanity_check_inputs()

    number_parameters = len(args.parameters)
    random.seed()

    if args.evenly_spaced:
        # generate evenly spaced grid
        print("Getting evenly spaced grid ...")
        x_ik = get_evenly_spaced_grid(number_parameters, args.domain_lo, args.domain_hi)
        print("Creating evenly spaced grid run directories ...")
        write_samples(number_parameters, x_ik, "evenly_spaced")
        print("Plotting evenly spaced sampling ...")
        plot_sampling(number_parameters, x_ik, "evenly_spaced")
        print("Finished creating evenly spaced grid.\n")

    if args.uniform_random:
        # generate uniform random samples
        print("Getting uniform random samples ...")
        x_ik = get_uniform_random_samples(number_parameters, args.domain_lo, args.domain_hi)
        print("Creating uniform random samples run directories ...")
        write_samples(number_parameters, x_ik, "uniform_random")
        print("Plotting uniform random sampling ...")
        plot_sampling(number_parameters, x_ik, "uniform_random")
        print("Finished creating uniform random grid.\n")

    if args.cauchy_random:
        # generate Cauchy random samples
        print("Getting Cauchy random samples ...")
        x_ik = get_cauchy_samples(number_parameters, args.domain_lo, args.domain_hi)
        print("Creating Cauchy random samples run directories ...")
        write_samples(number_parameters, x_ik, "cauchy")
        print("Plotting cauchy sampling ...")
        plot_sampling(number_parameters, x_ik, "cauchy")
        print("Finished creating cauchy grid.\n")
