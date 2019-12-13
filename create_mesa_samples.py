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

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--suite_name", type=str, required=True, help="Name of this suite of runs.")
parser.add_argument("-tdir", "--template_directory", type=str, required=True, help="Name of template directory for running MESA.")
parser.add_argument("-tinlist", "--template_inlist", type=str, required=True, help="Name of template inlist to use for MESA.")
parser.add_argument("-n", "--number_samples", type=int, required=True, help="Number of samples to create.")
parser.add_argument("-u", "--uniform_random", action="store_true", help="Do uniform random sampling.")
parser.add_argument("-c", "--cauchy_random", action="store_true", help="Do Cauchy random sampling.")
parser.add_argument("-e", "--evenly_spaced", action="store_true", help="Do evenly spaced grid sampling.")
parser.add_argument("-lo", "--domain_lo", type=float, nargs=2, required=True, help="Domain lower bounds in each dimension: [xlo_0, xlo_1]")
parser.add_argument("-hi", "--domain_hi", type=float, nargs=2, required=True, help="Domain upper bounds in each dimension: [xhi_0, xhi_1]")
args = parser.parse_args()

# hardcode to use 2 parameters for now
Npar = 2


def ChangeValue(inlist,newinl,param,newval):
    with open(inlist) as f:
        q = open(newinl,'a')
        for line in f:
            l = line
            if param in l:
                q.write('      '+param+' = '+str(newval)+'\n')
            if not param in l:
                q.write(l)

def get_evenly_spaced_grid(lo, hi):
    # For variables x_0 and x_1, args are lo and hi vectors
    # lo = xlo_0, xlo_1
    # hi = xhi_0, xhi_1

    NperDim = int(np.power(args.number_samples, 1.0/Npar))
    print("For evenly spaced grid, using {} samples per dimension.")
    axes_samples = []
    for i in Npar:
        axes_samples.append(np.linspace(args.domain_lo[i], args.domain_hi[i], num=NperDim))

    x_ik = np.meshgrid(*axes_samples)
    return x_ik

def get_uniform_random_samples(lo, hi):
    # For variables x_0 and x_1, args are lo and hi vectors
    # lo = xlo_0, xlo_1
    # hi = xhi_0, xhi_1

    x_ik = np.zeros( (Npar, args.number_samples) )
    for i in range(Npar):
        x_ik[i,:] = np.random.uniform(args.domain_lo[i], args.domain_hi[i], args.number_samples)
    return x_ik


def get_cauchy_samples(lo, hi):
    # For variables x_0 and x_1, args are lo and hi vectors
    # lo = xlo_0, xlo_1
    # hi = xhi_0, xhi_1

    # Calculate Cauchy distributed variables, see FK 2004 for algorithm
    # Number of experiments is args.number_samples
    n_inp = 2 # number of inputs

    x_lim = np.zeros( (n_inp, 2) )
    x_lim[0,:] = [lo[0], hi[0]]
    x_lim[1,:] = [lo[1], hi[1]]

    # Initialize arrays
    x = np.zeros( (n_inp, 2) )

    r_ik = np.zeros( (n_inp, args.number_samples) )
    c_ik = np.zeros( (n_inp, args.number_samples) )
    d_ik = np.zeros( (n_inp, args.number_samples) )
    x_ik = np.zeros( (n_inp, args.number_samples) )

    delta = np.zeros( (n_inp) ) # this is the interval half-width

    delta[0] = ( (x_lim[0,1] - x_lim[0,0]) / 2.0 )
    delta[1] = ( (x_lim[1,1] - x_lim[1,0]) / 2.0 )

    xtilde_ik = ( x_lim[:,1] + x_lim[:,0] )/ 2.0  # This is the midpoint

    Kvals = []

    for k in range(args.number_samples):

        for i in range(n_inp):

            r_ik[i,k] = random.uniform(0.0,1.0)

        c_ik[:,k] = np.tan( math.pi * (r_ik[:,k] - 0.5) )

        K = max( abs( c_ik[:,k] ))

        Kvals.append(K)

        d_ik[:,k] = ( delta[:] * c_ik[:,k] ) / K

        x_ik[:,k] = xtilde_ik[:] + d_ik[:,k]
    return x_ik


def write_samples(x, label):
    # given random samples in x, create the run directories
    # with x a numpy array of shape (Npar, args.number_samples)

    root_dir = os.getcwd()
    fpath = os.path.join(root_dir, args.suite_name, label)

    # These are the template files that will be copied and made into the series of directories
    template = args.template_directory # 'prems_to_wd_template'
    main_list = args.template_inlist # 'inlist_template'

    for i in range(args.number_samples):
        name = 'c'+str(i)
        rx = str(x_ik[0,i])
        bx = str(x_ik[1,i])

        # Copy templates and create directory called c#
        shutil.copytree(template,os.path.join(args.suite_name, name))
        shutil.copy(main_list,os.path.join(fpath, name, '.'))
        os.chdir(os.path.join(fpath, name))

        # Change Values in inlist
        ChangeValue(main_list,'CHANGE_R','Reimers_scaling_factor',rx)
        ChangeValue('CHANGE_R','inlist_1.0','Blocker_scaling_factor',bx)

        templ = open('batch_cmd.sh','r')
        cluster = open('cluster.sh','r')
        runfile = open('run_script','a')

        os.system('rm inlist_template')
        os.system('rm CHANGE_R')

        templ.close()
        runfile.close()

        print(os.getcwd())
        os.chdir(root_dir)

        with open('sample_summary_{}.txt'.format(label),'w') as b:
            b.write('Folder name: '+name+'\n')
            b.write('Reimers: '+str(rx)+'\n')
            b.write('Blocker: '+str(bx)+'\n')
            b.write('-----------------------------------------\n')


if __name__=="__main__":
    random.seed()

    if args.evenly_spaced:
        # generate evenly spaced grid
        print("Getting evenly spaced grid ...")
        x_ik = get_evenly_spaced_grid(args.domain_lo, args.domain_hi)
        print("Creating evenly spaced grid run directories ...")
        write_samples(x_ik, "evenly_spaced")
        print("Created evenly spaced grid run directories.")

    if args.uniform_random:
        # generate uniform random samples
        print("Getting uniform random samples ...")
        x_ik = get_uniform_random_samples(args.domain_lo, args.domain_hi)
        print("Creating uniform random samples run directories ...")
        write_samples(x_ik, "uniform_random")
        print("Created uniform random samples run directories.")

    if args.cauchy_random:
        # generate Cauchy random samples
        print("Getting Cauchy random samples ...")
        x_ik = get_cauchy_samples(args.domain_lo, args.domain_hi)
        print("Creating Cauchy random samples run directories ...")
        write_samples(x_ik, "cauchy")
        print("Created Cauchy random samples run directories.")
