#!/usr/bin/python
# June 2016
# Melissa Hoffman
# Make Random folders and run script
# 

import numpy as np
import random
import os, shutil, re
import math

# These are the template files that will be copied and made into the series of directories

template = 'prems_to_wd_template'
main_list = 'inlist_template'

def ChangeValue(inlist,newinl,param,newval):
    with open(inlist) as f:
        q = open(newinl,'a')
        for line in f:
            l = line
            if param in l:
                q.write('      '+param+' = '+str(newval)+'\n')
            if not param in l:
                q.write(l)
        
cwd = os.getcwd()
# Change this name to whatever you what the test suite to be called
topdir = '1M_CDGrid_split1'
fpath = cwd+'/'+topdir

# Number of folders w. unique Blockers & Reimers combinations
# for our study, we would want ~200
pts = 54
l = range(pts)
runlist = l[0::6]

print('Now creating top directory... '+topdir)
print('Number of points in paramters space... '+str(pts))

# Ask user if running on a cluster or not?
Q1 = raw_input("Are you running on a cluster [y/n] ? ")
Q2 = raw_input("Will you be using uniform[u] or cauchy[c] distribution [u/c] ? ")

random.seed()

# Calculate Cauchy distributed variables, see FK 2004 for algorithm
# Number of experiments is pts
n_inp = 2 # number of inputs

x_lim = np.zeros( (n_inp, 2) )
x_lim[0,:] = [0.3, 0.7]   #RGB winds
x_lim[1,:] = [0.01, 0.04]  #AGB winds

# Initialize arrays
x = np.zeros( (n_inp, 2) )

r_ik = np.zeros( (n_inp, pts) )
c_ik = np.zeros( (n_inp, pts) )
d_ik = np.zeros( (n_inp, pts) )
x_ik = np.zeros( (n_inp, pts) )

delta = np.zeros( (n_inp) ) # this is the interval half-width

delta[0] = ( (x_lim[0,1] - x_lim[0,0]) / 2.0 )
delta[1] = ( (x_lim[1,1] - x_lim[1,0]) / 2.0 )

xtilde_ik = ( x_lim[:,1] + x_lim[:,0] )/ 2.0  # This is the midpoint

Kvals = []
 
for k in range(pts):

    for i in range(n_inp):

        r_ik[i,k] = random.uniform(0.0,1.0)
    
    c_ik[:,k] = np.tan( math.pi * (r_ik[:,k] - 0.5) )
    
    K = max( abs( c_ik[:,k] ))
    
    Kvals.append(K)

    d_ik[:,k] = ( delta[:] * c_ik[:,k] ) / K

    x_ik[:,k] = xtilde_ik[:] + d_ik[:,k]



for i in range(pts):
    name = 'c'+str(i)
    # Change the intervales of rx and bx to match Reimers and Blockers intervals
    # rx and bx are random numbers selected from these intervals OR they are 
    # a Cauchy distributed value from the above calculation

    if Q2 == 'u':
        rx = str(np.random.uniform(0.3,0.9))
        bx = str(np.random.uniform(0.0,0.1))
    if Q2 == 'c':
        rx = str(x_ik[0,i])
        bx = str(x_ik[1,i])

    # Copy templates and create directory called c# 
    shutil.copytree(template,topdir+'/'+name)
    shutil.copy(main_list,fpath+'/'+name+'/.')
    os.chdir(fpath+'/'+name)

    # Change Values in inlist
    ChangeValue(main_list,'CHANGE_R','Reimers_scaling_factor',rx)
    ChangeValue('CHANGE_R','inlist_1.0','Blocker_scaling_factor',bx)

    templ = open('batch_cmd.sh','r')
    cluster = open('cluster.sh','r')
    runfile = open('run_script','a')
    if Q1 == 'y':
        for line in cluster:
            if './rn' in line:
                runfile.write('mpirun -np 1 ./rn > run_out_'+str(i)+'.log &\n')
                runfile.write('cd ../c'+str(i+1)+'\n')
                runfile.write('mpirun -np 1 ./rn > run_out_'+str(i+1)+'.log &\n')
                runfile.write('cd ../c'+str(i+2)+'\n')
                runfile.write('mpirun -np 1 ./rn > run_out_'+str(i+2)+'.log &\n')
                runfile.write('cd ../c'+str(i+3)+'\n')
                runfile.write('mpirun -np 1 ./rn > run_out_'+str(i+3)+'.log &\n')
                runfile.write('wait')
            else:
                runfile.write(line)
    if Q1 == 'n':
        for line in templ:
            if './rn' in line:
                runfile.write('./rn > run_out_'+str(i)+'.log\n')
            else:
                runfile.write(line)
    os.system('rm inlist_template')
    os.system('rm CHANGE_R')
    os.system('chmod +x run_script')
    templ.close()
    runfile.close()
    # If this folder is in nums, i.e. if it's the 4th folder, actually run the script        
    # Depending on whether or not you're running on a cluster, the command to run is different
    print(os.getcwd())
    os.chdir(cwd)
    with open('logfile','a') as b:
        b.write('Folder name: '+name+'\n')
        b.write('Reimers: '+str(rx)+'\n')
        b.write('Blocker: '+str(bx)+'\n')
        b.write('-----------------------------------------\n')
    np.savetxt('kvalues.out',Kvals)


f = open('runlog','a')
for i in os.listdir(fpath):
    newd = fpath+'/'+i
    n = re.sub('c','',i)
    nn = float(n)
    if Q1 == 'n':
        os.chdir(newd)
        os.system('at -f run_script -m -q a now &> at.out')
        #print('Run this')
        os.chdir(fpath)
        f.write('Ran in folder '+newd+'\n')
    if Q1 == 'y':                                
        if nn in runlist:
            os.chdir(newd)
            os.system('qsub run_script')
            os.chdir(fpath)
            f.write('Ran in folder '+newd+'\n') 
        else:
            #print('Don\'t run')
            os.chdir(fpath)
            f.write('Skipped folder '+newd+'\n')
f.close()    

