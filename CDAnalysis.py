#!/usr/bin/python

# Cauchy Deviates analysis
# Using a lot of Max Katz's code

import numpy as np
import os, re, shutil
import mesa as ms

# The Maximum Likelihood equation, as seen in Ferson and Kreinovich 2004).
# Moves everything to the left hand side so that if d is correct the function will be 0

def MLE(c,d,N):
    result = 0.0
    for i in range(len(c)):
        result += 1.0/ (1.0 + (c[i]/d)**2)
    result += -N/2.0
    return result

def ReadInls(inlist,value):
    inl = open(inlist,'r')
    for line in inl:
        if str(value) in line:
            a = line
            val = float( a.split()[-1] )
            return val


# We compute the c_k values using the results of the MESA run,
# and the results of the midpoints of the intervals, ytilde
# Meaning if our x_1 = [0.3,0.9] and x_2 = [0.01,0.1] then
# ytile = final mass from MESA run with x_1 = 0.6 and x_2 = 0.05

midrun = '/data/mhoffman/NUQ/new_midpoints/tile4' # This should be a path the the ytilde folder
os.chdir(midrun)
s = ms.history_data()
mass = s.get('star_mass')
ytilde = mass[ (len(mass)-1) ] # obtain the last mass

# now read in all the star masses
# Name of top directory and current directory

#path = raw_input('Please provide path of top folder using: ')
#os.chdir(path)

paths = ['/data/mhoffman/NUQ/cdtiles/tile4']

kvals = []
masses = []
reims = []
block = []
nums = []

for i in paths:
    os.chdir(i)

    
    # Read in k values from kvalues.out
    with open('kvalues.out') as f:
        #kvals = []
        for line in f:
            line = float(line)
            kvals.append(line)

    # Get the final masses from the folders
    #masses = []
    for file in os.listdir(i):
    
        # Check if it's a directory, if it's a file, we'll ignore it
        if os.path.isdir(i+'/'+file) == True:

            os.chdir(i+'/'+file)
            currd = os.getcwd()
            print(currd)

            for file in os.listdir(currd):
                logt = re.match('\Arun\_out\_([0-9]*)\.log\Z',file)
                if logt:
                
                    cn = logt.groups()[0]
                    print(cn)

            nums.append(cn)
            s = ms.history_data()
            lum = s.get('log_L')
            mass = s.get('star_mass')

            fl = lum[ len(lum)-1 ]
            fm = mass[ len(mass)-1 ]

            rv = ReadInls('inlist_1.0','Reimers_scaling')
            bv = ReadInls('inlist_1.0','Blocker_scaling')
            
            if fl < 0. :
                masses.append(fm)
                reims.append(rv)
                block.append(bv)
            else:
                print('In '+currd+' luminosity too low, excluding point!')
                #masses.append(-1.)
                #reims.append(-1.)
                #block.append(-1.)
                os.chdir(i)

        else:
            print(file+' is not a folder, ignoring it!')


# Compute simulated errors

c_k = np.zeros( len(masses) )

for i in range(len(c_k)):

    c_k[i] = kvals[i] * ( masses[i] - ytilde )

# Now solve MLE with the bisection method

Nexp = len(masses)

N_iter_max = 1000000

N_iter = 1

dlo = 1e-4
dhi = max(abs(c_k))

tol = 1.0e-8

while (N_iter <= N_iter_max):
    
    dmid = (dlo + dhi) / 2.0 

    func_lo = MLE(c_k, dlo, Nexp)
    func_mid = MLE(c_k, dmid, Nexp)

    if ((dhi - dlo) / 2.0 < tol):
        break
    if (func_lo * func_mid < 0 ):
        dhi = dmid
    else:
        dlo = dmid

    N_iter = N_iter + 1

if (N_iter == N_iter_max):
    print "Result not converged; stopping"
    exit

sigma = dmid * np.sqrt(2.0 / Nexp)

print('The result for ytilde is: '+str(ytilde))
print('The result for standard deviation is: '+str(sigma))
print('The result for delta is: '+str(dmid))
#
#
#
#file = open('results.out', 'w')

#np.savetxt('masses.out',masses,delimiter=',')
#np.savetxt('reims.out',reims,delimiter=',')
#np.savetxt('block.out',block,delimiter=',')

        
            
