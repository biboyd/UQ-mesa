#!/usr/bin/python

import os, shutil, re
import mesa_reader as mr
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sys import argv

def ReadInls(inlist,value):
        inl = open(inlist,'r')
        for line in inl:
                if str(value)in line:
                        a = line
                        val = a.split()[-1]
                        v = re.sub('d','e',val)
                        v = float(v)
                        return v

# Make arrays to store Final Stellar Mass, Reimers, and Blocker
masses = []
reims = []
block = []
lumi = []
failm = []
failr = []
failb = []

path = argv[1] 
outdir = argv[2]

os.chdir(path)

# Get the final masses from the folders
#masses = []
i=0
for file in sorted(os.listdir(path)):
    if file == "run_utils" or file == "multitask_.jugdata":
        pass
    else:
        i+=1

        # Check if it's a directory, if it's a file, we'll ignore it
        if os.path.isdir(path+'/'+file):
                
            os.chdir(path+'/'+file)
            currd = os.getcwd()
            print('Now in directory... '+os.getcwd())
            #os.system('rm ./LOGS/history.datasa')
            
            try: 
                 s = mr.MesaData('LOGS/history.data')
            except FileNotFoundError:
                 print("Can't fine LOGS/history.data . Will skip")
                 continue
            except:
                 print("found file but something else messed up")
                 continue
      
            lum = s.data('log_L')
            mass = s.data('star_mass')
            T = s.data('log_Teff')
            plt.plot(T, lum)
            xlo, xhi = plt.xlim()
            plt.xlim(xhi, xlo)
            plt.xlabel("log_Teff")
            plt.ylabel("log_L")
            plt.savefig(f"{outdir}/plots/plt{i}.png")
            plt.close()

            fl = lum[-1]
            fm = mass[-1]

            rv = ReadInls('inlist_to_wd','Reimers_scaling')
            bv = ReadInls('inlist_to_wd','Blocker_scaling')
            
            if fl < 0. :
                masses.append(fm)
                reims.append(rv)
                block.append(bv)
                print('Final mass is: '+str(fm))
                print('Reimers: '+str(rv))
                print('Blocker: '+str(bv))
                print('Lum: '+str(fl))
            else:
                print('Luminosity too low here, failed point!', fl)
                failm.append(fm)
                failb.append(bv)
                failr.append(rv)

        else:
            print(file+' is not a folder, ignoring it!')
        
np.savetxt(f"{outdir}/StarM.out",masses,delimiter=',')
np.save(f"{outdir}/StarM.npy", masses)

np.savetxt(f"{outdir}/Reims.out",reims,delimiter=',')
np.save(f"{outdir}/Reims.npy", reims)

np.savetxt(f"{outdir}/Block.out",block,delimiter=',')
np.save(f"{outdir}/Block.npy", block)

np.savetxt(f"{outdir}/FailM.out",failm,delimiter=',')
np.save(f"{outdir}/FailM.npy", failm)

np.savetxt(f"{outdir}/FailB.out",failb,delimiter=',')
np.save(f"{outdir}/FailB.npy", failb)

np.savetxt(f"{outdir}/FailR.out",failr,delimiter=',')
np.save(f"{outdir}/FailR.npy", failr)
