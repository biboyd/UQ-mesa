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

# Make arrays to store Final Stellar Mass, mesh , and var control
masses = []
mesh = []
var = []
lumi = []

path = argv[1] 
outdir = argv[2]

# Get the final masses from the folders
#masses = []
i=0
for f in sorted(os.listdir(path)):
    # regex to check if proper directory
    pattern = re.compile(r"^default[\w]{3,}")

    # returns None if not a match
    valid_dir = re.fullmatch(pattern, f)

    if valid_dir is not None:
        i+=1

        # Check if it's a directory, if it's a f, we'll ignore it
        if os.path.isdir(path+'/'+f):
            try: 
                 s = mr.MesaData(f"{path}/{f}/LOGS/history.data")
            except FileNotFoundError:
                 print("Can't fine LOGS/history.data . Will skip")
                 continue
            except:
                 print("found LOGS/history.data file but something else messed up")
                 continue
      
            lum = s.data('log_L')
            mass = s.data('star_mass')
            T = s.data('log_Teff')
            #plt.plot(T, lum)
            #xlo, xhi = plt.xlim()
            #plt.xlim(xhi, xlo)
            #plt.xlabel("log_Teff")
            #plt.ylabel("log_L")
            #plt.savefig(f"{outdir}/plots/plt{i}.png")
            #plt.close()

            fl = lum[-1]
            fm = mass[-1]

            mv = ReadInls(f"{path}/{f}/inlist_to_wd",'mesh_delta_coeff')
            vv = ReadInls(f"{path}/{f}/inlist_to_wd",'varcontrol_target')
            
            if fl < 0. :
                masses.append(fm)
                mesh.append(mv)
                var.append(vv)
                print('Final mass is: '+str(fm))
                print('mesh_delta_coeff: '+str(mv))
                print('varcontrol_target: '+str(vv))
                print('Lum: '+str(fl))

        else:
            print(f+' is not a folder, ignoring it!')
np.savetxt(f"{outdir}/StarM.out",masses,delimiter=',')
np.save(f"{outdir}/StarM.npy", masses)

np.savetxt(f"{outdir}/res_m.out",mesh,delimiter=',')
np.save(f"{outdir}/res_m.npy", mesh)

np.savetxt(f"{outdir}/res_v.out",var,delimiter=',')
np.save(f"{outdir}/res_v.npy", var)

