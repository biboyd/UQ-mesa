#!/bin/usr/python
# Make H-R diagram for chose folder
# M. Hoffman March 19 2016
# Try to make it so you can plot multiple folders or just the one

from nugridpy import mesa as ms
import numpy as np
import matplotlib.pyplot as plt
import os, re, shutil
import datetime
#from pylab import *

Q = raw_input('Do you want a single H-R plot [S] per folder or all H-R paths on one diagram[A]? [S/A] :')

# if not os.path.exists('HRs'):
#         os.mkdir('HRs')        
youarehere = os.getcwd()

now = datetime.datetime.now()
mo = str(now.month)
da = str(now.day)
hr = str(now.hour)
mn = str(now.minute)                                        

#cols = ['voilet','r','orange','y','g','b','indigo','grey','k']
#nums = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1]
#keys = dict(zip(nums,cols))                

def ReadInls(inlist,value):
        inl = open(inlist,'r')
        for line in inl:
                if str(value)+'_scaling_' in line:
                        a = line
                        lastV = re.sub('      '+value+'\_scaling\_factor \= ','',a)
                        Val = re.sub('d','e',lastV)
                        V = float(Val)
                        return V
                
def MakeHRPlot(L,T,kind,M,savename):
        fig, ax = plt.subplots()
        if kind == 'l':
                ax.plot(T,L,color=M) 
        if kind == 's':
                ax.scatter(T,L,s=40,c=M,cmap=plt.cm.gnuplot_r,vmin=min(M),vmax=max(M),edgecolor='none')
                fig.colorbar()
                cbar.set_label(Mass)
        ax.set_xlabel('Temperature')
        ax.set_xlim([max(T)+0.1,min(T)-0.1])
        ax.xaxis.label.set_fontsize(18)
        ax.set_ylabel('Luminosity')
        ax.set_ylim([min(L)-0.1,max(L)+0.1])
        ax.yaxis.label.set_fontsize(18)
        plt.savefig(savename)

#FV = raw_input('Folder path name: ')
FV = '/data/mhoffman/NUQ/fine_grid'

X = []
Y = []

os.chdir(FV)
print(os.getcwd())
for file in os.listdir(FV):

        os.chdir(file)
        file_cab = os.getcwd()

        for file in os.listdir(file_cab):
                matchc = re.match('\Ac([0-9]*)\Z',file)
                if matchc:
                        os.chdir(file)
                        print('Currently in...'+os.getcwd())
                        x = ReadInls('inlist_1.0','Blocker')
                        y = ReadInls('inlist_1.0','Reimers')

                        if x not in X:
                                X.append(x)
                        if y not in Y:
                                Y.append(y)
                        os.chdir(file_cab)

        os.chdir(FV)

print X
print Y
print len(X)

cmap = plt.get_cmap('jet')
Colors = [cmap(i) for i in Y]

print Colors

keys = dict(zip(Y,Colors))

print keys


if Q == 'S':
        os.chdir(FV)

        if not os.path.exists('HRs'):
                os.mkdir('HRs')
        today = mo+da+'_'+hr+mn 
        os.chdir(FV+'/HRs')

        if not os.path.exists(today):
                os.mkdir(today)
        os.chdir(FV+'/HRs/'+today)

        if not os.path.exists('failed'):
                os.mkdir('failed')
        os.chdir(FV)

        for file in os.listdir(FV):
                matchf = re.match(reg,file)
                if matchf:
                    os.chdir(file)
                    file_cab = os.getcwd()
                    for file in os.listdir(file_cab):
                            matchc = re.match('\Ac([0-9]*)\Z',file)
                            if matchc:
                                os.chdir(file)
                                print('Currently in...'+os.getcwd())

                                # Get Stellar Mass
                                s = ms.history_data()
                                Mass = s.get('star_mass')
                                Temp = s.get('log_Teff')
                                Lumi = s.get('log_L')
                                lastn = len(Lumi) - 1
                                final_L = Lumi[lastn]          

                                B = ReadInls('inlist_1.0','Blocker')
                                print B
                                R = ReadInls('inlist_1.0','Reimers')
                                print R

                                # Make an HR plot
                                os.chdir(FV+'/HRs/'+today)
                                name = 'R'+str(R)+'_B'+str(B)+'.png'
                                Sh = keys[R]

                                #MakeHRPlot(Lumi,Temp,'l',Sh,name)
                                if final_L > -0.999:
                                        os.chdir(FV+'/HRs/'+today+'/failed')
                                        MakeHRPlot(Lumi,Temp,'l',Sh,name)
                                        plt.close()
                                        os.chdir(file_cab)
                                else:
                                        MakeHRPlot(Lumi,Temp,'l',Sh,name)
                                        plt.close()
                                        os.chdir(file_cab)

                os.chdir(FV)

        os.chdir(youarehere)

if Q == 'A':
        
        Masses = []
        Temperatures = []
        Luminosities = []
        Reimers = []
        Blocker = []
        Textrema = []

        FV = input('Folder path name with \'\' : ')
        os.chdir(FV)

        if not os.path.exists('HRs'):
                os.mkdir('HRs')
        # for file in os.listdir(FV):
        #         matchf = re.match('\Abloc\-rand\_reim\-[0-9]\.[0-9]\Z',file)
        #         if matchf:
        #                 os.chdir(file)

        file_cab = os.getcwd()

        for file in os.listdir(file_cab):
                matchc = re.match('\Ac([0-9]*)\Z',file)
                if matchc:
                        os.chdir(file)
                        s = ms.history_data()
                        Mass = s.get('star_mass')
                        Temp = s.get('log_Teff')
                        TM = max(Temp)
                        Tm = min(Temp)
                        Lumi = s.get('log_L')
                        lastn = len(Lumi) - 1
                        final_L = Lumi[lastn]

                        if final_L < 0:
                                B = ReadInls('inlist_1.0','Blocker')
                                R = ReadInls('inlist_1.0','Reimers')
                                Ones = np.ones(lastn+1)
                                Bl = B * Ones
                                Rf = R * Ones
                                Blocker.append(Bl)
                                Reimers.append(Rf)
                                Masses.append(Mass)
                                Luminosities.append(Lumi)
                                Temperatures.append(Temp)
                                Textrema.append(TM)
                                Textrema.append(Tm)
                        else:
                                print('Luminosity too low, excluding point!')
                                os.chdir(file_cab)
                        os.chdir(file_cab)
                os.chdir(FV)
        os.chdir(FV+'/HRs/')

        name = 'HR_'+mo+da+'_'+hr+mn+'.png'
        Color = []

        for i in Reimers:
                row = []
                for j in i:
                        q = keys[j]
                        row.append(q)
                Color.append(row)

        for i in range(len(Masses)):
                A = Luminosities[i]
                B = Temperatures[i]
                C = Color[i][0]
                plt.plot(B,A,color=C)

        plt.xlim(max(Textrema)+0.1,min(Textrema)-0.1)
        plt.show()

        # ptl.gcf()
        # plt.savefig('HR_'+mo+da+'_'+hr+mn+'.png')
        # plt.show()
