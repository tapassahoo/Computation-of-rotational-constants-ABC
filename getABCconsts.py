import os
import sys
from subprocess import call
import math
import numpy as np
import scipy.stats

# h = 6.62607015*10^-34 J.s = 6.62607015*10^-34 kg.m^2.s^-2.s = 6.62607015*10^-34 kg.m^2.s^-1
# 1 amu = 1.660599*10^-27 kg 
# 1 anngstrom = 10^-10 m
# 1 cm^-1 = 2.99793*10^10 s^-1
unitconv=6.62607015/(8.0*math.pi*math.pi*1.660599*2.99793) # hbar/(4*pi*amu*angstrom^2) to Herz (s^-1); 
unitconv=unitconv*1000

## q-TIP4P/F parameters
#angHOH=107.4
#dOH=0.9419
##
angHOH=107.42
dOH=0.9582
ro_wf=np.zeros(3,dtype='float')
rh1_wf=np.zeros(3,dtype='float')
rh2_wf=np.zeros(3,dtype='float')
#
ang1=(angHOH*math.pi)/180.0
zH=ro_wf[2]-math.sqrt(0.5*dOH*dOH*(1.0+math.cos(ang1)))
xH=math.sqrt(dOH*dOH-(ro_wf[2]-zH)*(ro_wf[2]-zH))
print(xH,zH)
#
rh1_wf[0]=xH
rh1_wf[1]=0.0
rh1_wf[2]=zH
#
rh2_wf[0]=-rh1_wf[0]
rh2_wf[1]=rh1_wf[1]
rh2_wf[2]=rh1_wf[2]
print('ro ',ro_wf)
print('rh1 ',rh1_wf)
print('rh2 ',rh2_wf)
## Computations of Iij elements of the I tensor
mass_H=1.0078
mass_O=15.994915
mass_com=15.994915+2.0*1.0078
#Computation of Eq. 6.5 on page 255 in Zare's book
rcom = np.zeros(3,dtype='float')
rcom[0]=(mass_O*ro_wf[0]+mass_H*rh1_wf[0]+mass_H*rh2_wf[0])/mass_com
rcom[1]=(mass_O*ro_wf[1]+mass_H*rh1_wf[1]+mass_H*rh2_wf[1])/mass_com
rcom[2]=(mass_O*ro_wf[2]+mass_H*rh1_wf[2]+mass_H*rh2_wf[2])/mass_com
print('rcom ',rcom)

Itensor=np.zeros((3,3),dtype='float')
##computations of diagonal elements
Itensor[0,0]=mass_O*(ro_wf[1]*ro_wf[1]+ro_wf[2]*ro_wf[2])+mass_H*(rh1_wf[1]*rh1_wf[1]+rh1_wf[2]*rh1_wf[2])+mass_H*(rh2_wf[1]*rh2_wf[1]+rh2_wf[2]*rh2_wf[2])-mass_com*(rcom[1]*rcom[1]+rcom[2]*rcom[2])
Itensor[1,1]=mass_O*(ro_wf[0]*ro_wf[0]+ro_wf[2]*ro_wf[2])+mass_H*(rh1_wf[0]*rh1_wf[0]+rh1_wf[2]*rh1_wf[2])+mass_H*(rh2_wf[0]*rh2_wf[0]+rh2_wf[2]*rh2_wf[2])-mass_com*(rcom[0]*rcom[0]+rcom[2]*rcom[2])
Itensor[2,2]=mass_O*(ro_wf[0]*ro_wf[0]+ro_wf[1]*ro_wf[1])+mass_H*(rh1_wf[0]*rh1_wf[0]+rh1_wf[1]*rh1_wf[1])+mass_H*(rh2_wf[0]*rh2_wf[0]+rh2_wf[1]*rh2_wf[1])-mass_com*(rcom[0]*rcom[0]+rcom[1]*rcom[1])
##computations of off-diagonal elements
Itensor[0,1]=-(mass_O*ro_wf[0]*ro_wf[1]+mass_H*rh1_wf[0]*rh1_wf[1]+mass_H*rh2_wf[0]*rh2_wf[1])+mass_com*rcom[0]*rcom[1]
Itensor[0,2]=-(mass_O*ro_wf[0]*ro_wf[2]+mass_H*rh1_wf[0]*rh1_wf[2]+mass_H*rh2_wf[0]*rh2_wf[2])+mass_com*rcom[0]*rcom[2]
Itensor[1,2]=-(mass_O*ro_wf[1]*ro_wf[2]+mass_H*rh1_wf[1]*rh1_wf[2]+mass_H*rh2_wf[1]*rh2_wf[2])+mass_com*rcom[1]*rcom[2]
print(Itensor)
print(mass_com*(rcom[0]*rcom[0]+rcom[2]*rcom[2]))

print("Final A, B, C constants")
print(unitconv)
aconst = unitconv/Itensor[0,0]
bconst = unitconv/Itensor[2,2]
cconst = unitconv/Itensor[1,1]

##Spectroscopic values are: A = 27.877 cm^-1; B = 14.512 cm^-1; C = 9.285 cm^-1
print(aconst)
print(bconst)
print(cconst)

##Computation of spectroscopic geomrtry from the spectroscopic A,B,C constants.
aconst_spec = 27.877          # cm^-1
bconst_spec = 14.512          # cm^-1
hbar_au=1.0                   # hbar in atomic unit
au_wavenumber=2.1947463137e+5 # atomic unit to wavenumber
mh_au=1837.15264409           # mass of hydrogen in atomic unit
au_angstrom=0.52917720859     # atomic unit to angstrom
amu_au=1822.88848325          # amu to au
xhsq = 0.25*hbar_au*hbar_au*au_wavenumber/(bconst_spec*mh_au)
xh = np.sqrt(xhsq)*au_angstrom
mass_com_au = mass_com*amu_au
reduced_mass = (1.0-(2.0*mh_au/mass_com_au))
zhsq = 0.25*hbar_au*hbar_au*au_wavenumber/(aconst_spec*mh_au*reduced_mass)
zh = np.sqrt(zhsq)*au_angstrom
doh=np.sqrt(xh*xh+zh*zh)
cost=(zh*zh-xh*xh)/(doh*doh)
theta=math.acos(cost)*180.0/math.pi

print('xh and zh are')
print(xh,zh,doh,theta)
