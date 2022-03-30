import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
# todo: import astropy and plasmapy packages to reduce overhead

''' 
    MODEL INPUTS
'''
# numerical grid
zMax = 0.2
zSteps = 500
tMax = 6.28e-6
tSteps = 500
# pre-ionized plasma properties
zs0 = 0
ws0 = 0.002
pi0 = 0.01
zpi0 = 0
Te0 = 20
Ti0 = 0.01
# propellant gas profile
prop = 'Ar'
nn0 = 1e20
zn0 = 0.01
Tn = 298/11606
nn0z = lambda z: nn0*np.exp(-z/zn0)
# thruster circuit
V0 = 3e3
Lc = 0.5e-6
L0 = 50e-9
Rc = 0.005
C0 = 2e-6
# coil geometry
zc = 0.02
za = -0.002
ro = 0.1
ri = 0.03
# multi-pulse
nPulse = 1
fRep = 10e3


'''
    CONSTANTS
'''
ec = 1.6e-19
kb = 1.38e-23
mu0 = 4*np.pi*10**(-7)
eps0 = 8.854e-12
g0 = 9.81
gamma = 5/3
me = 9.11e-31
mp = 1.67e-27
As = np.pi*(ro**2-ri**2)
tLC = 2*np.pi*np.sqrt(C0*Lc)

'''
    PROPELLANT MODEL
'''
if prop.lower() in ['ar','argon']:
    aw = 40
    mi = aw*mp
    un = np.sqrt(ec*Tn/mi)
    sigEN = 4e-20
    sigCX = 5e-19
    sigQM = 1.57e-18
    epsIZ = 15.76
    epsEX = 12.14
    S_es = lambda Te: 2.336e-14*Te**1.609*np.exp(0.0618*np.log(Te)**2-0.1171*np.log(Te)**3)
    S_ion = lambda Te: 2.34e-14*Te**0.59*np.exp(-17.44/Te)
    S_ex = lambda Te: 2.48e-14*Te**0.33*np.exp(-12.78/Te)
    eps_ion = lambda Te: (S_ion(Te)*epsIZ+S_ex(Te)*epsEX+S_es(Te)*3*me*Te/mi)/S_ion(Te)
    def coul_log(ne, Te):
        if Te < 7.389:
            return 23-np.log(ne*10**(-6))**(1/2)*Te**(-3/2)
        else:
            return 24-np.log(ne*10**(-6))**(1/2)*Te**(-1)
    nu_ei = lambda ne, Te: 2.91e-12*ne*Te**(-3/2)*coul_log(ne, Te)
    nu_en = lambda nn, Te: nn*sigEN*np.sqrt(8*ec*Te/(np.pi*me))
    res_cl = lambda ne, nn, Te: (me*(nu_ei(ne, Te)+nu_en(nn, Te))/(ne*ec**2))
    d_sd = lambda Leff, Cp, ne, nn, Te: np.sqrt(2*res_cl(ne, nn, Te)*np.sqrt(Leff*Cp)/mu0)
    Rp = lambda Leff, Cp, ne, nn, Te, ws, ro, ri: np.pi*res_cl(ne, nn, Te)/np.minimum(d_sd(Leff, Cp, ne, nn, Te), ws)*(ro+ri)/(ro-ri)
    Dp = lambda ne, nn, Te: (Te*me/(mi*ne*ec*res_cl(ne, nn, Te)))
elif prop.lower() in ['xe','xenon']:
    aW = 131.293
    mi = aW*mp
    un = np.sqrt(ec*Tn/mi)
    sigEN = 5e-20
    sigCX = 7.5e-19
    sigQM = 3e-19
    epsIZ = 12.13
    epsEX = 8.84
    S_es = lambda Te: 6.6e-19*np.sqrt(8*ec*Te/(np.pi*me))*(Te/4-0.1)/(1+(Te/4)**1.6)
    S_ion = lambda Te: 1e-20*np.sqrt(8*ec*Te/(np.pi*me))*(-0.0001031*Te**2+6.386*np.exp(-12.127/Te))
    S_ex = lambda Te: 1.93e-19*np.sqrt(8*ec*Te/(np.pi*me))*np.exp(-11.6/Te)/np.sqrt(Te)
    eps_ion = lambda Te: (S_ion(Te)*epsIZ+S_ex(Te)*epsEX+S_es(Te)*3*me*Te/mi)/S_ion(Te)
    def coul_log(ne, Te):
        if Te < 7.389:
            return 23-np.log(ne*10**(-6))**(1/2)*Te**(-3/2)
        else:
            return 24-np.log(ne*10**(-6))**(1/2)*Te**(-1)
    nu_ei = lambda ne, Te: 2.91e-12*ne*Te**(-3/2)*coul_log(ne, Te)
    nu_en = lambda nn, Te: nn*sigEN*np.sqrt(8*ec*Te/(np.pi*me))
    res_cl = lambda ne, nn, Te: (me*(nu_ei(ne, Te)+nu_en(nn, Te))/(ne*ec**2))
    d_sd = lambda Leff, Cp, ne, nn, Te: np.sqrt(2*res_cl(ne, nn, Te)*np.sqrt(Leff*Cp)/mu0)
    Rp = lambda Leff, Cp, ne, nn, Te, ws, ro, ri: np.pi*res_cl(ne, nn, Te)/np.minimum(d_sd(Leff, Cp, ne, nn, Te), ws)*(ro+ri)/(ro-ri)
    Dp = lambda ne, nn, Te: (Te*me/(mi*ne*ec*res_cl(ne, nn, Te)))
else:
    print('invalid propellant choice.')

'''
    LUMPED-ELEMENT CIRCUIT MODEL
'''

'''
    INTER-PULSE MODEL
'''

'''
    PLOT STYLES
'''