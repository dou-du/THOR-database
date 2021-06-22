# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:34:25 2021

@author: Zsuzsanna Koczor-Benda, UCL
"""

import numpy as np
import math

phys_constants=dict(amu = 1.66053878e-27,    # kg
                    h   = 6.62606896e-34,    # Js
                    c   = 2.99792458e+8,     # m/s
                    k   = 1.3806504e-23      # J/K)
                    )
scaling_factors=dict(    
                    D=2.541746473,  # for dipole moment: 1 ea0 =2.54 Debye
                    A=0.529177210903, # 1 bohr = 0.529 Angstrom
                    IRfac=126.8 # factor from Philippe's paper for D2/A2amu to km/mol
                    )

def calc_scaling(T):
    # calculate scaling factors 
    #  R[cm^4/kg] = 10^-32/amu[kg] * R[Angstrom^4/amu]
    #  c[cm/s]=10^2 * c[m/s] 
    #  factor of 10^4 needed to agree with Principles of SERS book Eq. A.10
    amu=phys_constants['amu']
    h=phys_constants['h']
    c=phys_constants['c']
    k=phys_constants['k']
    scalingR=math.pow(10, 4)*h*math.pow(math.pi, 2)/((math.pow(10, 32)*amu)*(math.pow(10, 2)*c)*22.5)
    scalingR=45*math.pow(scaling_factors['A'],4)*scalingR
    scalingexp=-h*(math.pow(10, 2)*c)/(k*T)
    scalingIR=math.pow(scaling_factors['D']/scaling_factors['A'],2)*scaling_factors['IRfac']
    return scalingIR,scalingR,scalingexp

def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())

# Rotate molecule to reference orientation
# not finished yet
#def rotate_to_reference(D,P,Z,Q):
#    rot=np.zeros((3,3))
#    Prot=np.zeros_like(P)
#    Drot=np.zeros_like(D)
#    Prot=np.matmul(np.matmul(rot,P),np.transpose(rot))
#    Drot=np.matmul(rot,D)
#    return Drot,Prot

def single_rotD(p,a=0,b=0,c=0):
 #   e=np.array([0.,0.,1.0])
    Rz=np.array([[math.cos(a), -math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])
    Rx=np.array([[1,0,0],[0,math.cos(b), -math.sin(b)],[0,math.sin(b),math.cos(b)]])
    # e is invariant to Rz2 
    R=np.matmul(Rx,Rz)
 #   Rz2=np.array([[math.cos(c), -math.sin(c),0],[math.sin(c),math.cos(c),0],[0,0,1]])
 #   R=np.matmul(Rz2,np.matmul(Rx,Rz))
    rot=np.matmul(p,np.transpose(R))
    return rot

def single_rotP(p,a=0,b=0,c=0):
 #   e=np.array([0.,0.,1.0])
    Rz=np.array([[math.cos(a), -math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])
    Rx=np.array([[1,0,0],[0,math.cos(b), -math.sin(b)],[0,math.sin(b),math.cos(b)]])
    # e is invariant to Rz2 ?
    R=np.matmul(Rx,Rz)
   # Rz2=np.array([[math.cos(c), -math.sin(c),0],[math.sin(c),math.cos(c),0],[0,0,1]])
   # R=np.matmul(Rz2,np.matmul(Rx,Rz))
    rot=np.matmul(R,np.matmul(p,np.transpose(R)))
    return rot

def unit_rot(axis,rad):
    if axis==1 :
        R=np.array([[math.cos(rad),-math.sin(rad),0],[math.sin(rad),math.cos(rad),0],[0,0,1]]) 
    elif axis==2 :
        R=np.array([[math.cos(rad),0,math.sin(rad)],[0,1,0],[-math.sin(rad),0,math.cos(rad)]])
    elif axis==3 :
        R=np.array([[1,0,0],[0,math.cos(rad),-math.sin(rad)],[0,math.sin(rad),math.cos(rad)]])
    else :
        R=np.array([[1,0,0],[0,1,0],[0,0,1]])
    return R

