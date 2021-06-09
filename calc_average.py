# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:37:31 2021

@author: Zsuzsanna Koczor-Benda, UCL
"""

import numpy as np
import math

def full_average_IR(d):
    full_av=(math.pow(d[0],2) + math.pow(d[1],2) + math.pow(d[2],2))/3
    return full_av

# parallel in-out fields
def full_average_R(p): 
    full_av=(3*math.pow(p[0, 0],2) + math.pow(p[0, 1] + p[1, 0],2) + 
          3*math.pow(p[1, 1],2) + math.pow(p[0, 2] + p[2, 0],2) + 
          math.pow(p[1, 2] + p[2, 1],2) + 2*p[1, 1]*p[2, 2] + 
          3*math.pow(p[2, 2],2) + 2*p[0, 0]*(p[1, 1] + p[2, 2]))/15
    return full_av

# orthogonal in-out fields
def full_average_R_orth(p): 
    gamma=3*math.pow(p[0, 1],2)+3*math.pow(p[0, 2],2)+ 3*math.pow(p[1, 2],2)+ 0.5*(math.pow(p[0, 0] - p[1, 1],2) + 
             math.pow(p[1, 1] - p[2, 2],2) + 
             math.pow(p[2, 2] - p[0, 0],2))
    alpha=1/9* math.pow(p[0, 0] + p[1, 1]+p[2,2],2)
    rii=(45*alpha+4*gamma)/45
    rji=3*gamma/45
    full_av=rii+rji
  # depolarization ratio:  
    depol=1/(1+rji/rii)
    return full_av,depol  

    
def full_average(d,p):
    full_av=(3*(5*math.pow(d[0],2) + math.pow(d[1],2) + math.pow(d[2],2))*math.pow(p[0, 0],2) + 
      (3*math.pow(d[0],2) + 3*math.pow(d[1],2) + math.pow(d[2],2))*math.pow(p[0, 1],2) + 
      3*math.pow(d[0],2)*math.pow(p[0, 2],2) + math.pow(d[1],2)*math.pow(p[0, 2],2) + 
      3*math.pow(d[2],2)*math.pow(p[0, 2],2) + 2*(3*math.pow(d[0],2) + 3*math.pow(d[1],2) + 
        math.pow(d[2],2))*p[0, 1]*p[1, 0] + 4*d[1]*d[2]*p[0, 2]*
       p[1, 0] + 3*math.pow(d[0],2)*math.pow(p[1, 0],2) + 
      3*math.pow(d[1],2)*math.pow(p[1, 0],2) + math.pow(d[2],2)*math.pow(p[1, 0],2) + 
      4*d[0]*d[2]*p[0, 2]*p[1, 1] + 12*d[0]*d[1]*p[1, 0]*
       p[1, 1] + 3*math.pow(d[0],2)*math.pow(p[1, 1],2) + 
      15*math.pow(d[1],2)*math.pow(p[1, 1],2) + 3*math.pow(d[2],2)*math.pow(p[1, 1],2) + 
      4*d[0]*d[1]*p[0, 2]*p[1, 2] + 4*d[0]*d[2]*p[1, 0]*
       p[1, 2] + 12*d[1]*d[2]*p[1, 1]*p[1, 2] + 
      math.pow(d[0],2)*math.pow(p[1, 2],2) + 3*math.pow(d[1],2)*math.pow(p[1, 2],2) + 
      3*math.pow(d[2],2)*math.pow(p[1, 2],2) + 6*math.pow(d[0],2)*p[0, 2]*p[2, 0] + 
      2*math.pow(d[1],2)*p[0, 2]*p[2, 0] + 6*math.pow(d[2],2)*p[0, 2]*
       p[2, 0] + 4*d[1]*d[2]*p[1, 0]*p[2, 0] + 
      4*d[0]*d[2]*p[1, 1]*p[2, 0] + 4*d[0]*d[1]*p[1, 2]*
       p[2, 0] + 3*math.pow(d[0],2)*math.pow(p[2, 0],2) + math.pow(d[1],2)*math.pow(p[2, 0],2) + 
      3*math.pow(d[2],2)*math.pow(p[2, 0],2) + 4*d[0]*d[1]*p[0, 2]*p[2, 1] + 
      4*d[0]*d[2]*p[1, 0]*p[2, 1] + 12*d[1]*d[2]*p[1, 1]*
       p[2, 1] + 2*math.pow(d[0],2)*p[1, 2]*p[2, 1] + 
      6*math.pow(d[1],2)*p[1, 2]*p[2, 1] + 6*math.pow(d[2],2)*p[1, 2]*
       p[2, 1] + 4*d[0]*d[1]*p[2, 0]*p[2, 1] + 
      math.pow(d[0],2)*math.pow(p[2, 1],2) + 3*math.pow(d[1],2)*math.pow(p[2, 1],2) + 
      3*math.pow(d[2],2)*math.pow(p[2, 1],2) + 12*d[0]*d[2]*p[0, 2]*
       p[2, 2] + 4*d[0]*d[1]*p[1, 0]*p[2, 2] + 
      2*math.pow(d[0],2)*p[1, 1]*p[2, 2] + 6*math.pow(d[1],2)*p[1, 1]*
       p[2, 2] + 6*math.pow(d[2],2)*p[1, 1]*p[2, 2] + 
      12*d[1]*d[2]*p[1, 2]*p[2, 2] + 
      12*d[0]*d[2]*p[2, 0]*p[2, 2] + 
      12*d[1]*d[2]*p[2, 1]*p[2, 2] + 
      3*math.pow(d[0],2)*math.pow(p[2, 2],2) + 3*math.pow(d[1],2)*math.pow(p[2, 2],2) + 
      15*math.pow(d[2],2)*math.pow(p[2, 2],2) + 4*p[0, 1]*
       (3*d[0]*d[1]*p[1, 1] + d[2]*(d[0]*p[1, 2] + 
          d[1]*(p[0, 2] + p[2, 0]) + d[0]*p[2, 1]) + 
        d[0]*d[1]*p[2, 2]) + 2*p[0, 0]*
       (6*d[0]*d[1]*p[0, 1] + 6*d[0]*d[1]*p[1, 0] + 
        (3*math.pow(d[0],2) + 3*math.pow(d[1],2) + math.pow(d[2],2))*p[1, 1] + 
        2*d[2]*(3*d[0]*p[0, 2] + 3*d[0]*p[2, 0] + 
          d[1]*(p[1, 2] + p[2, 1])) + 
        (3*math.pow(d[0],2) + math.pow(d[1],2) + 3*math.pow(d[2],2))*p[2, 2]))/105       
    return full_av

def numerical_sector_average(d,p,k=1,l=1,a=0,b=0,c=0,nump=30):
    e=np.array([0.,0.,1.0])
    R0z=np.array([[math.cos(a), -math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])
    R0x=np.array([[1,0,0],[0,math.cos(b), -math.sin(b)],[0,math.sin(b),math.cos(b)]])
    R0z2=np.array([[math.cos(c), -math.sin(c),0],[math.sin(c),math.cos(c),0],[0,0,1]])
    R0=np.matmul(R0z2,np.matmul(R0x,R0z))
    q=np.matmul(np.transpose(R0),e)
    ir_av=0
    r_av=0
    p_av=0
    maxtheta=math.pi/k
    maxphi=2*math.pi/l
    minjj=math.cos(maxtheta)
    for i in range(nump):
        phi=maxphi*i/nump
    #    print("phi ",phi)
        Rz=np.array([[math.cos(phi), -math.sin(phi),0],[math.sin(phi),math.cos(phi),0],[0,0,1]])
        for j in range(0,nump):
            jj=1-(1-minjj)*j/nump
            theta=math.acos(jj)
            Rx=np.array([[1,0,0],[0,math.cos(theta), -math.sin(theta)],[0,math.sin(theta),math.cos(theta)]])
            for m in range(nump):
                xi=2*m*math.pi/nump
                Rz2=np.array([[math.cos(xi), -math.sin(xi),0],[math.sin(xi),math.cos(xi),0],[0,0,1]])
                R=np.matmul(Rz2,np.matmul(Rx,Rz))
               # R=np.matmul(Rx,Rz)
           #     print(xi,math.pow(np.matmul(q,np.matmul(np.matmul(R,np.matmul(p,np.transpose(R))),np.transpose(q))),2))
                ir=math.pow(np.matmul(q,np.matmul(R,d)),2)
                r=math.pow(np.matmul(q,np.matmul(np.matmul(R,np.matmul(p,np.transpose(R))),np.transpose(q))),2)
                ir_av+=ir
                r_av+=r
                p_av+=ir*r
    ir_av=ir_av/math.pow(nump,3)
    r_av=r_av/math.pow(nump,3)
    p_av=p_av/math.pow(nump,3)
    return ir_av,r_av,p_av
    