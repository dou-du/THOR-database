# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:34:25 2021

@author: Zsuzsanna Koczor-Benda, UCL
"""

import numpy as np
import math
import openbabel as ob

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
    scalingpolar=math.pow(scaling_factors['A'],4)
    scalingexp=-h*(math.pow(10, 2)*c)/(k*T)
    scalingIR=math.pow(scaling_factors['D']/scaling_factors['A'],2)*scaling_factors['IRfac']
    return scalingIR,scalingR,scalingexp, scalingpolar

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

def convert_to_mol(filename,molfilename):
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "mol") 
    mol = ob.OBMol()
    obConversion.ReadFile(mol, filename)
    outMDL = obConversion.WriteString(mol)
    obConversion.WriteFile(mol,molfilename)
    
def get_oriented_box(coords,vdw):
        coormin=np.zeros_like(coords)
        coormax=np.zeros_like(coords)
        for i in range(len(vdw)): 
            coormin[i,:]=coords[i,:]-vdw[i]
            coormax[i,:]=coords[i,:]+vdw[i]
        cell=np.max(coormax,axis=0)-np.min(coormin,axis=0)
        return cell    
    
def rotate_molecule(mol,filename,D,P,phi=0,theta=0):  
    # rotate to reference orientation first, then do custom rotation with phi, theta
    
    # for rotating to reference position
    # locate thiol S atom and its first- and second-level neighbours
    S_coords=np.zeros(3)
    neighb2_idxs=np.zeros(4,dtype=int)
    neighb2_num=np.zeros(4,dtype=int)
    thiol=False
    n1_indx=-1
    for a in ob.OBMolAtomIter(mol): 
        if a.GetAtomicNum()==16: 
            thiol=False
            n=0
            for neighbour in ob.OBAtomAtomIter(a):
                if neighbour.GetAtomicNum()==79: 
                      thiol=True
            if thiol==True:
                S_indx=a.GetIdx()
                S_coords[0]=a.GetX()
                S_coords[1]=a.GetY()
                S_coords[2]=a.GetZ()
                for neighbour in ob.OBAtomAtomIter(a):   
                    if neighbour.GetAtomicNum()!=79:
                        n1_indx=neighbour.GetIdx()
                        for neighbour2 in ob.OBAtomAtomIter(neighbour):
                            if neighbour2.GetIdx()==S_indx:
                                n+=1
                                continue
                            neighb2_idxs[n]=neighbour2.GetIdx()                     
                            neighb2_num[n]=neighbour2.GetAtomicNum()
                            n+=1
                        break     
    if n1_indx==-1:
        print("Error: S-Au not found in molecule")
        return 0
    
    n2_indx=neighb2_idxs[np.argmax(neighb2_num)]
    
    coords_orig = [[atom.GetX(),atom.GetY(),atom.GetZ()] for atom in ob.OBMolAtomIter(mol)] 
    
    # translate S to origin
    translate=np.array([-S_coords[0],-S_coords[1],-S_coords[2]] )
    atnums= [atom.GetAtomicNum() for atom in ob.OBMolAtomIter(mol)] 
    coords=np.zeros(np.shape(coords_orig))
    for i in range(0,len(coords_orig)):
        coords[i]=coords_orig[i]+translate
    
    # rotate n1 to z axis, n2 to x axis
    n1_coords=coords[n1_indx-1]
    n1_norm=np.linalg.norm(n1_coords)
    n1_normed=n1_coords/n1_norm
    cosphi=n1_normed[2]/np.sqrt(1-np.power(n1_normed[1],2))
    sinphi=n1_normed[0]/np.sqrt(1-np.power(n1_normed[1],2))

    rot_matrix=np.zeros((3,3))
    rot_matrix[0,0]=cosphi
    rot_matrix[0,2]=-sinphi
    rot_matrix[2,0]=sinphi
    rot_matrix[1,1]=1
    rot_matrix[2,2]=cosphi

    rotcoords=np.transpose(np.matmul(rot_matrix,np.transpose(coords)))
    
    n1_coords2=rotcoords[n1_indx-1]
    n1_norm2=np.linalg.norm(n1_coords2)
    n1_normed2=n1_coords2/n1_norm2
    cosrho=n1_normed2[2]/np.sqrt(1-np.power(n1_normed2[0],2))
    sinrho=n1_normed2[1]/np.sqrt(1-np.power(n1_normed2[0],2))

    rot_matrix2=np.zeros((3,3))
    rot_matrix2[0,0]=1
    rot_matrix2[1,1]=cosrho
    rot_matrix2[1,2]=-sinrho
    rot_matrix2[2,1]=sinrho
    rot_matrix2[2,2]=cosrho
    
    rotcoords2=np.transpose(np.matmul(rot_matrix2,np.transpose(rotcoords)))
    
    n2_coords3=rotcoords2[n2_indx-1]
    n2_norm3=np.linalg.norm(n2_coords3)
    n2_normed3=n2_coords3/n2_norm3
    costheta=n2_normed3[0]/np.sqrt(1-np.power(n2_normed3[2],2))
    sintheta=n2_normed3[1]/np.sqrt(1-np.power(n2_normed3[2],2))

    rot_matrix3=np.zeros((3,3))
    rot_matrix3[0,0]=costheta
    rot_matrix3[0,1]=sintheta
    rot_matrix3[1,0]=-sintheta
    rot_matrix3[1,1]=costheta
    rot_matrix3[2,2]=1
    
    # rotate original coordinates, dipole and polarizability derivatives to reference 
    rot=np.matmul(rot_matrix3,np.matmul(rot_matrix2,rot_matrix)) 
    Drot=np.zeros_like(D)
    Prot=np.zeros_like(P)
    for m in range(np.shape(D)[0]):
        Drot[m]=np.matmul(np.transpose(rot),D[m]) 
        Prot[m]=np.matmul(np.transpose(rot),np.matmul(P[m],rot)) 
    refcoords=np.transpose(np.matmul(rot,np.transpose(coords)))
#     molfilename="reference_{}.xyz".format(filename[:-4])
#     xyz="{} \n\n".format(len(atnums))
#     molfile = open(molfilename, "w+")   
#     molfile.write(xyz)
#     for m in range(0,len(atnums)):
#         molfile.write("  {}    {:.8f} {:.8f} {:.8f} \n".format(atnums[m],refcoords[m,0],refcoords[m,1],refcoords[m,2]))
#     molfile.close()
#     convert_to_mol(molfilename,molfilename[:-3]+"mol")
  
    # get Van der Waals radii
    vdw=np.zeros_like(atnums,dtype=float)
    etab=ob.OBElementTable()
    for i,a in enumerate(ob.OBMolAtomIter(mol)): 
        an=a.GetAtomicNum()
        vdw[i]= etab.GetVdwRad(an)
  
    # do custom rotation with phi and theta
    rotated=single_rotD(refcoords,a=phi,b=theta,c=0)

    # get dimensions of cell encapsulating the molecule
    rotc=np.zeros_like(coords)
    for s,c in enumerate(coords):
        if atnums[s]==79:
            vdw[s]=0     # do not count gold atom for cell size
            rotc[s]=np.array([0,0,0])
            continue 
        rotc[s]=rotated[s]
    cell=get_oriented_box(rotc,vdw) 
        
    # save in .xyz and .mol formats for displaying later
    filename="rotated_{}.xyz".format(filename[:-4])
    xyz="{} \n\n".format(len(atnums))
    molfile = open(filename, "w+")   
    molfile.write(xyz)
    for m in range(0,len(atnums)):
        molfile.write("  {}    {:.8f} {:.8f} {:.8f} \n".format(atnums[m],rotated[m,0],rotated[m,1],rotated[m,2]))
    molfile.close()
    molfilename=filename[:-3]+"mol"
    convert_to_mol(filename,molfilename)

    return Drot,Prot,molfilename,cell
