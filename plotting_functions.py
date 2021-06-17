# -*- coding: utf-8 -*-
"""
Created on Tue May 11 10:51:46 2021

@author: Zsuzsanna Koczor-Benda, UCL
"""
from __future__ import print_function
import time
import urllib 
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from rdkit import Chem

# read in frequency and intensity data from datafiles
def read_in_average_ints():  
    smiles=[]
    freqs=[]
    Rints=[]
    IRints=[]
    Pints=[]
    fname=[]
    filename="frequencies.txt"
    Rintfilename="Raman_Stokes_intensities.txt"
    IRintfilename="IR_intensities.txt"
    Pintfilename="Conversion_anti-Stokes_intensities.txt"

    with open(filename) as inpfile:
        line=inpfile.readline()
        line=inpfile.readline()
        mol=0
        while line:
                spl=line.split()
                fname.append(spl[0])
                smiles.append(spl[1]) 
                freqs.append([float(vib) for vib in spl[2:]])
                mol+=1
                line=inpfile.readline()

    with open(Rintfilename) as inpfile:
        line=inpfile.readline()
        line=inpfile.readline()
        while line: 
                Rints.append([float(vib) for vib in line.split()[2:]])
                line=inpfile.readline()

    with open(IRintfilename) as inpfile:
        line=inpfile.readline()
        line=inpfile.readline()
        while line: 
                IRints.append([float(vib) for vib in line.split()[2:]])
                line=inpfile.readline()

    with open(Pintfilename) as inpfile:
        line=inpfile.readline()
        line=inpfile.readline()
        while line: 
                Pints.append([float(vib) for vib in line.split()[2:]])
                line=inpfile.readline()
             
    freqs_np=np.array(freqs)
    Rints_np=np.array(Rints)
    IRints_np=np.array(IRints)
    Pints_np=np.array(Pints)
            
    print("{} molecules read, maximum number of normal modes: {}".format( np.shape(freqs_np)[0],np.shape(freqs_np)[1]))
    return freqs_np, Rints_np, IRints_np, Pints_np,smiles,fname


# calculate target quantities P,A,R 
# means and stds from randomly selected thiols
def get_targets(R_ints,IR_ints,prod_ints,
            Rmean=-28.91883480924085,Rstd=0.2625275063027046,  
            IRmean=2.290063042859388,IRstd=0.22266072753602675,
            Pmean=-28.367672289689146, Pstd=0.3809242121851108): 
    
    sConv=np.sum(prod_ints,axis=1)
    sIR=np.sum(IR_ints,axis=1)
    sRi=np.sum(R_ints,axis=1)
    
    P_=np.log10(sConv, out=np.full_like(sConv,np.NINF),where=sConv!=0)
    A_=np.log10(sIR, out=np.full_like(sIR,np.NINF),where=sIR!=0)
    R_=np.log10(sRi, out=np.full_like(sRi,np.NINF),where=sRi!=0)
    
    P=(P_-Pmean)/Pstd
    A=(A_-IRmean)/IRstd
    R=(R_-Rmean)/Rstd
    
    return P,A,R


# functions for calculating broadened spectrum for all input files
def lorentz0(res,gamma):
    rmax=gamma*30
    npoints=int(round(rmax/res))
    g=np.zeros((2*npoints))
    xs=[k*res for k in range(npoints)]
    for i,x in enumerate(xs):
        g[i]= 1/(math.pow((x),2) + math.pow(1/2*gamma, 2))
    return g

def displ_lorentz0(l0,x0,y,numpoints,xmin,res):
    disp=np.zeros((numpoints))
    lenl=len(l0)
    p0=int(round((x0-xmin)/res))
    if p0>numpoints+lenl or p0<-lenl:
        return disp
    if p0>numpoints:
        for p in range(numpoints+lenl-p0): 
            if numpoints-p-1 <0:
                break
            disp[numpoints-p-1]=y*l0[p0-numpoints+p]
    elif p0<0:
        for p in range(lenl+p0):  
            if p==numpoints:
                break
            disp[p]=y*l0[p-p0]
    else:
        for p in range(min(numpoints-p0,lenl)):  
            disp[p0+p]=y*l0[p]
        for p in range(min(p0,lenl)):            
            disp[p0-p-1]=y*l0[p]

    return disp
    
def calc_broadened_spectrum(freqs,rawints,xmin, xmax, res, gamma):
    l0=lorentz0(res,gamma)
    numfiles=len(freqs)
    numpoints=int(round((xmax-xmin)/res))
    wn=np.zeros(numpoints)
    sp_ints=np.zeros((numfiles,numpoints)) 
    x=xmin
    for i in range(0,numpoints):
        wn[i]=x
        x+=res      
    for f in range(0,numfiles):
        spectrum=np.zeros((numpoints))
        for state in range(0,len(freqs[f])):
            spectrum+=displ_lorentz0(l0,freqs[f,state],rawints[f,state],numpoints,xmin,res)
        sp_ints[f]=spectrum/math.pi * 1/2 * gamma
        
    return wn, sp_ints



def calc_broadened_conv_spectrum(freqs,rawints1,xmin, xmax, res, gamma1,gamma2):
    l0=lorentz0(res,gamma1)
    l02=lorentz0(res,gamma2)
    numfiles=len(freqs)
    numpoints=int(round((xmax-xmin)/res))
    wn=np.zeros(numpoints)
    sp_ints=np.zeros((numfiles,numpoints)) 
    x=xmin
    for i in range(0,numpoints):
        wn[i]=x
        x+=res      
    for f in range(0,numfiles):
        spectrum=np.zeros((numpoints))
        for state in range(0,len(freqs[f])):
            ones=1 #np.full((np.shape(rawints1[f,state])),1)
            
            spectrum+=rawints1[f,state]*displ_lorentz0(l0,freqs[f,state],ones,numpoints,xmin,res)*displ_lorentz0(l02,freqs[f,state],ones,numpoints,xmin,res)
        sp_ints[f]=spectrum/(math.pi**2) * 1/4 * gamma1*gamma2
    print(ones)    
    return wn, sp_ints

# create broadened spectrum and get target properties
def create_average_spec(xmin=30,xmax=1000,res=0.5,gammaIR=5,gammaR=5,sclf=0.98):
    freqs, Rints, IRints, Pints, smiles, fname=read_in_average_ints()
    
    # scale frequencies
    freqs=sclf*freqs
    fmin=xmin-100
    fmax=xmax+100 
    
    maxdof=np.shape(freqs)[1]
    R_fre=np.zeros((len(freqs),maxdof))
    IR_fre=np.zeros((len(freqs),maxdof))
    P_fre=np.zeros((len(freqs),maxdof))
    for i in range(0,len(freqs)):
        for l in range(0,maxdof):
            if(fmin<=freqs[i,l]<fmax):
                R_fre[i,l]=Rints[i,l]
                IR_fre[i,l]=IRints[i,l]
                P_fre[i,l]=Pints[i,l]
                
    prod_ints=np.zeros_like(IR_fre)
    R_ints=np.zeros_like(IR_fre)
    IR_ints=np.zeros_like(IR_fre)
    for mm,frmm in enumerate(freqs):
        for f,fr in enumerate(frmm):
            if fr<xmin or fr>xmax:
                continue
            prod_ints[mm,f]=P_fre[mm,f] 
            R_ints[mm,f]=R_fre[mm,f]
            IR_ints[mm,f]=IR_fre[mm,f]
            
    P,A,R=get_targets(R_ints,IR_ints,prod_ints)
    print("Target properties calculated")
    # average all orientations
    print("Calculating broadened spectra...")
    t1 = time.time()
    wn, IR_spec = calc_broadened_spectrum(freqs,IR_fre,xmin, xmax, res, gammaIR)
    wn, R_spec = calc_broadened_spectrum(freqs,R_fre,xmin, xmax, res, gammaR)
    wn, conv_spec=calc_broadened_conv_spectrum(freqs,P_fre,xmin, xmax, res, gammaIR,gammaR)
    t2 = time.time()
    print("Done \nTime for broadened spectra = {:.1f} s".format(t2-t1))
    
    return wn,R_spec,IR_spec,conv_spec,freqs,prod_ints,R_ints,IR_ints,smiles,fname,P,A,R

def create_average_spec_single(fr,  ir_av, r_av, conv_av,xmin=30,xmax=1000,res=0.5,gammaIR=5,gammaR=5,sclf=0.98):
    freqs=np.reshape(fr,(1,-1)) 
    Rints=np.reshape(r_av,(1,-1)) 
    IRints=np.reshape(ir_av,(1,-1)) 
    Pints=np.reshape(conv_av,(1,-1))  
    
    # scale frequencies
    freqs=sclf*freqs
    fmin=xmin-200
    fmax=xmax+200 
    
    maxdof=np.shape(freqs)[1]
    R_fre=np.zeros((len(freqs),maxdof))
    IR_fre=np.zeros((len(freqs),maxdof))
    P_fre=np.zeros((len(freqs),maxdof))
    for i in range(0,len(freqs)):
        for l in range(0,maxdof):
            if(fmin<=freqs[i,l]<fmax):
                R_fre[i,l]=Rints[i,l]
                IR_fre[i,l]=IRints[i,l]
                P_fre[i,l]=Pints[i,l]
                
    prod_ints=np.zeros_like(IR_fre)
    R_ints=np.zeros_like(IR_fre)
    IR_ints=np.zeros_like(IR_fre)
    for mm,frmm in enumerate(freqs):
        for f,fr in enumerate(frmm):
            if fr<xmin or fr>xmax:
                continue
            prod_ints[mm,f]=P_fre[mm,f] 
            R_ints[mm,f]=R_fre[mm,f]
            IR_ints[mm,f]=IR_fre[mm,f]
            
    P,A,R=get_targets(R_ints,IR_ints,prod_ints)
#    print("Target properties calculated")
    # average all orientations
#    print("Calculating broadened spectra...")
#    t1 = time.time()
    wn, IR_spec = calc_broadened_spectrum(freqs,IR_fre,xmin, xmax, res, gammaIR)
    wn, R_spec = calc_broadened_spectrum(freqs,R_fre,xmin, xmax, res, gammaR)
    wn, conv_spec=calc_broadened_conv_spectrum(freqs,P_fre,xmin, xmax, res, gammaIR,gammaR)
#    t2 = time.time()
#    print("Done \nTime for broadened spectra = {:.1f} s".format(t2-t1))
    
    return wn,R_spec[0],IR_spec[0],conv_spec[0],freqs[0],prod_ints[0],R_ints[0],IR_ints[0],P,A,R
    


# build molecules from smiles using RDKit
def build_molecules(smiles):
    mols=[]
    errs=[]
    for i in range(0,len(smiles)):
        if Chem.MolFromSmiles(smiles[i]) is None: 
            errs.append(i)
            mols.append(None)
            continue
        else:
            mols.append(Chem.MolFromSmiles(smiles[i])) 
    return mols, errs

# get intensities in custom frequency range
def get_intens_range(freqs,intens,fmin,fmax):
    intens_range=np.zeros_like(freqs)
    for f in range(0,len(freqs)):
        for mode in range(0,len(freqs[f])):
            if (freqs[f,mode]>= fmin) and (freqs[f,mode]<=fmax):
                intens_range[f,mode]=intens[f,mode]
    return intens_range

# calculate target quantities P, A, or R  
# means and stds from 1.3k randomly selected thiols
def get_target(intens,target_type="P"):    
    if target_type=="P": # conversion
        Tmean=-28.367672289689146 
        Tstd=0.3809242121851108
    elif target_type=="R": # Raman Stokes
        Tmean=-28.91883480924085
        Tstd=0.2625275063027046  
    else: # IR absorption
        Tmean=2.290063042859388
        Tstd=0.22266072753602675
    sint=np.sum(intens,axis=1)
    target_=np.log10(sint, out=np.full_like(sint,np.NINF),where=sint!=0)    
    target=(target_-Tmean)/Tstd
    return target

# formatter function takes tick label and tick position
def func(x, pos):  
   s = '{:.1e}'.format(int(x))
   return s

y_format = tkr.FuncFormatter(func)  # make formatter

# plot IR, Raman, THz Conversion spectra
def plot_spectra(mm,s_wn_av,R_ints_av,IR_ints_av,freqs_np,prod_ints,fmin,fmax,res,mcode,savespec=False):
    plt.rcParams.update({'font.size': 16})
    fig=plt.figure()
    fig.set_size_inches(8, 5)
    ax2=fig.add_subplot(311)
    ax3=fig.add_subplot(312,sharex=ax2)
    ax1=fig.add_subplot(313,sharex=ax2)
    ax3.plot(s_wn_av,R_ints_av[mm,:],alpha=1,label='Raman')
    ax2.plot(s_wn_av,IR_ints_av[mm,:],alpha=1,color='r',label='IR')  
    markerline, stemline, baseline, =ax1.stem(freqs_np[mm,:],prod_ints[mm,:],
                                                 'k',markerfmt='ok',basefmt='k',
                                                 use_line_collection=True,bottom=0,label='Conv.')
    plt.setp(stemline, linewidth = 1.25)
    plt.setp(markerline, markersize = 6)    
    plt.xlim(fmin,fmax)
    pmin=int(fmin/res)
    pmax=int(fmax/res)
    maxpr=np.max(prod_ints[mm])
    maxI=np.max(IR_ints_av[mm,pmin:pmax])
    maxR=np.max(R_ints_av[mm,pmin:pmax])
    ax1.set_ylim(-maxpr/100,1.2*maxpr)
    ax2.set_ylim(-maxI/100,1.2*maxI)
    ax3.set_ylim(-maxR/100,1.2*maxR)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0),useMathText=True)
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0),useMathText=True)
    ax3.ticklabel_format(axis="y", style="sci", scilimits=(0,0),useMathText=True)
    ax2.set_title('Molecule {}'.format(mcode.split("-",1)[1]))
    plt.xlabel('Wavenumber (cm$^{-1}$)')
    ax2.set_ylabel('IR',color='r')
    ax3.set_ylabel('R',color='C0')
    ax1.set_ylabel('Conv.')
    ax1.yaxis.set_label_coords(-0.07,0.5)
    ax2.yaxis.set_label_coords(-0.07,0.5)
    ax3.yaxis.set_label_coords(-0.07,0.5)
    fig.tight_layout(pad=1.8, w_pad=0.01, h_pad=0.01)
    if savespec:
        print("Save spectrum figure")
        plt.savefig("./figures/{}_spectrum.png".format(mcode),dpi=100)
    plt.show()
    
# write html summary of suitability for THz detection
def write_html(bestmols,smiles,P,A,R,fnam,fmin,fmax,target,tmin,tmax,target_type,sigma=0):
    htmlheader = """
                           <table border=1>
                                 <tr>
                                 <th> Molecule </th>
                                 <th> Supplier Code </th>  
                                 <th> {} <br>{:.1f}-{:.1f} cm-1</th>
                                 <th> P  <br>{:.1f}-{:.1f} cm-1</th>
                                 <th> A  <br>{:.1f}-{:.1f} cm-1</th>
                                 <th> R  <br>{:.1f}-{:.1f} cm-1</th>
                                 <th> Spectrum </th>
                                 </tr>     
                    """.format(target_type,tmin,tmax,fmin,fmax,fmin,fmax,fmin,fmax)
    htmlfilename = "summary_{}_{}.html".format(int(fmin), int(fmax))
    with open(htmlfilename, "w+") as htmlfile:
        htmlfile.write(htmlheader)
        for nm,mm in enumerate(bestmols):
                
                # mcode is the CAS number for Sigma-Aldrich molecules
                if sigma:
                    mcode="Sigma-"+fnam[mm]
                else:
                    mcode=fnam[mm].split("_",1)[0]
                mcode2=mcode.split("-",1)[1]
                # eMolecules link
                urlsmiles=urllib.request.quote(smiles[mm])
                link="https://orderbb.emolecules.com/search/#?query={}&system-type=BB&p=1".format(urlsmiles)
                # MolPort link
                if "-" in mcode2:
                    link="https://www.molport.com/shop/moleculelink/about-this-molecule/"+"".join(mcode2.split("-"))
                # Sigma-Aldrich link
                if sigma:
                    link="https://www.sigmaaldrich.com/catalog/search?term={}&interface=CAS%20No.&N=0&mode=match%20partialmax&lang=en&region=GB&focus=product".format(mcode2)
 
              #  print(link)

                molfile=".\\figures\\{}_mol.png".format(mcode)
                spectrumfile=".\\figures\\{}_spectrum.png".format(mcode) 

                MLhtml = """
                     <tr>
                       <th><img src={} alt={} height="140"></th>
                       <th><a href={}>{}</a></th>
                       <th>{:.3f}</th>
                       <th>{:.3f}</th>  
                       <th>{:.3f}</th>
                       <th>{:.3f}</th>
                       <th><img src={} alt="spectrum" height="140"></th>
                     </tr>     
                """.format(molfile, smiles[mm],link,mcode2, target[mm],P[mm],A[mm],R[mm],
                           spectrumfile)
                htmlfile.write(MLhtml) 
        htmlfoot="               </table> "
        htmlfile.write(htmlfoot)
        print("HTML file {} written".format(htmlfilename))
        print("Please open in browser")
