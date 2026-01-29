#sys.path.append(os.getenv('HOME')+'/Y1KPplots/py/')
#Need to get pack in path, via, e.g., PYTHONPATH=$PYTHONPATH:$HOME/Y1KPplots/py/

from desi_y1_plotting.kp3 import KP3Style 
kp3 = KP3Style()
import numpy as np
from matplotlib import pyplot as plt
import sys
import fitsio
import os

lssdir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/'
version = 'v0.6/blinded'
dircov = '/global/cfs/cdirs/desi/users/mrash/RascalC/Y1/blinded/v0.6'

def getmeanmockxi(tp='LRG_ffa',zr='0.4_0.6',nmock=25,wt='default_FKP',mockmin=0):
    xil = []
    for i in range(mockmin,mockmin+nmock):
        dirxi = '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/mock'+str(i)+'/'
        xi = np.loadtxt(dirxi+'/xi/smu/xipoles_'+tp+'_GCcomb_'+zr+'_'+wt+'_lin4_njack0_nran4_split20.txt').transpose()
        xil.append(xi)
    xi = sum(xil)/nmock
    err0 = np.zeros(len(xi))
    err2 = np.zeros(len(xi))
    err4 = np.zeros(len(xi))
    for i in range(mockmin,mockmin+nmock):
        dirxi = '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/mock'+str(i)+'/'
        xii = np.loadtxt(dirxi+'/xi/smu/xipoles_'+tp+'_GCcomb_'+zr+'_'+wt+'_lin4_njack0_nran4_split20.txt').transpose()
        err0 += (xi[2]-xii[2])**2
        err2 += (xi[3]-xii[3])**2
        err4 += (xi[4]-xii[4])**2
    err0 = np.sqrt(err0/nmock)
    err2 = np.sqrt(err2/nmock)
    err4 = np.sqrt(err4/nmock)
    return xi,err0,err2,err4
    
def getmeanmockxi_diff(tp='LRG_ffa',zr='0.4_0.6',nmock=25,wt='default_FKP',wt2='default_FKP_addRF',mockmin=0):
    xil = []
    for i in range(mockmin,mockmin+nmock):
        dirxi = '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/mock'+str(i)+'/'
        xi = np.loadtxt(dirxi+'/xi/smu/xipoles_'+tp+'_GCcomb_'+zr+'_'+wt+'_lin4_njack0_nran4_split20.txt').transpose()
        xi2 = np.loadtxt(dirxi+'/xi/smu/xipoles_'+tp+'_GCcomb_'+zr+'_'+wt2+'_lin4_njack0_nran4_split20.txt').transpose()
        xil.append(xi-xi2)
    xi = sum(xil)/nmock
    err0 = np.zeros(len(xi[0]))
    err2 = np.zeros(len(xi[0]))
    err4 = np.zeros(len(xi[0]))
    for i in range(mockmin,mockmin+nmock):
        dirxi = '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/mock'+str(i)+'/'
        xii = np.loadtxt(dirxi+'/xi/smu/xipoles_'+tp+'_GCcomb_'+zr+'_'+wt+'_lin4_njack0_nran4_split20.txt').transpose()
        xi2 = np.loadtxt(dirxi+'/xi/smu/xipoles_'+tp+'_GCcomb_'+zr+'_'+wt2+'_lin4_njack0_nran4_split20.txt').transpose()
        err0 += (xi[2]-(xii[2]-xi2[2]))**2
        err2 += (xi[3]-(xii[3]-xi2[3]))**2
        err4 += (xi[4]-(xii[4]-xi2[4]))**2
    err0 = np.sqrt(err0/nmock)
    err2 = np.sqrt(err2/nmock)
    err4 = np.sqrt(err4/nmock)
   
    return xi,err0,err2,err4

    
def comp_datamockimsysdiff(tp='LRG',zmin=0.4,zmax=0.6,wts=['noweight','RF']):
    plt.clf()
    mockwts = []
    datawts = []
    zr=str(zmin)+'_'+str(zmax)
    for wt in wts:
        if wt == 'noweight':
            mockwts.append('default_FKP')
            datawts.append('default_removeSN_FKP')
        if wt == 'RF':
            mockwts.append('default_FKP_addRF')
            datawts.append('default_swapinRF_FKP')
        if wt == 'SN':
            mockwts.append('default_FKP_addSN')
            datawts.append('default_FKP')
    mockmean = getmeanmockxi_diff(tp+'_ffa',zr=zr,wt=mockwts[0],wt2=mockwts[1])
    xi1 = np.loadtxt(lssdir+version+'/xi/smu/xipoles_'+tp+'_GCcomb_'+zr+'_'+datawts[0]+'_lin4_njack0_nran4_split20.txt').transpose()
    xi2 = np.loadtxt(lssdir+version+'/xi/smu/xipoles_'+tp+'_GCcomb_'+zr+'_'+datawts[1]+'_lin4_njack0_nran4_split20.txt').transpose()
    xidiff = xi1-xi2
    outf = lssdir+version+'/xi/smu//mockcomp/xipoles_'+tp+'_GCcomb_'+zr+wts[0]+wts[1]+'.png'
    clr = kp3.colors[(tp,(zmin,zmax))]

    plt.errorbar(xi1[0],xi1[0]*xidiff[2],xi1[0]*mockmean[1],fmt=kp3.points[0],color=clr)
    plt.plot(xi1[0],xi1[0]*mockmean[0][2],kp3.linestyles[0],color=clr)
    plt.errorbar(xi1[0],xi1[0]*xidiff[3],xi1[0]*mockmean[2],fmt=kp3.points[2],alpha=kp3.alphas[2],color=clr)
    plt.plot(xi1[0],xi1[0]*mockmean[0][3],kp3.linestyles[2],alpha=kp3.alphas[2],color=clr)
    plt.xlabel('s (Mpc/h)')
    plt.ylabel(r's ($\xi$'+wts[0]+r'-$\xi$'+wts[1]+')')
    plt.title(tp+' '+zr)
    plt.savefig(outf)
    plt.show()
    plt.clf()
    return True
 
def comp_dataimsysdiff(tp='LRG',zmin=0.4,zmax=0.6,wts=['SN','RF']):
    plt.clf()
    mockwts = []
    datawts = []
    zr=str(zmin)+'_'+str(zmax)
    for wt in wts:
        if wt == 'noweight':
            mockwts.append('default_FKP')
            datawts.append('default_removeSN_FKP')
        if wt == 'RF':
            mockwts.append('default_FKP_addRF')
            datawts.append('default_swapinRF_FKP')
        if wt == 'SN':
            mockwts.append('default_FKP_addSN')
            datawts.append('default_FKP')
    mockmean = getmeanmockxi_diff(tp+'_ffa',zr=zr,wt=mockwts[0],wt2=mockwts[1])
    xi1 = np.loadtxt(lssdir+version+'/xi/smu/xipoles_'+tp+'_GCcomb_'+zr+'_'+datawts[0]+'_lin4_njack0_nran4_split20.txt').transpose()
    xi2 = np.loadtxt(lssdir+version+'/xi/smu/xipoles_'+tp+'_GCcomb_'+zr+'_'+datawts[1]+'_lin4_njack0_nran4_split20.txt').transpose()
    xidiff = xi1-xi2
    outf = lssdir+version+'/xi/smu//mockcomp/xipoles_'+tp+'_GCcomb_'+zr+wts[0]+wts[1]+'_diff.png'
    clr = kp3.colors[(tp,(zmin,zmax))]

    plt.errorbar(xi1[0],xi1[0]*xidiff[2],xi1[0]*mockmean[1],fmt=kp3.points[0],color=clr)
    #plt.plot(xi1[0],xi1[0]*mockmean[0][2],kp3.linestyles[0],color=clr)
    plt.errorbar(xi1[0],xi1[0]*xidiff[3],xi1[0]*mockmean[2],fmt=kp3.points[2],alpha=kp3.alphas[2],color=clr)
    #plt.plot(xi1[0],xi1[0]*mockmean[0][3],kp3.linestyles[2],alpha=kp3.alphas[2],color=clr)
    plt.xlabel('s (Mpc/h)')
    plt.ylabel(r's ($\xi$'+wts[0]+r'-$\xi$'+wts[1]+')')
    plt.title(tp+' '+zr)
    plt.savefig(outf)
    plt.show()
    plt.clf()
    return True

def comp_dataimsysdiff_calib(tp='LRG',zmin=0.4,zmax=0.6,wts=['SN','RF']):
    plt.clf()
    mockwts = []
    datawts = []
    zr=str(zmin)+'_'+str(zmax)
    for wt in wts:
        if wt == 'noweight':
            mockwts.append('default_FKP')
            datawts.append('default_removeSN_FKP')
        if wt == 'RF':
            mockwts.append('default_FKP_addRF')
            datawts.append('default_swapinRF_FKP')
        if wt == 'SN':
            mockwts.append('default_FKP_addSN')
            datawts.append('default_FKP')
    mockmean = getmeanmockxi_diff(tp+'_ffa',zr=zr,wt=mockwts[0],wt2=mockwts[1])
    xi1 = np.loadtxt(lssdir+version+'/xi/smu/xipoles_'+tp+'_GCcomb_'+zr+'_'+datawts[0]+'_lin4_njack0_nran4_split20.txt').transpose()
    xi2 = np.loadtxt(lssdir+version+'/xi/smu/xipoles_'+tp+'_GCcomb_'+zr+'_'+datawts[1]+'_lin4_njack0_nran4_split20.txt').transpose()
    xidiff = xi1-xi2
    outf = lssdir+version+'/xi/smu//mockcomp/xipoles_'+tp+'_GCcomb_'+zr+wts[0]+wts[1]+'_diffcalib.png'
    clr = kp3.colors[(tp,(zmin,zmax))]

    plt.errorbar(xi1[0],xi1[0]*(xidiff[2]-mockmean[0][2]),xi1[0]*mockmean[1],fmt=kp3.points[0],color=clr)
    #plt.plot(xi1[0],xi1[0]*mockmean[0][2],kp3.linestyles[0],color=clr)
    plt.errorbar(xi1[0],xi1[0]*(xidiff[3]-mockmean[0][3]),xi1[0]*mockmean[2],fmt=kp3.points[2],alpha=kp3.alphas[2],color=clr)
    #plt.plot(xi1[0],xi1[0]*mockmean[0][3],kp3.linestyles[2],alpha=kp3.alphas[2],color=clr)
    plt.xlabel('s (Mpc/h)')
    plt.ylabel(r's ($\xi$'+wts[0]+r'-$\xi$'+wts[1]+')')
    plt.title(tp+' '+zr)
    plt.grid()
    plt.savefig(outf)
    plt.show()
    plt.clf()
    return True

def comp_dataimsysdiff_calib_dsig(tp='LRG',zmin=0.4,zmax=0.6,wts=['SN','RF']):
    zr=str(zmin)+'_'+str(zmax)
    covf = dircov+'/xi024_'+tp+'_GCcomb_'+zr+'_default_FKP_lin4_s20-200_cov_RascalC_rescaled.txt'
    cov = np.loadtxt(covf)
    diag0 = np.zeros(45)
    diag2 = np.zeros(45)
    for i in range(0,45):
        diag0[i] = np.sqrt(cov[i][i])
        diag2[i] = np.sqrt(cov[i+45][i+45])
    cov0 = cov[:45,:45]
    cov2 = cov[45:90,45:90]
    plt.clf()
    mockwts = []
    datawts = []
    
    for wt in wts:
        if wt == 'noweight':
            mockwts.append('default_FKP')
            datawts.append('default_removeSN_FKP')
        if wt == 'RF':
            mockwts.append('default_FKP_addRF')
            datawts.append('default_swapinRF_FKP')
        if wt == 'SN':
            mockwts.append('default_FKP_addSN')
            datawts.append('default_FKP')
    mockmean = getmeanmockxi_diff(tp+'_ffa',zr=zr,wt=mockwts[0],wt2=mockwts[1])
    xi1 = np.loadtxt(lssdir+version+'/xi/smu/xipoles_'+tp+'_GCcomb_'+zr+'_'+datawts[0]+'_lin4_njack0_nran4_split20.txt').transpose()
    xi2 = np.loadtxt(lssdir+version+'/xi/smu/xipoles_'+tp+'_GCcomb_'+zr+'_'+datawts[1]+'_lin4_njack0_nran4_split20.txt').transpose()
    xidiff = xi1-xi2
    outf = lssdir+version+'/xi/smu//mockcomp/xipoles_'+tp+'_GCcomb_'+zr+wts[0]+wts[1]+'_diffcalibdsig.png'
    clr = kp3.colors[(tp,(zmin,zmax))]
    icov0 = np.linalg.inv(cov0)
    diff0 = xidiff[2][5:]-mockmean[0][2][5:]
    chi2 = np.dot(diff0,np.dot(diff0,icov0))
    print(chi2)
    plt.plot(xi1[0][5:],(diff0)/diag0,kp3.linestyles[0],color=clr,label=r'$\xi_0$, $\chi^2=$'+str(round(chi2,3)))
    icov2 = np.linalg.inv(cov2)
    diff2 = xidiff[3][5:]-mockmean[0][3][5:]
    chi2 = np.dot(diff2,np.dot(diff2,icov2))
    print(chi2) 
    #plt.plot(xi1[0],xi1[0]*mockmean[0][2],kp3.linestyles[0],color=clr)
    plt.plot(xi1[0][5:],(diff2)/diag2,kp3.linestyles[2],alpha=kp3.alphas[2],color=clr,label=r'$\xi_2$, $\chi^2=$'+str(round(chi2,3)))
    plt.legend()
    #plt.plot(xi1[0],xi1[0]*mockmean[0][3],kp3.linestyles[2],alpha=kp3.alphas[2],color=clr)
    plt.xlabel('s (Mpc/h)')
    plt.ylabel(r' ($\xi$'+wts[0]+r'-$\xi$'+wts[1]+')/$\sigma$')
    plt.title(tp+' '+zr)
    plt.grid()
    plt.savefig(outf)
    plt.show()
    plt.clf()
    return True

def comp_dataimsysdiff_dsig(tp='LRG',zmin=0.4,zmax=0.6,wts=['SN','RF']):
    zr=str(zmin)+'_'+str(zmax)
    covf = dircov+'/xi024_'+tp+'_GCcomb_'+zr+'_default_FKP_lin4_s20-200_cov_RascalC_rescaled.txt'
    cov = np.loadtxt(covf)
    diag0 = np.zeros(45)
    diag2 = np.zeros(45)
    for i in range(0,45):
        diag0[i] = np.sqrt(cov[i][i])
        diag2[i] = np.sqrt(cov[i+45][i+45])
    cov0 = cov[:45,:45]
    cov2 = cov[45:90,45:90]
    plt.clf()
    mockwts = []
    datawts = []
    
    for wt in wts:
        if wt == 'noweight':
            mockwts.append('default_FKP')
            datawts.append('default_removeSN_FKP')
        if wt == 'RF':
            mockwts.append('default_FKP_addRF')
            datawts.append('default_swapinRF_FKP')
        if wt == 'SN':
            mockwts.append('default_FKP_addSN')
            datawts.append('default_FKP')
    mockmean = getmeanmockxi_diff(tp+'_ffa',zr=zr,wt=mockwts[0],wt2=mockwts[1])
    xi1 = np.loadtxt(lssdir+version+'/xi/smu/xipoles_'+tp+'_GCcomb_'+zr+'_'+datawts[0]+'_lin4_njack0_nran4_split20.txt').transpose()
    xi2 = np.loadtxt(lssdir+version+'/xi/smu/xipoles_'+tp+'_GCcomb_'+zr+'_'+datawts[1]+'_lin4_njack0_nran4_split20.txt').transpose()
    xidiff = xi1-xi2
    outf = lssdir+version+'/xi/smu//mockcomp/xipoles_'+tp+'_GCcomb_'+zr+wts[0]+wts[1]+'_diffdsig.png'
    clr = kp3.colors[(tp,(zmin,zmax))]
    icov0 = np.linalg.inv(cov0)
    diff0 = xidiff[2][5:]#-mockmean[0][2][5:]
    chi2 = np.dot(diff0,np.dot(diff0,icov0))
    print(chi2)
    plt.plot(xi1[0][5:],(diff0)/diag0,kp3.linestyles[0],color=clr,label=r'$\xi_0$, $\chi^2=$'+str(round(chi2,3)))
    icov2 = np.linalg.inv(cov2)
    diff2 = xidiff[3][5:]#-mockmean[0][3][5:]
    chi2 = np.dot(diff2,np.dot(diff2,icov2))
    print(chi2) 
    #plt.plot(xi1[0],xi1[0]*mockmean[0][2],kp3.linestyles[0],color=clr)
    plt.plot(xi1[0][5:],(diff2)/diag2,kp3.linestyles[2],alpha=kp3.alphas[2],color=clr,label=r'$\xi_2$, $\chi^2=$'+str(round(chi2,3)))
    plt.legend()
    #plt.plot(xi1[0],xi1[0]*mockmean[0][3],kp3.linestyles[2],alpha=kp3.alphas[2],color=clr)
    plt.xlabel('s (Mpc/h)')
    plt.ylabel(r' ($\xi$'+wts[0]+r'-$\xi$'+wts[1]+')/$\sigma$')
    plt.title(tp+' '+zr)
    plt.grid()
    plt.savefig(outf)
    plt.show()
    plt.clf()
    return True



