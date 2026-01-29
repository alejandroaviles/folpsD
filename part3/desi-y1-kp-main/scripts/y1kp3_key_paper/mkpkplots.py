import fitsio
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import sys,os
from desilike.observables import ObservableCovariance
from pypower import PowerSpectrumMultipoles,PowerSpectrumStatistics
from desi_y1_plotting.kp3 import KP3StylePaper
style = KP3StylePaper()
#set LSSCODE, e.g. via export LSSCODE=$HOME on the command line if that is where you cloned the repo 
#sys.path.append(os.environ['LSSCODE']+'/LSS/py')
#from LSS import common_tools as common

#from LSS.tabulated_cosmo import TabulatedDESI
#cosmo = TabulatedDESI()
#dis_dc = cosmo.comoving_radial_distance

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--indir", help="base directory for catalogs", default='/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/')
parser.add_argument("--outdir", help="output directory for the plots", default=None)
parser.add_argument("--mkLRG", help="whether to make the QSO plot", default='y')
parser.add_argument("--mkQSO", help="whether to make the LRG plot", default='y')
parser.add_argument("--mkELG", help="whether to make the ELG plot", default='y')
parser.add_argument("--mkBGS", help="whether to make the BGS plot", default='y')
parser.add_argument("--mkmockcomp", help="whether to make the plot comparing mock version", default='n')
args = parser.parse_args()

lssdir = args.indir
dircov = '/global/cfs/cdirs/desi/users/mrash/RascalC/Y1/'#blinded/v0.6/'
dircovpk = '/global/cfs/cdirs/desi/users/oalves/thecovs/y1_' #unblinded/post/v1.2
outroot = 'pk024_compamtl_'

if args.outdir is None:
    outdir = (args.indir.replace('dvs_ro','global'))+'KPplots/' 
  
else:
    outdir = args.outdir

def points_with_errobars(x,y,errors,color='k',ptype='s',size=10,**kwargs):
    plt.errorbar(x,y,errors,fmt=ptype,color=color,markersize=size,capsize=size/1.5,elinewidth=size/5,capthick=size/5,**kwargs)

def plot_pkpole_points_werr(k,pk,ell=0,mockmean=None,cov=None,clr='k',bf=1.):
    #style = KP3Style()
    ls = style.linestyles[ell] #if 'line' in markers else 'none'
    ps = style.points[ell] #if 'point' in markers else 'none'
    alph = style.alphas[ell]
    biasfac = 1
    if ell == 0:
        biasfac = bf**2.
    if ell == 2:
        biasfac = bf
    plt.plot(k,k*pk,ps,color=clr,alpha=alph)
    if mockmean is not None:
        plt.plot(k,k*mockmean*biasfac,ls,color='k',zorder=1000)#clr,alpha=alph)
    if cov is not None:    
        diag = np.zeros(len(k))
        for i in range(0,len(cov)):
            diag[i] = np.sqrt(cov[i][i])
        diff = pk-biasfac*mockmean
        icov = np.linalg.pinv(cov)
        print(len(diff),len(icov))
        chi2 = np.dot(diff,np.dot(diff,icov))

        plt.errorbar(k,k*pk,k*diag,fmt=ps,color=clr,alpha=alph,label=r'$\ell=$'+str(ell)+r',$\chi^2$/dof='+str(round(chi2,1))+'/'+str(len(diff)))
        print(bf,chi2,len(icov))

def getpkcov_an(tracer,zrange,reg='GCcomb',rec='pre',blinded='blinded',version='v1.2'):
    cov = np.loadtxt(dircovpk+blinded+'/'+rec+'/'+version+'/cov_gaussian_'+tracer+'_'+reg+'_'+zrange+'.txt')
    return cov

def getpk_desipipe_baseline(tp,zrange,reg='GCcomb',rpcut='',version='unblinded',cut='',rec='',bs=5,kmin=0,kmax=0.4):
    from pypower import PowerSpectrumStatistics
    fn = lssdir+version+'/desipipe/baseline_2pt/'+rec+'/pk/pkpoles_'+tp+'_'+reg+'_z'+zrange+'.npy'
    result = PowerSpectrumStatistics.load(fn)
    factor = bs
    rebinned = result[:(result.shape[0] // factor) * factor:factor]
    k,pks = rebinned.select((kmin, kmax))(ell=(0, 2, 4), return_k=True)
    return k,pks

def get_pkave_desipipe_ab_baseline(zr='0.4-0.6',tp='LRG',rec='',bs=5,nmock=25,flavor='altmtl',mockversion='v4_1',reg='GCcomb',thetacut='',mocktype='AbacusSummit',kmin=0,kmax=0.4):
    from pypower import PowerSpectrumStatistics
    if tp[:3] == 'BGS':
        mocktype += 'BGS'

    dirm = '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/'+mocktype+'/desipipe/'+mockversion+'/'+flavor+'/baseline_2pt/mock'
    xil = []
    sepl = []    
    for i in range(0,nmock):
        fnm = dirm + str(i)+'/'+rec+'/pk/pkpoles_'+tp+'_GCcomb_z'+zr+thetacut+'.npy' #loading first to get binning setup
        result = PowerSpectrumStatistics.load(fnm)
        factor = bs
        rebinned = result[:(result.shape[0] // factor) * factor:factor]
        k,pks = rebinned.select((kmin, kmax))(ell=(0, 2, 4), return_k=True)
        xil.append(pks)
        sepl.append(k)
    xi = sum(xil)/nmock
    sep = sum(sepl)/nmock
    return sep,xi

def getpk_EZcov(zr='0.4-0.6',tp='LRG',rec='',bs=5,nmock=1000,reg='GCcomb',thetacut='',EZversion='v1',dataversion='v1.5',kmin=0,kmax=0.4,ells=[0,2,4]):
    covfn = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/'+dataversion+'/desipipe/cov_2pt/ezmock/'+EZversion+'/covariance_power_'+tp+'_'+reg+'_z'+zr+'_default_FKP_lin'+thetacut+'.npy'
    covariance = ObservableCovariance.load(covfn)
    covariance = covariance.select(xlim=(kmin, kmax), projs=ells)
    return covariance.view()

def get_pkave_desipipe_EZ_baseline(zr='0.4-0.6',tp='LRG',rec='',bs=5,nmock=1000,reg='GCcomb',thetacut=''):
    from pypower import PowerSpectrumStatistics
    if tp[:3] == 'BGS':
        mocktype += 'BGS'

    dirm = '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/EZmock/desipipe/v1/ffa/baseline_2pt/mock'
    xil = []
    sepl = []    
    for i in range(1,1+nmock):
        fnm = dirm + str(i)+'/'+rec+'/pk/pkpoles_'+tp+'_GCcomb_z'+zr+thetacut+'.npy' #loading first to get binning setup
        result = PowerSpectrumStatistics.load(fnm)
        factor = bs
        rebinned = result[:(result.shape[0] // factor) * factor:factor]
        k,pks = rebinned(ell=(0, 2, 4), return_k=True)
        xil.append(pks)
        sepl.append(k)
    xi = sum(xil)/nmock
    sep = sum(sepl)/nmock
    return sep,xi

@mpl.rc_context(style._rcparams)
def plot_desipipe_pkdatamock_comp_3ell(tp,zr,rec='',size=5,bf=1,version='unblinded',flavor='altmtl',mockversion='v4_2',kmin=0,kmax=0.4,covfac=1):
    ells = [0,2,4]
    sep,ximock = get_pkave_desipipe_ab_baseline(zr=str(zr[0])+'-'+str(zr[1]),tp=tp,rec=rec,flavor=flavor,mockversion=mockversion,kmin=kmin,kmax=kmax)
    clr = style.colors[tp, (zr[0],zr[1])]
    print(clr)
    s,xiall = getpk_desipipe_baseline(tp,str(zr[0])+'-'+str(zr[1]),version=version,rec=rec,kmin=kmin,kmax=kmax)#nran=nran)
    crec = 'pre'
    if rec != '':
        crec = 'post'
    fullcov = getpk_EZcov(zr=str(zr[0])+'-'+str(zr[1]),tp=tp,rec=rec,dataversion='v1.5/unblinded',kmin=kmin,kmax=kmax)#getpkcov_an(tp,str(zr[0])+'-'+str(zr[1]),rec=crec,blinded='blinded',version='v1.2')
    totk = len(fullcov)//3
    #if klim is None:
    #    kmax = totk
    #else:
    #    kmax = klim
    #print(s[kmin],s[kmax])
    for ell in ells:
        xi = xiall[ell//2]#[5:]
        mockxi = ximock[ell//2]#[5:]
        indmin = ell//2*totk
        indmax = indmin + totk
        cov = fullcov[indmin:indmax,indmin:indmax]*covfac
        plot_pkpole_points_werr(s,xi.real,ell=ell,mockmean=mockxi.real,cov=cov,clr=clr,bf=bf)#,size=size,mec=clr)
    plt.xlabel(r'$k$ ($h$Mpc$^{-1}$)')
    plt.ylabel(r'$kP(k)$  ($h^{-2}$Mpc$^2$)')
               #'+tp+ '  '+str(zr[0])+'<z<'+str(zr[1]))
    plt.legend()

if args.mkBGS == 'y':
    plot_desipipe_pkdatamock_comp_3ell('BGS_BRIGHT-21.5',(0.1,0.4),mockversion='v1',version= 'unblinded',rec='',covfac=1.39)
    plt.title(r'BGS $0.1<z<0.4$')
    plt.savefig(outdir+outroot+'BGS.pdf', bbox_inches='tight')
    plt.clf()

if args.mkLRG == 'y':
    zrl = [(0.4,0.6),(0.6,0.8),(0.8,1.1)]
    facl = [1.15,1.21,1.19]
    c = 0
    for zr,cf in zip(zrl,facl):
        plot_desipipe_pkdatamock_comp_3ell('LRG',zr,mockversion='v4_2',version= 'unblinded',rec='',covfac=cf)
        plt.title(r'LRG $'+str(zr[0])+'<z<'+str(zr[1])+'$')
        plt.savefig(outdir+outroot+'LRG'+str(c)+'.pdf', bbox_inches='tight')
        c += 1
        plt.clf()

if args.mkELG == 'y':
    zrl = [(0.8,1.1),(1.1,1.6)]
    facl = [1.25,1.29]
    c = 0
    for zr,cf in zip(zrl,facl):
        plot_desipipe_pkdatamock_comp_3ell('ELG_LOPnotqso',zr,mockversion='v4_2',version= 'unblinded',rec='',covfac=cf)
        plt.title(r'ELG $'+str(zr[0])+'<z<'+str(zr[1])+'$')
        plt.savefig(outdir+outroot+'ELG'+str(c)+'.pdf', bbox_inches='tight')
        c += 1
        plt.clf()

if args.mkQSO == 'y':
    zr = (0.8,2.1)
    plot_desipipe_pkdatamock_comp_3ell('QSO',zr,mockversion='v4_1fixran',version= 'unblinded',rec='',covfac=1.11)
    plt.title(r'QSO $'+str(zr[0])+'<z<'+str(zr[1])+'$')
    plt.savefig(outdir+outroot+'QSO.pdf', bbox_inches='tight')
    plt.clf()

@mpl.rc_context(style._rcparams)
def plot_desipipe_pk_4mockflav(tp,zr,rec='',size=5,mockversion='v4_2',klim=None,kmin=0):
    ells = [0,2,4]
    tpu = tp
    if tp == 'ELG_LOPnotqso':
        tpu = 'ELG_LOP'

    sep,ximock = get_pkave_desipipe_EZ_baseline(zr=str(zr[0])+'-'+str(zr[1]),tp=tpu,rec=rec)
    for ell in ells:
        ls = style.linestyles[ell]
        if ell == 0:
            plt.plot(sep,sep*ximock[ell//2],ls,label='EZmock (FFA)',color='0.5')
        else:
            plt.plot(sep,sep*ximock[ell//2],ls,color='0.5')

    flavs = ['altmtl','ffa','complete']
    #mvers = ['v4_1fixran','v4_2','v4_1fixran']
    tpu = tp
    clrs = ['b','r','k']
    mversion = mockversion
    for flav,clr in zip(flavs,clrs):
        #if flav != 'ffa':
        #    mversion = mockversion
        #else:
            #mversion = 'v3'
        if tp == 'ELG_LOPnotqso' and flav == 'ffa':
            tpu = 'ELG_LOP'
                
        sep,ximock = get_pkave_desipipe_ab_baseline(zr=str(zr[0])+'-'+str(zr[1]),tp=tpu,rec=rec,flavor=flav,mockversion=mversion)
        for ell in ells:
            ls = style.linestyles[ell]
            if ell == 0:
                lflav = flav
                if flav == 'ffa':
                    lflav = 'FFA'
                plt.plot(sep,sep*ximock[ell//2],ls,label=lflav,color=clr)
                if flav == 'complete':
                    plt.plot(sep,sep*ximock[ell//2],ls,label=r'$\ell=0$',color=clr)
            else:
                if flav == 'complete':
                    plt.plot(sep,sep*ximock[ell//2],ls,label=r'$\ell=$'+str(ell),color=clr)
                else:
                    plt.plot(sep,sep*ximock[ell//2],ls,color=clr)
    plt.legend()
    plt.xlabel(r'$k$ ($h$Mpc$^{-1}$)')
    plt.ylabel(r'$kP(k)$  ($h^{-2}$Mpc$^2$)')

if args.mkmockcomp == 'y':
    plot_desipipe_pk_4mockflav('ELG_LOPnotqso',(1.1,1.6))
    plt.title(r'ELG $1.1<z<1.6$')
    plt.savefig(outdir+'pkELGbin2mocks.pdf', bbox_inches='tight')
    plt.clf()

