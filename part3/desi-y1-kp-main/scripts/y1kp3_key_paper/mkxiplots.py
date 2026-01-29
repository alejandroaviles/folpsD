import fitsio
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import sys,os
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
parser.add_argument("--mkLRG", help="whether to make the QSO veto plot", default='y')
parser.add_argument("--mkQSO", help="whether to make the LRG veto plot", default='y')
parser.add_argument("--mkELG", help="whether to make the ELG veto plot", default='y')
parser.add_argument("--mkBGS", help="whether to make the BGS veto plot", default='y')
args = parser.parse_args()

lssdir = args.indir
dircov = '/global/cfs/cdirs/desi/users/mrash/RascalC/Y1/'
dircovpk = '/global/cfs/cdirs/desi/users/oalves/thecovs/y1_' 
outroot = 'xi024_compamtl_'

if args.outdir is None:
    outdir = (args.indir.replace('dvs_ro','global'))+'KPplots/' 
else:
    outdir = args.outdir

def points_with_errobars(x,y,errors,color='k',ptype='s',size=10,**kwargs):
    plt.errorbar(x,y,errors,fmt=ptype,color=color,markersize=size,capsize=size/1.5,elinewidth=size/5,capthick=size/5,**kwargs)

def plot_xipole_points_werr(s,xi,ell=0,mockmean=None,cov=None,clr='k',bf=1.,lab='',size=10,mec='k'):
    #style = KP3Style()
    ls = style.linestyles[ell] #if 'line' in markers else 'none'
    ps = style.points[ell] #if 'point' in markers else 'none'
    alph = style.alphas[ell]
    biasfac = 1
    if ell == 0:
        biasfac = bf**2.
    if ell == 2:
        biasfac = bf
    
    
    if mockmean is not None:
        plt.plot(s,s**2.*mockmean*biasfac,ls,color='k')#clr,alpha=alph)
    if cov is not None:    
        diag = np.zeros(len(s))
        for i in range(0,len(cov)):
            diag[i] = np.sqrt(cov[i][i])
        if mockmean is not None:
            diff = xi-biasfac*mockmean
            icov = np.linalg.inv(cov)
            chi2 = np.dot(diff,np.dot(diff,icov))
            print(bf,chi2)
            lab = lab + r'$\ell=$'+str(ell)#+r',$\chi^2$/dof='+str(round(chi2,1))+'/'+str(len(diff))
        #plt.errorbar(s,s**2.*xi,s**2*diag,fmt=ps,color=clr,alpha=alph,label=lab)
        points_with_errobars(s,s**2.*xi,s**2*diag,ptype=ps,color=clr,size=size,mec=mec,alpha=alph,label=lab)
    else:
        plt.plot(s,s**2.*xi,ps,color=clr,alpha=alph)

def getcov_rascalC(tracer,zrange,nran=4,smin=20,bs=4,smax=200,reg='GCcomb',wt='default_FKP',nran_sn=4,ells=[0],rpcut='',covmd='rescaled',rec='',version='unblinded/v1.2'):
    if rec != '':
        if tracer == 'QSO':
            sm = '30'
        else:
            sm = '15'
        rec = '_IFFT_recsym_sm'+sm
    cov = np.loadtxt(dircov+version+'/xi024_'+tracer+rec+'_'+reg+'_'+zrange+'_'+wt+'_lin4_s20-200_cov_RascalC_'+covmd+'.txt')
    if ells == [0,2,4] and smin == 20 and smax == 200:
        return cov
    if smin != 20 or smax != 200 and len(ells) > 1:
        return 'need to write in limit choices for more than 1 multipole'
    if len(ells) == 1:
        indmin = 45*ells[0]//2 + (smin-20)//bs
        indmax = 45*(1+ells[0]//2) - (200-smax)//bs
        cov = cov[indmin:indmax,indmin:indmax]
        return cov

def getxi_desipipe_baseline(tp,zrange,reg='GCcomb',rpcut='',version='unblinded',cut='',bs=4,rec=''):
    from pycorr import TwoPointCorrelationFunction
    fn = lssdir+version+'/desipipe/baseline_2pt/'+rec+'/xi/smu/allcounts_'+tp+'_GCcomb_z'+zrange+cut+'.npy'
    result = TwoPointCorrelationFunction.load(fn)
    factor = bs
    rebinned = result[:(result.shape[0] // factor) * factor:factor]
    sep, xis = rebinned(ells=(0, 2, 4), return_sep=True, return_std=False)
    return sep,xis

def get_xiave_desipipe_ab_baseline(zr='0.4-0.6',tp='LRG',rec='',nmock=25,flavor='altmtl',mockversion='v4_1fixran',reg='GCcomb',thetacut=''):
    from pycorr import TwoPointCorrelationFunction
    mocks = 'AbacusSummit'
    if tp[:3] == 'BGS':
        mocks += 'BGS'
    dirr = '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/'+mocks+'/desipipe/'+mockversion+'/'+flavor+'/baseline_2pt/mock'
    xil = []
    sepl = []
    for i in range(0,nmock):
        fn = dirr + str(i)+'/'+rec+'/xi/smu/allcounts_'+tp+'_'+reg+'_z'+zr+thetacut+'.npy'
        result = TwoPointCorrelationFunction.load(fn)
        factor = 4
        rebinned = result[:(result.shape[0] // factor) * factor:factor]
        sep, xis = rebinned(ells=(0, 2, 4), return_sep=True, return_std=False)
        xil.append(xis)
        sepl.append(sep)
    xi = sum(xil)/nmock
    sep = sum(sepl)/nmock
    return sep,xi

@mpl.rc_context(style._rcparams)
def plot_desipipe_xidatamock_comp_3ell(tp,zr,rec='',size=5,bf=1,version='unblinded',flavor='altmtl',mockversion='v4_1fixran',titl='',doleg=1):
    ells = [0,2,4]
    sep,ximock = get_xiave_desipipe_ab_baseline(zr=str(zr[0])+'-'+str(zr[1]),tp=tp,rec=rec,flavor=flavor,mockversion=mockversion)
    
    clr = style.colors[tp, (zr[0],zr[1])]
    print(clr)
    s,xiall = getxi_desipipe_baseline(tp,str(zr[0])+'-'+str(zr[1]),version=version,rec=rec)#nran=nran)
    s = s[5:]
    
    for ell in ells:
        xi = xiall[ell//2][5:]
        mockxi = ximock[ell//2][5:]
        cov = getcov_rascalC(tp,str(zr[0])+'_'+str(zr[1]),ells=[ell],rec=rec)
        plot_xipole_points_werr(s,xi,ell=ell,mockmean=mockxi,cov=cov,clr=clr,bf=bf,size=size,mec=clr)
    plt.xlabel(r's ($h^{-1}$Mpc)')
    plt.ylabel(r'$s^2\xi$  ($h^{-2}$Mpc$^2$)')
    if doleg == 1:
        plt.legend()
    return

plt.clf()
print(mpl.rcParams['figure.figsize'])
if args.mkBGS == 'y':
    plot_desipipe_xidatamock_comp_3ell('BGS_BRIGHT-21.5',(0.1,0.4),mockversion='v1',version= 'unblinded',rec='',titl=r'BGS $0.1<z<0.4$')
    #plt.title(r'BGS $0.1<z<0.4$')
    plt.ylim(-120,120)
    plt.savefig(outdir+outroot+'BGS.pdf', bbox_inches='tight')
plt.clf()
if args.mkLRG == 'y':

    zrl = [(0.4,0.6),(0.6,0.8),(0.8,1.1)]
    c = 0
    doleg = 1
    for zr in zrl:
        plot_desipipe_xidatamock_comp_3ell('LRG',zr,mockversion='v4_1fixran',version= 'unblinded',rec='',titl=r'LRG $'+str(zr[0])+'<z<'+str(zr[1])+'$',doleg=doleg)
        #plt.title(r'LRG $'+str(zr[0])+'<z<'+str(zr[1])+'$')
        plt.ylim(-120,120)
        plt.savefig(outdir+outroot+'LRG'+str(c)+'.pdf', bbox_inches='tight')
        plt.show()
        c += 1 
        plt.clf()
        doleg = 0 #only do the legend for the first plot

plt.clf()
if args.mkELG == 'y':

    zrl = [(0.8,1.1),(1.1,1.6)]
    c = 0
    doleg = 1
    for zr in zrl:
        plot_desipipe_xidatamock_comp_3ell('ELG_LOPnotqso',zr,mockversion='v4_1fixran',version= 'unblinded',rec='',titl=r'ELG $'+str(zr[0])+'<z<'+str(zr[1])+'$',doleg=doleg)
        #plt.title(r'ELG $'+str(zr[0])+'<z<'+str(zr[1])+'$')
        plt.ylim(-65,40)
        plt.savefig(outdir+outroot+'ELG'+str(c)+'.pdf', bbox_inches='tight')
        plt.show()
        c += 1
        plt.clf()
        doleg = 0


plt.clf()
if args.mkQSO == 'y':
    
    zr = (0.8,2.1)
    plot_desipipe_xidatamock_comp_3ell('QSO',zr,mockversion='v4_1fixran',version= 'unblinded',rec='',titl=r'QSO $'+str(zr[0])+'<z<'+str(zr[1])+'$')
    #plt.title(r'QSO $'+str(zr[0])+'<z<'+str(zr[1])+'$')
    plt.savefig(outdir+outroot+'QSO.pdf', bbox_inches='tight')
    plt.show()
