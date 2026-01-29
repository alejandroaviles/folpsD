import fitsio
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import sys,os
from desi_y1_plotting.kp3 import KP3StylePaper
style = KP3StylePaper()
#set LSSCODE, e.g. via export LSSCODE=$HOME on the command line if that is where you cloned the repo 
sys.path.append(os.environ['LSSCODE']+'/LSS/py')
from LSS import common_tools as common

from LSS.tabulated_cosmo import TabulatedDESI
cosmo = TabulatedDESI()
dis_dc = cosmo.comoving_radial_distance

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--indir", help="base directory for catalogs", default='/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/')
parser.add_argument("--outdir", help="output directory for the plots", default=None)
parser.add_argument("--mkLRG", help="whether to make the QSO veto plot", default='n')
parser.add_argument("--mkQSO", help="whether to make the LRG veto plot", default='n')
parser.add_argument("--mkELG", help="whether to make the ELG veto plot", default='n')
parser.add_argument("--mkBGS", help="whether to make the BGS veto plot", default='n')
parser.add_argument("--mkLRGresid", help="whether to make the QSO veto plot", default='n')
parser.add_argument("--mkBGSresid", help="whether to make the QSO veto plot", default='n')
parser.add_argument("--mkQSOresid", help="whether to make the QSO veto plot", default='n')
parser.add_argument("--mkELGresid", help="whether to make the QSO veto plot", default='n')
args = parser.parse_args()


if args.outdir is None:
    outdir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/KPplots/' #imaging systematics were run on v1.3 and used again for v1.4
  
else:
    outdir = args.outdir

if not os.path.exists(outdir):
    os.makedirs(outdir)
    print('made '+outdir)

mockver='LSScats'
mocks='/AbacusSummit_v4_1/altmtl'
dirdata = args.indir+'/plots/imaging/'

def getmeanchi2_imsys(tp,mp,zmin,zmax,reg,nmock=25,datawt='',mockwts='noweights',notqso=''):
    #compare data chi2 to mean of mock results for a particular map
    chi2tot = 0
    tpm = tp
    if tp == 'ELG_LOPnotqso' and 'altmtl' not in mocks:
        tpm = 'ELG_LOP'
    
    for i in range(0,nmock):
        if 'altmtl' not in mocks:
            dirf = '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/'+mocks+'/mock'+str(i)+'/'+mockver+'/plots/imaging/'
        else:
            dirf = '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/'+mocks+str(i)+'/mock'+str(i)+'/'+mockver+'/plots/imaging/'
        fn = dirf+tpm+notqso+str(zmin)+'<z<'+str(zmax)+'_densclusvsall_'+reg+'_validate'+mockwts+'_chi2.txt'
        df = open(fn)
        for line in df:
            ls = line.split()
            if ls[0] == mp:
                chi2i = float(ls[1])
                #print(i,chi2i)
        chi2tot += chi2i
    
    fn = dirdata+tp+str(zmin)+'<z<'+str(zmax)+'_densfullvsall_'+reg+'_validate'+datawt+'_chi2.txt'
    df = open(fn)
    for line in df:
        ls = line.split()
        if ls[0] == mp:
            chi2d = float(ls[1])

    return chi2tot/nmock,chi2d

def getchi2high_imsys(tp,mp,zmin,zmax,reg,nmock=25,datawt='',mockwts='noweights',notqso=''):
    #find the number of mocks that have a higher chi2 for a given map
    
    fn = dirdata+tp+notqso+str(zmin)+'<z<'+str(zmax)+'_densfullvsall_'+reg+'_validate'+datawt+'_chi2.txt'
    df = open(fn)
    for line in df:
        ls = line.split()
        if ls[0] == mp:
            chi2d = float(ls[1])
            chi2dnw = float(ls[2])

    chi2tot = 0
    nhigh = 0
    nhighnw = 0
    tpm = tp
    if tp == 'ELG_LOPnotqso' and 'altmtl' not in mocks:
        tpm = 'ELG_LOP'
    nmocks = 0
    lowchi2 = 1000
    highchi2 = -1

    for i in range(0,nmock):
        if 'altmtl' not in mocks:
            dirf = '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/'+mocks+'/mock'+str(i)+'/'+mockver+'/plots/imaging/'
        else:
            dirf = '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/'+mocks+str(i)+'/mock'+str(i)+'/'+mockver+'/plots/imaging/'
        fn = dirf+tpm+str(zmin)+'<z<'+str(zmax)+'_densclusvsall_'+reg+'_validate'+mockwts+'_chi2.txt'
        if os.path.isfile(fn):
        #try:
            df = open(fn)
            for line in df:
                ls = line.split()
                if ls[0] == mp:
                    chi2i = float(ls[1])
                    #print(i,chi2i)
            if chi2i < lowchi2:
                lowchi2 = chi2i
            if chi2i > highchi2:
                highchi2 = chi2i

            if chi2i > chi2d:
                nhigh += 1
            if chi2i > chi2dnw:
                nhighnw += 1
            chi2tot += chi2i
            nmocks += 1
        #except:
        else:
            print(fn+' not found')
    return nhigh,nhighnw,chi2tot/nmocks,chi2d,lowchi2,highchi2,chi2dnw

def gettotalchi2_imsys(tp,zmin,zmax,reg,nmock=25,datawt='',mockwts='noweights',notqso=''):
    #compare the total chi2 across all maps from the data to the mean of the mocks
    fn = dirdata+tp+notqso+str(zmin)+'<z<'+str(zmax)+'_densfullvsall_'+reg+'_validate'+datawt+'_chi2.txt'
    df = open(fn).readlines()
    chi2d = float(df[-1].split()[3])
    chi2dnw = 0
    for i in range(0,len(df)-1):
        chi2dnw += float(df[i].split()[-1])
    #chi2dnw = float(df.split()[4])
    chi2tot = 0
    chi2totnw = 0
    tpm = tp
    #if tp == 'ELG_LOPnotqso' and mockver != 'LSScats':
    #    tpm = 'ELG_LOP'
    nmocks = 0
    nhigh = 0
    nhighnw = 0
    if tp == 'ELG_LOPnotqso' and 'altmtl' not in mocks:
        tpm = 'ELG_LOP'

    lowchi2 = 1000
    highchi2 = -1
    for i in range(0,nmock):
        if 'altmtl' not in mocks:
            dirf = '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/'+mocks+'/mock'+str(i)+'/'+mockver+'/plots/imaging/'
        else:
            dirf = '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/'+mocks+str(i)+'/mock'+str(i)+'/'+mockver+'/plots/imaging/'
        fn = dirf+tpm+str(zmin)+'<z<'+str(zmax)+'_densclusvsall_'+reg+'_validate'+mockwts+'_chi2.txt'
        try:
            dfa = open(fn).readlines()
            df = dfa[-1]
            chi2i = float(df.split()[3])
            if chi2i < lowchi2:
                lowchi2 = chi2i
            if chi2i > highchi2:
                highchi2 = chi2i

            chi2tot += chi2i
            chi2inw = 0
            for j in range(0,len(dfa)-1):
                chi2inw += float(dfa[j].split()[-1])
            chi2totnw += chi2inw
            nmocks += 1
            #print(i,chi2i)
            if chi2i > chi2d:
                nhigh += 1
            if chi2inw > chi2dnw:
                nhighnw += 1
        except:
            print(fn+' '+str(i)+' not found')
    print('number of mocks found '+str(nmocks))
    return chi2tot/nmock,chi2d,chi2totnw/nmock,lowchi2,highchi2,chi2dnw,nhigh,nhighnw

@mpl.rc_context(style._rcparams)
def plot_chi2results_datamock(tp,zr,mapl,mapnl,wts='WEIGHT_IMLIN',fname=None):
    ind = 0
    plt.clf()
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot()
    indl = []
    for mp in mapl:
    
        for reg in regl:
        
            ans = getchi2high_imsys(tp,mp,zr[0],zr[1],reg,datawt=wts,mockwts=wts)
            if reg == 'N':
                off = -.225
                #ax.plot(ind+off,ans[0],'o',mfc='none',mec='k')
                if ind == 0:
                    ax.bar(ind+off,ans[0],0.4,color='w',edgecolor='b',label='BASS/MzLS',linewidth=3)
                else:
                    ax.bar(ind+off,ans[0],0.4,color='w',edgecolor='b',linewidth=3)
                if ans[0] < 3:
                    ax.plot(ind+off,ans[0],'b*',markersize=20,linewidth=3,mec='k')
                #ax.plot(ind,ans[-2],'d',mfc='none',mec='r')
            else:    
                off = 0.225
                if ind == 0:
                    ax.bar(ind+off,ans[0],0.4,color='w',edgecolor='r',label='DECam',linewidth=3)
                    #ax.plot(ind,ans[0],'ko',label=r'# of mocks with $>\chi^2$')
                    #ax.plot(ind,ans[-2],'rd',label=r'data $\chi^2$')
                else:
                    ax.bar(ind+off,ans[0],0.4,color='w',edgecolor='r',linewidth=3)#,label='DECam')
                    #ax.plot(ind,ans[0],'ko')
                    #ax.plot(ind,ans[-2],'rd')
                if ans[0] < 3:
                    ax.plot(ind+off,ans[0],'r*',markersize=20,linewidth=3,mec='k')

                    #(axis='x')
        indl.append(ind)
        ind += 1
    indl.append(ind)
    #mapnl.append('total')
    #mapnl.append('') #for final formatting
    for reg in regl:    
        ans = gettotalchi2_imsys(tp,zr[0],zr[1],reg,datawt=wts,mockwts=wts)[-2]
        if reg == 'N':
            off = -.225
            ax.bar(ind+off,ans,0.4,color='w',edgecolor='b',linewidth=3)
            if ans < 3:
                ax.plot(ind+off,ans,'b*',markersize=20,linewidth=3,mec='k')
            #ax.plot(ind,ans[-2],'d',mfc='none',mec='r')
        else:    
            off = 0.225
            ax.bar(ind+off,ans,0.4,color='w',edgecolor='r',linewidth=3)#,label='DECaLS')
            if ans < 3:
                ax.plot(ind+off,ans,'r*',markersize=20,linewidth=3,mec='k')
    ind += 1
    indl.append(ind)
    for reg in regl:    
        ans = gettotalchi2_imsys(tp,zr[0],zr[1],reg,datawt=wts,mockwts=wts)[-1]
        if reg == 'N':
            off = -.225
            ax.bar(ind+off,ans,0.4,color='w',edgecolor='b',linewidth=3)
            if ans < 3:
                ax.plot(ind+off,ans,'b*',markersize=20,linewidth=3,mec='k')
            #ax.plot(ind,ans[-2],'d',mfc='none',mec='r')
        else:    
            off = 0.225
            ax.bar(ind+off,ans,0.4,color='w',edgecolor='r',linewidth=3)#,label='DECaLS')
            if ans < 3:
                ax.plot(ind+off,ans,'r*',markersize=20,linewidth=3,mec='k')
   
    tickl = np.arange(len(mapnl))-.5
    ax.plot(tickl,np.zeros(len(tickl)),'k:')
    ax.plot(tickl,np.ones(len(tickl)),'k:') 
    ax.plot(tickl,np.ones(len(tickl))*5,'k:')
    plt.ylabel(r'# of mocks with $>\chi^2$')
    ax.grid(axis='x')
    
    ax.set_xticks(ticks=tickl, labels=[key for key in mapnl], rotation=50, ha='left')
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    #plt.ylabel(r'data $\chi^2$ or number of mocks with higher $\chi^2$')
    plt.legend()
    plt.title(tp+' '+str(zr[0])+'<z<'+str(zr[1]))
    plt.ylim(-1,26)
    if fname is not None:
        print(fname)
        plt.savefig(fname, bbox_inches="tight")
    plt.show()
    return

@mpl.rc_context(style._rcparams)
def plot_chi2results_datamock_mean(tp,zr,mapl,mapnl,wts='WEIGHT_IMLIN',fname=None):
    ind = 0
    plt.clf()
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot()
    indl = []
    for mp in mapl:
    
        for reg in regl:
        
            ans = getchi2high_imsys(tp,mp,zr[0],zr[1],reg,datawt=wts,mockwts=wts)
            if reg == 'N':
                off = -.225
                #ax.plot(ind+off,ans[0],'o',mfc='none',mec='k')
                if ind == 0:
                    ax.bar(ind+off,ans[3]/ans[2],0.4,color='w',edgecolor='b',linewidth=3)#,label='BASS/MzLS')
                else:
                    ax.bar(ind+off,ans[3]/ans[2],0.4,color='w',edgecolor='b',linewidth=3)
                #if ans[0] < 1:
                #    ax.plot(ind+off,ans[3]/ans[2],'b*',markersize=20,linewidth=3,mec='k')
                ax.plot([ind+off,ind+off],[ans[4]/ans[2],ans[5]/ans[2]],'b:')
                #ax.plot(ind,ans[-2],'d',mfc='none',mec='r')
            else:    
                off = 0.225
                if ind == 0:
                    ax.bar(ind+off,ans[3]/ans[2],0.4,color='w',edgecolor='r',linewidth=3)#,label='DECam')
                    #ax.plot(ind,ans[0],'ko',label=r'# of mocks with $>\chi^2$')
                    #ax.plot(ind,ans[-2],'rd',label=r'data $\chi^2$')
                else:
                    ax.bar(ind+off,ans[3]/ans[2],0.4,color='w',edgecolor='r',linewidth=3)#,label='DECaLS')
                    #ax.plot(ind,ans[0],'ko')
                    #ax.plot(ind,ans[-2],'rd')
                #if ans[0] < 1:
                #    ax.plot(ind+off,ans[3]/ans[2],'r*',markersize=20,linewidth=3,mec='k')
                ax.plot([ind+off,ind+off],[ans[4]/ans[2],ans[5]/ans[2]],'r:')    

                    #(axis='x')
        indl.append(ind)
        ind += 1
    indl.append(ind)
    #mapnl.append('total')
    #mapnl.append('') #for final formatting
    for reg in regl:    
        ans = gettotalchi2_imsys(tp,zr[0],zr[1],reg,datawt=wts,mockwts=wts)#[-2]
        if reg == 'N':
            off = -.225
            ax.bar(ind+off,ans[1]/ans[0],0.4,color='w',edgecolor='b',linewidth=3)
            #if ans[-2] < 1:
            #    ax.plot(ind+off,ans[1]/ans[0],'b*',markersize=20,linewidth=3,mec='k')
            ax.plot([ind+off,ind+off],[ans[3]/ans[0],ans[4]/ans[0]],'b:')
            #ax.plot(ind,ans[-2],'d',mfc='none',mec='r')
        else:    
            off = 0.225
            ax.bar(ind+off,ans[1]/ans[0],0.4,color='w',edgecolor='r',linewidth=3)#,label='DECaLS')
            #if ans[-2] < 1:
            #    ax.plot(ind+off,ans[1]/ans[0],'r*',markersize=20,linewidth=3,mec='k')
            ax.plot([ind+off,ind+off],[ans[3]/ans[0],ans[4]/ans[0]],'r:')
    ind += 1
    indl.append(ind)
    for reg in regl:    
        ans = gettotalchi2_imsys(tp,zr[0],zr[1],reg,datawt=wts,mockwts=wts)#[-1]
        nwratio = ans[5]/ans[0]
        #if nwratio > 5:
        #    nwratio = 5

        if reg == 'N':
            off = -.225
            ax.bar(ind+off,nwratio,0.4,color='w',edgecolor='b',linewidth=3)
            #if ans[-1] < 3:
            #    ax.plot(ind+off,nwratio,'b*',markersize=20,linewidth=3,mec='k')
            #ax.plot(ind,ans[-2],'d',mfc='none',mec='r')
        else:    
            off = 0.225
            ax.bar(ind+off,nwratio,0.4,color='w',edgecolor='r',linewidth=3)#,label='DECaLS')
            #if ans[-1] < 3:
            #    ax.plot(ind+off,nwratio,'r*',markersize=20,linewidth=3,mec='k')
   
    tickl = np.arange(len(mapnl))-.5
    ax.plot(tickl,np.zeros(len(tickl)),'k:')
    ax.plot(tickl,np.ones(len(tickl)),'k:') 
    #ax.plot(tickl,np.ones(len(tickl))*5,'k:')
    plt.ylabel(r'$\chi^2/\langle \chi^2_{\rm mocks}\rangle $')
    #ax.grid(axis='x')
    
    ax.set_xticks(ticks=tickl, labels=[key for key in mapnl], rotation=50, ha='left')
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    #plt.ylabel(r'data $\chi^2$ or number of mocks with higher $\chi^2$')
    l1, = ax.bar(ind,6.1,bottom=6,color='w',edgecolor='k',linewidth=3,label='data',zorder=0)#,label='BASS/MzLS')
    l3, = ax.plot(ind,6,'b-',label='BASS/MzLS',zorder=2)
    l4, = ax.plot(ind,6,'r-',label='DECam',zorder=3)

    l2, = ax.plot([ind,ind],[6,6.5],'k:',label='range in mocks',zorder=1)
    ax.legend()#handles=[l1,l2,l3,l4])
    plt.text(6,5,tp[:3]+' '+str(zr[0])+'<z<'+str(zr[1]),bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2})
    plt.ylim(-.5,5.5)
    #plt.ylim(-1,26)
    if fname is not None:
        print(fname)
        plt.savefig(fname, bbox_inches="tight")
    plt.show()
    return

if args.mkLRG == 'y':
    mapl = ['PSFSIZE_G','PSFSIZE_R','PSFSIZE_Z','PSFDEPTH_W1','GALDEPTH_Z','GALDEPTH_R','GALDEPTH_G','STARDENS','HI','EBV_CHIANG_SFDcorr','EBV_DESI_GR-EBV_SFD']
    mapnlu = ['PSF_G','PSF_R','PSF_Z','DEPTH_W1','DEPTH_Z','DEPTH_R','DEPTH_G','STARDENS','HI',r'${\rm EBVnoCIB}}$',r'$\Delta$ EBV GR','total','total, unweighted','']
    tp = 'LRG'
    regl = ['N','S']
    zrl = [(0.4,0.6),(0.6,0.8),(0.8,1.1)]
    indl = []
    for zr in zrl:
        #fname = outdir+'imsys_mocktest_'+tp+str(zr[0])+str(zr[1])+'.png'
        #plot_chi2results_datamock(tp,zr,mapl,mapnlu,fname=fname)
        fname = outdir+'imsys_mocktest_relmean_'+tp+str(zr[0])+str(zr[1])+'.png'
        plot_chi2results_datamock_mean(tp,zr,mapl,mapnlu,fname=fname)

if args.mkELG == 'y':
    mapl = ['PSFSIZE_G','PSFSIZE_R','PSFSIZE_Z','GALDEPTH_Z','GALDEPTH_R','GALDEPTH_G','STARDENS','HI','EBV_CHIANG_SFDcorr','EBV_DESI_GR-EBV_SFD']
    mapnl = ['PSF_G','PSF_R','PSF_Z','DEPTH_Z','DEPTH_R','DEPTH_G','STARDENS','HI',r'${\rm EBVnoCIB}}$',r'$\Delta$ EBV GR','total','total, unweighted',''] #final empty string for formatting
    tp = 'ELG_LOPnotqso'
    regl = ['N','S']
    zrl = [(0.8,1.1),(1.1,1.6)]
    indl = []
    for zr in zrl:
        #fname = outdir+'imsys_mocktest_'+tp+str(zr[0])+str(zr[1])+'.png'
        #plot_chi2results_datamock(tp,zr,mapl,mapnl,fname=fname,wts='WEIGHT_SN')
        fname = outdir+'imsys_mocktest_relmean_'+tp+str(zr[0])+str(zr[1])+'.png'
        plot_chi2results_datamock_mean(tp,zr,mapl,mapnl,fname=fname,wts='WEIGHT_SN')

if args.mkQSO == 'y':
    mapl = ['PSFSIZE_G','PSFSIZE_R','PSFSIZE_Z','PSFDEPTH_W1','PSFDEPTH_W2','PSFDEPTH_Z','PSFDEPTH_R','PSFDEPTH_G','STARDENS','HI','EBV_CHIANG_SFDcorr','EBV_DESI_GR-EBV_SFD']
    mapnl = ['PSF_G','PSF_R','PSF_Z','DEPTH_W1', 'DEPTH_W2','DEPTH_Z','DEPTH_R','DEPTH_G','STARDENS','HI',r'${\rm EBVnoCIB}}$',r'$\Delta$ EBV GR','total','total, unweighted',''] #final empty string for formatting
    tp = 'QSO'
    regl = ['N','S']
    zrl = [(0.8,2.1)]#,(1.6,2.1)]
    indl = []
    for zr in zrl:
        #fname = outdir+'/imsys_mocktest_'+tp+str(zr[0])+str(zr[1])+'.png'
        #plot_chi2results_datamock(tp,zr,mapl,mapnl,wts='WEIGHT_RF',fname=fname)
        fname = outdir+'imsys_mocktest_relmean_'+tp+str(zr[0])+str(zr[1])+'.png'
        plot_chi2results_datamock_mean(tp,zr,mapl,mapnl,fname=fname,wts='WEIGHT_RF')

if args.mkBGS == 'y':
    #mockver='LSScats'
    mocks='/AbacusSummitBGS/altmtl'
    mapl = ['PSFSIZE_G','PSFSIZE_R','PSFSIZE_Z','GALDEPTH_Z','GALDEPTH_R','GALDEPTH_G','STARDENS','HI','EBV_CHIANG_SFDcorr','EBV_DESI_GR-EBV_SFD']
    mapnl = ['PSF_G','PSF_R','PSF_Z','DEPTH_Z','DEPTH_R','DEPTH_G','STARDENS','HI',r'${\rm EBVnoCIB}}$',r'$\Delta$ EBV GR','total','total, unweighted',''] #final empty string for formatting
    tp = 'BGS_BRIGHT-21.5'
    regl = ['N','S']
    zrl = [(0.1,0.4)]#,(1.6,2.1)]
    indl = []
    for zr in zrl:
        #fname = outdir+'imsys_mocktest_'+tp+str(zr[0])+str(zr[1])+'.png'
        #plot_chi2results_datamock(tp,zr,mapl,mapnl,wts='WEIGHT_IMLIN',fname=fname)
        fname = outdir+'imsys_mocktest_relmean_'+tp+str(zr[0])+str(zr[1])+'.png'
        plot_chi2results_datamock_mean(tp,zr,mapl,mapnl,fname=fname,wts='WEIGHT_IMLIN')

@mpl.rc_context(style._rcparams)
def mkLRGresid():
    fig = plt.figure(figsize=(8.5,3.5))
    ncolumns = 4 # number of horizontal subpanels
    nrows = 1 # number of vertical subpanels

    #one row is for the four worst maps in terms of chi2 relative to mean of the mocks
    clr1 = style.colors['LRG', (0.4,0.6)]
    clr2 = style.colors['LRG', (0.6,0.8)]
    clr3 = style.colors['LRG', (0.8,1.1)]
    #first plot 0.4<z<0.6
    ax = fig.add_subplot(nrows,ncolumns,1)
    dird = dirdata+'ngalvsysfiles/'
    d = np.loadtxt(dird+'ngalvs_EBV_CHIANG_SFDcorrY1LRG0.4z0.6S.txt').transpose()
    
    
    
    ax.errorbar(d[0],d[1],d[3],fmt='o',color=clr1,label='corrected',mec='k')
    chi2w = np.sum(((d[1]-1)/d[3])**2.)
    chi2nw = np.sum(((d[2]-1)/d[3])**2.)
    print('for ngalvs_EBV_CHIANG_SFDcorrY1LRG0.4z0.6S.txt :')
    print('chi2 weighted is '+str(chi2w))
    print('chi2 unweighted is '+str(chi2nw))
    ax.plot(d[0],d[2],'--',color=clr1,label='raw')
    ax.plot(d[0],np.ones(len(d[0])),'k:')
    ax.set_xlabel('EBV, no CIB \n (mag)')
    ax.set_ylabel(r'LRG $n_{\rm gal}$/$\langle n_{\rm gal} \rangle$')
    ax.legend()
    ax.text(0.02,.85,'DECam 0.4<z<0.6')
    d = np.loadtxt(dird+'ngalvs_GALDEPTH_RY1LRG0.4z0.6S.txt').transpose()
    ax2 = fig.add_subplot(nrows,ncolumns,2,sharey=ax)
    ax2.errorbar(d[0],d[1],d[3],fmt='o',color=clr1,mec='k')
    ax2.plot(d[0],d[2],'--',color=clr1)
    chi2w = np.sum(((d[1]-1)/d[3])**2.)
    chi2nw = np.sum(((d[2]-1)/d[3])**2.)
    print('for ngalvs_GALDEPTH_RY1LRG0.4z0.6S.txt :')
    print('chi2 weighted is '+str(chi2w))
    print('chi2 unweighted is '+str(chi2nw))
    
    ax2.plot(d[0],np.ones(len(d[0])),'k:')
    ax2.set_xlabel('DEPTH_R \n (nanomaggies)')
    ax2.text(350,.85,'DECam 0.4<z<0.6')
    for ylabel_i in ax2.axes.get_yticklabels():
        ylabel_i.set_visible(False)
    #0.6<z<0.8 against stellar density
    d = np.loadtxt(dird+'ngalvs_STARDENSY1LRG0.6z0.8S.txt').transpose()
    ax4 = fig.add_subplot(nrows,ncolumns,3,sharey=ax)
    ax4.errorbar(d[0],d[1],d[3],fmt='o',color=clr2,mec='k')
    ax4.plot(d[0],d[2],'--',color=clr2)
    chi2w = np.sum(((d[1]-1)/d[3])**2.)
    chi2nw = np.sum(((d[2]-1)/d[3])**2.)
    print('for ngalvs_STARDENSY1LRG0.6z0.8S.txt :')
    print('chi2 weighted is '+str(chi2w))
    print('chi2 unweighted is '+str(chi2nw))
    
    ax4.plot(d[0],np.ones(len(d[0])),'k:')
    ax4.set_xlabel('stellar density \n '+r'(# Gaia/deg$^2$)')
    ax4.text(5000,.85,'DECam 0.6<z<0.8')
    for ylabel_i in ax4.axes.get_yticklabels():
        ylabel_i.set_visible(False)

    #relationships to plot for 0.8 < z < 1.1    
    d = np.loadtxt(dird+'ngalvs_EBV_DESI_GREBV_SFDY1LRG0.8z1.1S.txt').transpose()
    ax3 = fig.add_subplot(nrows,ncolumns,4,sharey=ax)
    ax3.errorbar(d[0],d[1],d[3],fmt='o',color=clr3,mec='k')
    ax3.plot(d[0],d[2],'--',color=clr3)
    chi2w = np.sum(((d[1]-1)/d[3])**2.)
    chi2nw = np.sum(((d[2]-1)/d[3])**2.)
    print('for ngalvs_EBV_DESI_GREBV_SFDY1LRG0.8z1.1S.txt :')
    print('chi2 weighted is '+str(chi2w))
    print('chi2 unweighted is '+str(chi2nw))

    ax3.plot(d[0],np.ones(len(d[0])),'k:')
    d = np.loadtxt(dird+'ngalvs_EBV_DESI_GREBV_SFDnoCIBY1LRG0.8z1.1S.txt').transpose()
    ax3.plot(d[0],d[1],'-',color=clr3,label='(no CIB)')
    ax3.legend()
    chi2w = np.sum(((d[1]-1)/d[3])**2.)
    chi2nw = np.sum(((d[2]-1)/d[3])**2.)
    print('for ngalvs_EBV_DESI_GREBV_SFDY1LRG0.8z1.1S.txt :')
    print('chi2 weighted is '+str(chi2w))
    print('chi2 unweighted is '+str(chi2nw))

    ax3.set_xlabel(r'$\Delta$ EBV GR'+' \n (mag)')
    ax3.text(-0.03,.85,'DECam 0.8<z<1.1')
    for ylabel_i in ax3.axes.get_yticklabels():
        ylabel_i.set_visible(False)
    plt.ylim(.8,1.1)


    fig.subplots_adjust(wspace=0)
    plt.ylim(.8,1.1)
    fname = outdir+'imsys_LRGresid.png'
    plt.savefig(fname,bbox_inches="tight")
    return 

if args.mkLRGresid == 'y':
    mkLRGresid()

@mpl.rc_context(style._rcparams)
def mkBGSresid():
    dird = dirdata+'ngalvsysfiles/'
    fig = plt.figure(figsize=(8.5,3.5))
    ncolumns = 3 # number of horizontal subpanels
    nrows = 1 # number of vertical subpanels

    clr1 = style.colors['BGS_BRIGHT-21.5', (0.1,0.4)] #'darkgoldenrod'

    ax = fig.add_subplot(nrows,ncolumns,1)
    d = np.loadtxt(dird+'ngalvs_HIY1BGS_BRIGHT-21.50.1z0.4S.txt').transpose()
    ax.errorbar(d[0],d[1],d[3],fmt='o',color=clr1,label='corrected',mec='k')
    ax.plot(d[0],d[2],'--',color=clr1,label='raw')
    chi2w = np.sum(((d[1]-1)/d[3])**2.)
    chi2nw = np.sum(((d[2]-1)/d[3])**2.)
    print('for ngalvs_HIY1BGS_BRIGHT-21.50.1z0.4S.txt :')
    print('chi2 weighted is '+str(chi2w))
    print('chi2 unweighted is '+str(chi2nw))

    ax.plot(d[0],np.ones(len(d[0])),'k:')
    ax.set_xlabel('HI col. dens. \n'+r' (cm$^{-2}$)')
    ax.set_ylabel(r'BGS $n_{\rm gal}$/$\langle n_{\rm gal} \rangle$')
    ax.legend()
    ax.text(2e20,.92,'DECam ')
    d = np.loadtxt(dird+'ngalvs_EBV_DESI_GREBV_SFDY1BGS_BRIGHT-21.50.1z0.4S.txt').transpose()
    ax2 = fig.add_subplot(nrows,ncolumns,2,sharey=ax)
    ax2.errorbar(d[0],d[1],d[3],fmt='o',color=clr1,mec='k')
    ax2.plot(d[0],d[2],'--',color=clr1)
    chi2w = np.sum(((d[1]-1)/d[3])**2.)
    chi2nw = np.sum(((d[2]-1)/d[3])**2.)
    print('for ngalvs_EBV_DESI_GREBV_SFDY1BGS_BRIGHT-21.50.1z0.4S.txt :')
    print('chi2 weighted is '+str(chi2w))
    print('chi2 unweighted is '+str(chi2nw))

    ax2.plot(d[0],np.ones(len(d[0])),'k:')
    ax2.set_xlabel(r'$\Delta$ EBV GR'+'\n (mag.)')
    ax2.text(-0.03,.92,'DECam ')
    for ylabel_i in ax2.axes.get_yticklabels():
        ylabel_i.set_visible(False)
    d = np.loadtxt(dird+'ngalvs_EBV_DESI_GREBV_SFDnoCIBY1BGS_BRIGHT-21.50.1z0.4S.txt').transpose()
    ax2.plot(d[0],d[1],'-',color=clr1,label='(no CIB)')
    ax2.legend()
    chi2w = np.sum(((d[1]-1)/d[3])**2.)
    chi2nw = np.sum(((d[2]-1)/d[3])**2.)
    print('for ngalvs_EBV_DESI_GREBV_SFDnoCIBY1BGS_BRIGHT-21.50.1z0.4S.txt :')
    print('chi2 weighted is '+str(chi2w))

    d = np.loadtxt(dird+'ngalvs_EBV_DESI_GREBV_SFDY1BGS_BRIGHT-21.50.1z0.4N.txt').transpose()
    ax3 = fig.add_subplot(nrows,ncolumns,3,sharey=ax)
    ax3.errorbar(d[0],d[1],d[3],fmt='^',color=clr1,mec='k')
    ax3.plot(d[0],d[2],'--',color=clr1)
    chi2w = np.sum(((d[1]-1)/d[3])**2.)
    chi2nw = np.sum(((d[2]-1)/d[3])**2.)
    print('for ngalvs_EBV_DESI_GREBV_SFDY1BGS_BRIGHT-21.50.1z0.4N.txt :')
    print('chi2 weighted is '+str(chi2w))

    ax3.plot(d[0],np.ones(len(d[0])),'k:')
    d = np.loadtxt(dird+'ngalvs_EBV_DESI_GREBV_SFDnoCIBY1BGS_BRIGHT-21.50.1z0.4N.txt').transpose()
    ax3.plot(d[0],d[1],'-',color=clr1,label='(no CIB)')
    chi2w = np.sum(((d[1]-1)/d[3])**2.)
    chi2nw = np.sum(((d[2]-1)/d[3])**2.)
    print('for ngalvs_EBV_DESI_GREBV_SFDnoCIBY1BGS_BRIGHT-21.50.1z0.4N.txt :')
    print('chi2 weighted is '+str(chi2w))

    ax3.set_xlabel(r'$\Delta$ EBV GR'+'\n (mag.)')
    ax3.text(-0.02,.92,'BASS/MzLS')
    for ylabel_i in ax3.axes.get_yticklabels():
        ylabel_i.set_visible(False)
    plt.ylim(.8,1.1)
    fname = outdir + 'imsys_BGSresid.png'
    plt.savefig(fname,bbox_inches="tight")
    return

if args.mkBGSresid == 'y':
    mkBGSresid()

@mpl.rc_context(style._rcparams)
def mkQSOresid():
    dird = dirdata+'ngalvsysfiles/'
    fig = plt.figure(figsize=(8.5,3.5))
    ncolumns = 4 # number of horizontal subpanels
    nrows = 1 # number of vertical subpanels

    clr1 = style.colors['QSO', (0.8,2.1)]

    ax = fig.add_subplot(nrows,ncolumns,1)
    d = np.loadtxt(dird+'ngalvs_PSFSIZE_ZY1QSO0.8z2.1S.txt').transpose()
    chi2w = np.sum(((d[1]-1)/d[3])**2.)
    chi2nw = np.sum(((d[2]-1)/d[3])**2.)
    print('for ngalvs_PSFSIZE_ZY1QSO0.8z2.1S.txt :')
    print('chi2 weighted is '+str(chi2w))
    print('chi2 unweighted is '+str(chi2nw))
    ax.errorbar(d[0],d[1],d[3],fmt='o',color=clr1,label='corrected',mec='k')
    ax.plot(d[0],d[2],'--',color=clr1,label='raw')
    ax.plot(d[0],np.ones(len(d[0])),'k:')
    ax.set_xlabel('PSF_Z \n (arcsec)')
    ax.set_ylabel(r'QSO $n_{\rm gal}$/$\langle n_{\rm gal} \rangle$')
    ax.legend()
    ax.text(1.1,.95,'DECam ')
    d = np.loadtxt(dird+'ngalvs_EBV_CHIANG_SFDcorrY1QSO0.8z2.1S.txt').transpose()
    chi2w = np.sum(((d[1]-1)/d[3])**2.)
    chi2nw = np.sum(((d[2]-1)/d[3])**2.)
    print('for ngalvs_EBV_CHIANG_SFDcorrY1QSO0.8z2.1S.txt :')
    print('chi2 weighted is '+str(chi2w))
    print('chi2 unweighted is '+str(chi2nw))
    ax4 = fig.add_subplot(nrows,ncolumns,2,sharey=ax)
    ax4.errorbar(d[0],d[1],d[3],fmt='o',color=clr1,mec='k')
    ax4.plot(d[0],d[2],'--',color=clr1)
    ax4.plot(d[0],np.ones(len(d[0])),'k:')
    ax4.set_xlabel('EBV, no CIB \n (mag)')
    ax4.text(.03,.95,'DECam ')
    for ylabel_i in ax4.axes.get_yticklabels():
        ylabel_i.set_visible(False)
   
    d = np.loadtxt(dird+'ngalvs_PSFSIZE_RY1QSO0.8z2.1N.txt').transpose()
    chi2w = np.sum(((d[1]-1)/d[3])**2.)
    chi2nw = np.sum(((d[2]-1)/d[3])**2.)
    print('for ngalvs_PSFSIZE_RY1QSO0.8z2.1N.txt :')
    print('chi2 weighted is '+str(chi2w))
    print('chi2 unweighted is '+str(chi2nw))

    ax2 = fig.add_subplot(nrows,ncolumns,3,sharey=ax)
    ax2.errorbar(d[0],d[1],d[3],fmt='^',color=clr1,mec='k')
    ax2.plot(d[0],d[2],'--',color=clr1)
    ax2.plot(d[0],np.ones(len(d[0])),'k:')
    ax2.set_xlabel('PSF_R \n (arcsec)')
    ax2.text(1.3,.95,'BASS/MzLS ')
    for ylabel_i in ax2.axes.get_yticklabels():
        ylabel_i.set_visible(False)

    #relationships to plot for 0.8 < z < 1.1    
    d = np.loadtxt(dird+'ngalvs_PSFSIZE_GY1QSO0.8z2.1N.txt').transpose()
    chi2w = np.sum(((d[1]-1)/d[3])**2.)
    chi2nw = np.sum(((d[2]-1)/d[3])**2.)
    print('for ngalvs_PSFSIZE_GY1QSO0.8z2.1N.txt :')
    print('chi2 weighted is '+str(chi2w))
    print('chi2 unweighted is '+str(chi2nw))

    ax3 = fig.add_subplot(nrows,ncolumns,4,sharey=ax)
    ax3.errorbar(d[0],d[1],d[3],fmt='^',color=clr1,mec='k')
    ax3.plot(d[0],d[2],'--',color=clr1)
    ax3.plot(d[0],np.ones(len(d[0])),'k:')
    ax3.set_xlabel('PSF_G \n (arcsec)')
    ax3.text(1.5,.95,'BASS/MzLS')
    for ylabel_i in ax3.axes.get_yticklabels():
        ylabel_i.set_visible(False)
    plt.ylim(.9,1.1)

    fname = outdir + 'imsys_QSOresid.png'
    fig.subplots_adjust(wspace=0)
    plt.savefig(fname,bbox_inches="tight")
    return

if args.mkQSOresid == 'y':
    mkQSOresid()

@mpl.rc_context(style._rcparams)
def mkELGresid():
    dird = dirdata+'ngalvsysfiles/'
    fig = plt.figure(figsize=(8.5,3.5))
    ncolumns = 4 # number of horizontal subpanels
    nrows = 1 # number of vertical subpanels
    clr1 = style.colors['ELG_LOPnotqso', (0.8,1.1)]
    clr2 = style.colors['ELG_LOPnotqso', (1.1,1.6)]

    #first plot 0.8<z<1.1
    ax = fig.add_subplot(nrows,ncolumns,1)
    d = np.loadtxt(dird+'ngalvs_EBV_CHIANG_SFDcorrY1ELG_LOPnotqso0.8z1.1N.txt').transpose()
    chi2w = np.sum(((d[1]-1)/d[3])**2.)
    chi2nw = np.sum(((d[2]-1)/d[3])**2.)
    print('for ngalvs_EBV_CHIANG_SFDcorrY1ELG_LOPnotqso0.8z1.1N.txt :')
    print('chi2 weighted is '+str(chi2w))
    print('chi2 unweighted is '+str(chi2nw))

    ax.errorbar(d[0],d[1],d[3],fmt='^',color=clr1,label='corrected',mec='k')
    ax.plot(d[0],d[2],'--',color=clr1,label='raw')
    ax.plot(d[0],np.ones(len(d[0])),'k:')
    ax.set_xlabel('EBV, no CIB \n (mag.)')
    ax.set_ylabel(r'ELG $n_{\rm gal}$/$\langle n_{\rm gal} \rangle$')
    ax.legend()
    ax.text(0.02,.8,'BASS/MzLS \n 0.8<z<1.1')

    d = np.loadtxt(dird+'ngalvs_EBV_DESI_GREBV_SFDY1ELG_LOPnotqso0.8z1.1N.txt').transpose()
    chi2w = np.sum(((d[1]-1)/d[3])**2.)
    chi2nw = np.sum(((d[2]-1)/d[3])**2.)
    print('for ngalvs_EBV_DESI_GREBV_SFDY1ELG_LOPnotqso0.8z1.1N.txt :')
    print('chi2 weighted is '+str(chi2w))
    print('chi2 unweighted is '+str(chi2nw))

    ax2 = fig.add_subplot(nrows,ncolumns,2,sharey=ax)
    ax2.errorbar(d[0],d[1],d[3],fmt='^',color=clr1,mec='k')
    ax2.plot(d[0],d[2],'--',color=clr1)
    ax2.plot(d[0],np.ones(len(d[0])),'k:')
    ax2.set_xlabel(r'$\Delta$ EBV GR'+'\n (mag.)')
    ax2.text(-0.02,.8,'BASS/MzLS \n 0.8<z<1.1')
    for ylabel_i in ax2.axes.get_yticklabels():
        ylabel_i.set_visible(False)

    d = np.loadtxt(dird+'ngalvs_STARDENSY1ELG_LOPnotqso0.8z1.1S.txt').transpose()
    chi2w = np.sum(((d[1]-1)/d[3])**2.)
    chi2nw = np.sum(((d[2]-1)/d[3])**2.)
    print('for ngalvs_STARDENSY1ELG_LOPnotqso0.8z1.1S.txt :')
    print('chi2 weighted is '+str(chi2w))
    print('chi2 unweighted is '+str(chi2nw))
    ax4 = fig.add_subplot(nrows,ncolumns,3,sharey=ax)
    ax4.errorbar(d[0],d[1],d[3],fmt='o',color=clr1,mec='k')
    ax4.plot(d[0],d[2],'--',color=clr1)
    ax4.plot(d[0],np.ones(len(d[0])),'k:')
    ax4.set_xlabel('stellar density \n '+r'(# Gaia/deg$^2$)')
    ax4.text(5000,.8,'DECam \n 0.8<z<1.1')
    for ylabel_i in ax4.axes.get_yticklabels():
        ylabel_i.set_visible(False)


    #relationships to plot for 1.1 < z < 1.6    
    d = np.loadtxt(dird+'ngalvs_GALDEPTH_GY1ELG_LOPnotqso1.1z1.6N.txt').transpose()
    chi2w = np.sum(((d[1]-1)/d[3])**2.)
    chi2nw = np.sum(((d[2]-1)/d[3])**2.)
    print('for ngalvs_GALDEPTH_GY1ELG_LOPnotqso1.1z1.6N.txt :')
    print('chi2 weighted is '+str(chi2w))
    print('chi2 unweighted is '+str(chi2nw))
    ax3 = fig.add_subplot(nrows,ncolumns,4,sharey=ax)
    ax3.errorbar(d[0],d[1],d[3],fmt='^',color=clr2,mec='k')
    ax3.plot(d[0],d[2],'--',color=clr2)
    ax3.plot(d[0],np.ones(len(d[0])),'k:')
    ax3.set_xlabel('DEPTH_G \n (nannomaggies)')
    ax3.text(300,.8,'BASS/MzLS \n 1.1<z<1.6')
    for ylabel_i in ax3.axes.get_yticklabels():
        ylabel_i.set_visible(False)

    
    
    fig.subplots_adjust(wspace=0)
    plt.ylim(.7,1.2)
    fname = outdir +'imsys_ELGresid.png'
    plt.savefig(fname,bbox_inches="tight")
    return

if args.mkELGresid == 'y':
    mkELGresid()

