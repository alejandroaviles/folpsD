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
parser.add_argument("--mknz_log", help="whether to make the log scale plot", default='n')
parser.add_argument("--mknz_GC", help="whether to make the linear scale plot", default='n')
parser.add_argument("--mkkp4", help="whether to make the linear scale plot", default='n')
parser.add_argument("--mkQSOveto", help="whether to make the QSO veto plot", default='n')
parser.add_argument("--mkLRGveto", help="whether to make the LRG veto plot", default='n')
parser.add_argument("--mkELGveto", help="whether to make the ELG veto plot", default='n')
parser.add_argument("--mkBGSveto", help="whether to make the BGS veto plot", default='n')
parser.add_argument("--fontsize", help="fontsize rc param", default=None)
args = parser.parse_args()


if args.outdir is None:
    outdir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/KPplots/'
else:
    outdir = args.outdir

if not os.path.exists(outdir):
    os.makedirs(outdir)
    print('made '+outdir)


@mpl.rc_context(style._rcparams)
def plotnz_NS_alltypes(zmin=0.02,bs=0.01,randens=2500,zcol='Z_not4clus',fontsize=None):
    if fontsize is not None:
        mpl.rcParams.update({'font.size': int(fontsize)})

    tpl = ['BGS_BRIGHT-21.5','LRG','ELG_LOPnotqso','QSO']
    for tp in tpl:
        zmax = 1.6
        if tp == 'QSO':
            zmax = 3.5
        clr = style.colors[tp[:3]]
        datcols = ['Z_not4clus','PHOTSYS','ZWARN','FRACZ_TILELOCID','FRAC_TLOBS_TILES','DELTACHI2']
        if tp[:3] == 'ELG':
            datcols.append('o2c')
        dat = fitsio.read(args.indir+tp+'_full_HPmapcut.dat.fits',columns=datcols)
        gz = common.goodz_infull(tp[:3],dat)
        selobs = dat['ZWARN'] != 999999
        dat = dat[gz&selobs]
        selN_dat = dat['PHOTSYS'] == 'N'
        ran = fitsio.read(args.indir+tp.strip('-21.5')+'_0_full_HPmapcut.ran.fits',columns=["PHOTSYS"])
        selN = ran['PHOTSYS'] == 'N'
        areaN = len(ran[selN])/randens
        areaS = len(ran[~selN])/randens
        nbin = int((zmax-zmin)/bs)



        wts = 1/dat['FRACZ_TILELOCID']*1/dat['FRAC_TLOBS_TILES']
        selnan = wts*0 != 0
        print('number of nans in weights '+str(np.sum(selnan)))
        wts[selnan] = 1.

        zhistN = np.histogram(dat[selN_dat][zcol],bins=nbin,range=(zmin,zmax),weights=wts[selN_dat])
        zhistS = np.histogram(dat[~selN_dat][zcol],bins=nbin,range=(zmin,zmax),weights=wts[~selN_dat])
        zl = zhistS[1][:-1]
        zh = zhistS[1][1:]
        zm = (zl+zh)/2.

        vol = 1/(360.*360./np.pi)*4.*np.pi/3.*(dis_dc(zh)**3.-dis_dc(zl)**3.)


        nzN = zhistN[0]/vol/areaN
        nzS = zhistS[0]/vol/areaS
        plt.plot(1/(1+zm),nzN,'--',color=clr)
        plt.plot(1/(1+zm),nzS,'-',color=clr,label=tp[:3])
    #plt.xticks([.87,.645,.5,.333,.25],('0.15','0.5','1','2','3'))
    yl = [2e-6,0.001]
    zbl = [0.1,0.4,0.6,0.8,1.1,1.6,2.1]
    zb_strl = []
    xtickl = []
    for zb in zbl:
        xt = 1/(1+zb)
        xtickl.append(xt)
        zb_str = str(zb)
        zb_strl.append(zb_str)
    plt.xticks(xtickl,zb_strl)

    #for zb in zbl:
    #    xl = [1/(1+zb),1/(1+zb)]
    #    plt.plot(xl,yl,'k:')
    plt.plot([-.1,-.5],[0,0],'k-',label='S')
    plt.plot([-.1,-.5],[0,0],'k--',label='N')
    #plt.grid()
    plt.yscale('log')
    plt.ylim(2e-6,0.001)
    #plt.xlim(0,3.5)
    plt.xlim(1/1.01,.2)
    #plt.text(1/3.,6e-5,'Applying completeness corrections',bbox=dict(facecolor='white', alpha=1))
    plt.title('Applying completeness corrections')
    plt.legend()
    plt.xlabel('Redshift')
    plt.ylabel(r'Comoving number density ($h$/Mpc)$^3$')
    #plt.xscale('symlog')
    plt.tight_layout()
    plt.savefig(outdir+'nzall.pdf', bbox_inches='tight')
    plt.clf()
    #plt.show()

if args.mknz_log == 'y':
    plotnz_NS_alltypes(fontsize=args.fontsize)

def plotnz_clus_raw(tp,zmin,zmax,bs,randens=2500,zcol='Z',fac=1,reg='',lt='-',lab='y',fontsize=None,wo='n'):
    if fontsize is not None:
        mpl.rcParams.update({'font.size': int(fontsize)})
    clr = style.colors[tp[:3]]
    
    dat = fitsio.read(args.indir+tp+reg+'_clustering.dat.fits')
    ran = fitsio.read(args.indir+tp+reg+'_0_clustering.ran.fits',columns=["PHOTSYS"])
    area = len(ran)/randens
    nbin = int((zmax-zmin)/bs)
    
                          
    
    #wts = dat['WEIGHT_COMP']*1/dat['FRAC_TLOBS_TILES']
    #selnan = wts*0 != 0
    #print('number of nans in weights '+str(np.sum(selnan)))
    #wts[selnan] = 1.
                          
    zhist = np.histogram(dat[zcol],bins=nbin,range=(zmin,zmax))#,weights=wts)
    zl = zhist[1][:-1]
    zh = zhist[1][1:]
    zm = (zl+zh)/2.
                          
    vol = 1/(360.*360./np.pi)*4.*np.pi/3.*(dis_dc(zh)**3.-dis_dc(zl)**3.)
                          
    
    nz = zhist[0]/vol/area
    if lab == 'y':
        labl = tp[:3]
    else:
        labl = None
    plt.plot(zm,nz*fac,lt,color=clr,label=labl)
    if wo == 'y':
        fo = open(outdir + tp+reg+'raw_nz_clus.txt','w')
        fo.write(tp+' n(z) in units of (h/Mpc)^3 /'+str(fac)+' \n')
        for i in range(0,len(zm)):
            fo.write(str(zm[i])+' '+str(nz*fac)+'\n')
        fo.close()
        print('wrote to '+outdir+tp+reg+'raw_nz_clus.txt')

@mpl.rc_context(style._rcparams)
def plotnz_clus_alltypes_raw_KP4(bs=0.01,randens=2500,zcol='Z',fac=1e4):
    tpl = ['BGS_BRIGHT-21.5','LRG','ELG_LOPnotqso','QSO']
    zrl = [(0.1,0.4),(0.4,1.1),(0.8,1.6),(0.8,2.1)]
    for tp,zr in zip(tpl,zrl):
        plotnz_clus_raw(tp,zr[0],zr[1],bs,fac=fac,wo='y')
        print(tp)
    yl = [0,0.00075*fac]
    zbl = [0.1,0.4,0.6,0.8,1.1,1.6,2.1]
    for zb in zbl:
        xl = [zb,zb]
        plt.plot(xl,yl,'k:')
    
    plt.legend(loc=(.75,.7))
    plt.xlabel('redshift')
    plt.ylabel(r'number density 10$^{-4}$($h$/Mpc)$^3$')
    plt.ylim(0.,0.00042*fac)
    plt.tight_layout()
    plt.savefig(outdir+'nzall_raw_4KP4.pdf', bbox_inches='tight')

if args.mkkp4 == 'y':
    plotnz_clus_alltypes_raw_KP4()

@mpl.rc_context(style._rcparams)
def plotnz_clus_alltypes_raw_KP3(bs=0.01,randens=2500,zcol='Z',fac=1,fontsize=None):
    if fontsize is not None:
        mpl.rcParams.update({'font.size': int(fontsize)})
    mpl.rcParams["axes.formatter.limits"] = [-2,2]
    tpl = ['BGS_BRIGHT-21.5','LRG','ELG_LOPnotqso','QSO']
    zrl = [(0.1,0.4),(0.4,1.1),(0.8,1.6),(0.8,2.1)]
    regl = ['_NGC','_SGC']
    ltl = ['-','--']
    for tp,zr in zip(tpl,zrl):
        for reg,lt in zip(regl,ltl):
            lab = 'y'
            if reg == '_SGC':
                lab = 'n'
            plotnz_clus_raw(tp,zr[0],zr[1],bs,randens=2500,zcol='Z',fac=fac,reg=reg,lt=lt,lab=lab)
        print(tp)
    yl = [0,0.00075*fac]
    zbl = [0.1,0.4,0.6,0.8,1.1,1.6,2.1]
    #for zb in zbl:
    #    xl = [zb,zb]
    #    plt.plot(xl,yl,'k:')
    xtickl = []
    zlsp = .2
    xt = zlsp
    #for zb in zbl:
    while xt < 2.2:    
        xtickl.append(xt)
        xt += zlsp 
        #zb_str = str(zb)
        #zb_strl.append(zb_str)
    plt.xticks(xtickl)#,zb_strl)
    
    plt.grid()
    #plt.grid(which='minor')
    #plt.yscale('log')
    #plt.ylim(2e-6,0.001)
    #custom grid
    #minor grid
    mgsp = 0.1
    pos4g = np.arange(0.1,2.2,.1)
    for pos in pos4g:
        plt.plot([pos,pos],[0,0.001],alpha=.2,color='k',lw=.2)
    for zb in zbl:
        plt.plot([zb,zb],[0,0.001],alpha=.7,color='k',lw=1)

    pos4g = np.arange(5e-5,5e-4,5e-5)
    for pos in pos4g:
        plt.plot([0,2.2],[pos,pos],alpha=.2,color='k',lw=.2)
    
    plt.grid()
    plt.plot([-1,-0.5],[0,0],'k-',label='NGC')
    plt.plot([-1,-0.5],[0,0],'k--',label='SGC')
    loc = 'best'
    if fontsize is None:
        loc = (.73,.6)
    if int(fontsize) == 13:
        loc = (.73,.5)
    plt.legend(loc=loc)#loc=(.73,.5))
    plt.xlabel('Redshift')
    plt.ylabel(r'Comoving number density ($h$/Mpc)$^3$')
    plt.ylim(0.,0.00045*fac)
    plt.xlim(0,2.2)
    plt.title('No completeness corrections')
    plt.tight_layout()
    plt.savefig(outdir+'nzall_raw_4KP3.pdf', bbox_inches='tight')
    #plt.show()

if args.mknz_GC == 'y':
    plotnz_clus_alltypes_raw_KP3(fontsize=args.fontsize)

def getnz(data,random,randens=2500,zcol='Z',tp='QSO',zmin=.8,zmax=3.5,nbin=27,label=''):
    selobs = data['ZWARN'] != 999999
    comp = len(data['ZWARN'][selobs])/len(data['ZWARN'])
    print('obs completeness is '+str(comp))
    if tp == 'QSO':
        goodz = data[zcol]*0 == 0
        goodz &= data[zcol] != 999999
        goodz &= data[zcol] != 1.e20
    if tp == 'LRG':
        goodz = data['ZWARN']==0
        goodz &= data['DELTACHI2']>15
        goodz &= data[zcol]<1.5
    if tp == 'ELG':
        goodz = data['o2c'] > 0.9
    if tp == 'BGS':    
        goodz = data['ZWARN']==0
        goodz &= data['DELTACHI2']>40
        
    bs = (zmax-zmin)/nbin
    a = np.histogram(data[goodz&selobs][zcol],range=(zmin,zmax),bins=nbin)
    #plt.plot(a[1][:-1]+bs/2,a[0]/comp/(len(random)/2500),label=label)
    print('good redshift fraction: '+str(len(data[goodz&selobs])/len(data[selobs])))
    return a[1][:-1]+bs/2,a[0]/comp/(len(random)/2500)

@mpl.rc_context(style._rcparams)
def mkqsoveto_plot(fontsize=None):
    if fontsize is not None:
        mpl.rcParams.update({'font.size': int(fontsize)})
    tp = 'QSO'
    ran = fitsio.read(args.indir+tp+'_0_full_noveto.ran.fits',columns=['GOODHARDLOC','MASKBITS','PHOTSYS'])
    dat = fitsio.read(args.indir+tp+'_full_noveto.dat.fits',columns=['GOODHARDLOC','MASKBITS','PHOTSYS','Z','ZWARN'])
    seld = dat['GOODHARDLOC'] == 1
    dat = dat[seld]
    selr = ran['GOODHARDLOC'] == 1
    ran = ran[selr]
    ebits = [8,9,11]
    dat_mask = np.ones(len(dat),dtype='bool')
    ran_mask = np.ones(len(ran),dtype='bool')
    for bit in ebits:
        dat_mask &= ((dat['MASKBITS'] & 2**bit)==0)
        ran_mask &= ((ran['MASKBITS'] & 2**bit)==0)
    clr = style.colors['QSO',(0.8,2.1)]
    selreg_dat = np.ones(len(dat),dtype='bool')
    selreg_ran = np.ones(len(ran),dtype='bool')
    outm = getnz(dat[dat_mask&selreg_dat],ran[ran_mask&selreg_ran])
    plt.plot(outm[0],outm[1]/0.1,label='not in LS masks 8,9,11',color=clr)
    inm = getnz(dat[~dat_mask&selreg_dat],ran[~ran_mask&selreg_ran])
    plt.plot(inm[0],inm[1]/0.1,':',label='in LS masks 8,9,11',color=clr)
    plt.legend()
    plt.xlabel('Redshift')
    plt.ylabel(r'QSO redshifts per ${\rm d} z$ deg$^2$')
    #plt.grid()
    plt.tight_layout()
    plt.savefig(outdir+'QSOmaskNZ.png')    

if args.mkQSOveto == 'y':
    mkqsoveto_plot(args.fontsize)

@mpl.rc_context(style._rcparams)
def mkLRGnzmask_plot(fontsize=None):
    if fontsize is not None:
        mpl.rcParams.update({'font.size': int(fontsize)})
    tp = 'LRG'
    ran = fitsio.read(args.indir+tp+'_0_full_noveto.ran.fits',columns=['GOODHARDLOC','MASKBITS','PHOTSYS','lrg_mask'])
    dat = fitsio.read(args.indir+tp+'_full_noveto.dat.fits',columns=['GOODHARDLOC','MASKBITS','PHOTSYS','ZWARN','DELTACHI2','Z','lrg_mask'])
    seld = dat['GOODHARDLOC'] == 1
    dat = dat[seld]
    selr = ran['GOODHARDLOC'] == 1
    ran = ran[selr]
    dat_mask = dat['lrg_mask'] == 0
    ran_mask = ran['lrg_mask'] == 0

    plt.clf()
    selreg_dat = np.ones(len(dat),dtype='bool')
    selreg_ran = np.ones(len(ran),dtype='bool')
    outm = getnz(dat[dat_mask&selreg_dat],ran[ran_mask&selreg_ran],tp='LRG',zmin=.4,zmax=1.1,nbin=70)
    clr = style.colors['LRG']
    plt.plot(outm[0],outm[1]/.01,label='not in imaging mask',color=clr)
    inm =getnz(dat[~dat_mask&selreg_dat],ran[~ran_mask&selreg_ran],tp='LRG',zmin=.4,zmax=1.1,nbin=70)
    plt.plot(inm[0],inm[1]/.01,':',label='in imaging mask',color=clr)
    plt.legend()
    plt.xlabel('Redshift')
    plt.ylabel(r'LRG redshifts per ${\rm d} z$ deg$^2$')
    #plt.title(reg)
    #plt.grid()
    plt.tight_layout()
    plt.savefig(outdir+'LRGmaskNZ.png')

if args.mkLRGveto == 'y':
    mkLRGnzmask_plot(args.fontsize)

@mpl.rc_context(style._rcparams)
def mkELGnzmask_plot(fontsize=None):
    if fontsize is not None:
        mpl.rcParams.update({'font.size': int(fontsize)})
    tp = 'ELG_LOPnotqso'
    ran = fitsio.read(args.indir+tp+'_0_full_noveto.ran.fits',columns=['MASKBITS','GOODHARDLOC','PHOTSYS'])
    dat = fitsio.read(args.indir+tp+'_full_noveto.dat.fits',columns=['Z','o2c','ZWARN','MASKBITS','GOODHARDLOC','PHOTSYS'])
    ebits = [11]
    clr = style.colors['ELG']
    for bit in ebits:

        dat_mask = ((dat['MASKBITS'] & 2**bit)==0)
        ran_mask = ((ran['MASKBITS'] & 2**bit)==0)
        print(bit,len(dat),len(dat[dat_mask]))
        print(bit,len(ran),len(ran[ran_mask]))
        #regl = ['N','S']
        #for reg in regl:
        #selreg_dat = dat['PHOTSYS'] == reg
        #selreg_ran = ran['PHOTSYS'] == reg
        selreg_dat = np.ones(len(dat),dtype='bool')
        selreg_ran = np.ones(len(ran),dtype='bool')
        outm = getnz(dat[dat_mask&selreg_dat],ran[ran_mask&selreg_ran],tp='ELG',zmin=.8,zmax=1.6,nbin=80)
        plt.plot(outm[0],outm[1]/0.01,'-',color=clr,label='not in LS mask '+str(bit))
        inm = getnz(dat[~dat_mask&selreg_dat],ran[~ran_mask&selreg_ran],tp='ELG',zmin=.8,zmax=1.6,nbin=80)
        plt.plot(inm[0],inm[1]/0.01,':',color=clr,label='in LS mask '+str(bit))
        plt.legend()
        #plt.grid()
        plt.xlabel('Redshift')
        plt.ylabel(r'ELG redshifts per ${\rm d} z$ deg$^2$')
        plt.tight_layout()
        plt.savefig(outdir+'ELGmaskNZ.png')
        #plt.title(reg)
        
if args.mkELGveto == 'y':
    mkELGnzmask_plot(args.fontsize)

@mpl.rc_context(style._rcparams)
def bgsnzmask_plot(fontsize=None):
    if fontsize is not None:
        mpl.rcParams.update({'font.size': int(fontsize)})
    tp = 'BGS_BRIGHT'
    ran = fitsio.read(args.indir+tp+'_0_full_noveto.ran.fits',columns=['MASKBITS','GOODHARDLOC','PHOTSYS'])
    dat = fitsio.read(args.indir+tp+'_full_noveto.dat.fits',columns=['Z','ZWARN','DELTACHI2','MASKBITS','GOODHARDLOC','PHOTSYS'])
    clr = style.colors['BGS_BRIGHT-21.5',(0.1,0.4)]
    ebits = [11]
    for bit in ebits:

        dat_mask = ((dat['MASKBITS'] & 2**bit)==0)
        ran_mask = ((ran['MASKBITS'] & 2**bit)==0)
        print(bit,len(dat),len(dat[dat_mask]))
        print(bit,len(ran),len(ran[ran_mask]))
        #regl = ['N','S']
        #for reg in regl:
        #    selreg_dat = dat['PHOTSYS'] == reg
        #    selreg_ran = ran['PHOTSYS'] == reg
        selreg_dat = np.ones(len(dat),dtype='bool')
        selreg_ran = np.ones(len(ran),dtype='bool')
        outm = getnz(dat[dat_mask&selreg_dat],ran[ran_mask&selreg_ran],tp='BGS',zmin=.1,zmax=.4,nbin=30)
        plt.plot(outm[0],outm[1]/.01,color=clr,label='not in LS mask '+str(bit))
        inm = getnz(dat[~dat_mask&selreg_dat],ran[~ran_mask&selreg_ran],tp='BGS',zmin=.1,zmax=.4,nbin=30)
        plt.plot(inm[0],inm[1]/.01,':',color=clr,label='in mask LS '+str(bit))
        plt.legend()
        plt.xlabel('Redshift')
        plt.ylabel( r'BGS redshifts per ${\rm d} z$ deg$^2$')
        #plt.title(reg)
        #plt.grid()
        plt.tight_layout()
        plt.savefig(outdir+'BGS_BRIGHTmaskNZ.png')

if args.mkBGSveto == 'y':
    bgsnzmask_plot(args.fontsize)
