import fitsio
from astropy.table import Table, hstack,vstack
import healpy as hp
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
import matplotlib as mpl

from desi_y1_plotting.kp3 import KP3StylePaper
style = KP3StylePaper()

bossdir = '/dvs_ro/cfs/cdirs/sdss/data/sdss/dr12/boss/lss/'
desidir = '/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/'
ebossdir = '/dvs_ro/cfs/cdirs/sdss/data/sdss/dr16/eboss/lss/catalogs/DR16/'

#load in boss data and concatenate NGC/SGC
bn = fitsio.read(bossdir+'galaxy_DR12v5_CMASSLOWZTOT_North.fits.gz')
bs = fitsio.read(bossdir+'galaxy_DR12v5_CMASSLOWZTOT_South.fits.gz')
bt = np.concatenate([bn,bs])

#load in eboss data and concatenate NGC/SGC, galaxies and QSO

egaln = fitsio.read(ebossdir+'/eBOSS_LRG_clustering_data-NGC-vDR16.fits',columns=['RA','DEC','Z'])
egals = fitsio.read(ebossdir+'/eBOSS_LRG_clustering_data-SGC-vDR16.fits',columns=['RA','DEC','Z'])
egalt = np.concatenate([egaln,egals])

eqson = fitsio.read(ebossdir+'/eBOSS_QSO_clustering_data-NGC-vDR16.fits',columns=['RA','DEC','Z'])
eqsos = fitsio.read(ebossdir+'/eBOSS_QSO_clustering_data-SGC-vDR16.fits',columns=['RA','DEC','Z'])
eqsot = np.concatenate([eqson,eqsos])

#load in DESI targets
btar = fitsio.read('/global/cfs/cdirs/desi/survey/catalogs/main/LSS/BGS_ANYtargetsDR9v1.1.1.fits',columns=['RA','DEC'])
dtar = fitsio.read('/global/cfs/cdirs/desi/survey/catalogs/main/LSS/LRGtargetsDR9v1.1.1.fits',columns=['RA','DEC'])
qtar = fitsio.read('/global/cfs/cdirs/desi/survey/catalogs/main/LSS/QSOtargetsDR9v1.1.1.fits',columns=['RA','DEC'])

#load in DESI clusteirng catalogs
desi_bgsall = fitsio.read(desidir+'BGS_ANY_clustering.dat.fits')
desi_bgskp = fitsio.read(desidir+'BGS_BRIGHT-21.5_clustering.dat.fits')
desi_LRG = fitsio.read(desidir+'LRG_clustering.dat.fits')
desi_QSO = fitsio.read(desidir+'QSO_clustering.dat.fits')

#coordinate formats for astropy matching
cbgs = SkyCoord(ra=btar['RA']*u.degree, dec=btar['DEC']*u.degree)
cd = SkyCoord(ra=dtar['RA']*u.degree, dec=dtar['DEC']*u.degree)
cbn = SkyCoord(ra=bn['RA']*u.degree, dec=bn['DEC']*u.degree)
cbs = SkyCoord(ra=bs['RA']*u.degree, dec=bs['DEC']*u.degree)
cbt = SkyCoord(ra=bt['RA']*u.degree, dec=bt['DEC']*u.degree)
ces = SkyCoord(ra=egals['RA']*u.degree, dec=egals['DEC']*u.degree)
cen = SkyCoord(ra=egaln['RA']*u.degree, dec=egaln['DEC']*u.degree)
cet = SkyCoord(ra=egalt['RA']*u.degree, dec=egalt['DEC']*u.degree)
cqso_et = SkyCoord(ra=eqsot['RA']*u.degree, dec=eqsot['DEC']*u.degree)
cqso_tar = SkyCoord(ra=qtar['RA']*u.degree, dec=qtar['DEC']*u.degree)

cdesi_bgskp = SkyCoord(ra=desi_bgskp['RA']*u.degree, dec=desi_bgskp['DEC']*u.degree)
cdesi_LRG = SkyCoord(ra=desi_LRG['RA']*u.degree, dec=desi_LRG['DEC']*u.degree)
cdesi_QSO = SkyCoord(ra=desi_QSO['RA']*u.degree, dec=desi_QSO['DEC']*u.degree)

#do match
idx, d2d, d3d = cbn.match_to_catalog_sky(cd)
idxb, d2db, d3d = cbn.match_to_catalog_sky(cbgs)
idxsb, d2dsb, d3d = cbs.match_to_catalog_sky(cbgs)
idxs, d2ds, d3d = cbs.match_to_catalog_sky(cd)
idxes, d2des, d3d = ces.match_to_catalog_sky(cd)
idxen, d2den, d3d = cen.match_to_catalog_sky(cd)

idx_bl, d2d_bl, d3d_bl = cbt.match_to_catalog_sky(cd)
idx_bb, d2d_bb, d3d_bb = cbt.match_to_catalog_sky(cbgs)
idx_el, d2d_el, d3d_el = cet.match_to_catalog_sky(cd)
idx_qso, d2d_qso, d3d_qso = cqso_et.match_to_catalog_sky(cqso_tar)

idx_bgskp, d2d_bgskp, d3d_bgskp = cbt.match_to_catalog_sky(cdesi_bgskp)
idx_lrg, d2d_lrg, d3d_lrg = cbt.match_to_catalog_sky(cdesi_LRG)
idx_elrg, d2d_elrg, d3d_elrg = cet.match_to_catalog_sky(cdesi_LRG)
idx_dqso, d2d_dqso, d3d_dqso = cqso_et.match_to_catalog_sky(cdesi_QSO)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", help="output directory for the plots", default=None)
parser.add_argument("--mkLRGtar", help="whether to make the LRG target overlap fraction plot", default='y')
parser.add_argument("--mkLRGfoot", help="whether to make the LRG footprint overlap fraction plot", default='y')
parser.add_argument("--mkQSOtar", help="whether to make the QSO target overlap fraction plot", default='y')
parser.add_argument("--mkQSOfoot", help="whether to make the QSO footprint overlap fraction plot", default='y')
args = parser.parse_args()


if args.outdir is None:
    outdir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/KPplots/' 


#set maximum separation of 1 arcsecond for match
max_sep1 = 1.0 * u.arcsec


zmin = 0.2

zsep = 0.01
zmax = zmin+zsep#0.3
zl = []
fl = []
plt.clf()
with mpl.rc_context(style._rcparams):
    if args.mkLRGtar == 'y':
        while zmax < 0.81:
            selz = bt['Z'] > zmin
            selz &= bt['Z'] < zmax
            sep_c1 = d2d_bl < max_sep1
            sep_c1 |= d2d_bb < max_sep1
            #print(zmin,zmax,len(cbt[sep_c1&selz])/len(cbt[selz]))
            zl.append((zmax+zmin)/2.)
            zmin += zsep
            zmax += zsep
    
            fl.append(len(cbt[sep_c1&selz])/len(cbt[selz]))
        plt.plot(zl,fl,'k-',label='BOSS LRG, targets')
                
            
        #    plt.plot(zl[sel],fl[sel],'-',color=clr)#,label='BOSS LRG, targets')
        #    zmin = zmax
        zmin = 0.2
        zmax = zmin+zsep#0.3
        #zsep = 0.1
        zl = []
        fl = []
        while zmax < 0.81:
            selz = bt['Z'] > zmin
            selz &= bt['Z'] < zmax
            sep_c1 = d2d_lrg < max_sep1
            sep_c1 |= d2d_bgskp < max_sep1
            #print(zmin,zmax,len(cbt[sep_c1&selz])/len(cbt[selz]))
            zl.append((zmax+zmin)/2.)
            zmin += zsep
            zmax += zsep
    
            fl.append(len(cbt[sep_c1&selz])/len(cbt[selz]))
        plt.plot(zl,fl,'k--',label='BOSS LRG, DR1 LSS')
    
        zspl = [0.4,0.6,0.8,1.1]
        zmin = 0.1
        zl = np.array(zl)
        fl = np.array(fl)
        for zmax in zspl:
            selz = bt['Z'] > zmin
            selz &= bt['Z'] < zmax
            
            if zmax == 0.4:
                sep_c1 = d2d_bgskp < max_sep1
                print('BOSS/BGS comp',len(cbt[sep_c1&selz]),len(desi_bgskp))
            else:
                sep_c1 = d2d_lrg < max_sep1
                selzd = desi_LRG['Z'] > zmin
                selzd &= desi_LRG['Z'] < zmax
                print('BOSS/LRG comp '+str(zmin)+'<z<'+str(zmax),len(cbt[sep_c1&selz]),len(desi_LRG[selzd]))
                
            zmin = zmax
        zmin = 0.6
        zmax = zmin+zsep#0.7
        #zsep = 0.1
        zl = []
        fl = []
    
        while zmax < 1:
            selz = egalt['Z'] > zmin
            selz &= egalt['Z'] < zmax
            sep_c1 = d2d_el < max_sep1
            #print(zmin,zmax,len(cet[sep_c1&selz])/len(cet[selz]))
            zl.append((zmax+zmin)/2.)
            zmin += zsep
            zmax += zsep
    
            fl.append(len(cet[sep_c1&selz])/len(cet[selz]))
        plt.plot(zl,fl,'-',label='eBOSS LRG, targets',color='crimson')    
    
        zmin = 0.6
        zmax = zmin+zsep#0.7
        #zsep = 0.1
        zl = []
        fl = []
    
        while zmax < 1:
            selz = egalt['Z'] > zmin
            selz &= egalt['Z'] < zmax
            sep_c1 = d2d_elrg < max_sep1
            #print(zmin,zmax,len(cet[sep_c1&selz])/len(cet[selz]))
            zl.append((zmax+zmin)/2.)
            zmin += zsep
            zmax += zsep
    
            fl.append(len(cet[sep_c1&selz])/len(cet[selz]))
        plt.plot(zl,fl,'--',label='eBOSS LRG, DR1 LSS',color='crimson')    
        
        
        plt.legend()#loc='lower left')
        plt.ylim(0.1,1)
        plt.xlabel('Redshift')
        plt.ylabel('Fraction that are in DESI')
        plt.tight_layout()
        plt.savefig(outdir+'SDSSLRGretar.png')
    
        plt.clf()
    
        zspl = [0.8,1.1]
        zmin = 0.6
        zl = np.array(zl)
        fl = np.array(fl)
        for zmax in zspl:
            selz = egalt['Z'] > zmin
            selz &= egalt['Z'] < zmax
            
            sep_c1 = d2d_elrg < max_sep1
            selzd = desi_LRG['Z'] > zmin
            selzd &= desi_LRG['Z'] < zmax
            print('eBOSS/LRG comp '+str(zmin)+'<z<'+str(zmax),len(cet[sep_c1&selz]),len(desi_LRG[selzd]))
                
            zmin = zmax

    if args.mkLRGfoot == 'y':
        sep_c1 = d2d_lrg < max_sep1
        ra = bt['RA']
        selra = ra > 300
        ra[selra] -=360
        ratar = desi_LRG['RA']
        seldra = ratar > 300
        ratar[seldra] -= 360
    
        plt.plot(ratar,desi_LRG['DEC'],',',label='DESI DR1 LRG',color='.5')
        plt.plot(ra,bt['DEC'],'k,',label='BOSS LRG')
        plt.plot(ra[sep_c1],bt[sep_c1]['DEC'],',',color='r',label='BOSS LRG & DESI DR1')
        plt.gca().invert_xaxis()
        plt.legend(labelcolor='linecolor')
        plt.xlabel('Right Ascension')
        plt.ylabel('Declination')
        plt.tight_layout()
        plt.savefig(outdir+'SDSSLRGoverlap.png')

#compare redshifts of matched LRGs
match_keep = d2d_lrg < max_sep1
_, keep_counts = np.unique(idx_lrg[match_keep], return_counts=True)
print(f"Matched {np.sum(match_keep)} entries from input catalog to DR12 BOSS.")

# If there are any double matches we'll need to handle that
if np.any(keep_counts) > 1:
    print("Double matches found...")

# Reduces the tables to the matched entries using the indices of matches
desi_keep = Table(desi_LRG[idx_lrg][match_keep])
boss_keep = Table(bt[match_keep])
boss_keep.rename_column("Z", "Z_SDSS")
joined = hstack([desi_keep, boss_keep])
sel = abs((joined['Z_SDSS']-joined['Z'])/(1+joined['Z'])) > 0.001
print('number and fraction with difference greater than 0.001*(1+z_desi)',len(joined[sel]),len(joined[sel])/len(joined))
print('mean and standard deviation of delta z / (1+z_desi) for non-outliers:')
print(np.mean((joined['Z_SDSS'][~sel]-joined['Z'][~sel])/(1+joined['Z'][~sel])),np.std((joined['Z_SDSS'][~sel]-joined['Z'][~sel])/(1+joined['Z'][~sel])))

#compare with BGS
match_keep = d2d_bgskp < max_sep1
_, keep_counts = np.unique(idx_bgskp[match_keep], return_counts=True)
print(f"Matched {np.sum(match_keep)} entries from input catalog to DR12 BOSS.")

# If there are any double matches we'll need to handle that
if np.any(keep_counts) > 1:
    print("Double matches found...")

# Reduces the tables to the matched entries using the indices of matches
desi_keep = Table(desi_bgskp[idx_bgskp][match_keep])
boss_keep = Table(bt[match_keep])
boss_keep.rename_column("Z", "Z_SDSS")
joined = hstack([desi_keep, boss_keep])

sel = abs((joined['Z_SDSS']-joined['Z'])/(1+joined['Z'])) > 0.001
print('number and fraction with difference greater than 0.001*(1+z_desi)',len(joined[sel]),len(joined[sel])/len(joined))
print('mean and standard deviation of delta z / (1+z_desi) for non-outliers:')
print(np.mean((joined['Z_SDSS'][~sel]-joined['Z'][~sel])/(1+joined['Z'][~sel])),np.std((joined['Z_SDSS'][~sel]-joined['Z'][~sel])/(1+joined['Z'][~sel])))

match_keep = d2d_dqso < max_sep1
_, keep_counts = np.unique(idx_dqso[match_keep], return_counts=True)
print(f"Matched {np.sum(match_keep)} entries from input catalog to DR16 eBOSS QSO.")

# If there are any double matches we'll need to handle that
if np.any(keep_counts) > 1:
    print("Double matches found...")

# Reduces the tables to the matched entries using the indices of matches
desi_keep = Table(desi_QSO[idx_dqso][match_keep])
boss_keep = Table(eqsot[match_keep])
boss_keep.rename_column("Z", "Z_SDSS")
joined = hstack([desi_keep, boss_keep])

sel = abs(joined['Z_SDSS']-joined['Z'])/(1+joined['Z']) > 0.01
print(len(joined[sel]),len(joined[sel])/len(joined))
print(np.mean((joined['Z_SDSS'][~sel]-joined['Z'][~sel])/(1+joined['Z'][~sel])),np.std((joined['Z_SDSS'][~sel]-joined['Z'][~sel])/(1+joined['Z'][~sel])))

zmin = 0.8

zsep = 0.05
zmax = zmin+zsep#0.3
zl = []
fl = []
plt.clf()
with mpl.rc_context(style._rcparams):
    if args.mkQSOtar == 'y':
        while zmax < 2.21:
            selz = eqsot['Z'] > zmin
            selz &= eqsot['Z'] < zmax
            sep_c1 = d2d_qso < max_sep1
            
            if len(cqso_et[selz]) > 0:
                zl.append((zmax+zmin)/2.)
                fl.append(len(cqso_et[sep_c1&selz])/len(cqso_et[selz]))
            #else:
            #print(zmin,zmax)
            zmin += zsep
            zmax += zsep
    
        plt.plot(zl,fl,'-',label='SDSS QSO, targets',color='seagreen')
    
        zmin = 0.8
    
        zsep = 0.05
        zmax = zmin+zsep#0.3
        zl = []
        fl = []
        while zmax < 2.21:
            selz = eqsot['Z'] > zmin
            selz &= eqsot['Z'] < zmax
            sep_c1 = d2d_dqso < max_sep1
            
            if len(cqso_et[selz]) > 0:
                zl.append((zmax+zmin)/2.)
                fl.append(len(cqso_et[sep_c1&selz])/len(cqso_et[selz]))
            #else:
            #print(zmin,zmax)
            zmin += zsep
            zmax += zsep
        plt.plot(zl,fl,'--',label='SDSS QSO, DR1 LSS',color='seagreen')
        plt.legend()#loc='upper left')
        plt.ylim(0.1,1)
        plt.xlabel('Redshift')
        plt.ylabel('Fraction that are in DESI')
        plt.tight_layout()
        plt.savefig(outdir+'SDSSQSOretar.png')
    
        plt.clf()
        zmin = 0.8
        zmax = 2.1
        selz = eqsot['Z'] > zmin
        selz &= eqsot['Z'] < zmax
    
        sep_c1 = d2d_dqso < max_sep1
        selzd = desi_QSO['Z'] > zmin
        selzd &= desi_QSO['Z'] < zmax
        print('eBOSS/QSO comp '+str(zmin)+'<z<'+str(zmax),len(cqso_et[sep_c1&selz]),len(desi_QSO[selzd]),len(cqso_et[selz]))

    if args.mkQSOtar == 'y':    
        ra = eqsot['RA']
        selra = ra > 300
        ra[selra] -=360
        sep_c1 = d2d_dqso < max_sep1
    
        plt.plot(ra,eqsot['DEC'],'k,',label='eBOSS QSO')
        plt.plot(ra[sep_c1],eqsot[sep_c1]['DEC'],',',color='seagreen',label='in DESI DR1')
        plt.gca().invert_xaxis()
        plt.legend(labelcolor='linecolor')
        plt.xlabel('Right Ascension')
        plt.ylabel('Declination')
        plt.tight_layout()
        plt.savefig(outdir+'SDSSQSOoverlap.png')
        
