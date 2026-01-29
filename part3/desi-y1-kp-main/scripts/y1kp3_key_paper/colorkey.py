from matplotlib import pyplot as plt
import matplotlib as mpl
from desi_y1_plotting.kp3 import KP3StylePaper
style = KP3StylePaper()


outdir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.4/KPplots/'

def points_with_errobars(x,y,errors,color='k',ptype='s',size=10,**kwargs):
    plt.errorbar(x,y,errors,fmt=ptype,color=color,markersize=size,capsize=size/1.5,elinewidth=size/5,capthick=size/5,**kwargs)

@mpl.rc_context(style._rcparams)
def color_test():
    yval = 0
    xval = 0
    erval = .4
    size = 10
    s = 0
    for key in style.colors.keys():
        if len(key) == 2:
            color=style.colors[key]
            tp = key[0][:3]
            if '+' in key[0]:
                tp ='LRG+ELG'
            zmin = key[1][0]
            zmax = key[1][1]
            if tp == 'Lya' and s == 0:
                lab = r'Lyman-$\alpha$'
                points_with_errobars([xval],[yval],[erval],color=color,size=size,label=lab)
                s = 1
                xval += 1
            elif tp != 'Lya':
                points_with_errobars([xval],[yval],[erval],color=color,size=size,label=tp+' '+str(zmin)+r'$<z<$'+str(zmax))
                xval += 1
            else:
                pass
            xval += 1
    plt.legend(ncol=2,loc='upper center')
    plt.ylim(-.5,1.5)
    plt.axis('off')
    plt.savefig(outdir+'colorkey.png', bbox_inches="tight")
    return
    
color_test()
