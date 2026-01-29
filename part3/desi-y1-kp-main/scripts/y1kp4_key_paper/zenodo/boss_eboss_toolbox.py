#load packages
import numpy as np
from astropy.table import Table
import fitsio


#define a function to read required data from the fits files
def basicreader(data_file,weightmode):

    #loads data (needs to be this way, else the admatti libaries complain)
    data_raw = Table(fitsio.read(data_file))
    
    #the datatype used for the function and can be understood by the rest of the code
    read_datatype=[('ra','f'),('dec','f'),('z','f'),('weights','f')]
    n_datapoint=len(data_raw)
    data_read=np.zeros(n_datapoint,dtype=read_datatype)

    data_read['ra']=data_raw['RA']
    data_read['dec']=data_raw['DEC']
    data_read['z']=data_raw['Z']#
    if (weightmode==0):
        data_read['weights']=np.ones(len(data_raw))
    if (weightmode==1): #used for BOSS randoms
        data_read['weights']=data_raw['WEIGHT_FKP']
    if (weightmode==2):
        data_read['weights']=data_raw['WEIGHT_FKP']*data_raw['WEIGHT_SYSTOT']
    if (weightmode==3):
        data_read['weights']=data_raw['WEIGHT_FKP_EBOSS']*data_raw['WEIGHT_SYSTOT']
    if (weightmode==4):
        data_read['weights']=data_raw['WEIGHT_SYSTOT']
    if (weightmode==5): #used for BOSS data
        data_read['weights']=data_raw['WEIGHT_FKP']*data_raw['WEIGHT_SYSTOT']*(data_raw['WEIGHT_CP']+data_raw['WEIGHT_NOZ']-1.0)
    if (weightmode==6):
        data_read['weights']=data_raw['WEIGHT_SYSTOT']*(data_raw['WEIGHT_CP']+data_raw['WEIGHT_NOZ']-1.0)
    if (weightmode==7): # used for eBOSS
        data_read['weights']=data_raw['WEIGHT_FKP']*data_raw['WEIGHT_SYSTOT']*data_raw['WEIGHT_CP']*data_raw['WEIGHT_NOZ']

    
    return data_read