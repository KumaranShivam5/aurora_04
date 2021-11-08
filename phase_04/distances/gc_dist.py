import pandas as pd 
import numpy as  np 
import astropy.units as u
from astropy.coordinates import SkyCoord

from astropy.io import fits
import os 
#os.system('clear')

data = pd.read_csv('../source_list/BH_data_clean_source_list.csv')
obj = data.copy()

#obj = pd.read_csv('../data/CVs/CV_mid_set_clean.csv').copy()

try:
    obj = obj.drop(columns=['near_gc' , 'near_gc_dist'])
except:
    print('Key not found')
obj_pos = obj[['B_RA' , 'B_DEC']]
obj_ra = obj['B_RA']
obj_dec = obj['B_DEC']
obj_name = obj['A_NAME']
gc =  pd.read_csv('gc_cat.csv')

def find_sep(ra1,dec1 , ra2, dec2):
    c1 = SkyCoord(ra1, dec1, frame='icrs' , unit=(u.deg))
    c2 = SkyCoord(ra2, dec2, frame='icrs' , unit=(u.hourangle , u.deg))
    sep = c1.separation(c2)
    return (sep)

i = 0
gc_near_name = []
gc_near_dist = []

for ra , dec , name in zip(obj_ra , obj_dec , obj_name):
    #ra1 ,dec1 = gc['RA(2000)'][0] , gc['DEC'][0]
    #ra2 , dec2 = obj['B_RA'][0] , obj['B_DEC'][0] 
    #print(i)
    
    sep = [find_sep(ra , dec , ra_o, dec_o).arcmin for ra_o , dec_o in zip(gc['RA'],gc['DEC'])]    
    sep_min = np.amin(sep)
    index = np.argmin(sep)
    
    if(sep_min<40):
        print(i)
        i+=1
        print('----------------------------------')
        print(name)
        print(sep_min)
        print(gc['ID'].iloc[index])
#       print(sep_min , name , '--', gc['ID'].iloc[index])  
        gc_near_name.append(gc['ID'].iloc[index])
        gc_near_dist.append(sep_min)
    else :
        gc_near_name.append('no_cluster')
        gc_near_dist.append(np.NaN)

obj.insert(obj.shape[1] , 'near_gc' ,gc_near_name)
obj.insert(obj.shape[1] , 'near_gc_dist' ,gc_near_dist)    
obj.to_csv('BH_with_dist.csv')
