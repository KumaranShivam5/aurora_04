import pandas as pd 
import numpy as  np 
import astropy.units as u
from astropy.coordinates import SkyCoord
from tqdm import tqdm

from astropy.io import fits
import os 
#os.system('clear')

def find_sep(ra1,dec1 , ra2, dec2):
    c1 = SkyCoord(ra1, dec1, frame='icrs' , unit=(u.deg))
    c2 = SkyCoord(ra2, dec2, frame='icrs' , unit=(u.hourangle , u.deg))
    sep = c1.separation(c2)
    return (sep)


hdu = fits.open('../data/binaries/all_xrb.fits')
srca = hdu[1].data
try:
    srca = srca.drop(columns=['near_srcb' , 'near_srcb_dist'])
except:
    print('Key not found')

srca_ra = srca['RA']
srca_dec = srca['DEC']
srca_name = srca['NAME']
srcb =  pd.read_csv('PULSAR_data_clean_source_list.csv')


i = 0
offset = []
srca_name = []
srcb_name = []
srca_ra = []
srca_dec = []
srcb_ra = []
srcb_dec = []
srcb_id = []
i = 0
tot = len(srca)
for ra , dec , name in tqdm(zip(srca_ra , srca_dec , srca_name)):
    #ra1 ,dec1 = srcb['RA(2000)'][0] , srcb['DEC'][0]
    #ra2 , dec2 = srca['B_RA'][0] , srca['B_DEC'][0] 
    #print(i)
    #i+=1
    sep = [find_sep(ra , dec , ra_o, dec_o).arcsrc for ra_o , dec_o in zip(srcb['B_RA'],srcb['B_DEC'])]    
    sep_min = np.amin(sep)
    index = np.argmin(sep)
    
    if(sep_min<10):
        print('['+str(i)+' / '+str(tot)+'] : ----------------------------------')
        print(name)
        print(sep_min)
        print(srcb['A_NAME'].iloc[index])
#       print(sep_min , name , '--', srcb['ID'].iloc[index])  
	srcb_name.append(srcb['A_NAME'].iloc[index])
	srcb_ra.append(srcb['B_RA'].iloc[index])
	srcb_dec.append(srcb['B_DEC'].iloc[index])
	srcb_id.append(srcb['SRC_ID'].iloc[index])	

	srca_name.append(name)
	srca_ra.append(ra)
	srca_dec.append(dec)	

        srcb_near_name.append(srcb['A_NAME'].iloc[index])
        srcb_near_dist.append(sep_min)
        i+=1


cross_match = pd.DataFrame()
cross_match.insert(0 , 'A_NAME' , srca_name)
cross_match.insert(1 , 'A_RA' , srca_ra)
cross_match.insert(2 , 'A_DEC' , srca_dec)

cross_match.insert(3 , 'B_NAME' , srcb_name)
cross_match.insert(4 , 'B_RA' , srcb_ra)
cross_match.insert(5 , 'B_DEC' , srcb_dec)
cross_match.insert(6 , 'B_ID' , srcb_id)


cross_match.to_csv('xrb-pulsar.csv')
'''
srca.insert(srca.shape[1] , 'near_srcb' ,srcb_near_name)
srca.insert(srca.shape[1] , 'near_srcb_dist' ,srcb_near_dist)    
srca.to_csv('source_list/BH_data_clean_source_list.csv')
'''
