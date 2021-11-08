from posixpath import join
from astropy.io.fits import hdu
import pandas as pd 
#from matplotlib import pyplot as plt 
#import numpy as np 
from ciao_contrib.runtool import search_csc 

from astropy.io import fits 
from fields import fields 

def hr():
    print('-------------------------------------------------------')

#hdu1 = fits.open('../data/binaries/LMXB_NS_v2.fits')[1].data
#hdu1 = fits.open('../data/binaries/LMXRB_BH.fits')[1].data
#hdu1 = fits.open('../data/CV/CV_chandra_heasarc_corr_all_10_arcsec.fits')[1].data
from astropy.table import Table, vstack


def download_data(data_fits , root):
    count = 0 
    for di in data_fits:
        ra = di['B_RA']
        dec =  di['B_DEC']
        name  =  di['A_NAME'].replace(' ' ,'_')
        ch_name = di['B_NAME'].replace(' ' ,'_')
        name = ch_name+'.csv'
        hr()
        print('source Number :  ' , count )
        print(ra,dec)
        try:
            
            s = search_csc(
                radunit="arcsec", 
                columns=fields, 
                bands="broad",
                clobber="yes" ,
                radius=1, 
                pos= str(ra)+','+str(dec)  ,
                outfile='temp.csv')
            #sys('search_csc outfile=trial.csv radunit=arcsec columns="SOS,SOP,SOV , o.gti_obs m.flux_aper_b" bands=broad clobber=yes radius=1 pos="65.428058,32.907468"')
            #print(s)
            
            data = pd.read_csv('temp.csv', delimiter='\t' , comment='#')
            data = data[data['match_type']=='u          ']
            data = data[data['instrument']=='ACIS']
            data.index.name= 'index'
            #f_name_fits = data_fits['A_NAME']
            hr()
            print(name)
            hr()
            #name = name.replace(' ' , '_')+'.csv'
            data.to_csv(root+name)
            #------master info 

            print(data)
        except Exception as e:
            print(e)
            print('Error Occured , manual inspetion needed ')
        hr()
        count+=1

hdu1 = fits.open('../data/pulsars/atnf.fits')[1].data

ra = hdu1['B_RA']
dec = hdu1['B_DEC']
download_data(hdu1 , 'pulsar_data_all/')

hr()
print('ATNF COMPLETED')
hr()

hdu1 = fits.open('../data/pulsars/fermi_chandra.fits')[1].data
ra = hdu1['B_RA']
dec = hdu1['B_DEC']
download_data(hdu1 , 'pulsar_data_all/')



def download_data_pulsar(RA,DEC,NAME,root):
    count = 0 
    for ra,dec,name in zip(RA,DEC,NAME):
        name = name.replace(' ' , '_')
        hr()
        print('source Number :  ' , count )
        print(ra,dec)
        try:
            
            s = search_csc(
                radunit="arcsec", 
                columns=fields, 
                bands="broad",
                clobber="yes" ,
                radius=1, 
                pos= str(ra)+','+str(dec)  ,
                outfile='temp.csv')
            #sys('search_csc outfile=trial.csv radunit=arcsec columns="SOS,SOP,SOV , o.gti_obs m.flux_aper_b" bands=broad clobber=yes radius=1 pos="65.428058,32.907468"')
            #print(s)
            
            data = pd.read_csv('temp.csv', delimiter='\t' , comment='#')
            data = data[data['match_type']=='u          ']
            data = data[data['instrument']=='ACIS']
            data.index.name= 'index'
            #f_name_fits = data_fits['A_NAME']
            hr()
            print(name)
            hr()
            #name = name.replace(' ' , '_')+'.csv'
            data.to_csv(root+name)
            #------master info 

            print(data)
        except Exception as e:
            print(e)
            print('Error Occured , manual inspetion needed ')
        hr()
        count+=1

'''
pulsar_data = pd.read_csv('../data/pulsars/chandra_fermi_pulsars.csv')
RA = pulsar_data['ra']
DEC = pulsar_data['dec']
NAME = pulsar_data['name']
print(pulsar_data)
'''

'''
hdu1 = fits.open('../data/pulsars/atnf.fits')[1].data

ra = hdu1['RA']
dec = hdu1['DEC']
name = hdu['NAME']

download_data_pulsar(ra , dec , name , 'pulsar_data_all/fermi/')

'''