'''
Created on 11. jun. 2018

@author: ELP
'''

import xarray as xr
import numpy as np   
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import  relax
from relax import create_relax_array 
from relax import create_ranges 

def call_arctic() :
    
    # open the file to get levels from the model , interpolation further will be to these levels 
    dss = xr.open_dataset(r'C:\Users\elp\OneDrive\Python_workspace\arctic2030\Data\ROMS_Laptev_Sea_NETCDF3_CLASSIC_east_each_day.nc')
    levels = sorted(dss.depth.values)
    
    fig  = plt.figure(figsize=(5,8), dpi=100 )
    
    gs = gridspec.GridSpec(3,2)
    gs.update(hspace=0.3,top = 0.95,bottom = 0.05)
    ax = fig.add_subplot(gs[0]) 
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])     
    ax3 = fig.add_subplot(gs[3])     
    ax4 = fig.add_subplot(gs[4])     
    ax5 = fig.add_subplot(gs[5])     
   
    ncfile = 'data_from_WOD_COLLECTION_Laptev.nc'    
    create_relax_array(ncfile,'Oxygen',
                       True, True, levels, ax,1,double_int = True, only_clima_mean = True) # Arctic
    ax.set_title(r'O$_2\ \mu M$')  
    create_relax_array(ncfile,'po4',
                       True, True, levels,ax1,3, double_int = True, only_clima_mean =True) # Arctic,
    ax1.set_title(r'PO$_4\ \mu M$') 
    create_relax_array(ncfile,'si',
                       True, True, levels,ax2,10, double_int = True, only_clima_mean = True) # Arctic
    ax2.set_title(r'Si $\mu M$') 
    create_relax_array(ncfile,'no3',
                       True, True, levels,ax3, 3, double_int = True, only_clima_mean = True) # Arctic
    ax3.set_title(r'NO$_3\ \mu M$') 
    create_relax_array(ncfile,'alk',
                       True, True, levels,ax4, 1, double_int = True, only_clima_mean = True) # Arctic
    ax4.set_title(r'Alkalinity $\mu M$') 
    create_relax_array(ncfile,'pH',
                       True, True, levels,ax5, 3, double_int = True, only_clima_mean = True) # Arctic    
    ax5.set_title(r'pH') 
    #plt.show()
    for axis in (ax,ax1,ax2,ax3,ax4,ax5):
        axis.set_ylim(90,0)
    plt.savefig('data/{}/arctic_wod.png'.format(str(ncfile)[:-3])) 

def call_smeaheia_ranges() :
    ncfile = 'data_from_WOD_COLLECTION_1500-1600_Smeaheia.nc'
    # from BROM
    #levels = [0.00, 1.95, 3.83, 5.69, 7.53, 9.40,11.34,13.40,
    #          15.65,18.17,21.05,24.38,28.26,32.79,38.04,44.10,
    #          51.00,58.74,67.27,76.49,86.19,96.08,105.74,114.60,121.94]
    levels = np.arange(0,350,5)
    save = False #True
     
    fig  = plt.figure(figsize=(6,7), dpi=100 )    
    gs = gridspec.GridSpec(2,3)
    gs.update(left = 0.09, right = 0.97,hspace=0.2,
              wspace=0.3,bottom = 0.05,top = 0.96)
    ax = fig.add_subplot(gs[0]) 
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])     
    ax3 = fig.add_subplot(gs[3])     
    ax4 = fig.add_subplot(gs[4])     
    ax5 = fig.add_subplot(gs[5])    
        
    create_ranges(ncfile,'Oxygen', ax) 
    ax.set_title(r'O$_2\ \mu M$')   
    
    create_ranges(ncfile,'po4',ax1)
    ax1.set_title(r'PO$_4\ \mu M$') 
    
    create_ranges(ncfile,'si',ax2) 
    ax2.set_title(r'Si $ \mu M$') 
    ax2.set_xlim(0,12)
    create_ranges(ncfile,'no3',ax3)
    ax3.set_title(r'NO$_3\ \mu M$') 
    
    create_ranges(ncfile,'alk',ax5) 
    ax5.set_title(r'Alkalinity $\mu M$') 
    #ax5.set_xlim(1900,2600)
    
    create_ranges(ncfile,'pH',ax4) 
    ax4.set_title(r'pH') 
    
    for axis in (ax,ax1,ax2,ax3,ax4,ax5): #,ax4,ax5):
        axis.set_ylim(360,0)

    fig.savefig('data/{}/smeaheia_ranges.png'.format(str(ncfile)[:-3]),transparent = False)
    #plt.show()   
    plt.clf()
    
def call_smeaheia_relax() :
    ncfile = 'data_from_WOD_COLLECTION_1500-1600_Smeaheia.nc'
    levels = np.arange(0,300,2)
    save = False
     
    fig  = plt.figure(figsize=(10,6), dpi=100 )    
    gs = gridspec.GridSpec(2,3)
    gs.update(left = 0.04, right = 0.97,hspace=0.3)
    ax = fig.add_subplot(gs[0]) 
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])     
    ax3 = fig.add_subplot(gs[3])     
    ax4 = fig.add_subplot(gs[4])     
    ax5 = fig.add_subplot(gs[5])     
        
    create_relax_array(ncfile,'Oxygen',
                       True, save, levels, ax,1,double_int = True, only_clima_mean = False) 
    ax.set_title(r'O$_2\ \mu M$')   
    #ax.legend(['mean','1','2','3','4','5','6','7','8','9','10','11','12'])
    #ax.legend()
     
    create_relax_array(ncfile,'po4',
                       True, save, levels,ax1,1, double_int = True,only_clima_mean =False)
    ax1.set_title(r'PO$_4\ \mu M$') 
    create_relax_array(ncfile,'si',
                       True, save, levels,ax2,3 , double_int = True,only_clima_mean = False) 
    ax2.set_title(r'Si $ \mu M$') 
    ax2.set_xlim(0,12)
    create_relax_array(ncfile,'no3',
                       True, save, levels,ax3, 6, double_int = True,only_clima_mean = False)
    ax3.set_title(r'NO$_3\ \mu M$') 
    create_relax_array(ncfile,'alk',
                       True, save, levels,ax4, 1, double_int = True,only_clima_mean = False) 
    ax4.set_title(r'Alkalinity $\mu M$') 
    ax4.set_xlim(1900,2600)
    create_relax_array(ncfile,'pH',
                       True, save, levels,ax5, 1, double_int = True,only_clima_mean = False) 
    ax5.set_title(r'pH') 

    for axis in (ax,ax1,ax2,ax3,ax4,ax5):
        axis.set_ylim(450,0)
    fig.savefig('data/{}/smeaheia_wod.png'.format(str(ncfile)[:-3]),transparent = False)
    #plt.show()    
    plt.clf()
        
def call_jossingfjorden():
    levels = (0,5,10,20,30,50,75,100,125,150,165) #
    fig  = plt.figure(figsize=(10,6), dpi=100 )

    ncfile = 'jossingfjorden-wod.nc'
    gs = gridspec.GridSpec(3,3)
    gs.update(hspace=0.3)
    ax = fig.add_subplot(gs[0]) 
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])     
    ax3 = fig.add_subplot(gs[3])     
    ax4 = fig.add_subplot(gs[4])     
    ax5 = fig.add_subplot(gs[5])
    ax6 = fig.add_subplot(gs[6])    
    
    create_relax_array(ncfile,'Oxygen',
                       True, True, levels, ax,1,double_int = True, only_clima_mean = False) 
    ax.set_title(r'O$_2\ \mu M$')     
    create_relax_array(ncfile,'po4',
                       True, True, levels,ax1,1, double_int = True,only_clima_mean =False)
    ax1.set_title(r'PO$_4\ \mu M$') 
    create_relax_array(ncfile,'si',
                       True, True, levels,ax2,1 , double_int = True,only_clima_mean = False) 
    ax2.set_title(r'Si $ \mu M$') 
    ax2.set_xlim(0,12)
    create_relax_array(ncfile,'no3',
                       True, True, levels,ax3,1, double_int = True,only_clima_mean = False)
    ax3.set_title(r'NO$_3\ \mu M$') 
    create_relax_array(ncfile,'alk',
                       True, True, levels,ax4, 1, double_int = True,only_clima_mean = False) 
    ax4.set_title(r'Alkalinity $\mu M$') 
     
    create_relax_array(ncfile,'pH',
                           True, True, levels,ax5,1, double_int = True,only_clima_mean = False)  
    ax5.set_title(r'pH') 
    
    create_relax_array(ncfile,'chl',
                           True, True, levels,ax6,1, double_int = False ,only_clima_mean = False)     
    ax6.set_title(r'Chlorophyll')
    ax6.set_xlim(0,10)
    for axis in (ax,ax1,ax2,ax3,ax4,ax5,ax6):
        axis.set_ylim(170,0)        
    #plt.show()  
    plt.savefig('data/{}/jossingsfjorden_wod.png'.format(str(ncfile)[:-3])) 
    
def call_osterfjorden():

    ncfile = 'modest-wod.nc'
    
    #map_of_stations(ncfile,60.5,5.3,'Osterfjorden','i')
    levels = np.arange(0,600,2) #(0,5,10,20,30,50,75,100,125,150,165) #
    fig  = plt.figure(figsize=(10,6), dpi=100 )

    gs = gridspec.GridSpec(3,2)
    gs.update(hspace=0.3)
    ax = fig.add_subplot(gs[0]) 
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])     
    ax3 = fig.add_subplot(gs[3])     
    ax4 = fig.add_subplot(gs[4])     
    ax5 = fig.add_subplot(gs[5])
    #ax6 = fig.add_subplot(gs[6])    
    
    create_relax_array(ncfile,'Oxygen',
                       True, True, levels, ax,1,double_int = True, only_clima_mean = False) 
    ax.set_title(r'O$_2\ \mu M$')     
    create_relax_array(ncfile,'po4',
                       True, True, levels,ax1,1, double_int = True,only_clima_mean =False)
    ax1.set_title(r'PO$_4\ \mu M$') 
    create_relax_array(ncfile,'si',
                       True, True, levels,ax2,2 , double_int = True,only_clima_mean = False) 
    ax2.set_title(r'Si $ \mu M$') 
    ax2.set_xlim(0,12)
    
    create_relax_array(ncfile,'no3',
                       True, True, levels,ax3,4, double_int = True,only_clima_mean = False)
    
    ax3.set_title(r'NO$_3\ \mu M$') 
    
    create_relax_array(ncfile,'alk',
                       True, True, levels,ax4, 1, double_int = False,only_clima_mean = False) 
    
    ax4.set_title(r'Alkalinity $\mu M$') 
     
    create_relax_array(ncfile,'pH',
                           True, True, levels,ax5,1, double_int = True,only_clima_mean = False)  
    ax5.set_title(r'pH') 
    
    #create_relax_array(ncfile,'chl',
    #                       True, True, levels,ax6,1, double_int = True ,only_clima_mean = False)     
    #ax6.set_title(r'Chlorophyll')
    #ax6.set_xlim(0,10)
    for axis in (ax,ax1,ax2,ax3,ax4,ax5):
        axis.set_ylim(600,0)        
    #plt.show()  
    plt.savefig('data/{}/Osterfjorden_wod.png'.format(str(ncfile)[:-3]))    
    plt.clf()


if __name__ == '__main__': 
    #call_jossingfjorden()    
    #call_smeaheia_2d()
    call_smeaheia_ranges()
    call_smeaheia_relax()
    #call_arctic()     
    #call_osterfjorden() 
   