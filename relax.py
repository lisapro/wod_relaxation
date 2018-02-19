'''
Created on 31. jan. 2018

@author: ELP
'''
from statsmodels.discrete.tests.test_constrained import junk
from matplotlib import gridspec as gs
from scipy.interpolate import UnivariateSpline  
from scipy import interpolate 
'''
Script for making input data for relaxation for the BROM model 
Raw Source data - Geographically sorted data from World Ocean Database 
https://www.nodc.noaa.gov/OC5/WOD/datageo.html
.nc file created in Ocean Data View program

1. Read the nc file, needed variable. 
https://odv.awi.de/
'''

from netCDF4 import Dataset,num2date, date2num
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap
import numpy as np       
from datetime import datetime  
import pandas as pd
import xarray as xr

def map_of_stations(ncfile,f_lat,f_lon,title,res):
    fh = Dataset(ncfile, mode='r')
    
    lat_odv = fh.variables['latitude'][:]
    lon_odv = fh.variables['longitude'][:]   
        
    fig  = plt.figure(figsize=(10,6), dpi=100 )
    fh.close()
    gs = gridspec.GridSpec(1,2)
    ax = fig.add_subplot(gs[0]) 
    ax01 = fig.add_subplot(gs[1])
    n = 0.5
    lat_min = (lat_odv.min()-n)
    lat_max = (lat_odv.max()+n)
    lon_min = (lon_odv.min()-n)
    lon_max = (lon_odv.max()+n)
          
    map = Basemap(#width=150000,height=990000,
                resolution= res,projection='merc',\
                lat_0=70,lon_0=14,
                llcrnrlon = lon_min , llcrnrlat = lat_min,
                urcrnrlon = lon_max , urcrnrlat = lat_max,ax = ax)
    
    map2 = Basemap(#width=150000,height=990000,
                resolution='i',projection='stere',\
                lat_0=70,lon_0=14,
                llcrnrlon = 5 ,llcrnrlat = 54 ,
                urcrnrlon = 27 ,urcrnrlat =70,ax = ax01 )
        
    map.drawparallels(np.arange(50.,70.,0.5),labels = [1,1,1,1])
    map.drawmeridians(np.arange(-180.,180.,0.5),labels = [1,1,1,1])
    
    map2.drawparallels(np.arange(50.,70.,5),labels = [1,1,1,1])
    map2.drawmeridians(np.arange(-180.,180.,5),labels = [1,1,1,1])
    
    cont_col = '#bdad95'
    water_col = '#cae2f7'
    map.drawmapboundary(fill_color = water_col)
    map.fillcontinents(color = cont_col)
    map.drawcoastlines(linewidth=0.5)
    
    map2.drawmapboundary(fill_color = water_col)
    map2.fillcontinents(color = cont_col)
    map2.drawcoastlines(linewidth=0.5)
    
    x, y = map(lon_odv, lat_odv)   
    f_lat, f_lon = f_lat, f_lon 
    
    xf1,yf1 = map(f_lon + 0.2,f_lat + 0.1)
    xf2,yf2 = map2(f_lon + 5,f_lat + 1)
    
    map.scatter(x,y,3,marker='o',color='#0b956e',
                alpha = 0.7, edgecolors = '#054a37')

    map2.scatter(f_lon,f_lat,marker='*', latlon = True,
                  s= 250,color='#eb123a',
                  zorder = 10,edgecolors = "#8d0a22")
        
    map.scatter(f_lon,f_lat,marker='*', latlon = True,
                  s= 250,color='#eb123a',
                  zorder = 10,edgecolors = "#8d0a22") 
           
    ax.text(xf1,yf1, title, size=15, 
            va="center", ha="center", rotation=0,
        bbox=dict(boxstyle="Round", alpha=0.7,facecolor = '#cae2f7'))    
         
    ax01.text(xf2,yf2, title, size=15, 
            va="center", ha="center", rotation=0,
        bbox=dict(boxstyle="Round", alpha=0.7,facecolor = '#cae2f7'))

    #fig.suptitle('Data for creation arrays for Relaxation to climatic data ')
    plt.show()
    

def choose_month(ds,m_num,var,clima_var_m,levels,double_int = False):
    # get 1 month data 
    month_ds = ds.where(ds.date_time.dt.month == m_num, drop=True)
    # group by depth, find mean
    month_mean = month_ds.groupby(month_ds['var1']).mean()
    month_depth = month_mean.var1.values.copy()
    month_df = month_mean.to_dataframe()
    nonnan = len(month_df[var]) - np.count_nonzero(np.isnan(month_df[var])) 
    #print (nonnan)
    if nonnan > 5 : #print (nonnan,month_df[var])       
         
        # interpolate nans 
        var = (month_df[var].interpolate(kind = 'nearest'))
        if np.isnan(var[0]) :
            var[0] = var[1]
        f = interpolate.UnivariateSpline(month_depth,var,k=3)
        var_m = f(levels)
        if double_int == True:
            for n in range(0,3):
                f = interpolate.UnivariateSpline(levels,var_m,k=3)
                var_m = f(levels)
    else: 
  
        var_m = clima_var_m 
    return var_m,month_ds 
       
def create_relax_array(ncfile,varname,pl,save,double_int = False):
    
    funcs = {'Oxygen':'var4', 'Temperature': 'var2',
             'si':'var6','alk': 'var12','chl': 'var10',
             'po4':'var5', 'no3':'var7','pH': 'var9'}  
    var_from_odv = funcs[varname] 
          
    # read ncfile as xarray
    ds = xr.open_dataset(ncfile,drop_variables= ('metavar1', 'metavar1_QC',
       'metavar2', 'metavar2_QC', 'metavar3',
       'metavar3_QC', 'longitude', 'longitude_QC', 'latitude', 'latitude_QC',
       'metavar4', 'metavar4_QC', 'metavar5', 'metavar5_QC', 'metavar6',
       'metavar6_QC', 'metavar7', 'metavar7_QC', 'metavar8', 'metavar8_QC',
        'date_time_QC',  'var3_QC', 'var1_QC', 'var2_QC',
       'var4_QC', 'var5_QC',  'var6_QC',  'var7_QC',
       'var8', 'var8_QC', 'var9_QC',  'var10_QC', 'var11',
       'var11_QC',  'var12_QC', 'var13', 'var13_QC', 'var14',
       'var14_QC', 'var15', 'var15_QC', 'var16', 'var16_QC', 'var17',
       'var17_QC', 'var18', 'var18_QC', 'var19', 'var19_QC', 'var20',
       'var20_QC', 'var21', 'var21_QC', 'var22', 'var22_QC', 'var23',
       'var23_QC', 'var24', 'var24_QC', 'var25', 'var25_QC', 'var26',
       'var26_QC', 'var27', 'var27_QC'))
    
    max_depth = np.int(np.max(ds.var1))
    # usually we don't need the data from all depths 
    levels = np.arange(0,350,1) 
           
    # get only data from 1960 and later 
    ds = ds.where(ds.date_time.dt.year > 1960, drop=True)
    
    # group by depth and find mean for each depth 
    clima_mean = ds.groupby(ds['var1']).mean()
    clima_depth = clima_mean.var1.values.copy()
    clima_df = clima_mean.to_dataframe()
    
    # interpolate nans 
    clima_var = (clima_df[var_from_odv].interpolate(kind = 'nearest')) 
    
    # interpolations does not work if 1st variable is nan   
    if np.isnan(clima_var[0]) :
        clima_var[0] = clima_var[1]
        
    # interpolate values to standart levels    
    f = interpolate.UnivariateSpline(clima_depth,clima_var,k=3)
    clima_var_m = f(levels)
    if double_int == True:
        for n in range(0,3):
            f1 = interpolate.UnivariateSpline(levels,clima_var_m,k=3)
            clima_var_m = f1(levels)  
       
    means = [] 
    depths = [] 
    days = []
    nday = 0 
    numdays = {1:31,2:28,3:31,4:30,5:31,
                 6:30,7:31,8:31,9:30,10:31,
             11:30,12:31} 

    for n in range(1,13):
        #print (n)
        m = choose_month(ds,n,var_from_odv,clima_var_m,levels,double_int)
        var_m = m[0] 
        var =  m[1] 
        maxday = numdays[n]

                   
        for s in range(0,maxday): 
            nday += 1  
            means.append(var_m)
            depths.append(levels)  
            days.append(np.repeat(nday, len(levels)))
            
        
        
    if varname == 'Oxygen':
        means = np.array(means)*44.6     
        clima_var_m = np.array(clima_var_m)*44.6  
    elif varname == 'alk': 
        means = np.array(means)*1000     
        clima_var_m = np.array(clima_var_m)*1000            
    if pl == True:
        plt.clf()
        plt.title(varname)  
        plt.plot(clima_var_m,levels,'k',zorder = 10)
        plt.plot(np.array(means).T,np.array(depths).T,
                 '-',zorder = 8,alpha = 0.7)
        plt.ylim(450,0)
        #plt.show() 
        plt.savefig('data/{}_relax_{}.png'.format(str(ncfile)[:-3],varname))      
    means = np.ravel((np.array(means).T))
    levels = np.ravel((np.array(depths).T))
    days = np.ravel((np.array(days).T))
    arr = np.vstack((days,levels,means)).T

    if save == True:    
        np.savetxt('data/{}_relax_{}.txt'.format(str(ncfile)[:-3],varname), (arr), delimiter=' ')           

def time_to_run():    
    import timeit
    start = timeit.default_timer()
    stop = timeit.default_timer()
    print ('Seconds',stop - start) 

                                                   
#map_of_stations('jossingfjorden-wod.nc',58.3, 6.3, r"Jossingfjorden",'l')
#time_distribution('jossingfjorden-wod.nc',r"Jossingfjorden")

for var in ('alk','chl','po4','si', 'no3','Oxygen','pH'):   
    create_relax_array('jossingfjorden-wod.nc',var,
                       pl = True, save = True,double_int = True) 

#def call_fucns(var,nc):
#create_relax_array('jossingfjorden-wod.nc','si',pl = True, save = False,double_int = True) #
#create_relax_array('jossingfjorden-wod.nc','pH',pl = True, save = True,double_int = False)    
#сreate_relax_array('jossingfjorden-wod.nc','Oxygen',pl = True, save = True,double_int = False)