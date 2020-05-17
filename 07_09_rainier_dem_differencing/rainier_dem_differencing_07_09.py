#! /usr/bin/env python
# based on https://github.com/geohackweek/raster/blob/27486d8b249ece05ad9bd862a9b243d8bdc5e304/_episodes/05-pygeotools_rainier/rainier_dem.py

from osgeo import gdal
import numpy as np
import datetime
import matplotlib.pyplot as plt
from pygeotools.lib import iolib, warplib, geolib, timelib, malib


#Function to generate a 2-panel plot for input arrays
def plot_panels(n, dem_list, clim=None, titles=None, cmap='inferno', label=None, overlay=None, fn=None):
    fig, axa = plt.subplots(1, n, sharex=True, sharey=True, figsize=(10,5))
    alpha = 1.0
    if (str(type(axa)) == "<class 'matplotlib.axes._subplots.AxesSubplot'>"):
        axa = np.array([axa])
    for n, ax in enumerate(axa):
        #Gray background
        ax.set_facecolor('0.5')
        #Force aspect ratio to match images
        ax.set(adjustable='box', aspect='equal')
        #Turn off axes labels/ticks
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if titles is not None:
            ax.set_title(titles[n])
        #Plot background shaded relief map
        if overlay is not None:
            alpha = 0.7
            axa[n].imshow(overlay[n], cmap='gray', clim=(1,255)) 
    #Plot each array 
    im_list = [axa[i].imshow(dem_list[i], clim=clim, cmap=cmap, alpha=alpha) for i in range(len(dem_list))]
    fig.tight_layout()
    fig.colorbar(im_list[0], ax=axa.ravel().tolist(), label=label, extend='both', shrink=0.5)
    if fn is not None:
        fig.savefig(fn, bbox_inches='tight', pad_inches=0, dpi=150)


! ./Users/elischwat/Documents/UW/pygeotools/pygeotools/ogr_merge.sh

#Input DEM filenames
dem2007_fn = 'rainier_dem_differencing/07_joined.tif'
dem2009_fn = 'rainier_dem_differencing/09_joined.tif'
dem_fn_list = [dem2007_fn,dem2009_fn]

#This will return warped, in-memory GDAL dataset objects
#Can also resample all inputs to a lower resolution (res=256)
ds_list = warplib.memwarp_multi_fn(dem_fn_list, extent='intersection', res='min')

#Load datasets to NumPy arrays
dem_2007, dem_2009 = [iolib.ds_getma(i) for i in ds_list]
dem_list = [dem_2007, dem_2009]
#dem_list = [iolib.ds_getma(i) for i in ds_list]

import matplotlib
matplotlib.pyplot.imshow(dem_2007)

matplotlib.pyplot.imshow(dem_2009)

titles = ['2007', '2009']
clim = malib.calcperc(dem_list[0], (2,98))
plot_panels(2, dem_list, clim, titles, 'inferno', 'Elevation (m WGS84)', fn='dem.png')

# Its possible sometimes to get date times from TIFFS but apparently these DEMS don't have any datetimes assigned.

# We can get the date times from the original *(unmerged) files though.

[timelib.fn_getdatetime(fn) for fn in 
 [
    '/Users/elischwat/Downloads/datasetsA/lewis_2009/dtm/lewis_2009_dtm_44.tif',
    '/Users/elischwat/Downloads/datasetsA/rainier_2007/rainier_2007_dtm_12.tif',
    '/Users/elischwat/Downloads/datasetsA/rainier_2007/rainier_2007_dtm_13.tif',
    '/Users/elischwat/Downloads/datasetsA/rainier_2007/rainier_2007_dtm_14.tif',
    '/Users/elischwat/Downloads/datasetsA/rainier_2007/rainier_2007_dtm_7.tif',
    '/Users/elischwat/Downloads/datasetsA/rainier_2007/rainier_2007_dtm_8.tif',
    '/Users/elischwat/Downloads/datasetsA/rainier_2007/rainier_2007_dtm_9.tif'
 ]]

#Extract timestamps from filenames
t_list = np.array([
    datetime.datetime(2007, 4, 22, 0, 0),
    datetime.datetime(2009, 4, 22, 0, 0)])
#Compute time differences, convert decimal years
dt_list = [timelib.timedelta2decyear(d) for d in np.diff(t_list)]

#Calculate elevation difference for each time period 
dh_list = [dem_2009 - dem_2007]
titles = ['2007 to 2009 (%0.1f yr)' % dt_list[0]]
plot_panels(1, dh_list, (-30, 30), titles, 'RdBu', 'Elevation Change (m)', fn='dem_dh.png')

#Calculate annual rate of change
dhdt_list = np.ma.array(dh_list)/np.array(dt_list)[:,np.newaxis,np.newaxis]
plot_panels(1,dhdt_list, (-3, 3), titles, 'RdBu', 'Elevation Change Rate (m/yr)', fn='dem_dhdt.png')

sns.distplot(dhdt_list.compressed())

# Remove glacier areas by gettin a shapefile with glaciers, creating a mask from it, and inverting the mask

# +
#Let's clip our map to the glaciers using polygons from the Randolph Glacier Inventory (RGI)
shp_fn = '/Users/elischwat/data/rgi60/regions/02_rgi60_WesternCanadaUS.shp'
#Create binary mask from polygon shapefile to match our warped raster datasets
shp_mask = geolib.shp2array(shp_fn, ds_list[0])

# REVERSE THE MASK because we want to focus on NOT glaciers
shp_mask = ~shp_mask

#Now apply the mask to each array 
dhdt_list_shpclip = [np.ma.array(dhdt, mask=shp_mask) for dhdt in dhdt_list]
plot_panels(1, dhdt_list_shpclip, (-3, 3), titles, 'RdBu', 'Elevation Change Rate (m/yr)', fn='dem_dhdt_shpclip.png')
# -

# To integrate the changes over all pixels, which are 1 x 1 m, 
# simply sum the data to estimate the net volumetric change.
#
# In m^3:

dhdt_list_shpclip[0].compressed().sum()

# In km^3

-8531550.95917161 / 1000**3







#That looks pretty good, but context would be nice.
#Let's generate some shaded relief basemaps using gdaldem API functionality
dem_2007_hs_ds = gdal.DEMProcessing('', ds_list[0], 'hillshade', format='MEM')
dem_2007_hs = iolib.ds_getma(dem_2007_hs_ds)
dem_2009_hs_ds = gdal.DEMProcessing('', ds_list[1], 'hillshade', format='MEM')
dem_2009_hs = iolib.ds_getma(dem_2009_hs_ds)
hs_list = [dem_2007_hs, dem_2009_hs]

#Plot our clipped rates over shaded relief maps
titles = ['With 2007 hillshade', 'With 2009 hillshade']
plot_panels(2, [dhdt_list_shpclip[0], dhdt_list_shpclip[0]], (-2, 2), titles, 
            'RdBu', 'Elevation Change Rate (m/yr)', overlay=hs_list, fn='dem_dhdt_shpclip_hs.png')





#OK, so we have elevation change, what about volume and mass change during different periods? 
#Extract x and y pixel resolution (m) from geotransform
gt = ds_list[0].GetGeoTransform()
px_res = (gt[1], -gt[5])
#Calculate pixel area in m^2
px_area = px_res[0]*px_res[0]
dhdt_list_shpclip = np.ma.array(dhdt_list_shpclip).reshape(len(dhdt_list_shpclip), dhdt_list_shpclip[0].shape[0]*dhdt_list_shpclip[1].shape[1])
#Now, lets multiple pixel area by the observed elevation change for all valid pixels over glaciers
dhdt_mean = dhdt_list_shpclip.mean(axis=1)
#Compute area in km^2
area_total = px_area * dhdt_list_shpclip.count(axis=1) / 1E6
#Volume change rate in km^3/yr
vol_rate = dhdt_mean * area_total / 1E3
#Volume change in km^3
vol_total = vol_rate * dt_list 
#Assume intermediate density between ice and snow for volume change (Gt)
rho = 0.850
mass_rate = vol_rate * rho
mass_total = vol_total * rho

#Print some numbers (clean this up)
out = zip(titles, dhdt_mean, area_total, vol_rate, vol_total, mass_rate, mass_total)
for i in out:
    print(i[0])
    print('%0.2f m/yr mean elevation change rate' % i[1])
    print('%0.2f km^2 total area' % i[2])
    print('%0.2f km^3/yr mean volume change rate' % i[3])
    print('%0.2f km^3 total volume change' % i[4])
    print('%0.2f km^3/yr mean mass change rate' % i[5])
    print('%0.2f km^3 total mass change' % i[6])
    print('\n')

def plot_2dhist(ax, x, y, xlim, ylim, log=False):
    bins = (100, 100)
    common_mask = ~(malib.common_mask([x,y]))
    x = x[common_mask]
    y = y[common_mask]
    H, xedges, yedges = np.histogram2d(x,y,range=[xlim,ylim],bins=bins)
    H = np.rot90(H)
    H = np.flipud(H)
    Hmasked = np.ma.masked_where(H==0,H)
    Hmed_idx = np.ma.argmax(Hmasked, axis=0)
    ymax = (yedges[:-1]+np.diff(yedges))[Hmed_idx]
    #Hmasked = H
    H_clim = malib.calcperc(Hmasked, (2,98))
    if log:
        import matplotlib.colors as colors
        ax.pcolormesh(xedges,yedges,Hmasked,cmap='inferno',norm=colors.LogNorm(vmin=H_clim[0],vmax=H_clim[1]))
    else:
        ax.pcolormesh(xedges,yedges,Hmasked,cmap='inferno',vmin=H_clim[0],vmax=H_clim[1])
    ax.plot(xedges[:-1]+np.diff(xedges), ymax, color='dodgerblue',lw=1.0)

#Now, let's make some quick plots of elevation change vs. elevation for the two time periods
f, axa = plt.subplots(2, sharex=True, sharey=True)
dem_clim = (1000,4400)
dhdt_clim = (-3, 3)
plot_2dhist(axa[0], dem_list[0], dhdt_list_shpclip[0], dem_clim, dhdt_clim)
axa[0].set_title('1970 to 2008')
axa[0].set_ylabel('Elev. Change Rate (m/yr)')
axa[0].axhline(0,lw=0.5,ls='-',c='r',alpha=0.5)
plot_2dhist(axa[1], dem_list[1], dhdt_list_shpclip[1], dem_clim, dhdt_clim)
axa[1].set_title('2008 to 2015')
axa[1].set_ylabel('Elev. Change Rate (m/yr)')
axa[1].axhline(0,lw=0.5,ls='-',c='r',alpha=0.5)
axa[1].set_xlabel('Elevation (m WGS84)')
f.tight_layout()
f.savefig('dem_vs_dhdt_log.png', bbox_inches='tight', pad_inches=0, dpi=150)
plt.show()
