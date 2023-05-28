#!/usr/bin/env python
# coding: utf-8

# In[27]:


#base_dir = '/Users/jc/Downloads/S2B_MSIL1C_20230523T173909_N0509_R098_T13TGF_20230523T210702.SAFE/GRANULE/L1C_T13TGF_A032442_20230523T174707/IMG_DATA'

#base_dir = '/Users/jc/Downloads/S2B_MSIL2A_20230520T172859_N0509_R055_T14TKK_20230520T215118.SAFE/GRANULE/L2A_T14TKK_A032399_20230520T173827/IMG_DATA/R10m'

base_dir = \
'/Users/jc/Downloads/S2B_MSIL2A_20230520T172859_N0509_R055_T14TLK_20230520T215118.SAFE/\
GRANULE/L2A_T14TLK_A032399_20230520T173827/IMG_DATA/R10m'



# In[28]:


from glob import glob
import os

import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep

import rasterio as rio

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import plotly.graph_objects as go


# In[29]:


os.chdir(base_dir)
s2_sentinel_bands = glob("*B*.jp2")
print(s2_sentinel_bands)


# In[51]:


#src = rio.open("T13TGF_20230523T173909_B02.jp2")
src = rio.open(s2_sentinel_bands[2])


# In[52]:


from rasterio.plot import show
show(src)


# In[33]:


stacked_s2_sentinel = []
for img in s2_sentinel_bands:
    with rio.open(img, 'r') as f:
        stacked_s2_sentinel.append(f.read(1))
print(len(stacked_s2_sentinel))


# In[34]:


for i in range(len(stacked_s2_sentinel)):
    print(i, s2_sentinel_bands[i], stacked_s2_sentinel[i].shape)


# In[46]:


#stacked_s2_sentinel_img = np.stack((stacked_s2_sentinel[6], stacked_s2_sentinel[2], stacked_s2_sentinel[1]))
stacked_s2_sentinel_img = np.stack(stacked_s2_sentinel)

# RGB composite image with stretch-
ax = ep.plot_rgb(
        arr = stacked_s2_sentinel_img, rgb = (3,1,2),
        stretch = True, str_clip = 0.2,
        figsize =(20, 14)
        )

plt.savefig("test2.png")
plt.close()

#plt.show()


# In[44]:


# Normalize bands into 0.0 - 1.0 scale
def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)

def norm_mean(array, target_mean):
    return (target_mean / array.mean()) * array

blue = rio.open(s2_sentinel_bands[2])
green = rio.open(s2_sentinel_bands[1]) 
red = rio.open(s2_sentinel_bands[3]) 

#blue = rio.open('T13TGF_20230523T173909_B02.jp2')
#green = rio.open('T13TGF_20230523T173909_B03.jp2') 
#red = rio.open('T13TGF_20230523T173909_B04.jp2') 

cropx_min, cropx_max = 2000, 4000
cropy_min, cropy_max = 500, 2500
blue_img = norm_mean( normalize(blue.read(1)[cropx_min:cropx_max, cropy_min:cropy_max]), .5)
green_img = norm_mean( normalize(green.read(1)[cropx_min:cropx_max, cropy_min:cropy_max]), .5)
red_img = norm_mean( normalize(red.read(1)[cropx_min:cropx_max, cropy_min:cropy_max]), .5)
blue_img = np.clip(blue_img, 0, 1.0)
green_img = np.clip(green_img, 0, 1.0)
red_img = np.clip(red_img, 0, 1.0)

print(blue_img.min(), blue_img.max(), blue_img.mean())
print(green_img.min(), green_img.max(), green_img.mean())
print(red_img.min(), red_img.max(), red_img.mean())

print(blue.dtypes[0])

with rio.open('test3.tiff','w',driver='Gtiff', width=blue_img.shape[1], height=blue_img.shape[0], count=3, crs=blue.crs,transform=blue.transform, dtype='uint8') as rgb:
    rgb.write(255. * blue_img,3)
    rgb.write(255. * green_img,2) 
    rgb.write(255. * red_img,1) 
    rgb.close()
    


# In[25]:


from PIL import Image
im = Image.open('test3.tiff')
im.show()


# In[24]:


import matplotlib.pyplot as plt
I = plt.imread('test3.tiff')


# In[ ]:




