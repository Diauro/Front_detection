# Front_detection

This simple notebook shows how to extract the interface of a cell. According to disorder elastic system framework valuable information about the cell can be extracted by analyzing the behaviour of the interface of the cell It is therefore important to extract the interface correctly.

```python

from math import sqrt,pi
import numpy as np
from scipy  import optimize
import matplotlib.pyplot as plt
#import matplotlib.pylab as plt
#from skimage.io import imread
from skimage import filters
import scipy.ndimage
import pims
from skimage.morphology import watershed , closing, square,label
from skimage.filters import threshold_otsu, sobel
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d

frames = pims.ImageSequence('...*.tif', as_grey=True) 


```


The stack of images are loaded, one image is plotted as an example

```python
frame_no = 10

brightness_low = -30
brightness_up = 90
image = np.array(frames[10])



fig, ax = plt.subplots(1)
im = ax.imshow(image,  vmin= brightness_low , vmax= brightness_up, cmap=plt.cm.jet)
ax.xaxis.tick_top()
fig.colorbar(im, ax=ax)
plt.savefig('out_0.png', dpi=600)  
plt.show()



```


![Cell image](/images/out_0.png)

# Cell front detection using Skimage


```python
min_max = MinMaxScaler(copy=True, feature_range=(-1, 1))

image_scaled = min_max.fit_transform(image)
edges = sobel(image_scaled)
foreground, background = 1, 2
markers = np.zeros_like(image)
markers[image_scaled < 1.2] = foreground
markers[image_scaled > 2] = background


segmentation = watershed(image,markers)
seg1 = label(segmentation == foreground)


edges_ = np.copy(edges)
edges_[edges_ <0.35] = 0
edges_cut = edges_[11:112, 105:142]
edges_flip = np.flip(edges_cut, axis=1)

plt.figure(2)
plt.imshow(edges_flip, cmap = 'jet')
plt.savefig('out_1.png', dpi=600)  


```

The image for convenience is flipped (the result is the same) and cropped 

![Cell front](/images/out_1.png)

# For the particular application we need a univaluated border
We plot the border together with the original image to see the result


```python

last_element = [] 

for k in range(len(edges_flip)):
        
    edges_cut_list = edges_flip[k].tolist()
    last_element_arr = next(iter([x for x in edges_cut_list if x > 0 ]))
    temp = edges_cut_list.index(last_element_arr)
    last_element.append(temp)


    

max_x = len(edges_cut_list)
max_y = len(edges_cut)
    
  
plt.figure(3)
img = np.zeros((max_y ,max_x +20))        

img[0,0:len(edges_cut_list)] = edges_cut[0]
plt.imshow( edges_flip,cmap = 'jet')       
plt.title( 'Image + edge')
plt.colorbar()
x_axis = np.arange(0,len(last_element),1)
plt.plot(last_element,x_axis, 'r')
plt.savefig('out_2.png', dpi = 600)

```
![Cell front](/images/out_2.png)


# To obtain the interface in our case we fit it to a circumnference and do the element-wise substraction. 


![Cell front](/images/out_4.png)
