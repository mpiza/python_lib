# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:16:10 2020

@author: Apa
"""

import numpy as np
import matplotlib.pyplot as plt 
import nn_lib
import scipy as sp
import  imageio 
 
    
# the function we want to have (desired outcome)


# load the pixel image!
face = imageio.imread('imageio:chelsea.png')
pixel_image=np.transpose(face[:,:,0]) # have to transpose...
pixel_image=pixel_image[:,::-1] # and flip... to get the right view!
pixel_image-=pixel_image.min()
pixel_image=(pixel_image.astype(dtype='float'))/pixel_image.max() # normalize between 0 and 1!
Npixels=np.shape(pixel_image)[0] # assuming a square image!


# function to convert image to a 2D fnction
def myFunc(x0,x1, pixel_image, Npixels):
#    global pixel_image, Npixels
    # convert to integer coordinates (assuming input is 0..1)
    x0int=(x0*Npixels*0.9999).astype(dtype='int')
    x1int=(x1*Npixels*0.9999).astype(dtype='int')
    return(pixel_image[x0int,x1int]) # extract color values at these pixels    
    
    
# test plot image    
Npixels_test=194   
xrange=np.linspace(0,1,Npixels_test)
X0,X1=np.meshgrid(xrange,xrange)
fig=plt.figure(figsize=(10,10))
plt.imshow(myFunc(X0,X1,pixel_image, Npixels_test),interpolation='nearest',origin='lower')
plt.axis('off')
plt.colorbar()
plt.show()


def make_batch(batchsize,pixel_image, Npixels):
   

    inputs=np.random.uniform(low=0,high=1,size=[batchsize,2])
    targets=np.zeros([batchsize,1]) # must have right dimensions
    targets[:,0]=myFunc(inputs[:,0],inputs[:,1],pixel_image, Npixels)
    return(inputs,targets)

# set up all the weights and biases

LayerSizes=[2,150,150,100,1] # input-layer,hidden-1,hidden-2,...,output-layer
NumLayers = len(LayerSizes) - 1    # does not count input-layer (but does count output)

w_low = -0.1
w_high = +0.1
Weights=[np.random.uniform(low=w_low,high= w_high,size=[ LayerSizes[j],LayerSizes[j+1] ]) for j in range(NumLayers)]
Biases=[np.zeros(LayerSizes[j+1]) for j in range(NumLayers)]        
        
# set up all the helper variables

y_layer=[np.zeros(LayerSizes[j]) for j in range(NumLayers+1)]
df_layer=[np.zeros(LayerSizes[j+1]) for j in range(NumLayers)]
dw_layer=[np.zeros([LayerSizes[j],LayerSizes[j+1]]) for j in range(NumLayers)]
db_layer=[np.zeros(LayerSizes[j+1]) for j in range(NumLayers)]


eta=.5

batchsize = 1000  # number of points taken in  "image" 
batches=2000 # number of image samples
costs=np.zeros(batches)

for k in range(batches):
    y_in,y_target= make_batch(batchsize,pixel_image, Npixels)
    costs[k]=nn_lib.train_net(y_in,y_target,eta, y_layer, df_layer, Weights, Biases, NumLayers,dw_layer, db_layer, batches)
    if (k%10 == 0): 
        print(k, costs[k])
fig=plt.figure(figsize=(10,10))
plt.plot(np.sqrt(costs))
plt.show()


# a 'test' batch that includes all the points on the image grid
test_batchsize=np.shape(X0)[0]*np.shape(X0)[1]
testsample=np.zeros([test_batchsize,2])
testsample[:,0]=X0.flatten()
testsample[:,1]=X1.flatten()

# check the output of this net
testoutput=nn_lib.apply_net_simple(testsample,y_layer,Weights, Biases, NumLayers)

fig=plt.figure(figsize=(10,10))
myim=plt.imshow(np.reshape(testoutput,np.shape(X0)),origin='lower',interpolation='nearest',vmin=0.0,vmax=1.0)
plt.axis('off')
plt.colorbar()
plt.show()

