#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 19:00:34 2019

@author: ali
"""
import numpy as np
import math
import skimage
def im3dRot(img, boundaryType):

    h,w,d=img.shape
    if w>h:
        s=1.5*w
    else:
        s=1.5*h
    source=np.zeros((4,4))

    source[0,:]=[-w/2,h/2,0,1]
    source[1,:]=[w/2,h/2,0,1]
    source[2,:]=[w/2,-h/2,0,1]
    source[3,:]=[-w/2,-h/2,0,1]



    theta=np.pi/4 -np.random.rand(3)*np.pi/2
#    theta=[0,45*np.pi/180,0]
    R_x = np.array([[1,         0,                  0,                   0],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]), 0],
                    [0,         math.sin(theta[0]), math.cos(theta[0]),  0],
                    [1,         1,                  1,                   1]
                    ])
         
         
                     
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1]), 0],
                    [0,                     1,      0,                  0],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1]), 0],
                    [1,                     1,      1,                  1]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0,  0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0,  0],
                    [0,                     0,                      1,  0],
                    [1,                     1,                      1,  1]
                    ])
                     
                     
    R = np.matmul(R_z, np.matmul( R_y, R_x ))
    destination=np.matmul(R,np.transpose(source))
    destination=np.transpose(destination)  
    destination[:,2] += s
    destination[0,:] /= destination[0,2]/s
    destination[1,:] /= destination[1,2]/s
    destination[2,:] /= destination[2,2]/s
    destination[3,:] /= destination[3,2]/s
    tform=skimage.transform.ProjectiveTransform()
    tform.estimate(source[:,:2]+[w/2,h/2], destination[:,:2]+[w/2,h/2])


    Boundary=np.zeros((4,2))
    Boundary[0,:]=[0,0]
    Boundary[1,:]=[w,0]
    Boundary[2,:]=[w,h]
    Boundary[3,:]=[0,h]
    BoundaryTmp=tform(Boundary)
    invshift_x=int(-np.min(BoundaryTmp[:,0]))
    invshift_y=int(-np.min(BoundaryTmp[:,1]))                            
    rows=int(np.max(BoundaryTmp[:,1])-np.min(BoundaryTmp[:,1])+1)                            
    cols=int(np.max(BoundaryTmp[:,0])-np.min(BoundaryTmp[:,0])+1)  
    tf_shift_inv = skimage.transform.SimilarityTransform(translation=[invshift_x, invshift_y])
    tform=(tform + tf_shift_inv)
    ho=rows
    wo=cols
    imgF = skimage.transform.warp(img, tform.inverse,order=1,output_shape=(ho,wo),mode=boundaryType,cval=0, preserve_range=True)
#    imgF = skimage.transform.warp(img, tform.inverse,order=1,output_shape=(ho,wo),mode='constant',cval=0, preserve_range=True)
#    imgF = skimage.transform.warp(img, tform.inverse,order=1,output_shape=(ho,wo),mode='edge',cval=0, preserve_range=True)

    return np.asarray(imgF, dtype='uint8')


if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    im_path='/mnt/Dataset/textBoxesEval/icdar/2013/Challenge2_Test_Task12_Images/img_3.jpg'
    im = cv2.imread(im_path)
    
    plt.close('all')
    plt.figure()
    imgF=im3dRot(im, 'reflect')
    plt.subplot(1,2,1)
    plt.imshow(im)
    plt.subplot(1,2,2)
    plt.imshow(imgF)
    
    plt.figure()
    imgF=im3dRot(im, 'edge')
    plt.subplot(1,2,1)
    plt.imshow(im)
    plt.subplot(1,2,2)
    plt.imshow(imgF)
    
    plt.figure()
    imgF=im3dRot(im, 'constant')
    plt.subplot(1,2,1)
    plt.imshow(im)
    plt.subplot(1,2,2)
    plt.imshow(imgF)