
import matplotlib.pyplot as plt
from PIL import Image
 
import numpy as np
import os
import cv2
 
def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask




def mask_and_crop(im, cropx, cropy):
    w,h = im.size
    mask = create_circular_mask(h, w, center=None, radius=324)
    plt.imshow( np.array(im)  )
    # plt.show()
    img = np.array(im) 
    # apply mask
    img[~mask] = 0
    # plt.imshow( np.array(img)  )
    # plt.show()
    yc = int(img.shape[0]/2)
    xc = int(img.shape[1]/2)
    oim = img[ cropx : -cropx ,cropy : -cropy , :]
    # plt.imshow(oim)
    # plt.show()
    return oim


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam
