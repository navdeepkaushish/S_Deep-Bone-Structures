# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:14:38 2024

@author: Navdeep Kumar
"""

import os
import glob
import numpy as np
import pandas as pd
import cv2 as cv
import albumentations as A
from sklearn.model_selection import train_test_split
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from shapely.geometry import Polygon
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models
from unet_model import UNet
from f_unet_model import f_UNet
from utils import *
from loss_functions import *

batch_size = 2
N = 13
root = 'D:/missing_structures/dataset/full_masks'
str_names = ['p', ['op2', 'op1'], ['oc2', 'oc1'], 'n', ['m2', 'm1'], ['hm2', 'hm1'], ['en2', 'en1'], ['d2', 'd1'], ['cl2', 'cl1'], 
             ['ch2','ch1'], ['cb2', 'cb1'], ['br2b', 'br2a'], ['br1b', 'br1a']]
image_ids_paths = glob.glob(os.path.join(root, '*'))
image_ids = os.listdir(root)
idxs = [i for i in range(len(image_ids))]

seed=22    
train_idxs, test_idxs = train_test_split(idxs, test_size=0.15, random_state=seed)
tr_idxs, val_idxs = train_test_split(train_idxs, test_size=0.15, random_state=seed)

train_ids = [image_ids_paths[i] for i in tr_idxs]
val_ids = [image_ids_paths[i] for i in val_idxs]
test_ids = [image_ids_paths[i] for i in test_idxs]


threshold = 0.45

f_model = f_UNet(3,1)
f_model_path = "D:/missing_structures/v_fullmask/saved_models/new/best_model.pt"
f_checkpoint = torch.load(f_model_path, map_location=torch.device('cpu'))
f_model.load_state_dict(f_checkpoint['state_dict'])
f_model.to(device)
f_model.eval()
#======== model for cropped images ==========================================
cr_model = UNet(3,N)
cr_model_path = "D:/missing_structures/saved_models/V_cropped/focal/best_model.pt"
cr_checkpoint = torch.load(cr_model_path, map_location=torch.device('cpu'))
cr_model.load_state_dict(cr_checkpoint['state_dict'])
cr_model.to(device)
cr_model.eval()

final_contours = {}
for i in range(len(test_ids)):
     img = cv.imread(os.path.join(test_ids[i], 'image.png'))  
     img = cv.cvtColor(img, cv.COLOR_BGR2RGB) 
     img_cp = img.copy()
     ids = os.path.basename(test_ids[i])
     H, W = img.shape[:2]
     
     re_img, _ = rescale_pad(img, None, 512)
     image = re_img/255.0
     image = torch.from_numpy(image)
     image = image.permute(2,0,1)
     image = torch.unsqueeze(image, 0)
     image = image.type(torch.float32)
     image = image.to(device)
     out = f_model(image)
     logits = torch.squeeze(out)
     r_logits = logits.cpu().detach().numpy()
     pred = torch.sigmoid(logits)
     pred = pred.cpu()
     pred = pred.detach().numpy()
     pred[pred < threshold] = 0
     pred[pred >= threshold] = 1
     
     u_mask = up_mask(pred, H, W)
     u_mask = u_mask.astype(np.uint8)
     
     smooth = cv.GaussianBlur(u_mask, (5, 5), 0)
     
     pts = crop_pts(smooth) #extract coordinates for cropping
     
     cr_img = img[pts[1]:pts[3],pts[0]:pts[2]]
     h, w = cr_img.shape[:2]
     re_img, _ = rescale_pad(cr_img, None, 512)
     image = re_img/255.0
     image = torch.from_numpy(image)
     image = image.permute(2,0,1)
     image = torch.unsqueeze(image, 0)
     image = image.type(torch.float32)
     image = image.to(device)
     pred = cr_model(image)
     pred = torch.squeeze(pred)
     pred = pred.permute(1,2,0)
     r_logits = pred.cpu().detach().numpy()
     pred = torch.sigmoid(pred)
     pred = pred.cpu()
     pred = pred.detach().numpy()
     pred[pred < threshold] = 0
     pred[pred >= threshold] = 1
     
     u_mask = up_mask(pred, h, w)
     u_mask = u_mask.astype(np.uint8)
     smooth = cv.GaussianBlur(u_mask, (5, 5), 0)
     
     cr_mask = cr_up(smooth, pts, H, W)
     
     poly_dict = {}  #to store polygons in a dictionary
     for j in range(N):
         pred = cr_mask[:,:,j].astype(np.uint8)
         
         ret, thresh = cv.threshold(pred, 1, 255, cv.THRESH_OTSU)
         pred_str_per_mask, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
         pred_str_per_mask = list(pred_str_per_mask)
         if (len(pred_str_per_mask) >= 2):
             pred_str_per_mask = [a for a in pred_str_per_mask if len(a)>25]
             
         if pred_str_per_mask: # if atleast one strucutre is predicted
             
            if (str_names[j] == 'p') or (str_names[j]== 'n'): #for single strucutres
                if (len(pred_str_per_mask) > 1):
                    pred_str_per_mask = [a for a in pred_str_per_mask if len(a) == max([len(a) for a in pred_str_per_mask])] #take only the largest strucutre
                    
                pred_str = np.squeeze(pred_str_per_mask[0])
                poly = Polygon(pred_str)
                poly_dict[str_names[j]] = poly
                
            else:
                if(len(pred_str_per_mask) == 1): #only one str is predicted by model
                    pred_str = np.squeeze(pred_str_per_mask[0])
                    poly = Polygon(pred_str)
                    min_coords = pred_str[:,1].min()   # to check if the present struc is upper or lower one
                    max_coords = pred_str[:,1].max()
                    if (min_coords <= H // 2 ) and (max_coords <= H // 2):
                        poly_dict[str_names[j][1]] = poly # pred str is in upper half of the image
                        poly_dict[str_names[j][0]] = None
                        
                    else:
                        poly_dict[str_names[j][0]] = poly # pred str is in lower half of the image
                        poly_dict[str_names[j][1]] = None
                        
                elif (len(pred_str_per_mask) == 2):
                    for k in range(len(pred_str_per_mask)):
                        pred_str = np.squeeze(pred_str_per_mask[k])
                        poly = Polygon(pred_str)
                        poly_dict[str_names[j][k]] = poly
                        
                else:
                    list_pred_str = finding_largest(pred_str_per_mask)  #finding first two largest polygons from predictions
                    
                    for k in range(len(list_pred_str)):
                        pred_str = np.squeeze(pred_str_per_mask[k])
                        poly = Polygon(pred_str)
                        
                        min_coords = pred_str[:,1].min()   # to check if the present struc is upper or lower one
                        max_coords = pred_str[:,1].max()
                        
                        if (min_coords <= H // 2 ) and (max_coords <= H // 2):
                            poly_dict[str_names[j][1]] = poly
                            
                        else:
                            poly_dict[str_names[j][0]] = poly
                            
         else:
             poly_dict[str_names[j][0]] = None
             poly_dict[str_names[j][1]] = None
             
     final_contours[ids] = poly_dict
             
            
                        
                        
                        
                        
                        
                        
                
                    
                        
                        
                    
                    
                    
                    
                    
                    
                
                
            
                
                
                
            
             
        
     
     
     
