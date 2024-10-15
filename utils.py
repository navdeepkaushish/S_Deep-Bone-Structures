# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 14:21:58 2023

@author: Navdeep Kumar
"""
from __future__ import print_function, division
import os
import glob
import numpy as np
#import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import shutil
import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import random
import colorsys

writer = SummaryWriter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#================ rescale and pad image ====================================
def rescale_pad(image, mask, desired_size):
    
    h, w = image.shape[:2]
    
    aspect = w / h
    
    if aspect > 1 : #horizontal image
        new_w = desired_size
        new_h = int(desired_size * h / w)
        offset = int(new_w - new_h)
        if offset %  2 != 0: #odd offset
            top = offset // 2 + 1
            bottom = offset // 2
        else:
            top = bottom = offset // 2
        
        dim = (new_w, new_h)
        re_img = cv.resize(image, dim, interpolation = cv.INTER_NEAREST)
        pad_img = cv.copyMakeBorder(re_img, top, bottom, 0, 0, cv.BORDER_REPLICATE)
        if mask is not None:
            re_mask = cv.resize(mask, dim, interpolation = cv.INTER_NEAREST)
            pad_mask = cv.copyMakeBorder(re_mask, top, bottom, 0, 0, cv.BORDER_REPLICATE)
            
        else:
            pad_mask = None

            
    elif aspect < 1:  #vertical image
        new_h = desired_size
        new_w = int(desired_size * w / h)
        offset = int(new_h - new_w)
        if offset %  2 != 0: #odd offset
            left = offset //2 + 1
            right = offset // 2
        else:
            left = right = offset // 2
        
        dim = (new_w, new_h)
        re_img = cv.resize(image, dim, interpolation = cv.INTER_NEAREST)
        pad_img = cv.copyMakeBorder(re_img, 0, 0, left, right, cv.BORDER_REPLICATE)
        if mask is not None:
            re_mask = cv.resize(mask, dim, interpolation = cv.INTER_NEAREST)
            pad_mask = cv.copyMakeBorder(re_mask, 0, 0, left, right, cv.BORDER_REPLICATE)
        else:
            pad_mask = None

    
    
    return pad_img, pad_mask
#======== upsize the mask to orginal size ====================================
def up_mask(mask, H, W):
    
    asp_ratio = W / H
    size = mask.shape[0]
    
    if asp_ratio > 1: # horizontal image
            
        cr_w = size
        cr_h = int(size / asp_ratio)
        
        offset = int(cr_w - cr_h)
        if offset %  2 != 0: #odd offset
            top = int(offset // 2 + 1)
        else:
            top = int(offset // 2)
            
        cr_mask = mask[top:top+cr_h, :]
        
            
    elif asp_ratio < 1 : # vertical image
        
        cr_w = int(size / asp_ratio)
        cr_h = size
        
        offset = int(cr_h - cr_w)
        if offset %  2 != 0: #odd offset
            left = int(offset //2 + 1)
        else:
            left  = int(offset // 2)
            
        cr_mask = mask[:, left:left+cr_w]
        
    dim = (W, H)
        
    up_mask = cv.resize(cr_mask, dim, interpolation = cv.INTER_NEAREST)
        
    return up_mask

#========= combine individual masks into one ================================
def create_mask(masks_path, H, W):
    n = len(masks_path)
    full_mask =  np.zeros(shape=(H, W, n))
    for i in range(len(masks_path)):
        
        mask = cv.imread(masks_path[i], 0) #grayscale
        mask[mask==255] = 1
        full_mask[:, :, i] = mask
        
    return full_mask


def crop_pts(mask):
     """
 	return coordinate points from predicted mask
 	"""
     #mask = mask.numpy().astype(np.float32)
     #mask = np.squeeze(mask, axis=2)

     coord = np.where(mask == [1])
     y_min = min(coord[0]) -60
     y_max = max(coord[0]) +60
     x_max = max(coord[1]) -600
     x_min = min(coord[1])
     if x_min > 60:
         x_min = min(coord[1]) - 60
         
     
     pts = [x_min, y_min, x_max, y_max]
     pts = np.asarray(pts).astype(int)
     
     return pts
#==============================================================================
class SegDataset(Dataset):
    """Segmentation Dataset"""

    def __init__(self, id_paths, transform=None):
        """
        Args:
            image_id_paths (string): Path to all the image id folders.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.id_paths = id_paths
        self.transform = transform
    def __len__(self):
        return len(self.id_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.id_paths[idx], 'image.png')
        image = cv.imread(img_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        H, W = image.shape[:2]
        print(H, W)
        mask_path = glob.glob(os.path.join(self.id_paths[idx], '[!i]*'))
        mask = create_mask(mask_path, H, W)
        image, mask = rescale_pad(image, mask, 512)
        mask[mask!=0] = 1
        

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            #image = image.reshape((image_size))
            mask = transformed['mask']
            mask = mask.to(torch.float32).permute(2,0,1)
            
        
        image = image/255.0
        sample = {'image': image, 'mask': mask}
        #sample = {'image': image, 'landmarks': landmarks}
            
            

        return sample
#=========================== saving and loading checkpoints ==================
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)
 #=============================================================================       
def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()    
#==============================================================================
def train(start_epochs, n_epochs, valid_loss_min_input, loaders, model, optimizer, criterion, checkpoint_path, best_model_path, scheduler=None):
    """
    Keyword arguments:
    start_epochs -- the real part (default 0.0)
    n_epochs -- the imaginary part (default 0.0)
    valid_loss_min_input
    loaders
    model
    optimizer
    criterion
    scheduler
    checkpoint_path
    best_model_path
    
    returns trained model
    """
    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input 
    
    for epoch in range(start_epochs, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, sample_batched in enumerate(loaders['train']):
            # move to GPU
            
            data, target_mask = sample_batched['image'].to(device), sample_batched['mask'].to(device)
            ## find the loss and update the model parameters accordingly
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            out_mask = model(data)
            # calculate the batch loss
            loss = criterion(out_mask, target_mask)
    

        
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()

            train_loss += loss.item()
        #compute avarage loss over batches
        avg_loss = train_loss/len(loaders['train'].dataset) 

    
        ######################    
        # validate the model #
        ######################
        model.eval()
        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(loaders['val']):
            # move to GPU
            
                data, target_mask =sample_batched['image'].to(device), sample_batched['mask'].to(device)
    
            # forward pass: compute predicted outputs by passing inputs to the model
                out_mask = model(data)
                
                loss = criterion(out_mask, target_mask)
                valid_loss += loss.item()
                
                
        # calculate average val loss
        avg_vloss = valid_loss/len(loaders['val'].dataset)
        if scheduler:
            scheduler.step(avg_vloss)
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            avg_loss,
            avg_vloss
            ))
        
        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': avg_vloss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        
        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        
        ## TODO: save the model if validation loss has decreased
        if avg_vloss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, avg_vloss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = avg_vloss
        writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch + 1)
        writer.flush()
            
    # return trained model
    return model
#==================== Dice score =============================================
def dice(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())
#==============================================================================

def evaluate_model(model, ckp_path, loader, threshold):
    PATH = os.path.join(ckp_path, 'best_model/best_model.pt')
    checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    dice_score = []
    model.eval()
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(loader):
            data, target_mask = sample_batched['image'].to(device), sample_batched['mask'].to(device)
            print(target_mask.shape)
            out_mask = model(data)
            print(out_mask.shape)
            out_mask = torch.sigmoid(out_mask)
            out_mask = torch.squeeze(out_mask).cpu().numpy()
            target_mask = torch.squeeze(target_mask).cpu().numpy()
            
            dice_per_slice = []
            for i in range(len(out_mask)):
                
                pred_mask = out_mask[i]
                gt_mask = target_mask[12-i]
                
                pred_mask[pred_mask < threshold] = 0
                pred_mask[pred_mask >= threshold] = 1
                
                dice_scr = dice(pred_mask, gt_mask)
                #print(dice_scr)
                if np.isnan(dice_scr):
                    dice_scr = 0
                dice_per_slice.append(dice_scr)
            dice_scr_per_mask = np.sum(dice_per_slice) / len(out_mask)
            print('Dice score per mask: {:.6f}'.format(dice_scr_per_mask))
            dice_score.append(dice_scr_per_mask)
                
    avg_dice_score = np.sum(dice_score) / len(loader.dataset)
    print('Average Dice Score for test data {:.6f}.  Evalaution terminated ...'.format(avg_dice_score))
    return avg_dice_score
#===========================Find first two largest elements from a list of arrays===========
def finding_largest(arr_list):
    largest = [a for a in arr_list if len(a) == max([len(a) for a in arr_list])]
    largest = np.squeeze(np.array(largest))
    largest_idx = np.argmax([len(a) for a in arr_list])
    arr_list.pop(largest_idx)
    second_largest = [a for a in arr_list if len(a) == max([len(a) for a in arr_list])]
    second_largest = np.squeeze(np.array(second_largest))
    return [largest, second_largest]  

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors              
                
def draw_contours(image, contours):
    colors = random_colors(24)
    contours_list = []    
    for key, value in contours.items():
        contours_list.append(value)
    for i in range(len(contours_list)):
        if contours_list[i] is not None:
            poly = contours_list[i]
            coords = list(poly.exterior.coords)
            coords = np.array(coords).astype(np.int32)
            coords = coords.reshape((-1, 1, 2))
            cv.drawContours(image, coords, -1, colors[i], 3, cv.LINE_AA)
            
        else:
            continue           
                
def cr_up(mask, pts, H, W):
    top = pts[1]
    bottom = H - pts[3]
    left = pts[0]
    right = W - pts[2]
    
    pad_mask = cv.copyMakeBorder(mask, top, bottom, left, right, cv.BORDER_CONSTANT, None, value = 0)
    
    return pad_mask
#==============================================================================           
            
            
            
            
            
            
            
    
    


