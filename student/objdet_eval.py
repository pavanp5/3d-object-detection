# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Evaluate performance of object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import numpy as np
import matplotlib
#matplotlib.use('wxagg') # change backend so that figure maximizing works on Mac as well     
import matplotlib.pyplot as plt

import torch
from shapely.geometry import Polygon
from operator import itemgetter

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# object detection tools and helper functions
import misc.objdet_tools as tools
import cv2

def get_bev_img(bev_maps, configs):

    bev_map = (bev_maps.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    bev_map = cv2.resize(bev_map, (configs.bev_width, configs.bev_height))
    bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
    return bev_map

def drawRotatedBox(img, x, y, w, l, yaw, color):
    bev_corners = get_corners(x, y, w, l, yaw)
    corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
    cv2.polylines(img, [corners_int], True, color, 2)
    corners_int = bev_corners.reshape(-1, 2)
    cv2.line(img, (corners_int[0, 0], corners_int[0, 1]), (corners_int[3, 0], corners_int[3, 1]), (255, 255, 0), 2)
    return img 

def get_corners(x, y, w, l, yaw):
    box_corners = np.zeros((4, 2), dtype=np.float32)

    box_corners[0, 0] = x - (w / 2 )* np.cos(yaw) - (l / 2 )* np.sin(yaw)
    box_corners[0, 1] = y - (w / 2 )* np.sin(yaw) + (l / 2 )* np.cos(yaw)


    box_corners[1, 0] = x - (w / 2) * np.cos(yaw) + (l / 2) * np.sin(yaw)
    box_corners[1, 1] = y - (w / 2) * np.sin(yaw) - (l / 2) * np.cos(yaw)


    box_corners[2, 0] = x + (w / 2) * np.cos(yaw) + (l / 2) * np.sin(yaw)
    box_corners[2, 1] = y + (w / 2) * np.sin(yaw) - (l / 2) * np.cos(yaw)


    box_corners[3, 0] = x + (w / 2) * np.cos(yaw) - (l / 2) * np.sin(yaw)
    box_corners[3, 1] = y + (w / 2) * np.sin(yaw) + (l / 2) * np.cos(yaw)
    

    return box_corners

from shapely.geometry import Polygon

def calculate_iou(box_1,box_2):
    
    box_1 = Polygon(box_1)
    box_2 = Polygon(box_2)
    iou = box_1.intersection(box_2).area / box_1.union(box_2).area

    return iou

def show_objects_and_labels_in_bev(detections,lidar_bev,labels,labels_valid,configs_det):
    color = [0, 0, 255]
    color_l = [0,255,0]
    img =get_bev_img(lidar_bev, configs_det)
    for d1 in detections:
        i,_x,_y, _z, _h, _w, _l, _yaw = d1
        img = drawRotatedBox(img, _x,_y,_w,_l, (-1)*_yaw,color)
    print(len(labels))
    
    for label, valid in zip(labels, labels_valid):
        
        if valid: # exclude all labels from statistics which are not considered valid
            img = drawRotatedBox(img, label.box.center_x,label.box.center_y,label.box.width,label.box.length, label.box.heading,color_l)
            
    cv2.imshow('labels (green) vs. detected objects (red)', img)
    cv2.waitKey(0)

    
# compute various performance measures to assess object detection
def measure_detection_performance(detections, labels, labels_valid,lidar_bev, configs_det,min_iou=0.5,show_bev_det=False):
    
    # find best detection for each valid label 
    true_positives = 0 # no. of correctly detected objects
    center_devs = []
    _ious = []
    
    #
    if show_bev_det==True:
        img =get_bev_img(lidar_bev, configs_det)
        
    color = [0, 255, 255]
    yhat_corners = []
    all_positives = 0
    false_positives = 0
    false_negatives = 0

    for d1 in detections:
        i,_x,_y, _z, _h, _w, _l, _yaw = d1
        
        x = (_y+25)*(609/50)
        _y = (_x)*(609/50) 
        _x = x
        _x = 609 - _x
        _y = 609 -_y
        _w = _w*(609/50)
        _l = _l*(609/50)
        
        yhat_corners.append(get_corners(_x,_y,_w,_l, (-1)*_yaw))
        
        if show_bev_det==True:
            img = drawRotatedBox(img, _x,_y,_w,_l, (-1)*_yaw,color)
        #all_positives = all_positives + 1
   
    color_l = [0, 0, 255]
    color_w = [255, 255, 255]
    match_fn = 0
    match_fp = 0
    dist_x=0
    dist_y=0
    dist_z=0
    t = []
    ious = []
    true_positives = 0
    matches_lab_det = []
    for label, valid in zip(labels, labels_valid):
        matches_lab_det = []
        
        if valid: # exclude all labels from statistics which are not considered valid
            all_positives = all_positives + 1

            x = label.box.center_x
            label.box.center_x = configs_det.bev_width - (label.box.center_y  + 25 ) * (609/50)
            label.box.center_y = configs_det.bev_height - (x) * (609/50)
            label.box.width = label.box.width * (609/50)
            label.box.length = label.box.length * (609/50)
            #img = drawRotatedBox(img, label.box.center_x,label.box.center_y,label.box.width,label.box.length, label.box.heading,color_l)

            y_corners = get_corners(label.box.center_x,label.box.center_y,label.box.width,label.box.length, label.box.heading)
            
            match_fn = 0
            
            for yhat, d in zip(yhat_corners,detections):
                iou = calculate_iou(y_corners,yhat)
                
                if iou > .6:
                   
                    match_fn = 1
                    match_fp = match_fp + 1
                    i,_x,_y, _z, _h, _w, _l, _yaw = d
                    x = (_y+25)*(609/50)
                    _y = (_x)*(609/50) 
                    _x = x
                    _x = 609 - _x
                    _y = 609 -_y
            
                    dist_x = label.box.center_x - _x
                    dist_y = label.box.center_y - _y
                    dist_z = label.box.center_z - _z
                    if show_bev_det==True:
                        img = drawRotatedBox(img, _x,_y,_w,_l, (-1)*_yaw,color_w)
                    true_positives = true_positives + 1
                    matches_lab_det.append([iou,dist_x, dist_y, dist_z])
             
                   
            
            false_negatives = all_positives - true_positives                     
            
            # compute intersection over union (iou) and distance between centers

            ####### ID_S4_EX1 START #######     
            #######
            print("student task ID_S4_EX1 ")
     
            ## step 1 : extract the four corners of the current label bounding-box
            
            ## step 2 : loop over all detected objects

                ## step 3 : extract the four corners of the current detection
                
                ## step 4 : compute the center distance between label and detection bounding-box in x, y, and z
                
                ## step 5 : compute the intersection over union (IOU) between label and detection bounding-box
                
                ## step 6 : if IOU exceeds min_iou threshold, store [iou,dist_x, dist_y, dist_z] in matches_lab_det and increase the TP count
    
                
            #######
            ####### ID_S4_EX1 END #######     
        
        # find best match and compute metrics
        
        if matches_lab_det:
            #print(matches_lab_det)
            best_match = max(matches_lab_det,key=itemgetter(1)) # retrieve entry with max iou in case of multiple candidates   
            ious.append(best_match[0])
            center_devs.append(best_match[1:])

    ####### ID_S4_EX2 START #######     
    #######
    print("student task ID_S4_EX2") 
    false_positives = len(detections) - true_positives
    #if match_fp < len(yhat_corners):
    #    
    #else:
    #    false_positives = 0
    #for yhat in yhat_corners:
    #    false_positives = false_positives + 1
    
    
    # compute positives and negatives for precision/recall
    
    ## step 1 : compute the total number of positives present in the scene
    #all_positives = 0

    ## step 2 : compute the number of false negatives
    #false_negatives = 0

    ## step 3 : compute the number of false positives
    #false_positives = 0
    
    #######
    ####### ID_S4_EX2 END #######     
    
    pos_negs = [all_positives, true_positives, false_negatives, false_positives]
    det_performance = [ious, center_devs, pos_negs]
    #cv2.imshow('img_labels', img)
    #cv2.waitKey(0)
    return det_performance


# evaluate object detection performance based on all frames
def compute_performance_stats(det_performance_all):

    # extract elements
    ious = []
    center_devs = []
    pos_negs = []
    ap_t=0
    tp_t=0
    fn_t=0 
    fp_t=0
    precision = 0.0
    
    for item in det_performance_all:
        ious.append(item[0])
        center_devs.append(item[1])
        ap, tp, fn, fp = item[2]

        ap_t = ap + ap_t
        tp_t = tp + tp_t
        fn_t = fn + fn_t
        fp_t = fp + fp_t
        
        #pos_negs.append(item[2])
    
    ####### ID_S4_EX3 START #######     
    #######    
    print('student task ID_S4_EX3')

    ## step 1 : extract the total number of positives, true positives, false negatives and false positives
    
    ## step 2 : compute precision
    if (tp_t + fp_t) > 0:
        precision = tp_t / (tp_t + fp_t)

    ## step 3 : compute recall 
    if (tp_t + fn_t) > 0:
        recall = tp_t / (tp_t + fn_t)

    #######    
    ####### ID_S4_EX3 END #######      

    # serialize intersection-over-union and deviations in x,y,z
    ious_all = [element for tupl in ious for element in tupl]
    devs_x_all = []
    devs_y_all = []
    devs_z_all = []
    for tuple in center_devs:
        for elem in tuple:
            dev_x, dev_y, dev_z = elem
            
            devs_x_all.append(dev_x)
            devs_y_all.append(dev_y)
            devs_z_all.append(dev_z)
    

    # compute statistics
    stdev__ious = np.std(ious_all)
    mean__ious = np.mean(ious_all)

    stdev__devx = np.std(devs_x_all)
    mean__devx = np.mean(devs_x_all)

    stdev__devy = np.std(devs_y_all)
    mean__devy = np.mean(devs_y_all)

    stdev__devz = np.std(devs_z_all)
    mean__devz = np.mean(devs_z_all)
    print(recall)

    # plot results
    data = [precision, recall, ious_all, devs_x_all, devs_y_all, devs_z_all]
    titles = ['detection precision', 'detection recall', 'intersection over union', 'position errors in X', 'position errors in Y', 'position error in Z']
    textboxes = ['', '', '',
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_x_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_x_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_y_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_y_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_z_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_z_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), )))]

    f, a = plt.subplots(2, 3)
    a = a.ravel()
    num_bins = 20
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for idx, ax in enumerate(a):
        ax.hist(data[idx], num_bins)
        ax.set_title(titles[idx])
        if textboxes[idx]:
            ax.text(0.05, 0.95, textboxes[idx], transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.show()

