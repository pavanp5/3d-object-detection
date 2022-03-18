# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Process the point-cloud and prepare it for object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#


# general package imports
import cv2
import numpy as np
import torch

# add project directory to python path to enable relative imports
import os
import sys
import zlib
import math
import open3d as o3d
import matplotlib.pyplot as plt
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2

# object detection tools and helper functions
import misc.objdet_tools as tools

##get PCL from range image
def range_image_to_point_cloud(frame, lidar_name, vis=False):

    # extract range values from frame
    ri = load_range_image(frame, lidar_name)
    #ri[ri<0]=0.0
    ri_range = ri[:,:,0]

    # load calibration data
    calibration = [obj for obj in frame.context.laser_calibrations if obj.name == lidar_name][0]

    # compute vertical beam inclinations
    height = ri_range.shape[0]
    inclination_min = calibration.beam_inclination_min
    inclination_max = calibration.beam_inclination_max
    if len(calibration.beam_inclinations)>0:
        inclinations = np.array(calibration.beam_inclinations)
    else:
        inclinations = np.linspace(inclination_min, inclination_max, height)
    inclinations = np.flip(inclinations)

    # compute azimuth angle and correct it so that the range image center is aligned to the x-axis
    width = ri_range.shape[1]
    extrinsic = np.array(calibration.extrinsic.transform).reshape(4,4)
    az_correction = math.atan2(extrinsic[1,0], extrinsic[0,0])
    azimuth = np.linspace(np.pi,-np.pi,width) - az_correction
    #print(len(azimuth))

    # expand inclination and azimuth such that every range image cell has its own appropriate value pair
    azimuth_tiled = np.broadcast_to(azimuth[np.newaxis,:], (height,width))
    inclination_tiled = np.broadcast_to(inclinations[:,np.newaxis],(height,width))

    # perform coordinate conversion
    x = np.cos(azimuth_tiled) * np.cos(inclination_tiled) * ri_range
    y = np.sin(azimuth_tiled) * np.cos(inclination_tiled) * ri_range
    z = np.sin(inclination_tiled) * ri_range

    
    # transform 3d points into vehicle coordinate system
    xyz_sensor = np.stack([x,y,z,np.ones_like(z)])
    xyz_vehicle = np.einsum('ij,jkl->ikl', extrinsic, xyz_sensor)
    xyz_vehicle = xyz_vehicle.transpose(1,2,0)

    # extract points with range > 0
    idx_range = ri_range > 0
    pcl = xyz_vehicle[idx_range,:3]
    pcl_full = np.column_stack((pcl, ri[idx_range, 1]))
    pcl = pcl_full 
   
    # visualize point-cloud
    if vis:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl)
        o3d.visualization.draw_geometries([pcd])

    # stack lidar point intensity as last column
    #pcl_full = np.column_stack((pcl, ri[idx_range, 1])) 
  

    return pcl_full   


# visualize lidar point-cloud
def show_pcl(pcl,configs):
        
    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    pcl[:, 2] = pcl[:, 2] - configs.lim_z[0] 
    

    ####### ID_S1_EX2 START #######     
    #######
    def closeWindow(vis):
        vis.close()
        return False
    print("student task ID_S1_EX2")

    # step 1 : initialize open3d with key callback and create window
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_callback(262, closeWindow)
    vis.create_window()
    
    
    # step 2 : create instance of open3d point-cloud class
    pcd = o3d.geometry.PointCloud()
    
    # step 3 : set points in pcd instance by converting the point-cloud into 3d vectors (using open3d function Vector3dVector)
   
    pcd.points = o3d.utility.Vector3dVector(pcl[:,:3])
    
    
    # step 4 : for the first frame, add the pcd instance to visualization using add_geometry; for all other frames, use update_geometry instead   
    vis.add_geometry(pcd)
    #print("Create")
    vc = vis.get_view_control()
  

        
    vc.translate(40, 30, xo=0.0, yo=0.0)
  
    # step 5 : visualize point cloud and keep window open until right-arrow is pressed (key-code 262)
    vis.run()
    #######
    ####### ID_S1_EX2 END #######     
       

# visualize range image
def show_range_image(frame, lidar_name):

  ####### ID_S1_EX1 START #######     
  #######
  print("student task ID_S1_EX1")

  # step 1 : extract lidar data and range image for the roof-mounted lidar
  ri = load_range_image(frame, lidar_name)  
  
  # step 2 : extract the range and the intensity channel from the range image
  ri_intensity = ri[:,:,1]
  ri_range = ri[:,:,0]
       
  # step 3 : set values <0 to zero
  ri[ri<0]=0.0
    
    
  # step 4 : map the range channel onto an 8-bit scale and make sure that the full range of values is appropriately considered
  # map value range to 8bit
  ri_intensity = ri_intensity * (np.amax(ri_intensity)/2) *255 / (np.amax(ri_intensity) - np.amin(ri_intensity))
  ri_range = ri_range * 255 / (np.amax(ri_range) - np.amin(ri_range))
  img_range =(ri_range).astype(np.uint8)
  img_intensity = (ri_intensity).astype(np.uint8)

  #deg45
  deg45 = int(img_range.shape[1] / 8)
  ri_center = int(img_range.shape[1]/2)
  img_range = img_range[:,ri_center-deg45:ri_center+deg45]

  deg45_intensity = int(img_intensity.shape[1] / 8)
  ri_center_intensity = int(img_intensity.shape[1]/2)
  img_intensity = img_intensity[:,ri_center_intensity-deg45_intensity:ri_center_intensity+deg45_intensity]
  # step 5 : map the intensity channel onto an 8-bit scale and normalize with the difference between the 1- and 99-percentile to mitigate the influence of outliers
    
  # step 6 : stack the range and intensity image vertically using np.vstack and convert the result to an unsigned 8-bit integer
  img_range_intensity = np.vstack((img_range,img_intensity))
  
  #######
  ####### ID_S1_EX1 END #######     
    
  return img_range_intensity
                                                                  
def load_range_image(frame, lidar_name):
  lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0]
  ri = 0
  if len(lidar.ri_return1.range_image_compressed) > 0:
    ri = dataset_pb2.MatrixFloat()
    ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
    ri= np.array(ri.data).reshape(ri.shape.dims)
  return ri
                                                                  

#get BEV Map from PCL
def makeBEVMap(PointCloud_,configs):
    # convert sensor coordinates to bev-map coordinates (center is bottom-middle)
    ####### ID_S2_EX1 START #######     
    #######
    print("student task ID_S2_EX1")

    ## step 1 :  compute bev-map discretization by dividing x-range by the bev-image height (see configs)
    ## step 2 : create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates    
    # step 3 : perform the same operation as in step 2 for the y-coordinates but make sure that no negative bev-coordinates occur
    # step 4 : visualize point-cloud using the function show_pcl from a previous task

  
    

    Height = configs.bev_height + 1
    Width = configs.bev_width + 1
    discrete = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / discrete))
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / discrete) + (Width / 2))
    #######
    ####### ID_S2_EX1 END ####### 

    # Compute intensity layer of the BEV map
    ####### ID_S2_EX2 START #######     
    #######
    print("student task ID_S2_EX2")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    # step 2 : re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then -z (use numpy.lexsort)
    ## step 3 : extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    ##          also, store the number of points per x,y-cell in a variable named "counts" for use in the next task
    ## step 4 : assign the intensity value of each unique entry in lidar_top_pcl to the intensity map 
    ##          make sure that the intensity is scaled in such a way that objects of interest (e.g. vehicles) are clearly visible    
    ##          also, make sure that the influence of outliers is mitigated by normalizing intensity on the difference between the max. and min. value within the point cloud      
    ## step 5 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    counts = np.count_nonzero(PointCloud, axis=1,  keepdims=True)
    # sort-3times
    sorted_indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud_top = PointCloud[sorted_indices]
    _, unique_indices, unique_counts = np.unique(PointCloud_top[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud_top[unique_indices]
    
    sorted_indices_int = np.lexsort((-PointCloud_top[:, 3], PointCloud_top[:, 1], PointCloud_top[:, 0]))
    PointCloud_int = PointCloud_top[sorted_indices_int]
    _, unique_indices_int, unique_counts_int = np.unique(PointCloud_int[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_int = PointCloud_int[unique_indices_int]
    
    intensity_clip_99 = np.percentile(PointCloud_int[:, 3], 99)
    intensity_clip_1 = np.percentile(PointCloud_int[:, 3], 1)
    
 
    PointCloud_int[:, 3][PointCloud_int[:, 3]>intensity_clip_99]=intensity_clip_99
    PointCloud_int[:, 3][PointCloud_int[:, 3]<intensity_clip_1]=intensity_clip_1
    PointCloud_int[:, 3] = (PointCloud_int[:, 3]) / (1.3*(intensity_clip_99 - intensity_clip_1 ))


    ####### ID_S2_EX2 END ####### 

    # Compute height layer of the BEV map
    ####### ID_S2_EX3 START #######     
    #######
    print("student task ID_S2_EX3")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
  

    ## step 2 : assign the height value of each unique entry in lidar_top_pcl to the height map 
    ##          make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    ##          use the lidar_pcl_top data structure from the previous task to access the pixels of the height_map



    ## step 3 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background    
    
    # sort-3times
    #sorted_indices = np.lexsort((-PointCloud_int[:, 2], PointCloud_int[:, 1], PointCloud_int[:, 0]))
    #PointCloud_top = PointCloud_int[sorted_indices]
    #_, unique_indices, unique_counts = np.unique(PointCloud_top[:, 0:2], axis=0, return_index=True, return_counts=True)
    #PointCloud_top = PointCloud_top[unique_indices]
       

    # Height Map, Intensity Map & Density Map
    heightMap = np.zeros((Height, Width))
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    # some important problem is image coordinate is (y,x), not (x,y)
    #max_height = float(np.abs(boundary['maxZ'] - boundary['minZ']))
   
    #heightMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = (((PointCloud_top[:, 2])  / np.float_((np.amax(PointCloud_top[:, 2],axis=0))) ))
    
    heightMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = (PointCloud_top[:, 2])  / (configs.lim_z[1] - configs.lim_z[0])
    
    #######
    ####### ID_S2_EX3 END ####### 
    # Compute density layer of the BEV map    
    normalizedCounts =  np.minimum(1,(np.log(unique_counts + 1) /np.log(64)))
         
    intensityMap[np.int_(PointCloud_int[:, 0]), np.int_(PointCloud_int[:, 1])] = (PointCloud_int[:, 3])
    
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = (normalizedCounts)
    
    #cv2.imshow('density_img',densityMap)
    #cv2.waitKey(0)
    #cv2.imshow('int_img',intensityMap)
    #cv2.waitKey(0)
    #cv2.imshow('ht_img',heightMap)
    #cv2.waitKey(0)
    
    # assemble 3-channel bev-map from individual maps   

    RGB_Map = np.zeros((3, Height - 1, Width - 1))
    RGB_Map[2, :, :] = densityMap[:configs.bev_height, :configs.bev_width]  # r_map
    RGB_Map[1, :, :] = heightMap[:configs.bev_height, :configs.bev_width]  # g_map
    RGB_Map[0, :, :] = intensityMap[:configs.bev_height, :configs.bev_width]  # b_map
    
    # expand dimension of bev_map before converting into a tensor
    s1, s2, s3 = RGB_Map.shape
    bev_maps = np.zeros((1, s1, s2, s3))
    bev_maps[0] = RGB_Map   
    
    bev_maps = torch.from_numpy(bev_maps)  # create tensor from birds-eye view
    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()

    return input_bev_maps

def bev_from_pcl(lidar_pcl,configs):
    
# create birds-eye view of lidar data
#def bev_from_pcl(lidar_pcl, configs):

    # remove lidar points outside detection area and with too low reflectivity
    mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                    (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
                    (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
    lidar_pcl = lidar_pcl[mask]
    
    
    
    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0]  
    
    rgb_map = makeBEVMap( lidar_pcl,configs)
    return rgb_map




