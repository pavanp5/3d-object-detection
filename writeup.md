# 3D Object Detection write-up & output

Steps involved
 1. Extract Range Image & Intensity Channel 
 2. Convert Range Image to Point Cloud
 3. Visualize Point Cloud
 4. Convert Point Cloud to Bird Eye View(BEV)
 5. Instantiate 3d object detection model
 6. Compute IoU for detections
 7. Compute metrics for detection

PFB, the metric for 3d object detection on sequence 1 for frames 50 to 150. 
<img src="img/output/metric/metric_1_50_150.PNG"/>

**Range Image & Intensity Channel Image**
The range images & the intensity channel images are shown below. We examine a few features of vehicles.
The objects that relect more appear brighter in the intensity channel.
PFB in range & intensity channel images, notable features that can be seen on multiple frames 

 - Tail Lamps
 - Number plates
 - Glow coates on pedestrians
 - Glow signboards
 - Head lamps

<img src="img/output/rng_img/r2_s.PNG"/>

<img src="img/output/rng_img/r3_s.PNG"/>

<img src="img/output/rng_img/r4_s.PNG"/>

<img src="img/output/rng_img/r5_s.PNG"/>

<img src="img/output/rng_img/r6_s.PNG"/>

<img src="img/output/rng_img/r7_s.PNG"/>

<img src="img/output/rng_img/r8_s.PNG"/>

<img src="img/output/rng_img/r9_s.PNG"/>

<img src="img/output/rng_img/r11_s.PNG"/>

**Point Cloud Images**

PFB, a few features in the point cloud images. We have used view control and translate function of o3d visualization to visualize the vehicles at the varying degress of visibility.

 - Rear view
 - Rear bumper
 - Front View
 - Side View
 - Wheels
 - Multiple vehicles with varying degrees of visibility

<img src="img/output/pcl/pcl_3_1_5_side_mirrors_a.PNG"/>

<img src="img/output/pcl/pcl_3_1_front_bumper_a.PNG"/>

<img src="img/output/pcl/pcl_3_5_rear_a.PNG"/>

<img src="img/output/pcl/pcl_3_1_tail_light.PNG"/>

<img src="img/output/pcl/pcl_3_1_a.PNG"/>

<img src="img/output/pcl/pcl_3_1_b.PNG"/>

<img src="img/output/pcl/pcl_3_1_d.PNG"/>

<img src="img/output/pcl/pcl_3_1_front_view.PNG"/>

<img src="img/output/pcl/pcl_3_10_rear_view_a.PNG"/>

<img src="img/output/pcl/pcl_3_190_rear_view.PNG"/>

<img src="img/output/pcl/pcl_3_190_rear_view_b.PNG"/>







