from __future__ import division
from __future__ import print_function
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from enum import Enum
#import time

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, unary_from_softmax
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
from numbers import Number
import open3d as o3d

N_classes=8

# Smoothness Kernel Compatibility Matrix
smoothness_matrix = np.array([
    [0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],  # background
    [0.05, 1.0, 0.6, 0.4, 0.4, 0.8, 0.9, 0.6],        # wall
    [0.05, 0.6, 1.0, 0.7, 0.8, 0.4, 0.3, 0.6],        # floor
    [0.05, 0.4, 0.7, 1.0, 0.8, 0.4, 0.3, 0.7],        # chair
    [0.05, 0.4, 0.8, 0.8, 1.0, 0.5, 0.4, 0.7],        # table
    [0.05, 0.8, 0.4, 0.4, 0.5, 1.0, 0.9, 0.5],        # door
    [0.05, 0.9, 0.3, 0.3, 0.4, 0.9, 1.0, 0.6],        # window
    [0.05, 0.6, 0.6, 0.7, 0.7, 0.5, 0.6, 1.0],        # other_object
    ], dtype=np.float32)
compatib_matrix2 = np.array([
            [0.0,  8.0,  8.5,  9.0,  9.0,  9.5,  9.5, 10.0],  # background
            [8.0,  0.0,  2.5,  2.5,  3.0,  3.0,  2.5,  3.5],  # wall
            [8.5,  2.5,  0.0,  3.0,  3.5,  3.0,  3.5,  4.0],  # floor
            [9.0,  2.5,  3.0,  0.0,  2.0,  3.0,  3.5,  4.0],  # chair
            [9.0,  3.0,  3.5,  2.0,  0.0,  3.0,  3.0,  4.0],  # table
            [9.5,  3.0,  3.0,  3.0,  3.0,  0.0,  3.5,  4.0],  # door
            [9.5,  2.5,  3.5,  3.5,  3.0,  3.5,  0.0,  4.0],  # window
            [10.0, 3.5,  4.0,  4.0,  4.0,  4.0,  4.0,  0.0],  # other_object
        ], dtype=np.float32)

sdims = [1, 1 , 1]
schan = [12, 3, 3]
snormal =[0.05, 0.05, 0.05]

def color_map(N=8, normalized=False):
    """
    Return Color Map in PASCAL VOC format (rgb)
    \param N (int) number of classes
    \param normalized (bool) whether colors are normalized (float 0-1)
    \return (Nx3 numpy array) a color map
    """
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)
    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3
        cmap[i] = np.array([r, g, b])
    cmap = cmap/255.0 if normalized else cmap
    
    return cmap

def decode_segmap(temp, n_classes, cmap):
    """
    Given an image of class predictions, produce an bgr8 image with class colors
    \param temp (2d numpy int array) input image with semantic classes (as integer)
    \param n_classes (int) number of classes
    \cmap (Nx3 numpy array) input color map
    \return (numpy array bgr8) the decoded image with class colors
    """
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = cmap[l,0]
        g[temp == l] = cmap[l,1]
        b[temp == l] = cmap[l,2]
    bgr = np.zeros((temp.shape[0], temp.shape[1], 3))
    bgr[:, :, 0] = b
    bgr[:, :, 1] = g
    bgr[:, :, 2] = r
    return bgr.astype(np.uint8)

class PointType(Enum):
    COLOR = 0
    SEMANTICS_MAX = 1
    SEMANTICS_BAYESIAN = 2

class ColorPclGenerator:
    '''
    Generate a ros point cloud given a color image and a depth image
    \author Xuan Zhang
    \date May - July 2018
    '''
    def __init__(self, intrinsic, width = 640, height = 480, frame_id = "/kinect", point_type = PointType.SEMANTICS_BAYESIAN):
        '''
        width: (int) width of input images
        height: (int) height of input images
        '''
        self.width=width
        self.height=height
        self.point_type = point_type
        self.intrinsic = intrinsic
        self.num_semantic_colors = 3 # Number of semantic colors to be sent in the message
        self.cmap = color_map(N = N_classes, normalized = False) # Color map for semantic classes
        

        
        # Allocate arrays
        x_index = np.array(list(range(width)) * height, dtype='<f4')
        y_index = np.array([[i]*width for i in range(height)], dtype = '<f4').ravel()
        self.xy_index = np.vstack((x_index, y_index)).T # x,y
        self.xyd_vect = np.zeros([width*height, 3], dtype = '<f4') # x,y,depth
        self.XYZ_vect = np.zeros([width*height, 3], dtype = '<f4') # real world coord
        if self.point_type is PointType.SEMANTICS_BAYESIAN:
            self.ros_data = np.ones([width*height, 16], dtype = '<f4') # [x,y,z,0,bgr0,0,0,0,color0,color1,color2,0,confidence0,confidence1,confidence2,0]
        else:
            self.ros_data = np.ones([width*height, 8], dtype = '<f4') # [x,y,z,0,bgr0,0,0,0] or [x,y,z,0,bgr0,semantics,confidence,0]
        self.bgr0_vect = np.zeros([width*height, 4], dtype = '<u1') #bgr0
        self.semantic_color_vect = np.zeros([width*height, 4], dtype = '<u1') #bgr0
        self.semantic_colors_vect = np.zeros([width*height, 4 * self.num_semantic_colors], dtype = '<u1') #bgr0bgr0bgr0 ...
        self.confidences_vect = np.zeros([width*height, self.num_semantic_colors],dtype = '<f4') # class confidences
        self.appearence_features = np.zeros([6,width*height],dtype = '<f4') # appearence features
        self.d = dcrf.DenseCRF(width*height, N_classes) #len(merged_classes)
        # self.smoothness_features = np.zeros([6,width*height],dtype = '<f4') # smoothnessfeatures
        # self.unary_potentials = np.zeros([N_classes,width*height],dtype = '<f4') # unary potentials
        # Prepare ros cloud msg
        # Cloud data is serialized into a contiguous buffer, set fields to specify offsets in buffer
        self.cloud_ros = PointCloud2()
        self.cloud_ros.header.frame_id = frame_id
        self.cloud_ros.height = 1
        self.cloud_ros.width = width*height
        self.cloud_ros.fields.append(PointField(
                             name = "x",
                             offset = 0,
                             datatype = PointField.FLOAT32, count = 1))
        self.cloud_ros.fields.append(PointField(
                             name = "y",
                             offset = 4,
                             datatype = PointField.FLOAT32, count = 1))
        self.cloud_ros.fields.append(PointField(
                             name = "z",
                             offset = 8,
                             datatype = PointField.FLOAT32, count = 1))
        self.cloud_ros.fields.append(PointField(
                             name = "rgb",
                             offset = 16,
                             datatype = PointField.FLOAT32, count = 1))
        if self.point_type is PointType.SEMANTICS_MAX:
            self.cloud_ros.fields.append(PointField(
                            name = "semantic_color",
                            offset = 20,
                            datatype = PointField.FLOAT32, count = 1))
            self.cloud_ros.fields.append(PointField(
                            name = "confidence",
                            offset = 24,
                            datatype = PointField.FLOAT32, count = 1))
        elif self.point_type is PointType.SEMANTICS_BAYESIAN:
            self.cloud_ros.fields.append(PointField(
                                name = "semantic_color1",
                                offset = 32,
                                datatype = PointField.FLOAT32, count = 1))
            self.cloud_ros.fields.append(PointField(
                                name = "semantic_color2",
                                offset = 36,
                                datatype = PointField.FLOAT32, count = 1))
            self.cloud_ros.fields.append(PointField(
                                name = "semantic_color3",
                                offset = 40,
                                datatype = PointField.FLOAT32, count = 1))
            self.cloud_ros.fields.append(PointField(
                                name = "confidence1",
                                offset = 48,
                                datatype = PointField.FLOAT32, count = 1))
            self.cloud_ros.fields.append(PointField(
                                name = "confidence2",
                                offset = 52,
                                datatype = PointField.FLOAT32, count = 1))
            self.cloud_ros.fields.append(PointField(
                                name = "confidence3",
                                offset = 56,
                                datatype = PointField.FLOAT32, count = 1))

        self.cloud_ros.is_bigendian = False
        if self.point_type is PointType.SEMANTICS_BAYESIAN:
            self.cloud_ros.point_step = 16 * 4 # In bytes
        else:
            self.cloud_ros.point_step = 8 * 4 # In bytes
                
        self.cloud_ros.row_step = self.cloud_ros.point_step * self.cloud_ros.width * self.cloud_ros.height
        self.cloud_ros.is_dense = False

    def generate_cloud_data_common(self, bgr_img, depth_img):
        """
        Do depth registration, suppose that rgb_img and depth_img has the same intrinsic
        \param bgr_img (numpy array bgr8)
        \param depth_img (numpy array float32 2d)
        [x, y, Z] = [X, Y, Z] * intrinsic.T
        """
        bgr_img = bgr_img.view('<u1')
        depth_img = depth_img.view('<f4')
        #mean_depth = np.nanmean(depth_img)

        #print(f"mean depth value: {mean_depth}")

        depth_img= np.nan_to_num(depth_img, nan=0.0, copy=False)
       
        # print(f"bgr_img Shape: {bgr_img.shape}")
        # print(f"Sample bgr_img: {bgr_img[:10]}")
        # print(f"depth_img Shape: {depth_img.shape}")
        # print(f"Sample depth_img: {depth_img[:10]}")

        # Add depth information
        self.xyd_vect[:,0:2] = self.xy_index * depth_img.reshape(-1,1)
        self.xyd_vect[:,2:3] = depth_img.reshape(-1,1)
        self.XYZ_vect = self.xyd_vect.dot(self.intrinsic.I.T)
        # Convert to ROS point cloud message in a vectorialized manner
        # ros msg data: [x,y,z,0,bgr0,0,0,0,color0,color1,color2,0,confidence0,confidence1,confidenc2,0] (little endian float32)
        # Transform color
        self.bgr0_vect[:,0:1] = bgr_img[:,:,0].reshape(-1,1)
        self.bgr0_vect[:,1:2] = bgr_img[:,:,1].reshape(-1,1)
        self.bgr0_vect[:,2:3] = bgr_img[:,:,2].reshape(-1,1)
        # Concatenate data
        self.ros_data[:,0:3] = self.XYZ_vect
        self.ros_data[:,4:5] = self.bgr0_vect.view('<f4')

    def make_ros_cloud(self, stamp):
        # Assign data to ros msg
        # We should send directly in bytes, send in as a list is too slow, numpy tobytes is too slow, takes 0.3s.
        #self.cloud_ros.data = np.getbuffer(self.ros_data.ravel())[:]
        self.cloud_ros.data = self.ros_data.ravel().tobytes()
        self.cloud_ros.header.stamp = stamp
        return self.cloud_ros

    def generate_cloud_color(self, bgr_img, depth_img, stamp):
        """
        Generate color point cloud
        \param bgr_img (numpy array bgr8) input color image
        \param depth_img (numpy array float32) input depth image
        """
        self.generate_cloud_data_common(bgr_img, depth_img)
        return self.make_ros_cloud(stamp)
    
    def unary_from_softmax(self, sm, scale=0.6, clip=1e-5):

        """Converts softmax class-probabilities to unary potentials (NLL per node).

        Parameters
        ----------
        sm: numpy.array
            Output of a softmax where the first dimension is the classes,
            all others will be flattend. This means `sm.shape[0] == n_classes`.
        scale: float
            The certainty of the softmax output (default is None).
            If not None, the softmax outputs are scaled to range from uniform
            probability for 0 outputs to `scale` probability for 1 outputs.
        clip: float
            Minimum value to which probability should be clipped.
            This is because the unary is the negative log of the probability, and
            log(0) = inf, so we need to clip 0 probabilities to a positive value.
        """
        #sm_no_bg = sm[1:, ...]  # Remove the first row corresponding to the background
        #sm_no_bg = sm_no_bg / np.sum(sm_no_bg, axis=0, keepdims=True)

        num_cls = sm.shape[0]
        #num_cls_no_bg = sm_no_bg.shape[0]  # Number of classes excluding the background
        if scale is not None:
            assert 0 < scale <= 1, "`scale` needs to be in (0,1]"
            uniform = np.ones(sm.shape) / num_cls
            sm = scale * sm + (1 - scale) * uniform
            #sm_no_bg=scale * sm_no_bg + (1 - scale) * uniform

        if clip is not None:
            sm= np.clip(sm, clip, 1.0)

         # Compute the negative log probabilities
        unaries = -np.log(sm).reshape([num_cls, -1]).astype(np.float32)

        # Assign a very high penalty to the background class (row 0)
        unaries[0, :] *= 5 # Arbitrary large value to discourage selection

        return unaries

        #return -np.log(sm).reshape([num_cls, -1]).astype(np.float32)


       
    def create_pairwise_bilateral_appearence_kernel(self, sdims, schan):
        """
        Create pairwise bilateral potentials for a 3D point cloud.

        Parameters
        ----------
        sdims : list or tuple
            Scaling factors for the spatial dimensions [sx, sy, sz].
        schan : list or tuple
            Scaling factors for the feature dimensions (e.g., color).
        points : numpy.array
            3D point cloud of shape (n_points, 3), where each row represents [x, y, z].
        features : numpy.array
            Additional feature matrix of shape (n_points, n_features), e.g., RGB values.

        Returns
        -------
        numpy.array
            Feature matrix of shape (3 + n_features, n_points).
        """

                
        # Scale the spatial coordinates
        points = np.ascontiguousarray(self.ros_data[:,0:3]).astype(np.float32)
        # print(f"Points appearence Shape: {points.shape}")
        # print(f"Sample appearence Points: {points[:100,:]}")

        for i, s in enumerate(sdims):
            points[:, i] /=s
            
        # Scale the feature dimensions
        rgb_color = np.ascontiguousarray(self.ros_data[:,4:5]).view('<u1')[:,0:3].astype(np.uint8)
        #print(f"Sample RGB color: {rgb_color[:10,:]}")
        bgr_color = rgb_color.reshape((-1, 1, 3)).astype(np.uint8)  # Shape (n, 1, 3)
        lab_image = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2LAB)
        lab_table = lab_image.reshape((-1, 3)).astype(np.float32)  # Shape (n, 3)




        #print(f"LAB color Shape: {lab_table.shape}")
        #print(f"Sample LAB color: {lab_table[:10,:]}")
        
        
        if isinstance(schan, Number):
            lab_table /= schan
        else:

            for i, s in enumerate(schan):
                lab_table[:, i] /= s
        #print(f" lab_table after scaling Shape: {lab_table.shape}")
        #print(f"Sample lab_table: {lab_table[:10,:]}")

        # Combine spatial and feature dimensions
        combined_features = np.concatenate([points.T, lab_table.T], axis=0)
        return combined_features
    
    def create_pairwise_bilateral_smoothness_kernel(self, sdims, schan):

        # Scale the spatial coordinates
        scaled_points = np.ascontiguousarray(self.ros_data[:,0:3]).astype(np.float32)
        for i, s in enumerate(sdims):
            scaled_points[:, i] /=s
                                        
        #features based on 3D normals of the points
        pcd = o3d.geometry.PointCloud()
        points=self.ros_data[:,0:3] .copy().astype(np.float32)
        # print(f"Points Shape: {points.shape}")
        # print(f"Sample Points: {points[:10]}")

        pcd.points = o3d.utility.Vector3dVector(points)

        # Estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)) #Use 30 neighbors for normal estimation
        #o3d.visualization.draw_geometries([pcd], point_show_normal=True)


       
        normals=np.array(pcd.normals).astype(np.float32)
        
       
        if isinstance(schan, Number):
            normals /= schan
        else:
            for i, s in enumerate(schan):
                normals[:, i] /=s                                                                                                                                                                        

        # Combine spatial and feature dimensions
        combined_features = np.concatenate([scaled_points.T, normals.T], axis=0)
        return combined_features
    
     

    def generate_cloud_semantic_max(self, bgr_img, depth_img, semantic_color,class_labels, confidence, softmax, stamp):

        self.generate_cloud_data_common(bgr_img, depth_img)

                
        self.appearence_features=self.create_pairwise_bilateral_appearence_kernel(sdims, schan)
        self.appearence_features = np.ascontiguousarray(self.appearence_features)
        # # Detect NaN values
        # nan_mask = np.isnan(self.appearence_features)

        # # Get indices of NaN values
        # nan_indices = np.argwhere(nan_mask)

        # # Print results
        # print("NaN mask:\n", nan_mask)
        # print("Indices of NaN values:\n", nan_indices)
        # col_medians = np.nanmedian(self.appearence_features, axis=0)
        # self.appearence_features[np.isnan(self.appearence_features)] = np.take(col_medians, np.where(np.isnan(self.appearence_features))[1])

        # # Detect NaN values
        # nan_mask = np.isnan(self.appearence_features)

        # # Get indices of NaN values
        # nan_indices = np.argwhere(nan_mask)


        smoothness_features=self.create_pairwise_bilateral_smoothness_kernel([0.2,0.2,0.2], snormal)
        smoothness_features = np.ascontiguousarray(smoothness_features)    

        scale=0.8
        self.unary_potentials= self.unary_from_softmax(softmax, scale, clip=1e-5)
        self.unary_potentials = np.ascontiguousarray(self.unary_potentials)
         
        
        print(" CRF setting")
        # print(f"Appearance features shape: {self.appearence_features.shape}, dtype: {self.appearence_features.dtype}")
        # # print(f"Smoothness features shape: {smoothness_features.shape}, dtype: {smoothness_features.dtype}")
        # # print(f"Unary potentials shape: {Unary_potentials.shape}, dtype: {Unary_potentials.dtype}")
        # # print(f"DCRF object: {self.d}")

        
        self.d.setUnaryEnergy(self.unary_potentials)            
       

        self.d.addPairwiseEnergy(smoothness_features, compat=3, #was smoothness matrix
                     kernel=dcrf.DIAG_KERNEL,
                     normalization=dcrf.NORMALIZE_SYMMETRIC)
        
        self.d.addPairwiseEnergy(self.appearence_features, compat=compatib_matrix2, #was 10
                       kernel=dcrf.DIAG_KERNEL,
                      normalization=dcrf.NORMALIZE_SYMMETRIC)
        print("CRF Inference")
        # # Run five inference steps.
        Q = self.d.inference(5)
        Q_array = np.array(Q)
        # # Find out the most probable class for each point.
        MAP = np.argmax(Q, axis=0)

        MAP_reshaped = MAP.reshape((self.height,self.width))
        MAP_decoded=decode_segmap(MAP_reshaped,N_classes,self.cmap)
        #print("Confidence shape: ", confidence.shape)
        #print("Q_array shape: ", Q_array.shape)
        #print("Q_array values ", Q_array[:10])

        confidence_updated= np.max(Q_array, axis=0).reshape((self.height,self.width))

        #Transform semantic color
        self.semantic_color_vect[:,0:1] = MAP_decoded[:,:,0].reshape(-1,1)
        self.semantic_color_vect[:,1:2] = MAP_decoded[:,:,1].reshape(-1,1)
        self.semantic_color_vect[:,2:3] = MAP_decoded[:,:,2].reshape(-1,1)
        # Concatenate data
        self.ros_data[:,5:6] = self.semantic_color_vect.view('<f4')
        self.ros_data[:,6:7] = confidence_updated.reshape(-1,1)

        return self.make_ros_cloud(stamp)
    
    


    # def generate_cloud_semantic_bayesian(self, bgr_img, depth_img, semantic_colors, confidences, softmax, stamp):
    #     """
    #     Generate semantic point cloud to be used to do bayesian fusion
    #     \param bgr_img (numpy array bgr8) input color image
    #     \param depth_img (numpy array float32) input depth image
    #     \param semantic_colors (list of bgr8 images) semantic colors of different levels of confidences, ordered by confidences (desc)
    #     \param confidences (a list of numpy array float32) confidence maps of associated semantic colors, ordered by values (desc)
    #     \stamp (ros time stamp)
    #     """
        
    #     self.generate_cloud_data_common(bgr_img, depth_img)
    #     self.appearence_features=self.create_pairwise_bilateral_appearence_kernel(sdims, schan)
    #     self.appearence_features = np.ascontiguousarray(self.appearence_features)
    #     smoothness_features=self.create_pairwise_bilateral_smoothness_kernel([0.2,0.2,0.2], snormal)
    #     smoothness_features = np.ascontiguousarray(smoothness_features)    

    #     scale=0.8
    #     self.unary_potentials= self.unary_from_softmax(softmax, scale, clip=1e-5)
    #     self.unary_potentials = np.ascontiguousarray(self.unary_potentials)
         
        
    #     print(" CRF setting")
    #     self.d.setUnaryEnergy(self.unary_potentials)            
       

    #     self.d.addPairwiseEnergy(smoothness_features, compat=smoothness_matrix ,
    #                  kernel=dcrf.DIAG_KERNEL,
    #                  normalization=dcrf.NORMALIZE_SYMMETRIC)
        
    #     self.d.addPairwiseEnergy(self.appearence_features, compat=10,
    #                    kernel=dcrf.DIAG_KERNEL,
    #                   normalization=dcrf.NORMALIZE_SYMMETRIC)
    #     print("CRF Inference")
    #     # # Run five inference steps.
    #     # Q = self.d.inference(5)
    #     # Q_array = np.array(Q)
    #     # # # Get the indices of the top 3 classes for each point
    #     # Q_array_top_3 = np.argsort(Q_array, axis=0)[-3:, :]  # Shape: (3, num_points)
    #     # print(f"Q_array_top_3 shape and type: {Q_array_top_3 .shape}, {Q_array_top_3 .dtype}")

    #     # print(f"Sample Q_array_top_3: {Q_array_top_3[:10]}")

    #     # # #Reshape the top 3 indices to (3, height, width)
    #     # MAP3_reshaped = Q_array_top_3.reshape((3,self.height, self.width)).astype(np.int32)
    #     # print(f"MAP3_reshaped  shape and type: {MAP3_reshaped .shape}, {MAP3_reshaped .dtype}")
    #     # top_3_confidences = np.take_along_axis(Q_array, Q_array_top_3, axis=0)  # Shape: (3, n_points)
    #     # MAP_confidences=top_3_confidences.reshape((3,self.height, self.width)).astype(np.float32)

    #     # #MAP3_decoded = np.zeros((3, 480, 640, 3), dtype=np.uint8)


    #     # # MAP3_decoded=decode_segmap(MAP3_reshaped,N_classes,self.cmap)
    #     # for i in range(self.num_semantic_colors):
    #     #     semantic_colors[i]=decode_segmap(MAP3_reshaped[i], N_classes, self.cmap)

    #     # Perform CRF inference
    #     Q = self.d.inference(5)
    #     Q_array = np.array(Q)

    #     # Get the indices of the top 3 classes for each point
    #     Q_array_top_3 = np.argsort(Q_array, axis=0)[-3:, :]  # Shape: (3, num_points)
    #     top_3_probs = np.take_along_axis(Q_array, Q_array_top_3, axis=0)  # Shape: (3, n_points)

    #     # Reshape the top-3 class indices into 3 separate 2D label maps
    #     MAP3_reshaped = Q_array_top_3.reshape((3, self.height, self.width))  # Shape: (3, height, width)

    #     # Initialize semantic_colors to hold 3 RGB images
    #     semantic_colors = np.zeros((3, self.height, self.width, 3), dtype=np.uint8)

    #     # Decode each of the 3 label maps into an RGB image
    #     for i in range(3):  # Loop over the top-3 classes
    #         semantic_colors[i] = decode_segmap(MAP3_reshaped[i], N_classes, self.cmap)
        

# Now semantic_colors contains the 3 RGB images (3 x height x width x 3)

        




        # # # Find out the most 3 probables classes for each point.
        # # MAP = np.argmax(Q, axis=0)

        # # MAP_reshaped = MAP.reshape((self.height,self.width))
        # # MAP_decoded=decode_segmap(MAP_reshaped,N_classes,self.cmap)
        # #print(f"semantic colors shape and type: {semantic_colors.shape}, {semantic_colors.dtype}")
        # #print(f"confidences shape and type: {confidences.shape}, {confidences.dtype}")


        # # Transform semantic colors
        # for i in range(self.num_semantic_colors):
        #     self.semantic_colors_vect[:,4*i:4*i+1] = semantic_colors[i][:,:,0].reshape(-1,1)
        #     self.semantic_colors_vect[:,4*i+1:4*i+2] = semantic_colors[i][:,:,1].reshape(-1,1)
        #     self.semantic_colors_vect[:,4*i+2:4*i+3] = semantic_colors[i][:,:,2].reshape(-1,1)
        # # Transform class confidence
        # for i in range(self.num_semantic_colors):
        #     self.confidences_vect[:,i:i+1] = confidences[i].reshape(-1,1)
        # # Concatenate data
        # self.ros_data[:,8:8+self.num_semantic_colors] = self.semantic_colors_vect.view('<f4')
        # #self.ros_data[:,12:12+self.num_semantic_colors] = self.confidences_vect
        # self.ros_data[:,12:12+self.num_semantic_colors] = top_3_probs.T
        # return self.make_ros_cloud(stamp)  