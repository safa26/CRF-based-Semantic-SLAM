#!/usr/bin/env python3

from __future__ import division
from __future__ import print_function

import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import numpy as np

from sensor_msgs.msg import PointCloud2
from color_pcl_generator import PointType, ColorPclGenerator
import message_filters
import time

from skimage.transform import resize
import cv2

import torch
from ptsemseg.models import get_model
from ptsemseg.utils import convert_state_dict



original_classes = ['background', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
                    'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror',
                    'floor_mat', 'clothes', 'ceiling', 'books', 'fridge', 'tv', 'paper', 'towel', 'shower_curtain',
                    'box', 'whiteboard', 'person', 'night_stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag']

Merged_classes=['background', 'wall', 'floor', 'chair', 'table', 'door', 'window', 'other_object']

N_classes=8

merge_classes = {
    0: 0,  # background -> background
    1: 1,  # wall -> wall
    2: 2,  # floor -> floor
    5: 3,  # chair -> chair
    6: 3,  # sofa -> chair
    7: 4,  # table -> table
    14: 4, # desk -> table
    8: 5,  # door -> door
    9: 6,  # window -> window
}

default_class = 7  # 'other_object'

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

class SemanticCloud:
    """
    Class for ros node to take in a color image (bgr) and do semantic segmantation on it to produce an image with semantic class colors (chair, desk etc.)
    Then produce point cloud based on depth information
    CNN: PSPNet (https://arxiv.org/abs/1612.01105) (with resnet50) pretrained on ADE20K, fine tuned on SUNRGBD or not
    """
    def __init__(self, gen_pcl = True):
        """
        Constructor
        \param gen_pcl (bool) whether generate point cloud, if set to true the node will subscribe to depth image
        """
        # Get point type
        point_type = rospy.get_param('/semantic_pcl/point_type')
       
        #point_type = 0
        if point_type == 0:
            self.point_type = PointType.COLOR
            print('Generate color point cloud.')
        elif point_type == 1:
            self.point_type = PointType.SEMANTICS_MAX
            print('Generate semantic point cloud [max fusion].')
        
        elif point_type == 2:
            self.point_type = PointType.SEMANTICS_BAYESIAN
            print('Generate semantic point cloud [bayesian fusion].')
        else:
            print("Invalid point type.")
            return
        # Get image size
        # self.img_width, self.img_height = rospy.get_param('/camera/width'), rospy.get_param('/camera/height')
        self.img_width, self.img_height = 640, 480
        # Set up CNN is use semantics
        if self.point_type is PointType.COLOR:
             print('Colored Point Cloud...')
             # Set device
             self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #GPU: device=cuda
            # # Get dataset
             dataset = rospy.get_param('/semantic_pcl/dataset')
            # # Setup model
             model_name ='pspnet'
             model_path = rospy.get_param('/semantic_pcl/model_path')
            # #model_path = '/home/yubao/data/SpacialAI/catkin_ws/src/dataset/pspnet_sunrgbd_best_model180625_5k.pth'
             if dataset == 'sunrgbd': # If use version fine tuned on sunrgbd dataset
                 self.n_classes = 38 # Semantic class number
                 self.model = get_model(model_name, self.n_classes, version = 'sunrgbd_res50')
                 state = torch.load(model_path, map_location="cpu")
                 self.model.load_state_dict(state)
                 self.cnn_input_size = (321, 321)
                 self.mean = np.array([104.00699, 116.66877, 122.67892]) # Mean value of dataset
             elif dataset == 'ade20k':
                 self.n_classes = 150 # Semantic class number
                 self.model = get_model(model_name, self.n_classes, version = 'ade20k')
                 state = torch.load(model_path, map_location="cpu")
                 self.model.load_state_dict(convert_state_dict(state['model_state'])) # Remove 'module' from dictionary keys
                 self.cnn_input_size = (473, 473)
                 self.mean = np.array([104.00699, 116.66877, 122.67892]) # Mean value of dataset
             self.model = self.model.to(self.device)
             self.model.eval()
             self.cmap = color_map(N = self.n_classes, normalized = False) # Color map for semantic classes

        if self.point_type is not PointType.COLOR:
            print('Setting up CNN model...')
            # Set device
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Get dataset
            dataset = rospy.get_param('/semantic_pcl/dataset')
            # Setup model
            model_name ='pspnet'
            model_path = rospy.get_param('/semantic_pcl/model_path')
            if dataset == 'sunrgbd': # If use version fine tuned on sunrgbd dataset
                print("!!SUNRGBD!!")
                self.n_classes = 8 # Semantic class number
                self.model = get_model(model_name, self.n_classes, version = 'sunrgbd_res50')
                state = torch.load(model_path, map_location='cpu')
                #self.model.load_state_dict(state)      
                self.model.load_state_dict(state)                                                             
                self.cnn_input_size = (321, 321)
                self.mean = np.array([104.00699, 116.66877, 122.67892]) # Mean value of dataset
            elif dataset == 'ade20k':
                self.n_classes = 150 # Semantic class number
                self.model = get_model(model_name, self.n_classes, version = 'ade20k')
                state = torch.load(model_path, map_location='cpu')
                self.model.load_state_dict(convert_state_dict(state['model_state'])) # Remove 'module' from dictionary keys
                self.cnn_input_size = (473, 473)
                self.mean = np.array([104.00699, 116.66877, 122.67892]) # Mean value of dataset
            self.model = self.model.to(self.device)
            self.model.eval()
            self.cmap = color_map(N = self.n_classes, normalized = False) # Color map for semantic classes
         # Declare array containers
        if self.point_type is PointType.SEMANTICS_BAYESIAN:
            self.semantic_colors = np.zeros((3, self.img_height, self.img_width, 3), dtype = np.uint8) # Numpy array to store 3 decoded semantic images with highest confidences
            self.confidences = np.zeros((3, self.img_height, self.img_width), dtype = np.float32) # Numpy array to store top 3 class confidences
                
        # Set up ROS
        print('Setting up ROS...')
        self.bridge = CvBridge() # CvBridge to transform ROS Image message to OpenCV image
        # Semantic image publisher
        self.sem_img_pub = rospy.Publisher("/semantic_pcl/semantic_image", Image, queue_size = 1)
        # Set up ros image subscriber
        # Set buff_size to average msg size to avoid accumulating delay
        if gen_pcl:
            # Point cloud frame id
            frame_id = rospy.get_param('/semantic_pcl/frame_id')
            print(frame_id)
            # Camera intrinsic matrix
            fx = rospy.get_param('/camera/fx')
            fy = rospy.get_param('/camera/fy')
            cx = rospy.get_param('/camera/cx')
            cy = rospy.get_param('/camera/cy')
            intrinsic = np.matrix([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype = np.float32)
            
            self.pcl_pub = rospy.Publisher("/semantic_pcl/semantic_pcl", PointCloud2, queue_size = 1)
            self.color_sub = message_filters.Subscriber(rospy.get_param('/semantic_pcl/color_image_topic'), Image, queue_size = 10, buff_size = 30*480*640)
            self.depth_sub = message_filters.Subscriber(rospy.get_param('/semantic_pcl/depth_image_topic'), Image, queue_size = 10, buff_size = 40*480*640 ) # increase buffer size to avoid del ay (despite queue_size = 1)
            self.ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size = 1, slop = 0.3) # Take in one color image and one depth image with a limite time gap between message time stamps
            self.ts.registerCallback(self.color_depth_callback)
            self.cloud_generator = ColorPclGenerator(intrinsic, self.img_width,self.img_height, frame_id , self.point_type)
            print(self.point_type)
        else:
            self.image_sub = rospy.Subscriber(rospy.get_param('/semantic_pcl/color_image_topic'), Image, self.color_callback, queue_size = 1, buff_size = 30*480*640)
        print('Ready.')

    def color_callback(self, color_img_ros):
        """
        Callback function for color image, de semantic segmantation and show the decoded image. For test purpose
        \param color_img_ros (sensor_msgs.Image) input ros color image message
        """
        print('callback')
        try:
            color_img = self.bridge.imgmsg_to_cv2(color_img_ros, "bgr8") # Convert ros msg to numpy array
        except CvBridgeError as e:
            print(e)

        # color_img = resize(color_img, (self.img_height, self.img_width),  mode = 'reflect', anti_aliasing=True, preserve_range = True)
        color_img = cv2.resize(color_img, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)

        # Do semantic segmantation
        class_probs = self.predict(color_img)
        confidence, label = class_probs.max(1)
        confidence, label = confidence.squeeze(0).cpu().numpy(), label.squeeze(0).cpu().numpy()
        label = resize(label, (self.img_height, self.img_width), order = 0, mode = 'reflect', anti_aliasing=False, preserve_range = True) # order = 0, nearest neighbour
        label = label.astype(np.int)

        #Add semantic class colors
        decoded = decode_segmap(label, self.n_classes, self.cmap)        # Show input image and decoded image

        class_labels = np.unique(label)
        print(class_labels)

        confidence = resize(confidence, (self.img_height, self.img_width),  mode = 'reflect', anti_aliasing=True, preserve_range = True)
        cv2.imshow('Camera image', color_img)
        cv2.imshow('confidence', confidence)
        cv2.imshow('Semantic segmantation', decoded)
        cv2.waitKey(3)

    def color_depth_callback(self, color_img_ros, depth_img_ros):
        """
        Callback function to produce point cloud registered with semantic class color based on input color image and depth image
        \param color_img_ros (sensor_msgs.Image) the input color image (bgr8)
        \param depth_img_ros (sensor_msgs.Image) the input depth image (registered to the color image frame) (float32) values are in meters
        """
        # Convert ros Image message to numpy array
        try:
            color_img = self.bridge.imgmsg_to_cv2(color_img_ros, "passthrough")
            depth_img = self.bridge.imgmsg_to_cv2(depth_img_ros, "32FC1")
        except CvBridgeError as e:
            print(e)

        # Resize RGB image
        if depth_img.shape[0] is not self.img_height or depth_img.shape[1] is not self.img_width:
            color_img = cv2.resize(color_img, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)

        # Resize depth
        if depth_img.shape[0] is not self.img_height or depth_img.shape[1] is not self.img_width:
            depth_img = resize(depth_img, (self.img_height, self.img_width), order = 0, mode = 'reflect', anti_aliasing=False, preserve_range = True) # order = 0, nearest neighbour
            depth_img = depth_img.astype(np.float32)

        if self.point_type is PointType.COLOR:
            cloud_ros = self.cloud_generator.generate_cloud_color(color_img, depth_img, color_img_ros.header.stamp)
        else:
            # Do semantic segmentation
            if self.point_type is PointType.SEMANTICS_MAX:
                #semantic_color, pred_confidence = self.predict_max(color_img)
                semantic_color, class_labels, pred_confidence, softmax = self.predict_max(color_img) 

                cloud_ros = self.cloud_generator.generate_cloud_semantic_max(color_img, depth_img, semantic_color,class_labels, pred_confidence, softmax,color_img_ros.header.stamp)
                
            elif self.point_type is PointType.SEMANTICS_BAYESIAN:
                softmax=self.predict_bayesian(color_img)
                # Produce point cloud with rgb colors, semantic colors and confidences
                cloud_ros = self.cloud_generator.generate_cloud_semantic_bayesian(color_img, depth_img, self.semantic_colors, self.confidences, softmax, color_img_ros.header.stamp)

            # Publish semantic image
            if self.sem_img_pub.get_num_connections() > 0:
                if self.point_type is PointType.SEMANTICS_MAX:
                    semantic_color_msg = self.bridge.cv2_to_imgmsg(semantic_color, encoding="bgr8")
                else:
                    semantic_color_msg = self.bridge.cv2_to_imgmsg(self.semantic_colors[0], encoding="bgr8")
                self.sem_img_pub.publish(semantic_color_msg)

        # Publish point cloud
        self.pcl_pub.publish(cloud_ros)

      

    def predict_max(self, img):
        """
        Do semantic prediction for max fusion
        \param img (numpy array rgb8)
        """
        class_probs = self.predict(img)
        class_probs_merged= self.remap_and_merge_classes(class_probs)
        Softmax_outputs= class_probs_merged.squeeze(0).cpu().numpy()
        Softmax_outputs = resize( Softmax_outputs, (self.n_classes,self.img_height, self.img_width), order = 0, mode = 'reflect', anti_aliasing=False, preserve_range = True) # order = 0, nearest neighbour
        # Reshape to (8, 321 * 321)
        Softmax_outputs = Softmax_outputs.reshape(self.n_classes, -1)
        # Take best prediction and confidence
        pred_confidence, pred_label = class_probs_merged.max(1)
        pred_confidence = pred_confidence.squeeze(0).cpu().numpy()
        pred_label = pred_label.squeeze(0).cpu().numpy()
        pred_label = resize(pred_label, (self.img_height, self.img_width), order = 0, mode = 'reflect', anti_aliasing=False, preserve_range = True) # order = 0, nearest neighbour
        pred_label = pred_label.astype(np.int)
        # Add semantic color
        semantic_color = decode_segmap(pred_label, self.n_classes, self.cmap)
        pred_confidence = resize(pred_confidence, (self.img_height, self.img_width),  mode = 'reflect', anti_aliasing=True, preserve_range = True)
        return (semantic_color, pred_label, pred_confidence,Softmax_outputs)


    def predict(self, img):
        """
        Do semantic segmantation
        \param img: (numpy array bgr8) The input cv image
        """
        img = img.copy() # Make a copy of image because the method will modify the image
        #orig_size = (img.shape[0], img.shape[1]) # Original image size
        # Prepare image: first resize to CNN input size then extract the mean value of SUNRGBD dataset. No normalization
        img = resize(img, self.cnn_input_size, mode = 'reflect', anti_aliasing=True, preserve_range = True) # Give float64
        img = img.astype(np.float32)
        img -= self.mean
        # Convert HWC -> CHW
        img = img.transpose(2, 0, 1)
        # Convert to tensor
        img = torch.tensor(img, dtype = torch.float32)
        img = img.unsqueeze(0) # Add batch dimension required by CNN
        with torch.no_grad():
            img = img.to(self.device)
            # Do inference
            since = time.time()
            outputs = self.model(img) #N,C,W,H
            # Apply softmax to obtain normalized probabilities
            outputs = torch.nn.functional.softmax(outputs, 1)
            return outputs
        

    def remap_and_merge_classes(self, class_probs): #shape (1,38,321,321)

        
        batch_size, num_classes, height, width = class_probs.shape
        assert batch_size == 1, "Batch size must be 1 for this function."

        #print(f"batchsize: {batch_size}, num_classes: {num_classes}, height: {height}, width: {width}")
        # Number of target classes (10) based on the merge_map
        num_target_classes = len(set(merge_classes.values()))

        #print(f"num_target_classes :{num_target_classes }")

        # Initialize the output tensor
        output_tensor = torch.zeros((batch_size, num_target_classes+1, height, width), dtype=class_probs.dtype, device=class_probs.device)

        #print(f"output_tensor shape: {output_tensor.shape}")

        # Iterate over the merge map and remap classes
        for orig_class, new_class in merge_classes.items():
            if orig_class in [3, 4, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]:
                output_tensor[:, default_class, :, :] += class_probs[:, orig_class, :, :]
            else:
                output_tensor[:, new_class, :, :] += class_probs[:, orig_class, :, :]

        output_tensor /= output_tensor.sum(dim=1, keepdim=True) + 1e-8

        
        return output_tensor

        
    def predict_bayesian(self, img):
        """
        Do semantic prediction for bayesian fusion
        \param img (numpy array rgb8)
        """
        class_probs = self.predict(img)
        class_probs_merged= self.remap_and_merge_classes(class_probs)
        Softmax_outputs= class_probs_merged.squeeze(0).cpu().numpy()
        Softmax_outputs = resize( Softmax_outputs, (self.n_classes,self.img_height, self.img_width), order = 0, mode = 'reflect', anti_aliasing=False, preserve_range = True) # order = 0, nearest neighbour
        # Reshape to (8, 321 * 321)
        Softmax_outputs = Softmax_outputs.reshape(self.n_classes, -1)
        # Take best prediction and confidence
        pred_confidences, pred_labels  = torch.topk(input = class_probs_merged, k = 3, dim = 1, largest = True, sorted = True)
        pred_labels = pred_labels.squeeze(0).cpu().numpy()
        pred_confidences = pred_confidences.squeeze(0).cpu().numpy()
              
        
        
        # Resize predicted labels and confidences to original image size
        for i in range(pred_labels.shape[0]):
            pred_labels_resized = resize(pred_labels[i], (self.img_height, self.img_width), order = 0, mode = 'reflect', anti_aliasing=False, preserve_range = True) # order = 0, nearest neighbour
            pred_labels_resized = pred_labels_resized.astype(np.int)
            # Add semantic class colors
            self.semantic_colors[i] = decode_segmap(pred_labels_resized, self.n_classes, self.cmap)
            
        for i in range(pred_confidences.shape[0]):
            self.confidences[i] = resize(pred_confidences[i], (self.img_height, self.img_width),  mode = 'reflect', anti_aliasing=True, preserve_range = True)

        return Softmax_outputs

        



def main(args):
    rospy.init_node('semantic_cloud', anonymous=True)
    seg_cnn = SemanticCloud(gen_pcl = True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)
