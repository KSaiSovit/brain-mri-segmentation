# import cv2
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
# from skimage import segmentation, color, graph
# from skimage.metrics import structural_similarity as ssim
# from sklearn.metrics import jaccard_score
# import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D, UpSampling2D, Input
# from tensorflow.keras.models import Model

# class ConvReLU(tf.keras.layers.Layer):
#     def __init__(self, filters, kernel_size=3, padding=1):
#         super(ConvReLU, self).__init__()
#         self.conv = Conv2D(filters, kernel_size=kernel_size, padding='same')
#         self.bn = BatchNormalization()
#         self.relu = ReLU()

#     def call(self, inputs):
#         x = self.conv(inputs)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x

# class EncoderBlock(tf.keras.layers.Layer):
#     def __init__(self, filters, depth=2, kernel_size=3, padding=1):
#         super(EncoderBlock, self).__init__()
#         self.layers = []
#         for _ in range(depth):
#             self.layers.append(ConvReLU(filters, kernel_size, padding))
#         self.pool = MaxPooling2D(pool_size=2, strides=2)

#     def call(self, inputs):
#         for layer in self.layers:
#             inputs = layer(inputs)
#         return self.pool(inputs)

# class DecoderBlock(tf.keras.layers.Layer):
#     def __init__(self, filters, depth=2, kernel_size=3, padding=1, classification=False):
#         super(DecoderBlock, self).__init__()
#         self.unpool = UpSampling2D(size=2)
#         self.layers = []
#         for i in range(depth):
#             if i == depth - 1 and classification:
#                 self.layers.append(Conv2D(filters, kernel_size=kernel_size, padding='same'))
#             elif i == depth - 1:
#                 self.layers.append(ConvReLU(filters, kernel_size, padding))
#             else:
#                 self.layers.append(ConvReLU(filters, kernel_size, padding))

#     def call(self, inputs):
#         inputs = self.unpool(inputs)
#         for layer in self.layers:
#             inputs = layer(inputs)
#         return inputs

# class SegNet(tf.keras.Model):
#     def __init__(self, in_channels=3, out_channels=1, features=64):
#         super(SegNet, self).__init__()

#         # Encoder
#         self.enc0 = EncoderBlock(features)
#         self.enc1 = EncoderBlock(features * 2)
#         self.enc2 = EncoderBlock(features * 4, depth=3)
#         self.enc3 = EncoderBlock(features * 8, depth=3)

#         # Bottleneck
#         self.bottleneck_enc = EncoderBlock(features * 8, depth=3)
#         self.bottleneck_dec = DecoderBlock(features * 8, depth=3)

#         # Decoder
#         self.dec0 = DecoderBlock(features * 4, depth=3)
#         self.dec1 = DecoderBlock(features * 2, depth=3)
#         self.dec2 = DecoderBlock(features)
#         self.dec3 = DecoderBlock(out_channels, classification=True)

#     def call(self, inputs):
#         # encoder
#         e0 = self.enc0(inputs)
#         e1 = self.enc1(e0)
#         e2 = self.enc2(e1)
#         e3 = self.enc3(e2)

#         # bottleneck
#         b0 = self.bottleneck_enc(e3)
#         b1 = self.bottleneck_dec(b0)

#         # decoder
#         d0 = self.dec0(b1)
#         d1 = self.dec1(d0)
#         d2 = self.dec2(d1)

#         # classification layer
#         output = self.dec3(d2)
#         return output

# def load_segnet_model(input_shape=(256, 256, 3)):
#     """Load and return the SegNet model"""
#     model = SegNet()
#     # Dummy call to build the model
#     model.build((None, *input_shape))
#     return model

# def kmeans_segmentation(image, k=4):
#     """Perform K-Means segmentation on the image"""
#     # Convert to 2D array
#     vals = image.reshape((-1, 3))
#     vals = np.float32(vals)
    
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
#     retval, labels, centers = cv2.kmeans(vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
#     # Convert data into 8-bit values
#     centers = np.uint8(centers)
#     segmented_data = centers[labels.flatten()]
    
#     # Reshape data into the original image dimensions
#     segmented_image = segmented_data.reshape((image.shape))
#     return segmented_image

# def mean_shift_segmentation(image, bandwidth=3):
#     """Perform Mean Shift segmentation on the image"""
#     # Flatten the image array for clustering
#     flat_image = image.reshape(-1, 3)
#     flat_image = np.float32(flat_image)
    
#     # Apply Mean Shift clustering
#     ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
#     ms.fit(flat_image)
    
#     # Get the cluster labels and reshape to the original image shape
#     labels = ms.labels_
#     segmented_image = labels.reshape(image.shape[:2])
    
#     # Convert to color image for visualization
#     segmented_image = color.label2rgb(segmented_image, image, kind='avg')
#     return segmented_image

# def ncut_segmentation(image, n_segments=50, compactness=10):
#     """Perform Normalized Cut segmentation on the image"""
#     # Apply SLIC to get superpixels
#     labels = segmentation.slic(image, compactness=compactness, n_segments=n_segments, 
#                               enforce_connectivity=True, convert2lab=True)
    
#     # Compute the Region Adjacency Graph using mean colors
#     rag = graph.rag_mean_color(image, labels, mode='similarity')
    
#     # Perform Normalized Graph cut on the Region Adjacency Graph
#     labels2 = graph.cut_normalized(labels, rag)
#     segmented_image = color.label2rgb(labels2, image, kind='avg')
#     return segmented_image

# def segnet_predict(model, image, size=(256, 256)):
#     """Run SegNet prediction on the image"""
#     # Preprocess the image
#     original_size = image.shape[:2]
#     image_resized = cv2.resize(image, size)
#     image_normalized = image_resized / 255.0
#     image_input = np.expand_dims(image_normalized, axis=0)
    
#     # Run prediction
#     prediction = model(image_input, training=False)
#     prediction = tf.squeeze(prediction).numpy()
    
#     # Resize back to original size and apply threshold
#     prediction_resized = cv2.resize(prediction, (original_size[1], original_size[0]))
#     segmented_image = (prediction_resized > 0.5).astype('uint8') * 255
    
#     return segmented_image

# def calculate_metrics(original, segmented):
#     """Calculate SSIM and IoU metrics between original and segmented images"""
#     # Convert to grayscale for SSIM
#     if len(original.shape) == 3:
#         original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
#     else:
#         original_gray = original
        
#     if len(segmented.shape) == 3:
#         segmented_gray = cv2.cvtColor(segmented, cv2.COLOR_RGB2GRAY)
#     else:
#         segmented_gray = segmented
    
#     # Calculate SSIM
#     ssim_value = ssim(original_gray, segmented_gray, data_range=segmented_gray.max() - segmented_gray.min())
    
#     # For IoU, we need to binarize the images
#     _, original_bin = cv2.threshold(original_gray, 127, 255, cv2.THRESH_BINARY)
#     _, segmented_bin = cv2.threshold(segmented_gray, 127, 255, cv2.THRESH_BINARY)
    
#     # Calculate IoU
#     intersection = np.logical_and(original_bin, segmented_bin)
#     union = np.logical_or(original_bin, segmented_bin)
#     iou = np.sum(intersection) / np.sum(union)
    
#     return ssim_value, iou

# ------------------
## Version 2
# ------------------

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from skimage import segmentation, color, graph
from skimage.metrics import structural_similarity as ssim, hausdorff_distance
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, ReLU, MaxPooling2D, UpSampling2D, 
    Input, Concatenate, Dropout, Activation, GlobalAveragePooling2D,
    Multiply, Add, Reshape, Permute, Lambda
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import pandas as pd
import os
from datetime import datetime
import json

# ---------------------------
# Enhanced U-Net with Attention
# ---------------------------
def conv_block(x, filters, kernel_size=3, dropout_rate=0.1, activation='relu'):
    """Convolutional block with batch normalization and dropout"""
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    if activation == 'relu':
        x = ReLU()(x)
    elif activation == 'leaky_relu':
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = Dropout(dropout_rate)(x)
    return x

def attention_block(g, x, filters):
    """Attention gate mechanism"""
    g_conv = Conv2D(filters, 1, padding='same')(g)
    g_conv = BatchNormalization()(g_conv)
    
    x_conv = Conv2D(filters, 1, padding='same')(x)
    x_conv = BatchNormalization()(x_conv)
    
    add = Add()([g_conv, x_conv])
    add = ReLU()(add)
    
    psi = Conv2D(1, 1, padding='same')(add)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)
    
    return Multiply()([x, psi])

def unet_attention(input_size=(256, 256, 3), num_classes=1, dropout_rate=0.1, filters=64):
    """U-Net with attention gates for better feature focus"""
    inputs = Input(input_size)
    
    # Encoder
    c1 = conv_block(inputs, filters, dropout_rate=dropout_rate)
    c1 = conv_block(c1, filters, dropout_rate=dropout_rate)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = conv_block(p1, filters*2, dropout_rate=dropout_rate)
    c2 = conv_block(c2, filters*2, dropout_rate=dropout_rate)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = conv_block(p2, filters*4, dropout_rate=dropout_rate)
    c3 = conv_block(c3, filters*4, dropout_rate=dropout_rate)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = conv_block(p3, filters*8, dropout_rate=dropout_rate)
    c4 = conv_block(c4, filters*8, dropout_rate=dropout_rate)
    p4 = MaxPooling2D((2, 2))(c4)
    
    # Bridge
    c5 = conv_block(p4, filters*16, dropout_rate=dropout_rate)
    c5 = conv_block(c5, filters*16, dropout_rate=dropout_rate)
    
    # Decoder with attention
    u6 = UpSampling2D((2, 2))(c5)
    a6 = attention_block(c4, u6, filters*8)
    u6 = Concatenate()([u6, a6])
    c6 = conv_block(u6, filters*8, dropout_rate=dropout_rate)
    c6 = conv_block(c6, filters*8, dropout_rate=dropout_rate)
    
    u7 = UpSampling2D((2, 2))(c6)
    a7 = attention_block(c3, u7, filters*4)
    u7 = Concatenate()([u7, a7])
    c7 = conv_block(u7, filters*4, dropout_rate=dropout_rate)
    c7 = conv_block(c7, filters*4, dropout_rate=dropout_rate)
    
    u8 = UpSampling2D((2, 2))(c7)
    a8 = attention_block(c2, u8, filters*2)
    u8 = Concatenate()([u8, a8])
    c8 = conv_block(u8, filters*2, dropout_rate=dropout_rate)
    c8 = conv_block(c8, filters*2, dropout_rate=dropout_rate)
    
    u9 = UpSampling2D((2, 2))(c8)
    a9 = attention_block(c1, u9, filters)
    u9 = Concatenate()([u9, a9])
    c9 = conv_block(u9, filters, dropout_rate=dropout_rate)
    c9 = conv_block(c9, filters, dropout_rate=dropout_rate)
    
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# ---------------------------
# DeepLabV3+ Implementation
# ---------------------------
def deeplab_v3_plus(input_size=(256, 256, 3), num_classes=1, backbone='resnet50'):
    """DeepLabV3+ architecture for semantic segmentation"""
    # Input
    inputs = Input(shape=input_size)
    
    # Backbone
    if backbone == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
    else:  # vgg16
        base_model = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
    
    # ASPP (Atrous Spatial Pyramid Pooling)
    # Get feature maps from different levels
    if backbone == 'resnet50':
        # For ResNet50
        h = base_model.get_layer('conv4_block6_out').output
        low_level_features = base_model.get_layer('conv2_block3_out').output
    else:
        # For VGG16
        h = base_model.get_layer('block5_pool').output
        low_level_features = base_model.get_layer('block3_pool').output
    
    # ASPP Branches
    # Branch 1: 1x1 convolution
    b1 = Conv2D(256, 1, padding='same', use_bias=False)(h)
    b1 = BatchNormalization()(b1)
    b1 = ReLU()(b1)
    
    # Branch 2: 3x3 convolution with rate=6
    b2 = Conv2D(256, 3, padding='same', dilation_rate=6, use_bias=False)(h)
    b2 = BatchNormalization()(b2)
    b2 = ReLU()(b2)
    
    # Branch 3: 3x3 convolution with rate=12
    b3 = Conv2D(256, 3, padding='same', dilation_rate=12, use_bias=False)(h)
    b3 = BatchNormalization()(b3)
    b3 = ReLU()(b3)
    
    # Branch 4: 3x3 convolution with rate=18
    b4 = Conv2D(256, 3, padding='same', dilation_rate=18, use_bias=False)(h)
    b4 = BatchNormalization()(b4)
    b4 = ReLU()(b4)
    
    # Branch 5: Global Average Pooling
    b5 = GlobalAveragePooling2D()(h)
    b5 = Reshape((1, 1, 2048 if backbone == 'resnet50' else 512))(b5)
    b5 = Conv2D(256, 1, use_bias=False)(b5)
    b5 = BatchNormalization()(b5)
    b5 = ReLU()(b5)
    b5 = UpSampling2D(size=(h.shape[1]//b5.shape[1], h.shape[2]//b5.shape[2]), 
                      interpolation='bilinear')(b5)
    
    # Concatenate ASPP branches
    x = Concatenate()([b1, b2, b3, b4, b5])
    x = Conv2D(256, 1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.1)(x)
    
    # Decoder
    # Low-level features
    low_level_features = Conv2D(48, 1, use_bias=False)(low_level_features)
    low_level_features = BatchNormalization()(low_level_features)
    low_level_features = ReLU()(low_level_features)
    
    # Upsample and concatenate
    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    x = Concatenate()([x, low_level_features])
    
    x = Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.1)(x)
    
    x = Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.1)(x)
    
    # Output
    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    outputs = Conv2D(num_classes, 1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# ---------------------------
# Enhanced Classical Methods
# ---------------------------
def improved_kmeans_segmentation(image, k=4, preprocessing=True):
    """Enhanced K-Means segmentation with preprocessing"""
    # Preprocessing
    if preprocessing:
        # Apply contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Convert to 2D array
    vals = image.reshape((-1, 3))
    vals = np.float32(vals)
    
    # Use k-means++ initialization for better results
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    retval, labels, centers = cv2.kmeans(vals, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    # Convert data into 8-bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    
    # Reshape data into the original image dimensions
    segmented_image = segmented_data.reshape((image.shape))
    
    # Post-processing
    if preprocessing:
        # Apply morphological operations to clean up the segmentation
        gray = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
        kernel = np.ones((3, 3), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        segmented_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    return segmented_image

def improved_mean_shift_segmentation(image, bandwidth=3, preprocessing=True):
    """Enhanced Mean Shift segmentation with preprocessing"""
    # Preprocessing
    if preprocessing:
        # Apply contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Flatten the image array for clustering
    flat_image = image.reshape(-1, 3)
    flat_image = np.float32(flat_image)
    
    # Automatically estimate bandwidth if not provided
    if bandwidth is None:
        bandwidth = estimate_bandwidth(flat_image, quantile=0.2, n_samples=500)
    
    # Apply Mean Shift clustering
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq=100)
    ms.fit(flat_image)
    
    # Get the cluster labels and reshape to the original image shape
    labels = ms.labels_
    segmented_image = labels.reshape(image.shape[:2])
    
    # Convert to color image for visualization
    segmented_image = color.label2rgb(segmented_image, image, kind='avg')
    
    # Post-processing
    if preprocessing:
        # Apply median filtering to reduce noise
        segmented_image = cv2.medianBlur((segmented_image * 255).astype(np.uint8), 5)
        segmented_image = segmented_image.ast(np.float32) / 255
    
    return segmented_image

def improved_ncut_segmentation(image, n_segments=50, compactness=10, preprocessing=True):
    """Enhanced Normalized Cut segmentation with preprocessing"""
    # Preprocessing
    if preprocessing:
        # Apply contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Apply SLIC to get superpixels
    labels = segmentation.slic(image, compactness=compactness, n_segments=n_segments, 
                              enforce_connectivity=True, convert2lab=True, sigma=1)
    
    # Compute the Region Adjacency Graph using mean colors
    rag = graph.rag_mean_color(image, labels, mode='similarity')
    
    # Perform Normalized Graph cut on the Region Adjacency Graph
    labels2 = graph.cut_normalized(labels, rag)
    segmented_image = color.label2rgb(labels2, image, kind='avg')
    
    return segmented_image

# ---------------------------
# Hybrid Approach: Deep Learning + Classical
# ---------------------------
def hybrid_segmentation(image, deep_learning_model, classical_method='kmeans', alpha=0.7):
    """
    Hybrid segmentation combining deep learning and classical methods
    alpha: Weight for deep learning result (0-1)
    """
    # Get deep learning segmentation
    dl_segmentation = segnet_predict(deep_learning_model, image)
    
    # Get classical segmentation
    if classical_method == 'kmeans':
        classical_seg = improved_kmeans_segmentation(image)
    elif classical_method == 'meanshift':
        classical_seg = improved_mean_shift_segmentation(image)
    elif classical_method == 'ncut':
        classical_seg = improved_ncut_segmentation(image)
    else:
        raise ValueError("Unknown classical method")
    
    # Convert to grayscale for combination
    if len(dl_segmentation.shape) == 3:
        dl_gray = cv2.cvtColor(dl_segmentation, cv2.COLOR_RGB2GRAY)
    else:
        dl_gray = dl_segmentation
        
    if len(classical_seg.shape) == 3:
        classical_gray = cv2.cvtColor(classical_seg, cv2.COLOR_RGB2GRAY)
    else:
        classical_gray = classical_seg
    
    # Normalize
    dl_gray = dl_gray.astype(np.float32) / 255
    classical_gray = classical_gray.astype(np.float32) / 255
    
    # Combine
    hybrid_seg = alpha * dl_gray + (1 - alpha) * classical_gray
    hybrid_seg = (hybrid_seg * 255).astype(np.uint8)
    
    # Apply threshold
    _, hybrid_seg = cv2.threshold(hybrid_seg, 127, 255, cv2.THRESH_BINARY)
    
    # Convert back to RGB for display
    hybrid_seg = cv2.cvtColor(hybrid_seg, cv2.COLOR_GRAY2RGB)
    
    return hybrid_seg

# ---------------------------
# Model Loading and Prediction
# ---------------------------
def load_advanced_model(model_type='unet_attention', input_shape=(256, 256, 3)):
    """Load advanced segmentation model"""
    if model_type == 'unet_attention':
        model = unet_attention(input_shape)
    elif model_type == 'deeplabv3plus':
        model = deeplab_v3_plus(input_shape)
    else:
        raise ValueError("Unknown model type")
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', iou_metric, dice_coef]
    )
    
    return model

def segnet_predict(model, image, size=(256, 256)):
    """Run model prediction on the image"""
    # Preprocess the image
    original_size = image.shape[:2]
    image_resized = cv2.resize(image, size)
    image_normalized = image_resized / 255.0
    image_input = np.expand_dims(image_normalized, axis=0)
    
    # Run prediction
    prediction = model.predict(image_input)
    prediction = np.squeeze(prediction)
    
    # Resize back to original size and apply threshold
    prediction_resized = cv2.resize(prediction, (original_size[1], original_size[0]))
    segmented_image = (prediction_resized > 0.5).astype('uint8') * 255
    
    return segmented_image

# ---------------------------
# Advanced Metrics
# ---------------------------
def iou_metric(y_true, y_pred):
    """Intersection over Union metric"""
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + tf.keras.backend.epsilon())

def dice_coef(y_true, y_pred):
    """Dice coefficient metric"""
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + tf.keras.backend.epsilon())

def calculate_all_metrics(original, segmented, ground_truth=None):
    """Calculate comprehensive metrics between original and segmented images"""
    metrics = {}
    
    # Convert to grayscale for some metrics
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    else:
        original_gray = original
        
    if len(segmented.shape) == 3:
        segmented_gray = cv2.cvtColor(segmented, cv2.COLOR_RGB2GRAY)
    else:
        segmented_gray = segmented
    
    # Calculate SSIM
    metrics['ssim'] = ssim(original_gray, segmented_gray, 
                          data_range=segmented_gray.max() - segmented_gray.min())
    
    # For binary metrics, we need to binarize the images
    _, original_bin = cv2.threshold(original_gray, 127, 255, cv2.THRESH_BINARY)
    _, segmented_bin = cv2.threshold(segmented_gray, 127, 255, cv2.THRESH_BINARY)
    
    # Calculate IoU
    intersection = np.logical_and(original_bin, segmented_bin)
    union = np.logical_or(original_bin, segmented_bin)
    metrics['iou'] = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    
    # Calculate Dice coefficient
    metrics['dice'] = (2 * np.sum(intersection)) / (np.sum(original_bin) + np.sum(segmented_bin))
    
    # Calculate precision, recall, and F1 score
    metrics['precision'] = precision_score(original_bin.flatten() > 0, segmented_bin.flatten() > 0, 
                                         zero_division=0)
    metrics['recall'] = recall_score(original_bin.flatten() > 0, segmented_bin.flatten() > 0,
                                   zero_division=0)
    metrics['f1'] = f1_score(original_bin.flatten() > 0, segmented_bin.flatten() > 0,
                           zero_division=0)
    
    # Calculate Hausdorff distance if ground truth is provided
    if ground_truth is not None:
        if len(ground_truth.shape) == 3:
            ground_truth_gray = cv2.cvtColor(ground_truth, cv2.COLOR_RGB2GRAY)
        else:
            ground_truth_gray = ground_truth
            
        _, ground_truth_bin = cv2.threshold(ground_truth_gray, 127, 255, cv2.THRESH_BINARY)
        try:
            metrics['hausdorff'] = hausdorff_distance(ground_truth_bin > 0, segmented_bin > 0)
        except:
            metrics['hausdorff'] = float('inf')
    
    # Calculate additional metrics
    metrics['accuracy'] = np.sum(original_bin == segmented_bin) / original_bin.size
    
    return metrics

# ---------------------------
# Training Functions
# ---------------------------
def train_advanced_model(model, train_data, val_data, epochs=50, model_save_path="advanced_model.h5"):
    """Train the advanced model"""
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
        ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7, monitor='val_loss')
    ]
    
    history = model.fit(
        train_data[0], train_data[1],
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
        batch_size=8  # Smaller batch size for better generalization
    )
    
    return model, history

def create_training_report(history, metrics, model_path):
    """Create a comprehensive training report"""
    report = {
        'training_date': datetime.now().isoformat(),
        'model_path': model_path,
        'training_history': {
            'loss': [float(val) for val in history.history['loss']],
            'val_loss': [float(val) for val in history.history['val_loss']],
            'accuracy': [float(val) for val in history.history['accuracy']],
            'val_accuracy': [float(val) for val in history.history['val_accuracy']],
            'iou': [float(val) for val in history.history.get('iou_metric', [])],
            'val_iou': [float(val) for val in history.history.get('val_iou_metric', [])],
            'dice': [float(val) for val in history.history.get('dice_coef', [])],
            'val_dice': [float(val) for val in history.history.get('val_dice_coef', [])]
        },
        'final_metrics': metrics,
        'model_summary': []
    }
    
    return report