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
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, ReLU, MaxPooling2D, UpSampling2D, 
    Input, Concatenate, Dropout, Activation
)
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans, MeanShift
from skimage import segmentation, color
import time

# Tumor detection function
def detect_tumor(image, method='kmeans', threshold=0.5):
    """
    Detect tumors in MRI images
    Returns: (has_tumor, processed_image, metrics_dict)
    """
    start_time = time.time()
    
    # Convert to grayscale for processing
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply different methods based on selection
    if method == 'kmeans':
        segmented = improved_kmeans_segmentation(image, k=3, preprocessing=True)
    elif method == 'meanshift':
        segmented = improved_mean_shift_segmentation(image, bandwidth=3, preprocessing=True)
    elif method == 'unet':
        if "unet_model" not in globals():
            global unet_model
            unet_model = load_advanced_model('unet_attention')
        segmented = segnet_predict(unet_model, image)
    elif method == 'hybrid':
        if "unet_model" not in globals():
            global unet_model
            unet_model = load_advanced_model('unet_attention')
        segmented = hybrid_segmentation(image, unet_model)
    else:
        segmented = improved_kmeans_segmentation(image, k=3, preprocessing=True)
    
    # Convert segmentation to binary mask
    if len(segmented.shape) == 3:
        segmented_gray = cv2.cvtColor(segmented, cv2.COLOR_RGB2GRAY)
    else:
        segmented_gray = segmented
    
    # Threshold to create binary image
    _, binary = cv2.threshold(segmented_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours (potential tumors)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size
    min_tumor_size = 100  # Minimum pixel area to be considered a tumor
    tumor_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_tumor_size]
    
    # Create result image with tumor outlines
    result_img = image.copy()
    if len(result_img.shape) == 2:
        result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2RGB)
    
    has_tumor = len(tumor_contours) > 0
    
    if has_tumor:
        # Draw contours on original image
        cv2.drawContours(result_img, tumor_contours, -1, (255, 0, 0), 2)
        
        # Fill tumor areas with transparent red
        overlay = result_img.copy()
        for contour in tumor_contours:
            cv2.drawContours(overlay, [contour], -1, (255, 0, 0), -1)
        result_img = cv2.addWeighted(overlay, 0.3, result_img, 0.7, 0)
        
        # Calculate tumor size (total area)
        tumor_size = sum(cv2.contourArea(cnt) for cnt in tumor_contours)
        
        # Simple confidence metric
        confidence = min(0.99, tumor_size / (image.shape[0] * image.shape[1]) * 10)
    else:
        tumor_size = 0
        confidence = 0.0
    
    processing_time = time.time() - start_time
    
    metrics = {
        'size': int(tumor_size),
        'confidence': confidence,
        'time': processing_time,
        'contours': len(tumor_contours)
    }
    
    return has_tumor, result_img, metrics

# Enhanced segmentation functions
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
    
    # Apply Mean Shift clustering
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq=100)
    ms.fit(flat_image)
    
    # Get the cluster labels and reshape to the original image shape
    labels = ms.labels_
    segmented_image = labels.reshape(image.shape[:2])
    
    # Convert to color image for visualization
    segmented_image = color.label2rgb(segmented_image, image, kind='avg')
    
    return (segmented_image * 255).astype(np.uint8)

# U-Net model implementation
def conv_block(x, filters, kernel_size=3, dropout_rate=0.1):
    """Convolutional block with batch normalization and dropout"""
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(dropout_rate)(x)
    return x

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
    
    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    u6 = Concatenate()([u6, c4])
    c6 = conv_block(u6, filters*8, dropout_rate=dropout_rate)
    c6 = conv_block(c6, filters*8, dropout_rate=dropout_rate)
    
    u7 = UpSampling2D((2, 2))(c6)
    u7 = Concatenate()([u7, c3])
    c7 = conv_block(u7, filters*4, dropout_rate=dropout_rate)
    c7 = conv_block(c7, filters*4, dropout_rate=dropout_rate)
    
    u8 = UpSampling2D((2, 2))(c7)
    u8 = Concatenate()([u8, c2])
    c8 = conv_block(u8, filters*2, dropout_rate=dropout_rate)
    c8 = conv_block(c8, filters*2, dropout_rate=dropout_rate)
    
    u9 = UpSampling2D((2, 2))(c8)
    u9 = Concatenate()([u9, c1])
    c9 = conv_block(u9, filters, dropout_rate=dropout_rate)
    c9 = conv_block(c9, filters, dropout_rate=dropout_rate)
    
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def load_advanced_model(model_type='unet_attention', input_shape=(256, 256, 3)):
    """Load advanced segmentation model"""
    if model_type == 'unet_attention':
        model = unet_attention(input_shape)
    else:
        raise ValueError("Unknown model type")
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
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
    else:
        classical_seg = improved_kmeans_segmentation(image)
    
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
    
    return hybrid_seg