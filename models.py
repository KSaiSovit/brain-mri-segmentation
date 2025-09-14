import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from skimage import segmentation, color, graph
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import jaccard_score
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.models import Model

class ConvReLU(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, padding=1):
        super(ConvReLU, self).__init__()
        self.conv = Conv2D(filters, kernel_size=kernel_size, padding='same')
        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters, depth=2, kernel_size=3, padding=1):
        super(EncoderBlock, self).__init__()
        self.layers = []
        for _ in range(depth):
            self.layers.append(ConvReLU(filters, kernel_size, padding))
        self.pool = MaxPooling2D(pool_size=2, strides=2)

    def call(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return self.pool(inputs)

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters, depth=2, kernel_size=3, padding=1, classification=False):
        super(DecoderBlock, self).__init__()
        self.unpool = UpSampling2D(size=2)
        self.layers = []
        for i in range(depth):
            if i == depth - 1 and classification:
                self.layers.append(Conv2D(filters, kernel_size=kernel_size, padding='same'))
            elif i == depth - 1:
                self.layers.append(ConvReLU(filters, kernel_size, padding))
            else:
                self.layers.append(ConvReLU(filters, kernel_size, padding))

    def call(self, inputs):
        inputs = self.unpool(inputs)
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

class SegNet(tf.keras.Model):
    def __init__(self, in_channels=3, out_channels=1, features=64):
        super(SegNet, self).__init__()

        # Encoder
        self.enc0 = EncoderBlock(features)
        self.enc1 = EncoderBlock(features * 2)
        self.enc2 = EncoderBlock(features * 4, depth=3)
        self.enc3 = EncoderBlock(features * 8, depth=3)

        # Bottleneck
        self.bottleneck_enc = EncoderBlock(features * 8, depth=3)
        self.bottleneck_dec = DecoderBlock(features * 8, depth=3)

        # Decoder
        self.dec0 = DecoderBlock(features * 4, depth=3)
        self.dec1 = DecoderBlock(features * 2, depth=3)
        self.dec2 = DecoderBlock(features)
        self.dec3 = DecoderBlock(out_channels, classification=True)

    def call(self, inputs):
        # encoder
        e0 = self.enc0(inputs)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # bottleneck
        b0 = self.bottleneck_enc(e3)
        b1 = self.bottleneck_dec(b0)

        # decoder
        d0 = self.dec0(b1)
        d1 = self.dec1(d0)
        d2 = self.dec2(d1)

        # classification layer
        output = self.dec3(d2)
        return output

def load_segnet_model(input_shape=(256, 256, 3)):
    """Load and return the SegNet model"""
    model = SegNet()
    # Dummy call to build the model
    model.build((None, *input_shape))
    return model

def kmeans_segmentation(image, k=4):
    """Perform K-Means segmentation on the image"""
    # Convert to 2D array
    vals = image.reshape((-1, 3))
    vals = np.float32(vals)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    retval, labels, centers = cv2.kmeans(vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert data into 8-bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    
    # Reshape data into the original image dimensions
    segmented_image = segmented_data.reshape((image.shape))
    return segmented_image

def mean_shift_segmentation(image, bandwidth=3):
    """Perform Mean Shift segmentation on the image"""
    # Flatten the image array for clustering
    flat_image = image.reshape(-1, 3)
    flat_image = np.float32(flat_image)
    
    # Apply Mean Shift clustering
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(flat_image)
    
    # Get the cluster labels and reshape to the original image shape
    labels = ms.labels_
    segmented_image = labels.reshape(image.shape[:2])
    
    # Convert to color image for visualization
    segmented_image = color.label2rgb(segmented_image, image, kind='avg')
    return segmented_image

def ncut_segmentation(image, n_segments=50, compactness=10):
    """Perform Normalized Cut segmentation on the image"""
    # Apply SLIC to get superpixels
    labels = segmentation.slic(image, compactness=compactness, n_segments=n_segments, 
                              enforce_connectivity=True, convert2lab=True)
    
    # Compute the Region Adjacency Graph using mean colors
    rag = graph.rag_mean_color(image, labels, mode='similarity')
    
    # Perform Normalized Graph cut on the Region Adjacency Graph
    labels2 = graph.cut_normalized(labels, rag)
    segmented_image = color.label2rgb(labels2, image, kind='avg')
    return segmented_image

def segnet_predict(model, image, size=(256, 256)):
    """Run SegNet prediction on the image"""
    # Preprocess the image
    original_size = image.shape[:2]
    image_resized = cv2.resize(image, size)
    image_normalized = image_resized / 255.0
    image_input = np.expand_dims(image_normalized, axis=0)
    
    # Run prediction
    prediction = model(image_input, training=False)
    prediction = tf.squeeze(prediction).numpy()
    
    # Resize back to original size and apply threshold
    prediction_resized = cv2.resize(prediction, (original_size[1], original_size[0]))
    segmented_image = (prediction_resized > 0.5).astype('uint8') * 255
    
    return segmented_image

def calculate_metrics(original, segmented):
    """Calculate SSIM and IoU metrics between original and segmented images"""
    # Convert to grayscale for SSIM
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    else:
        original_gray = original
        
    if len(segmented.shape) == 3:
        segmented_gray = cv2.cvtColor(segmented, cv2.COLOR_RGB2GRAY)
    else:
        segmented_gray = segmented
    
    # Calculate SSIM
    ssim_value = ssim(original_gray, segmented_gray, data_range=segmented_gray.max() - segmented_gray.min())
    
    # For IoU, we need to binarize the images
    _, original_bin = cv2.threshold(original_gray, 127, 255, cv2.THRESH_BINARY)
    _, segmented_bin = cv2.threshold(segmented_gray, 127, 255, cv2.THRESH_BINARY)
    
    # Calculate IoU
    intersection = np.logical_and(original_bin, segmented_bin)
    union = np.logical_or(original_bin, segmented_bin)
    iou = np.sum(intersection) / np.sum(union)
    
    return ssim_value, iou