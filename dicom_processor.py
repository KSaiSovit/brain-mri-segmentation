import pydicom
import numpy as np
import cv2
from PIL import Image
import io

def read_dicom_file(file_bytes):
    """Read DICOM file and convert to RGB image"""
    try:
        # Read DICOM file
        dicom_dataset = pydicom.dcmread(io.BytesIO(file_bytes))
        
        # Extract pixel data
        pixel_array = dicom_dataset.pixel_array
        
        # Normalize to 0-255
        if pixel_array.dtype != np.uint8:
            pixel_array = ((pixel_array - pixel_array.min()) / 
                          (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        
        # Handle different photometric interpretations
        if hasattr(dicom_dataset, 'PhotometricInterpretation'):
            if dicom_dataset.PhotometricInterpretation == 'MONOCHROME1':
                # Invert monochrome
                pixel_array = 255 - pixel_array
        
        # Convert to 3-channel if grayscale
        if len(pixel_array.shape) == 2:
            pixel_array = np.stack([pixel_array] * 3, axis=-1)
        
        return pixel_array, dicom_dataset
        
    except Exception as e:
        raise Exception(f"Error reading DICOM file: {str(e)}")

def extract_dicom_metadata(dicom_dataset):
    """Extract and format DICOM metadata"""
    metadata = {}
    exclude_tags = ['PixelData', 'PixelDataProviderURL']
    
    for elem in dicom_dataset:
        if elem.tag not in exclude_tags:
            key = f"{elem.tag} {elem.name}"
            try:
                metadata[key] = str(elem.value)
            except:
                metadata[key] = "Binary or unsupported data type"
    
    return metadata