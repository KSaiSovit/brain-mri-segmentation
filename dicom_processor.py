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

def convert_to_dicom(image_array, original_ds=None):
    """Convert image array to DICOM format"""
    # This is a simplified version - in production you'd need more complete DICOM creation
    if original_ds:
        # Copy original metadata
        new_ds = original_ds
    else:
        # Create new DICOM dataset
        new_ds = pydicom.Dataset()
        new_ds.SOPClassUID = "1.2.3.4.5"  # Placeholder
        new_ds.SOPInstanceUID = pydicom.uid.generate_uid()
    
    # Set pixel data
    if len(image_array.shape) == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    new_ds.PixelData = image_array.tobytes()
    new_ds.Rows, new_ds.Columns = image_array.shape
    
    # Update necessary tags
    new_ds.BitsAllocated = 8
    new_ds.BitsStored = 8
    new_ds.HighBit = 7
    new_ds.PixelRepresentation = 0
    new_ds.PhotometricInterpretation = "MONOCHROME2"
    
    return new_ds