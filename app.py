import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from models import kmeans_segmentation, mean_shift_segmentation, ncut_segmentation
from models import load_segnet_model, segnet_predict, calculate_metrics

# Page configuration
st.set_page_config(
    page_title="NeuroSegment: Brain MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🧠 NeuroSegment: Brain MRI Segmentation Suite")
st.markdown("""
This application demonstrates various image segmentation techniques for brain MRI analysis.
Upload an MRI image to see how different algorithms perform at segmenting the image.
""")

# Sidebar
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload a Brain MRI Image", type=['jpg', 'jpeg', 'png'])

# Model selection
st.sidebar.subheader("Segmentation Models")
model_options = {
    "K-Means Clustering": "kmeans",
    "Mean Shift Clustering": "mean_shift",
    "Normalized Cut": "ncut",
    "SegNet (Deep Learning)": "segnet",
    "Compare All Models": "all"
}
selected_model = st.sidebar.selectbox(
    "Choose a segmentation method:",
    list(model_options.keys())
)

# Parameters for different models
st.sidebar.subheader("Model Parameters")
if selected_model in ["K-Means Clustering", "Compare All Models"]:
    kmeans_clusters = st.sidebar.slider("K-Means Clusters", 2, 8, 4)

if selected_model in ["Mean Shift Clustering", "Compare All Models"]:
    mean_shift_bandwidth = st.sidebar.slider("Mean Shift Bandwidth", 1, 10, 3)

if selected_model in ["Normalized Cut", "Compare All Models"]:
    ncut_segments = st.sidebar.slider("NCut Segments", 10, 100, 50)
    ncut_compactness = st.sidebar.slider("NCut Compactness", 1, 20, 10)

# Main content area
col1, col2 = st.columns(2)

if uploaded_file is not None:
    # Load and display the original image
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # Convert to RGB if needed
    if len(image_np.shape) == 2:  # Grayscale
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:  # RGBA
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    with col1:
        st.subheader("Original Image")
        st.image(image_np, use_column_width=True)
    
    # Perform segmentation based on selected model
    with col2:
        st.subheader("Segmentation Result")
        
        if selected_model == "K-Means Clustering" or selected_model == "Compare All Models":
            if selected_model == "K-Means Clustering":
                with st.spinner("Performing K-Means clustering..."):
                    segmented = kmeans_segmentation(image_np, k=kmeans_clusters)
                st.image(segmented, use_column_width=True, caption="K-Means Segmentation")
                
                # Calculate metrics
                ssim_value, iou = calculate_metrics(image_np, segmented)
                st.write(f"**SSIM:** {ssim_value:.3f}, **IoU:** {iou:.3f}")
            
            elif selected_model == "Compare All Models":
                st.warning("Note: Comparing all models may take several moments.")
                
                # Create tabs for each model
                tab1, tab2, tab3, tab4 = st.tabs(["K-Means", "Mean Shift", "Normalized Cut", "SegNet"])
                
                with tab1:
                    with st.spinner("Running K-Means..."):
                        kmeans_result = kmeans_segmentation(image_np, k=kmeans_clusters)
                    st.image(kmeans_result, use_column_width=True, caption="K-Means Segmentation")
                    ssim_value, iou = calculate_metrics(image_np, kmeans_result)
                    st.write(f"**SSIM:** {ssim_value:.3f}, **IoU:** {iou:.3f}")
                
                with tab2:
                    with st.spinner("Running Mean Shift..."):
                        mean_shift_result = mean_shift_segmentation(image_np, bandwidth=mean_shift_bandwidth)
                    st.image(mean_shift_result, use_column_width=True, caption="Mean Shift Segmentation")
                    ssim_value, iou = calculate_metrics(image_np, mean_shift_result)
                    st.write(f"**SSIM:** {ssim_value:.3f}, **IoU:** {iou:.3f}")
                
                with tab3:
                    with st.spinner("Running Normalized Cut..."):
                        ncut_result = ncut_segmentation(image_np, n_segments=ncut_segments, compactness=ncut_compactness)
                    st.image(ncut_result, use_column_width=True, caption="Normalized Cut Segmentation")
                    ssim_value, iou = calculate_metrics(image_np, ncut_result)
                    st.write(f"**SSIM:** {ssim_value:.3f}, **IoU:** {iou:.3f}")
                
                with tab4:
                    with st.spinner("Loading SegNet model..."):
                        segnet_model = load_segnet_model()
                    with st.spinner("Running SegNet prediction..."):
                        segnet_result = segnet_predict(segnet_model, image_np)
                    st.image(segnet_result, use_column_width=True, caption="SegNet Segmentation", clamp=True)
                    ssim_value, iou = calculate_metrics(image_np, segnet_result)
                    st.write(f"**SSIM:** {ssim_value:.3f}, **IoU:** {iou:.3f}")
        
        elif selected_model == "Mean Shift Clustering":
            with st.spinner("Performing Mean Shift clustering..."):
                segmented = mean_shift_segmentation(image_np, bandwidth=mean_shift_bandwidth)
            st.image(segmented, use_column_width=True, caption="Mean Shift Segmentation")
            
            # Calculate metrics
            ssim_value, iou = calculate_metrics(image_np, segmented)
            st.write(f"**SSIM:** {ssim_value:.3f}, **IoU:** {iou:.3f}")
        
        elif selected_model == "Normalized Cut":
            with st.spinner("Performing Normalized Cut segmentation..."):
                segmented = ncut_segmentation(image_np, n_segments=ncut_segments, compactness=ncut_compactness)
            st.image(segmented, use_column_width=True, caption="Normalized Cut Segmentation")
            
            # Calculate metrics
            ssim_value, iou = calculate_metrics(image_np, segmented)
            st.write(f"**SSIM:** {ssim_value:.3f}, **IoU:** {iou:.3f}")
        
        elif selected_model == "SegNet (Deep Learning)":
            with st.spinner("Loading SegNet model..."):
                segnet_model = load_segnet_model()
            with st.spinner("Running SegNet prediction..."):
                segmented = segnet_predict(segnet_model, image_np)
            st.image(segmented, use_column_width=True, caption="SegNet Segmentation", clamp=True)
            
            # Calculate metrics
            ssim_value, iou = calculate_metrics(image_np, segmented)
            st.write(f"**SSIM:** {ssim_value:.3f}, **IoU:** {iou:.3f}")

else:
    st.info("👆 Please upload a Brain MRI image to get started.")
    
    # Show sample images
    st.subheader("Sample MRI Images")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image("https://github.com/ieee8023/covid-chestxray-dataset/raw/master/images/1-s2.0-S0140673620303706-fx1_lrg.jpg", 
                 caption="Sample Brain MRI 1", use_column_width=True)
    
    with col2:
        st.image("https://www.researchgate.net/profile/Andras-Jakab/publication/339486311/figure/fig1/AS:861038362673153@1582166217007/A-T2-weighted-MRI-scan-of-a-healthy-2-year-old-child-The-image-shows-the-typical.ppm", 
                 caption="Sample Brain MRI 2", use_column_width=True)
    
    with col3:
        st.image("https://prod-images-static.radiopaedia.org/images/102394/70e9ffc5c5b2c35d5f3b2e5d2b4f34_jumbo.jpg", 
                 caption="Sample Brain MRI 3", use_column_width=True)

# Footer
st.markdown("---")
st.markdown("""
**NeuroSegment** is a demonstration of various image segmentation techniques applied to brain MRI analysis.
This tool is for educational and research purposes only and should not be used for clinical diagnosis.
""")