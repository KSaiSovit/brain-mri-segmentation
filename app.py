# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# from models import kmeans_segmentation, mean_shift_segmentation, ncut_segmentation
# from models import load_segnet_model, segnet_predict, calculate_metrics

# # Page configuration
# st.set_page_config(
#     page_title="NeuroSegment: Brain MRI Analysis",
#     page_icon="🧠",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Title and description
# st.title("🧠 NeuroSegment: Brain MRI Segmentation Suite")
# st.markdown("""
# This application demonstrates various image segmentation techniques for brain MRI analysis.
# Upload an MRI image to see how different algorithms perform at segmenting the image.
# """)

# # Sidebar
# st.sidebar.header("Settings")
# uploaded_file = st.sidebar.file_uploader("Upload a Brain MRI Image", type=['jpg', 'jpeg', 'png'])

# # Model selection
# st.sidebar.subheader("Segmentation Models")
# model_options = {
#     "K-Means Clustering": "kmeans",
#     "Mean Shift Clustering": "mean_shift",
#     "Normalized Cut": "ncut",
#     "SegNet (Deep Learning)": "segnet",
#     "Compare All Models": "all"
# }
# selected_model = st.sidebar.selectbox(
#     "Choose a segmentation method:",
#     list(model_options.keys())
# )

# # Parameters for different models
# st.sidebar.subheader("Model Parameters")
# if selected_model in ["K-Means Clustering", "Compare All Models"]:
#     kmeans_clusters = st.sidebar.slider("K-Means Clusters", 2, 8, 4)

# if selected_model in ["Mean Shift Clustering", "Compare All Models"]:
#     mean_shift_bandwidth = st.sidebar.slider("Mean Shift Bandwidth", 1, 10, 3)

# if selected_model in ["Normalized Cut", "Compare All Models"]:
#     ncut_segments = st.sidebar.slider("NCut Segments", 10, 100, 50)
#     ncut_compactness = st.sidebar.slider("NCut Compactness", 1, 20, 10)

# # Main content area
# col1, col2 = st.columns(2)

# if uploaded_file is not None:
#     # Load and display the original image
#     image = Image.open(uploaded_file)
#     image_np = np.array(image)
    
#     # Convert to RGB if needed
#     if len(image_np.shape) == 2:  # Grayscale
#         image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
#     elif image_np.shape[2] == 4:  # RGBA
#         image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
#     with col1:
#         st.subheader("Original Image")
#         st.image(image_np, use_column_width=True)
    
#     # Perform segmentation based on selected model
#     with col2:
#         st.subheader("Segmentation Result")
        
#         if selected_model == "K-Means Clustering" or selected_model == "Compare All Models":
#             if selected_model == "K-Means Clustering":
#                 with st.spinner("Performing K-Means clustering..."):
#                     segmented = kmeans_segmentation(image_np, k=kmeans_clusters)
#                 st.image(segmented, use_column_width=True, caption="K-Means Segmentation")
                
#                 # Calculate metrics
#                 ssim_value, iou = calculate_metrics(image_np, segmented)
#                 st.write(f"**SSIM:** {ssim_value:.3f}, **IoU:** {iou:.3f}")
            
#             elif selected_model == "Compare All Models":
#                 st.warning("Note: Comparing all models may take several moments.")
                
#                 # Create tabs for each model
#                 tab1, tab2, tab3, tab4 = st.tabs(["K-Means", "Mean Shift", "Normalized Cut", "SegNet"])
                
#                 with tab1:
#                     with st.spinner("Running K-Means..."):
#                         kmeans_result = kmeans_segmentation(image_np, k=kmeans_clusters)
#                     st.image(kmeans_result, use_column_width=True, caption="K-Means Segmentation")
#                     ssim_value, iou = calculate_metrics(image_np, kmeans_result)
#                     st.write(f"**SSIM:** {ssim_value:.3f}, **IoU:** {iou:.3f}")
                
#                 with tab2:
#                     with st.spinner("Running Mean Shift..."):
#                         mean_shift_result = mean_shift_segmentation(image_np, bandwidth=mean_shift_bandwidth)
#                     st.image(mean_shift_result, use_column_width=True, caption="Mean Shift Segmentation")
#                     ssim_value, iou = calculate_metrics(image_np, mean_shift_result)
#                     st.write(f"**SSIM:** {ssim_value:.3f}, **IoU:** {iou:.3f}")
                
#                 with tab3:
#                     with st.spinner("Running Normalized Cut..."):
#                         ncut_result = ncut_segmentation(image_np, n_segments=ncut_segments, compactness=ncut_compactness)
#                     st.image(ncut_result, use_column_width=True, caption="Normalized Cut Segmentation")
#                     ssim_value, iou = calculate_metrics(image_np, ncut_result)
#                     st.write(f"**SSIM:** {ssim_value:.3f}, **IoU:** {iou:.3f}")
                
#                 with tab4:
#                     with st.spinner("Loading SegNet model..."):
#                         segnet_model = load_segnet_model()
#                     with st.spinner("Running SegNet prediction..."):
#                         segnet_result = segnet_predict(segnet_model, image_np)
#                     st.image(segnet_result, use_column_width=True, caption="SegNet Segmentation", clamp=True)
#                     ssim_value, iou = calculate_metrics(image_np, segnet_result)
#                     st.write(f"**SSIM:** {ssim_value:.3f}, **IoU:** {iou:.3f}")
        
#         elif selected_model == "Mean Shift Clustering":
#             with st.spinner("Performing Mean Shift clustering..."):
#                 segmented = mean_shift_segmentation(image_np, bandwidth=mean_shift_bandwidth)
#             st.image(segmented, use_column_width=True, caption="Mean Shift Segmentation")
            
#             # Calculate metrics
#             ssim_value, iou = calculate_metrics(image_np, segmented)
#             st.write(f"**SSIM:** {ssim_value:.3f}, **IoU:** {iou:.3f}")
        
#         elif selected_model == "Normalized Cut":
#             with st.spinner("Performing Normalized Cut segmentation..."):
#                 segmented = ncut_segmentation(image_np, n_segments=ncut_segments, compactness=ncut_compactness)
#             st.image(segmented, use_column_width=True, caption="Normalized Cut Segmentation")
            
#             # Calculate metrics
#             ssim_value, iou = calculate_metrics(image_np, segmented)
#             st.write(f"**SSIM:** {ssim_value:.3f}, **IoU:** {iou:.3f}")
        
#         elif selected_model == "SegNet (Deep Learning)":
#             with st.spinner("Loading SegNet model..."):
#                 segnet_model = load_segnet_model()
#             with st.spinner("Running SegNet prediction..."):
#                 segmented = segnet_predict(segnet_model, image_np)
#             st.image(segmented, use_column_width=True, caption="SegNet Segmentation", clamp=True)
            
#             # Calculate metrics
#             ssim_value, iou = calculate_metrics(image_np, segmented)
#             st.write(f"**SSIM:** {ssim_value:.3f}, **IoU:** {iou:.3f}")

# else:
#     st.info("👆 Please upload a Brain MRI image to get started.")
    
#     # Show sample images
#     st.subheader("Sample MRI Images")
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.image("https://github.com/ieee8023/covid-chestxray-dataset/raw/master/images/1-s2.0-S0140673620303706-fx1_lrg.jpg", 
#                  caption="Sample Brain MRI 1", use_column_width=True)
    
#     with col2:
#         st.image("https://www.researchgate.net/profile/Andras-Jakab/publication/339486311/figure/fig1/AS:861038362673153@1582166217007/A-T2-weighted-MRI-scan-of-a-healthy-2-year-old-child-The-image-shows-the-typical.ppm", 
#                  caption="Sample Brain MRI 2", use_column_width=True)
    
#     with col3:
#         st.image("https://prod-images-static.radiopaedia.org/images/102394/70e9ffc5c5b2c35d5f3b2e5d2b4f34_jumbo.jpg", 
#                  caption="Sample Brain MRI 3", use_column_width=True)

# # Footer
# st.markdown("---")
# st.markdown("""
# **NeuroSegment** is a demonstration of various image segmentation techniques applied to brain MRI analysis.
# This tool is for educational and research purposes only and should not be used for clinical diagnosis.
# """)

# ------------
## Version 2
# ------------

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import io
import time

# Import our modules
from dicom_processor import read_dicom_file, extract_dicom_metadata
from models import detect_tumor

# Page configuration
st.set_page_config(
    page_title="Brain MRI Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'processing_times' not in st.session_state:
    st.session_state.processing_times = {}

# Main application
st.sidebar.title("NeuroSegment - Tumor Detection")

# Navigation
app_mode = st.sidebar.selectbox(
    "Choose Mode",
    ["Single Image Analysis", "Model Comparison"]
)

# Single Image Analysis
if app_mode == "Single Image Analysis":
    st.title("🧠 Brain MRI Tumor Detection")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a Brain MRI Image", 
        type=['jpg', 'jpeg', 'png', 'dcm'],
        help="Support for JPEG, PNG, and DICOM formats"
    )
    
    if uploaded_file:
        # Check if file is DICOM
        is_dicom = uploaded_file.name.lower().endswith('.dcm')
        
        try:
            if is_dicom:
                # Process DICOM file
                image_array, dicom_dataset = read_dicom_file(uploaded_file.getvalue())
                # Display DICOM metadata
                with st.expander("DICOM Metadata"):
                    metadata = extract_dicom_metadata(dicom_dataset)
                    for key, value in list(metadata.items())[:10]:
                        st.text(f"{key}: {value}")
            else:
                # Process regular image file
                image = Image.open(uploaded_file)
                image_array = np.array(image)
                
                # Convert to RGB if needed
                if len(image_array.shape) == 2:  # Grayscale
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
                elif image_array.shape[2] == 4:  # RGBA
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            
            # Display original image
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original MRI")
                st.image(image_array, use_column_width=True)
            
            # Model selection
            selected_model = st.selectbox(
                "Choose a detection method:",
                ["K-Means", "Mean Shift", "U-Net", "Hybrid"]
            )
            
            # Process button
            if st.button("Detect Tumor"):
                with st.spinner("Analyzing image for tumors..."):
                    has_tumor, result_img, metrics = detect_tumor(
                        image_array, method=selected_model.lower()
                    )
                
                # Display results
                with col2:
                    st.subheader("Tumor Analysis Result")
                    
                    if has_tumor:
                        st.error("🚨 Tumor Detected!")
                        st.image(result_img, use_column_width=True, caption="Tumor areas highlighted in red")
                        
                        # Display metrics
                        st.metric("Tumor Size", f"{metrics['size']} pixels")
                        st.metric("Confidence", f"{metrics['confidence']:.2%}")
                        st.metric("Processing Time", f"{metrics['time']:.2f} seconds")
                        st.metric("Number of Regions", f"{metrics['contours']}")
                    else:
                        st.success("✅ No Tumor Detected")
                        st.image(image_array, use_column_width=True, caption="No tumors found")
                        st.metric("Processing Time", f"{metrics['time']:.2f} seconds")
                
                # Store processing time for comparison
                st.session_state.processing_times[selected_model] = metrics['time']
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("👆 Please upload a Brain MRI image to get started.")
        
        # Show sample images
        st.subheader("Sample MRI Images")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image("https://assets.technologynetworks.com/production/dynamic/images/content/341355/brain-mri-scans-how-to-read-341355-960x540.jpg?cb=11234532", 
                     caption="Sample Brain MRI 1", use_column_width=True)
        
        with col2:
            st.image("https://www.researchgate.net/profile/Andras-Jakab/publication/339486311/figure/fig1/AS:861038362673153@1582166217007/A-T2-weighted-MRI-scan-of-a-healthy-2-year-old-child-The-image-shows-the-typical.ppm", 
                     caption="Sample Brain MRI 2", use_column_width=True)
        
        with col3:
            st.image("https://prod-images-static.radiopaedia.org/images/102394/70e9ffc5c5b2c35d5f3b2e5d2b4f34_jumbo.jpg", 
                     caption="Sample Brain MRI 3", use_column_width=True)

# Model Comparison
elif app_mode == "Model Comparison":
    st.title("🧠 Model Comparison for Tumor Detection")
    
    uploaded_file = st.file_uploader(
        "Upload a Brain MRI Image for Comparison", 
        type=['jpg', 'jpeg', 'png', 'dcm']
    )
    
    if uploaded_file:
        try:
            # Process the image
            is_dicom = uploaded_file.name.lower().endswith('.dcm')
            
            if is_dicom:
                image_array, _ = read_dicom_file(uploaded_file.getvalue())
            else:
                image = Image.open(uploaded_file)
                image_array = np.array(image)
                if len(image_array.shape) == 2:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
                elif image_array.shape[2] == 4:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            
            st.subheader("Original Image")
            st.image(image_array, use_column_width=True)
            
            # Select models to compare
            models_to_compare = st.multiselect(
                "Select models to compare:",
                ["K-Means", "Mean Shift", "U-Net", "Hybrid"],
                default=["K-Means", "U-Net", "Hybrid"]
            )
            
            if st.button("Run Comparison"):
                results = {}
                metrics_data = []
                
                # Process with each selected model
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, model_name in enumerate(models_to_compare):
                    status_text.text(f"Processing with {model_name}...")
                    
                    has_tumor, segmented, metrics = detect_tumor(
                        image_array, method=model_name.lower()
                    )
                    
                    results[model_name] = {
                        'image': segmented,
                        'has_tumor': has_tumor,
                        'metrics': metrics
                    }
                    
                    metrics['model'] = model_name
                    metrics['tumor_detected'] = has_tumor
                    metrics_data.append(metrics)
                    
                    progress_bar.progress((i + 1) / len(models_to_compare))
                
                status_text.text("Comparison complete!")
                
                # Display results
                st.subheader("Segmentation Results Comparison")
                cols = st.columns(len(models_to_compare))
                
                for i, (model_name, result) in enumerate(results.items()):
                    with cols[i]:
                        st.write(f"**{model_name}**")
                        if result['has_tumor']:
                            st.error("Tumor Detected")
                        else:
                            st.success("No Tumor")
                        st.image(result['image'], use_column_width=True)
                        st.caption(f"Time: {result['metrics']['time']:.2f}s")
                
                # Display metrics comparison
                st.subheader("Performance Metrics Comparison")
                metrics_df = pd.DataFrame(metrics_data)
                metrics_df.set_index('model', inplace=True)
                st.dataframe(metrics_df)
                
                # Visual comparison
                st.subheader("Metrics Visualization")
                
                # Create bar charts for each metric
                fig = px.bar(metrics_df, x=metrics_df.index, y='time', 
                             title='Processing Time Comparison')
                st.plotly_chart(fig, use_container_width=True)
                
                if any(metrics_df['tumor_detected']):
                    fig2 = px.bar(metrics_df, x=metrics_df.index, y='size',
                                 title='Tumor Size Detection Comparison')
                    st.plotly_chart(fig2, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error in model comparison: {str(e)}")
    else:
        st.info("Please upload an image to compare models.")

# Footer
st.markdown("---")
st.markdown("""
**NeuroSegment** is a brain MRI analysis tool for tumor detection.
This tool is for educational and research purposes only and should not be used for clinical diagnosis.
""")