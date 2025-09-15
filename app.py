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
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import time
import os
from datetime import datetime

# Import our modules
from auth import setup_authentication, user_registration_form, check_admin
from dicom_processor import read_dicom_file, extract_dicom_metadata, convert_to_dicom
from models import (
    improved_kmeans_segmentation, improved_mean_shift_segmentation, 
    improved_ncut_segmentation, hybrid_segmentation,
    load_advanced_model, segnet_predict, calculate_all_metrics,
    train_advanced_model, create_training_report,
    unet_attention, deeplab_v3_plus
)

# Page configuration
st.set_page_config(
    page_title="NeuroSegment: Brain MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'advanced_models' not in st.session_state:
    st.session_state.advanced_models = {}

# Authentication
authenticator = setup_authentication()

if not st.session_state.authenticated:
    # Login form
    name, authentication_status, username = authenticator.login('Login', 'main')
    
    if authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')
    else:
        st.session_state.authenticated = True
        st.session_state.username = username
        st.rerun()
else:
    # Main application
    st.sidebar.title(f"Welcome, {st.session_state.username}!")
    
    # Navigation
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["Single Image Analysis", "Batch Processing", "Model Training", "User Management", "Settings", "Model Comparison"]
    )
    
    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()
    
    # Single Image Analysis
    if app_mode == "Single Image Analysis":
        st.title("🧠 NeuroSegment: Single Image Analysis")
        
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
                        for key, value in list(metadata.items())[:20]:  # Show first 20 metadata items
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
                    st.subheader("Original Image")
                    st.image(image_array, use_column_width=True)
                
                # Model selection
                model_options = {
                    "K-Means Clustering": "kmeans",
                    "Mean Shift Clustering": "mean_shift",
                    "Normalized Cut": "ncut",
                    "U-Net with Attention": "unet_attention",
                    "DeepLabV3+": "deeplabv3plus",
                    "Hybrid Approach": "hybrid"
                }
                selected_model = st.selectbox(
                    "Choose a segmentation method:",
                    list(model_options.keys())
                )
                
                # Model parameters
                with st.expander("Advanced Parameters"):
                    if selected_model in ["K-Means Clustering"]:
                        kmeans_clusters = st.slider("Number of Clusters", 2, 8, 4)
                        kmeans_preprocessing = st.checkbox("Enable Preprocessing", value=True)
                    
                    if selected_model in ["Mean Shift Clustering"]:
                        mean_shift_bandwidth = st.slider("Bandwidth", 1, 10, 3)
                        mean_shift_preprocessing = st.checkbox("Enable Preprocessing", value=True)
                    
                    if selected_model in ["Normalized Cut"]:
                        ncut_segments = st.slider("Number of Segments", 10, 100, 50)
                        ncut_compactness = st.slider("Compactness", 1, 20, 10)
                        ncut_preprocessing = st.checkbox("Enable Preprocessing", value=True)
                    
                    if selected_model in ["Hybrid Approach"]:
                        classical_method = st.selectbox(
                            "Classical Method",
                            ["K-Means", "Mean Shift", "Normalized Cut"]
                        )
                        alpha = st.slider("Deep Learning Weight", 0.0, 1.0, 0.7)
                
                # Process button
                if st.button("Segment Image"):
                    with st.spinner("Processing image..."):
                        if selected_model == "K-Means Clustering":
                            segmented = improved_kmeans_segmentation(
                                image_array, k=kmeans_clusters, preprocessing=kmeans_preprocessing
                            )
                        elif selected_model == "Mean Shift Clustering":
                            segmented = improved_mean_shift_segmentation(
                                image_array, bandwidth=mean_shift_bandwidth, 
                                preprocessing=mean_shift_preprocessing
                            )
                        elif selected_model == "Normalized Cut":
                            segmented = improved_ncut_segmentation(
                                image_array, n_segments=ncut_segments, 
                                compactness=ncut_compactness, preprocessing=ncut_preprocessing
                            )
                        elif selected_model == "U-Net with Attention":
                            if "unet_attention" not in st.session_state.advanced_models:
                                with st.spinner("Loading U-Net with Attention..."):
                                    st.session_state.advanced_models["unet_attention"] = load_advanced_model(
                                        'unet_attention'
                                    )
                            segmented = segnet_predict(
                                st.session_state.advanced_models["unet_attention"], image_array
                            )
                        elif selected_model == "DeepLabV3+":
                            if "deeplabv3plus" not in st.session_state.advanced_models:
                                with st.spinner("Loading DeepLabV3+..."):
                                    st.session_state.advanced_models["deeplabv3plus"] = load_advanced_model(
                                        'deeplabv3plus'
                                    )
                            segmented = segnet_predict(
                                st.session_state.advanced_models["deeplabv3plus"], image_array
                            )
                        elif selected_model == "Hybrid Approach":
                            # Ensure we have a base model
                            if "unet_attention" not in st.session_state.advanced_models:
                                with st.spinner("Loading base model for hybrid approach..."):
                                    st.session_state.advanced_models["unet_attention"] = load_advanced_model(
                                        'unet_attention'
                                    )
                            segmented = hybrid_segmentation(
                                image_array, 
                                st.session_state.advanced_models["unet_attention"],
                                classical_method=classical_method.lower(),
                                alpha=alpha
                            )
                    
                    # Display results
                    with col2:
                        st.subheader("Segmentation Result")
                        st.image(segmented, use_column_width=True, clamp=True)
                    
                    # Calculate metrics
                    metrics = calculate_all_metrics(image_array, segmented)
                    
                    # Display metrics
                    st.subheader("Performance Metrics")
                    metric_cols = st.columns(4)
                    metric_cols[0].metric("SSIM", f"{metrics['ssim']:.3f}")
                    metric_cols[1].metric("IoU", f"{metrics['iou']:.3f}")
                    metric_cols[2].metric("Dice", f"{metrics['dice']:.3f}")
                    metric_cols[3].metric("F1 Score", f"{metrics['f1']:.3f}")
                    
                    # Detailed metrics table
                    detailed_metrics = pd.DataFrame({
                        'Metric': ['SSIM', 'IoU', 'Dice Coefficient', 'Precision', 'Recall', 'F1 Score', 'Accuracy'],
                        'Value': [
                            metrics['ssim'], 
                            metrics['iou'], 
                            metrics['dice'],
                            metrics['precision'],
                            metrics['recall'],
                            metrics['f1'],
                            metrics['accuracy']
                        ]
                    })
                    st.dataframe(detailed_metrics, use_container_width=True)
                    
                    # Export options
                    st.subheader("Export Results")
                    export_col1, export_col2 = st.columns(2)
                    
                    with export_col1:
                        # Export segmented image
                        seg_pil = Image.fromarray(segmented)
                        buf = io.BytesIO()
                        seg_pil.save(buf, format="PNG")
                        st.download_button(
                            label="Download Segmented Image",
                            data=buf.getvalue(),
                            file_name=f"segmented_{uploaded_file.name}",
                            mime="image/png"
                        )
                    
                    with export_col2:
                        # Export metrics as CSV
                        csv = detailed_metrics.to_csv(index=False)
                        st.download_button(
                            label="Download Metrics (CSV)",
                            data=cufdssv,
                            file_name=f"metrics_{uploaded_file.name.split('.')[0]}.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Model Comparison
    elif app_mode == "Model Comparison":
        st.title("🧠 NeuroSegment: Model Comparison")
        
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
                    ["K-Means", "Mean Shift", "Normalized Cut", "U-Net with Attention", "DeepLabV3+", "Hybrid"],
                    default=["K-Means", "U-Net with Attention", "Hybrid"]
                )
                
                if st.button("Run Comparison"):
                    results = {}
                    metrics_data = []
                    
                    # Ensure advanced models are loaded
                    if "U-Net with Attention" in models_to_compare and "unet_attention" not in st.session_state.advanced_models:
                        with st.spinner("Loading U-Net with Attention..."):
                            st.session_state.advanced_models["unet_attention"] = load_advanced_model('unet_attention')
                    
                    if "DeepLabV3+" in models_to_compare and "deeplabv3plus" not in st.session_state.advanced_models:
                        with st.spinner("Loading DeepLabV3+..."):
                            st.session_state.advanced_models["deeplabv3plus"] = load_advanced_model('deeplabv3plus')
                    
                    # Process with each selected model
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, model_name in enumerate(models_to_compare):
                        status_text.text(f"Processing with {model_name}...")
                        
                        if model_name == "K-Means":
                            segmented = improved_kmeans_segmentation(image_array, k=4, preprocessing=True)
                        elif model_name == "Mean Shift":
                            segmented = improved_mean_shift_segmentation(image_array, bandwidth=3, preprocessing=True)
                        elif model_name == "Normalized Cut":
                            segmented = improved_ncut_segmentation(image_array, n_segments=50, compactness=10, preprocessing=True)
                        elif model_name == "U-Net with Attention":
                            segmented = segnet_predict(st.session_state.advanced_models["unet_attention"], image_array)
                        elif model_name == "DeepLabV3+":
                            segmented = segnet_predict(st.session_state.advanced_models["deeplabv3plus"], image_array)
                        elif model_name == "Hybrid":
                            segmented = hybrid_segmentation(
                                image_array, 
                                st.session_state.advanced_models["unet_attention"],
                                classical_method="kmeans",
                                alpha=0.7
                            )
                        
                        results[model_name] = segmented
                        metrics = calculate_all_metrics(image_array, segmented)
                        metrics['model'] = model_name
                        metrics_data.append(metrics)
                        
                        progress_bar.progress((i + 1) / len(models_to_compare))
                    
                    status_text.text("Comparison complete!")
                    
                    # Display results
                    st.subheader("Segmentation Results Comparison")
                    cols = st.columns(len(models_to_compare))
                    
                    for i, (model_name, segmented) in enumerate(results.items()):
                        with cols[i]:
                            st.write(f"**{model_name}**")
                            st.image(segmented, use_column_width=True)
                    
                    # Display metrics comparison
                    st.subheader("Performance Metrics Comparison")
                    metrics_df = pd.DataFrame(metrics_data)
                    metrics_df.set_index('model', inplace=True)
                    st.dataframe(metrics_df)
                    
                    # Visual comparison
                    st.subheader("Metrics Visualization")
                    
                    # Create bar charts for each metric
                    metrics_to_plot = ['ssim', 'iou', 'dice', 'f1', 'accuracy']
                    fig = make_subplots(
                        rows=2, cols=3,
                        subplot_titles=[metric.upper() for metric in metrics_to_plot]
                    )
                    
                    for i, metric in enumerate(metrics_to_plot):
                        row = i // 3 + 1
                        col = i % 3 + 1
                        
                        fig.add_trace(
                            go.Bar(x=metrics_df.index, y=metrics_df[metric], name=metric.upper()),
                            row=row, col=col
                        )
                    
                    fig.update_layout(height=600, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error in model comparison: {str(e)}")
    
    # Batch Processing
    elif app_mode == "Batch Processing":
        st.title("🧠 NeuroSegment: Batch Processing")
        
        uploaded_files = st.file_uploader(
            "Upload multiple Brain MRI Images", 
            type=['jpg', 'jpeg', 'png', 'dcm'],
            accept_multiple_files=True,
            help="Select multiple files for batch processing"
        )
        
        if uploaded_files:
            st.info(f"Selected {len(uploaded_files)} files for processing")
            
            # Model selection
            selected_model = st.selectbox(
                "Choose a segmentation method:",
                ["K-Means Clustering", "Mean Shift Clustering", "Normalized Cut", "U-Net with Attention", "DeepLabV3+", "Hybrid Approach"]
            )
            
            # Process all button
            if st.button("Process All Images"):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                    
                    try:
                        # Process each file
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
                        
                        # Apply segmentation
                        if selected_model == "K-Means Clustering":
                            segmented = improved_kmeans_segmentation(image_array, k=4, preprocessing=True)
                        elif selected_model == "Mean Shift Clustering":
                            segmented = improved_mean_shift_segmentation(image_array, bandwidth=3, preprocessing=True)
                        elif selected_model == "Normalized Cut":
                            segmented = improved_ncut_segmentation(image_array, n_segments=50, compactness=10, preprocessing=True)
                        elif selected_model == "U-Net with Attention":
                            if "unet_attention" not in st.session_state.advanced_models:
                                with st.spinner("Loading U-Net with Attention..."):
                                    st.session_state.advanced_models["unet_attention"] = load_advanced_model('unet_attention')
                            segmented = segnet_predict(st.session_state.advanced_models["unet_attention"], image_array)
                        elif selected_model == "DeepLabV3+":
                            if "deeplabv3plus" not in st.session_state.advanced_models:
                                with st.spinner("Loading DeepLabV3+..."):
                                    st.session_state.advanced_models["deeplabv3plus"] = load_advanced_model('deeplabv3plus')
                            segmented = segnet_predict(st.session_state.advanced_models["deeplabv3plus"], image_array)
                        elif selected_model == "Hybrid Approach":
                            if "unet_attention" not in st.session_state.advanced_models:
                                with st.spinner("Loading base model for hybrid approach..."):
                                    st.session_state.advanced_models["unet_attention"] = load_advanced_model('unet_attention')
                            segmented = hybrid_segmentation(
                                image_array, 
                                st.session_state.advanced_models["unet_attention"],
                                classical_method="kmeans",
                                alpha=0.7
                            )
                        
                        # Calculate metrics
                        metrics = calculate_all_metrics(image_array, segmented)
                        metrics['filename'] = uploaded_file.name
                        
                        # Store results
                        results.append(metrics)
                    
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("Processing complete!")
                
                # Display results
                if results:
                    results_df = pd.DataFrame(results)
                    st.subheader("Batch Processing Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary statistics
                    st.subheader("Summary Statistics")
                    summary_df = results_df.drop('filename', axis=1).describe()
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Visualizations
                    st.subheader("Performance Visualization")
                    fig = make_subplots(
                        rows=2, cols=3,
                        subplot_titles=('IoU Distribution', 'Dice Distribution', 
                                       'SSIM Distribution', 'F1 Score Distribution', 'Accuracy Distribution')
                    )
                    
                    fig.add_trace(go.Histogram(x=results_df['iou'], name='IoU'), row=1, col=1)
                    fig.add_trace(go.Histogram(x=results_df['dice'], name='Dice'), row=1, col=2)
                    fig.add_trace(go.Histogram(x=results_df['ssim'], name='SSIM'), row=1, col=3)
                    fig.add_trace(go.Histogram(x=results_df['f1'], name='F1'), row=2, col=1)
                    fig.add_trace(go.Histogram(x=results_df['accuracy'], name='Accuracy'), row=2, col=2)
                    
                    fig.update_layout(height=600, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Export options
                    st.subheader("Export Results")
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Full Results (CSV)",
                        data=csv,
                        file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    # Model Training
    elif app_mode == "Model Training" and check_admin(st.session_state.username):
        st.title("🧠 NeuroSegment: Model Training")
        
        st.info("This section allows administrators to train new segmentation models.")
        
        # Upload training data
        train_files = st.file_uploader(
            "Upload Training Data (images and masks)",
            type=['zip'],
            accept_multiple_files=False,
            help="Upload a ZIP file containing images and corresponding masks"
        )
        
        if train_files:
            st.success("Training data uploaded successfully!")
            
            # Model selection
            model_type = st.selectbox(
                "Select model to train:",
                ["U-Net with Attention", "DeepLabV3+"]
            )
            
            # Training parameters
            st.subheader("Training Parameters")
            epochs = st.slider("Number of Epochs", 10, 200, 50)
            batch_size = st.slider("Batch Size", 8, 64, 16)
            learning_rate = st.slider("Learning Rate", 1e-5, 1e-2, 1e-4, format="%.5f")
            
            # Start training
            if st.button("Start Training"):
                with st.spinner("Training model... This may take a while."):
                    # In a real application, you would:
                    # 1. Extract the ZIP file
                    # 2. Load and preprocess images and masks
                    # 3. Split into training and validation sets
                    # 4. Train the model
                    
                    # For demonstration, we'll simulate training
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Load the appropriate model
                    if model_type == "U-Net with Attention":
                        model = load_advanced_model('unet_attention')
                    else:  # DeepLabV3+
                        model = load_advanced_model('deeplabv3plus')
                    
                    # Simulate training progress
                    for i in range(epochs):
                        # Simulate training progress
                        time.sleep(0.1)
                        progress = (i + 1) / epochs
                        progress_bar.progress(progress)
                        status_text.text(f"Epoch {i+1}/{epochs}")
                    
                    # Simulate results
                    simulated_metrics = {
                        'loss': 0.123, 'val_loss': 0.145,
                        'accuracy': 0.921, 'val_accuracy': 0.894,
                        'iou': 0.856, 'val_iou': 0.832,
                        'dice': 0.892, 'val_dice': 0.876
                    }
                    
                    # Display results
                    st.success("Training completed successfully!")
                    st.subheader("Training Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Training Loss", f"{simulated_metrics['loss']:.3f}")
                        st.metric("Training Accuracy", f"{simulated_metrics['accuracy']:.3f}")
                        st.metric("Training IoU", f"{simulated_metrics['iou']:.3f}")
                    with col2:
                        st.metric("Validation Loss", f"{simulated_metrics['val_loss']:.3f}")
                        st.metric("Validation Accuracy", f"{simulated_metrics['val_accuracy']:.3f}")
                        st.metric("Validation IoU", f"{simulated_metrics['val_iou']:.3f}")
                    
                    # Training history plot (simulated)
                    history_data = pd.DataFrame({
                        'epoch': range(1, epochs+1),
                        'loss': np.linspace(0.8, simulated_metrics['loss'], epochs),
                        'val_loss': np.linspace(0.85, simulated_metrics['val_loss'], epochs),
                        'accuracy': np.linspace(0.7, simulated_metrics['accuracy'], epochs),
                        'val_accuracy': np.linspace(0.65, simulated_metrics['val_accuracy'], epochs),
                        'iou': np.linspace(0.6, simulated_metrics['iou'], epochs),
                        'val_iou': np.linspace(0.55, simulated_metrics['val_iou'], epochs)
                    })
                    
                    fig = make_subplots(rows=2, cols=2, subplot_titles=('Loss', 'Accuracy', 'IoU', 'Dice'))
                    fig.add_trace(go.Scatter(x=history_data['epoch'], y=history_data['loss'], name='Training Loss'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=history_data['epoch'], y=history_data['val_loss'], name='Validation Loss'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=history_data['epoch'], y=history_data['accuracy'], name='Training Accuracy'), row=1, col=2)
                    fig.add_trace(go.Scatter(x=history_data['epoch'], y=history_data['val_accuracy'], name='Validation Accuracy'), row=1, col=2)
                    fig.add_trace(go.Scatter(x=history_data['epoch'], y=history_data['iou'], name='Training IoU'), row=2, col=1)
                    fig.add_trace(go.Scatter(x=history_data['epoch'], y=history_data['val_iou'], name='Validation IoU'), row=2, col=1)
                    
                    fig.update_layout(height=600, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save model option
                    if st.button("Save Trained Model"):
                        # In a real application, you would save the actual model
                        if model_type == "U-Net with Attention":
                            st.session_state.advanced_models["unet_attention"] = model
                        else:
                            st.session_state.advanced_models["deeplabv3plus"] = model
                        st.success("Model saved successfully and is now active!")
    
    # User Management
    elif app_mode == "User Management" and check_admin(st.session_state.username):
        st.title("🧠 NeuroSegment: User Management")
        user_registration_form(authenticator)
    
    # Settings
    elif app_mode == "Settings":
        st.title("🧠 NeuroSegment: Settings")
        
        st.subheader("Model Management")
        
        # Display loaded models
        st.write("Loaded Models:")
        for model_name in st.session_state.advanced_models.keys():
            st.write(f"- {model_name}")
        
        if not st.session_state.advanced_models:
            st.warning("No advanced models are currently loaded")
        
        # Model upload
        model_file = st.file_uploader("Upload a trained model", type=['h5'])
        model_type = st.selectbox(
            "Select model type:",
            ["U-Net with Attention", "DeepLabV3+"]
        )
        
        if model_file and st.button("Load Model"):
            # In a real application, you would load the model
            if model_type == "U-Net with Attention":
                st.session_state.advanced_models["unet_attention"] = load_advanced_model('unet_attention')
            else:
                st.session_state.advanced_models["deeplabv3plus"] = load_advanced_model('deeplabv3plus')
            st.success("Model loaded successfully!")

# Footer
st.markdown("---")
st.markdown("""
**NeuroSegment** is a comprehensive brain MRI analysis tool demonstrating various segmentation techniques.
This tool is for educational and research purposes only and should not be used for clinical diagnosis.
""")