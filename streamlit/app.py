import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import colorsys
import cv2

def rgb_to_hex(rgb):
    """Convert RGB values to HEX format"""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def extract_colors(image, n_colors, quant_level, mode='kmeans', hue_threshold=30):
    """
    Extract dominant colors from image
    Parameters:
    - image: PIL Image object
    - n_colors: number of colors to extract
    - quant_level: color quantization level
    - mode: 'kmeans' or 'quantization'
    - hue_threshold: minimum hue difference in degrees
    """
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Reshape the image to be a list of pixels
    pixels = img_array.reshape(-1, 3)
    
    if mode == 'kmeans':
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_
        
    else:  # quantization mode
        # Reduce color space
        div = 256 // quant_level
        pixels = (pixels // div) * div
        # Get unique colors
        colors = np.unique(pixels, axis=0)
        # Sort by frequency and take top n_colors
        unique, counts = np.unique(pixels, axis=0, return_counts=True)
        colors = unique[np.argsort(-counts)][:n_colors]
    
    return colors.astype(int)

# Set up Streamlit interface
st.title("Color Palette Extractor")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Sidebar for parameters
    st.sidebar.header("Parameters")
    
    n_colors = st.sidebar.slider("Number of colors to extract", 2, 10, 5)
    quant_level = st.sidebar.slider("Quantization level", 2, 32, 8)
    mode = st.sidebar.selectbox("Color selection mode", ['kmeans', 'quantization'])
    hue_threshold = st.sidebar.slider("Hue difference threshold (degrees)", 0, 360, 30)
    output_format = st.sidebar.selectbox("Output format", ['HEX', 'RGB'])
    
    if st.button("Extract Colors"):
        # Extract colors
        colors = extract_colors(image, n_colors, quant_level, mode, hue_threshold)
        
        # Prepare color output
        if output_format == 'HEX':
            color_list = [rgb_to_hex(color) for color in colors]
        else:  # RGB
            color_list = [tuple(color) for color in colors]
        
        # Display results
        st.header("Extracted Color Palette")
        
        # Display color swatches
        cols = st.columns(len(color_list))
        for idx, (col, color) in enumerate(zip(cols, colors)):
            col.markdown(
                f'<div style="background-color: rgb{tuple(color)}; height: 50px; border-radius: 5px;"></div>',
                unsafe_allow_html=True
            )
        
        # Display color values
        st.text_area("Color Values (Copy this)", str(color_list))