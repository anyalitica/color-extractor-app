import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import colorsys
import json
import time

# Configure page
st.set_page_config(
    page_title="Color Palette Extractor",
    page_icon="🎨",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    .color-swatch {
        height: 80px; 
        border-radius: 10px;
        border: 2px solid #ddd;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        font-size: 12px;
    }
    .settings-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .help-box {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .method-explanation {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def rgb_to_hex(rgb):
    """Convert RGB values to HEX format"""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def resize_image_for_processing(image, max_size=800):
    """Resize large images for faster processing"""
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image

def filter_similar_colors(colors, hue_threshold):
    """Remove colors with similar hues"""
    if len(colors) <= 1 or hue_threshold == 0:
        return colors
    
    filtered_colors = [colors[0]]
    
    for color in colors[1:]:
        # Convert to HSV for better hue comparison
        hsv_new = colorsys.rgb_to_hsv(color[0]/255, color[1]/255, color[2]/255)
        
        is_different = True
        for existing_color in filtered_colors:
            hsv_existing = colorsys.rgb_to_hsv(existing_color[0]/255, existing_color[1]/255, existing_color[2]/255)
            
            # Calculate hue difference (considering circular nature of hue)
            hue_diff = abs(hsv_new[0] - hsv_existing[0]) * 360
            hue_diff = min(hue_diff, 360 - hue_diff)  # Take shorter arc
            
            if hue_diff < hue_threshold:
                is_different = False
                break
        
        if is_different:
            filtered_colors.append(color)
    
    return np.array(filtered_colors)

@st.cache_data
def extract_colors_from_array(image_bytes, image_shape, n_colors, quant_level, mode, hue_threshold):
    """Cached version of color extraction"""
    # Reconstruct array from bytes
    img_array = np.frombuffer(image_bytes, dtype=np.uint8).reshape(image_shape)
    
    # Reshape the image to be a list of pixels
    pixels = img_array.reshape(-1, 3)
    
    if mode == 'kmeans':
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_
        
    else:  # quantization mode
        # Reduce color space
        div = 256 // quant_level
        pixels = (pixels // div) * div
        # Get unique colors and their counts
        unique, counts = np.unique(pixels, axis=0, return_counts=True)
        # Sort by frequency and take top n_colors * 2 (in case filtering removes some)
        colors = unique[np.argsort(-counts)][:(n_colors * 2)]
    
    colors = colors.astype(int)
    
    # Apply hue filtering if threshold > 0
    if hue_threshold > 0:
        colors = filter_similar_colors(colors, hue_threshold)
    
    # Ensure we have the requested number of colors
    return colors[:n_colors]

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
    
    # Use cached function
    return extract_colors_from_array(
        img_array.tobytes(), 
        img_array.shape, 
        n_colors, 
        quant_level, 
        mode, 
        hue_threshold
    )

def get_color_harmony_info(colors):
    """Provide information about color harmony"""
    if len(colors) >= 2:
        warm_colors = sum(1 for color in colors if color[0] > color[2])  # More red than blue
        cool_colors = len(colors) - warm_colors
        
        # Calculate average brightness
        avg_brightness = np.mean([sum(color) for color in colors]) / 3
        brightness_desc = "bright" if avg_brightness > 127 else "dark"
        
        return f"🎨 **Palette Analysis:** {warm_colors} warm, {cool_colors} cool colors. Overall tone: {brightness_desc}"
    return ""

def create_sample_image():
    """Create a sample gradient image for demo purposes"""
    # Create a simple gradient
    width, height = 400, 200
    image = Image.new('RGB', (width, height))
    pixels = []
    
    for y in range(height):
        for x in range(width):
            r = int(255 * (x / width))
            g = int(255 * (y / height))
            b = int(255 * ((x + y) / (width + height)))
            pixels.append((r, g, b))
    
    image.putdata(pixels)
    return image

# Main UI
st.markdown('<h1 class="main-header">🎨 Color Palette Extractor</h1>', unsafe_allow_html=True)

st.markdown("""
Extract dominant colors from any image using advanced clustering algorithms. 
Upload an image or try the sample to get started!
""")

# Add expandable help section
with st.expander("ℹ️ How It Works & Tips", expanded=False):
    st.markdown("""
    ### 🔧 How Color Extraction Works
    
    This app analyzes every pixel in your image and groups similar colors together using machine learning algorithms:
    
    **K-means Clustering:** Groups pixels into color clusters based on similarity in RGB color space
    **Color Quantization:** Reduces the color palette by rounding color values to nearest levels
    
    ### 💡 Tips for Best Results
    - **High contrast images** produce more distinct color palettes
    - **Well-lit photos** give more accurate color representation  
    - **Avoid heavily filtered images** as they may have artificial color casts
    - **For logo extraction:** Use clean, high-resolution logo files
    - **For web design:** Extract from hero images or key brand visuals
    
    ### 🎯 Common Use Cases
    - **Web Design:** Create CSS color variables from design mockups
    - **Brand Identity:** Extract official colors from logos and marketing materials
    - **Interior Design:** Match paint colors to inspiration photos
    - **Art Analysis:** Study color relationships in paintings and artwork
    """)

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col2:
    st.markdown('<div class="settings-box">', unsafe_allow_html=True)
    st.subheader("⚙️ Extraction Settings")
    
    # Number of colors
    n_colors = st.slider(
        "Number of colors to extract", 
        2, 10, 5,
        help="More colors create a detailed palette, fewer colors create a simplified color scheme"
    )
    
    # Extraction method with detailed explanation
    st.markdown("**Extraction Method**")
    mode = st.selectbox(
        "Choose algorithm",
        ['kmeans', 'quantization'],
        help="Select the algorithm used to identify dominant colors",
        label_visibility="collapsed"
    )
    
    # Method explanation box
    if mode == 'kmeans':
        st.markdown("""
        <div class="method-explanation">
        <strong>🧠 K-means Clustering</strong><br>
        <em>Best for: Complex images with many colors</em><br>
        • Groups similar pixels using machine learning<br>
        • Finds natural color clusters in your image<br>
        • Higher quality results, slightly slower processing<br>
        • Ideal for photographs and detailed artwork
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="method-explanation">
        <strong>⚡ Color Quantization</strong><br>
        <em>Best for: Simple images or quick results</em><br>
        • Reduces color precision to find dominant colors<br>
        • Faster processing, good for real-time use<br>
        • Works well with graphics and illustrations<br>
        • Perfect for images with distinct color areas
        </div>
        """, unsafe_allow_html=True)
    
    # Quantization level (only for quantization mode)
    if mode == 'quantization':
        st.markdown("**Quantization Precision**")
        quant_level = st.slider(
            "Color precision level",
            2, 32, 8,
            help="Higher values preserve more color nuances, lower values create broader color groupings"
        )
        st.markdown("""
        <div class="help-box">
        <strong>💡 Quantization Level Guide:</strong><br>
        • <strong>2-8:</strong> Bold, simplified palettes<br>
        • <strong>8-16:</strong> Balanced detail and simplicity<br>
        • <strong>16-32:</strong> Detailed, nuanced colors
        </div>
        """, unsafe_allow_html=True)
    else:
        quant_level = 8  # Default for kmeans
    
    # Hue threshold with detailed explanation
    st.markdown("**Hue Similarity Filter**")
    hue_threshold = st.slider(
        "Hue difference threshold (degrees)",
        0, 90, 30,
        help="Removes colors with similar hues. Higher values = fewer similar colors in result"
    )
    
    st.markdown("""
    <div class="help-box">
    <strong>🌈 Understanding Hue Filtering:</strong><br>
    Colors are compared by their hue (red, blue, green, etc.) on a 360° color wheel:<br>
    • <strong>0°:</strong> Keep all extracted colors<br>
    • <strong>15-30°:</strong> Remove very similar hues (recommended)<br>
    • <strong>45-90°:</strong> Keep only distinctly different hues<br><br>
    <em>Example: Red (0°) and Orange (30°) have a 30° hue difference</em>
    </div>
    """, unsafe_allow_html=True)
    
    # Output format
    st.markdown("**Output Format**")
    output_format = st.selectbox(
        "Color format for export",
        ['HEX', 'RGB'],
        help="HEX for web/CSS use, RGB for design software and data analysis"
    )
    
    if output_format == 'HEX':
        st.markdown("🌐 **HEX Format:** Perfect for CSS, web design, and color pickers")
    else:
        st.markdown("📊 **RGB Format:** Ideal for design software, data analysis, and programming")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sample image option
    st.subheader("🖼️ Try Sample")
    if st.button("Use Sample Gradient", help="Try the app with a colorful gradient image"):
        st.session_state.use_sample = True
    
    st.markdown("""
    <div class="help-box">
    <strong>🎨 Sample Gradient:</strong><br>
    Perfect for testing different settings and seeing how the algorithms work with smooth color transitions.
    </div>
    """, unsafe_allow_html=True)

with col1:
    # File uploader with detailed instructions
    st.markdown("### 📁 Upload Your Image")
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg', 'webp'],
        help="Supported formats: PNG, JPG, JPEG, WEBP. For best results, use high-quality images with good lighting."
    )
    
    # Handle sample image
    if st.session_state.get('use_sample', False) and uploaded_file is None:
        image = create_sample_image()
        st.image(image, caption='Sample Gradient Image - Perfect for Testing!', use_container_width=True)
        st.markdown("""
        <div class="help-box">
        This gradient contains smooth color transitions from red to blue to green. 
        Try different extraction methods and settings to see how they affect the results!
        </div>
        """, unsafe_allow_html=True)
        process_image = True
    elif uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Display original image with info
            st.image(image, caption=f'Uploaded: {uploaded_file.name} ({image.size[0]}×{image.size[1]} pixels)', 
                    use_container_width=True)
            
            # Image analysis info
            total_pixels = image.size[0] * image.size[1]
            st.markdown(f"""
            <div class="help-box">
            <strong>📊 Image Info:</strong> {total_pixels:,} pixels total<br>
            {f'<em>Large image - automatically resized to 800px for faster processing</em>' if max(image.size) > 800 else '<em>Good size for quick processing</em>'}
            </div>
            """, unsafe_allow_html=True)
            
            process_image = True
            
            # Reset sample flag
            if 'use_sample' in st.session_state:
                del st.session_state.use_sample
                
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")
            st.markdown("""
            **Troubleshooting:**
            - Make sure the file is a valid image format (PNG, JPG, JPEG, WEBP)
            - Try a different image if the file might be corrupted
            - Check that the file size isn't too large (>50MB)
            """)
            st.stop()
    else:
        process_image = False
        st.info("👆 Upload an image above or try the sample gradient to get started!")

# Process image if available
if process_image:
    # Resize for processing to improve performance
    processing_image = resize_image_for_processing(image.copy())
    
    if st.button("🎨 Extract Colors", type="primary", help="Analyze your image and extract the dominant colors"):
        try:
            with st.spinner('🔍 Analyzing image colors...'):
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                progress_text.text("📊 Processing image pixels...")
                progress_bar.progress(25)
                
                progress_text.text(f"🧠 Applying {mode} algorithm...")
                progress_bar.progress(50)
                
                # Extract colors
                colors = extract_colors(processing_image, n_colors, quant_level, mode, hue_threshold)
                
                progress_text.text("🎨 Generating color palette...")
                progress_bar.progress(75)
                
                # Prepare color output
                if output_format == 'HEX':
                    color_list = [rgb_to_hex(color) for color in colors]
                else:  # RGB
                    color_list = [tuple(int(c) for c in color) for color in colors]
                
                progress_text.text("✅ Complete!")
                progress_bar.progress(100)
                time.sleep(0.5)  # Brief pause to show completion
                progress_bar.empty()
                progress_text.empty()
                
                # Display results
                st.header("🎨 Extracted Color Palette")
                
                # Color harmony analysis
                harmony_info = get_color_harmony_info(colors)
                if harmony_info:
                    st.markdown(harmony_info)
                
                # Display color swatches
                if len(colors) > 0:
                    st.markdown(f"**Found {len(colors)} dominant colors using {mode} extraction:**")
                    
                    cols = st.columns(len(colors))
                    for idx, (col, color) in enumerate(zip(cols, colors)):
                        hex_color = rgb_to_hex(color)
                        # Determine text color based on background brightness
                        text_color = 'white' if sum(color) < 384 else 'black'
                        
                        col.markdown(
                            f'''
                            <div class="color-swatch" style="
                                background-color: {hex_color}; 
                                color: {text_color};
                            ">
                                {hex_color if output_format == 'HEX' else f'RGB{tuple(color)}'}
                            </div>
                            ''',
                            unsafe_allow_html=True
                        )
                    
                    # Display color values for copying
                    st.subheader("📋 Color Values")
                    color_text = '\n'.join([str(color) for color in color_list])
                    st.text_area(
                        "Copy these values:",
                        color_text, 
                        height=100,
                        help="Select all and copy these color values to use in your projects"
                    )
                    
                    # Export functionality with explanations
                    st.subheader("📥 Export Palette")
                    st.markdown("Download your color palette in different formats:")
                    
                    col_json, col_css = st.columns(2)
                    
                    with col_json:
                        st.markdown("**📄 JSON Format**")
                        st.markdown("*Perfect for developers and data analysis*")
                        try:
                            # Convert colors to native Python types for JSON serialization
                            if output_format == 'HEX':
                                serializable_colors = [str(color) for color in color_list]
                            else:  # RGB
                                serializable_colors = [
                                    [int(c) for c in color] if isinstance(color, (tuple, list, np.ndarray)) 
                                    else convert_numpy_types(color) 
                                    for color in color_list
                                ]
                            
                            palette_data = {
                                'colors': serializable_colors,
                                'format': output_format,
                                'extraction_method': mode,
                                'parameters': {
                                    'n_colors': int(n_colors),
                                    'quantization_level': int(quant_level) if mode == 'quantization' else None,
                                    'hue_threshold': int(hue_threshold)
                                }
                            }
                            
                            json_data = json.dumps(palette_data, indent=2)
                            st.download_button(
                                label="📄 Download JSON",
                                data=json_data,
                                file_name="color_palette.json",
                                mime="application/json",
                                help="Download structured data with colors and extraction settings"
                            )
                        except Exception as e:
                            st.error(f"Error creating JSON export: {str(e)}")
                    
                    with col_css:
                        st.markdown("**🎨 CSS Format**")
                        st.markdown("*Ready-to-use CSS custom properties*")
                        try:
                            # Generate CSS variables
                            css_vars = '\n'.join([
                                f'  --color-{i+1}: {color};' 
                                for i, color in enumerate(color_list)
                            ])
                            css_content = f':root {{\n{css_vars}\n}}'
                            
                            st.download_button(
                                label="🎨 Download CSS",
                                data=css_content,
                                file_name="color_palette.css",
                                mime="text/css",
                                help="Download CSS variables ready to use in your stylesheets"
                            )
                            
                            # Show preview of CSS
                            with st.expander("👀 CSS Preview"):
                                st.code(css_content, language='css')
                                st.markdown("Use in your CSS like: `background-color: var(--color-1);`")
                                
                        except Exception as e:
                            st.error(f"Error creating CSS export: {str(e)}")
                else:
                    st.warning("⚠️ No colors could be extracted with the current settings.")
                    st.markdown("""
                    **Try these solutions:**
                    - Reduce the hue threshold to allow more similar colors
                    - Use a different extraction method
                    - Try with an image that has more color variety
                    - Increase the number of colors to extract
                    """)
                    
        except Exception as e:
            st.error(f"❌ Error extracting colors: {str(e)}")
            st.markdown("""
            **Troubleshooting:**
            - Try reducing the image size or using a different image
            - Switch between K-means and quantization methods
            - Reduce the number of colors to extract
            """)

# Footer with additional tips
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🎨 <strong>Color Palette Extractor</strong> | Built with Streamlit</p>
    <p><small>💡 <strong>Pro Tip:</strong> For brand color extraction, use high-resolution logos with clean backgrounds for best results.</small></p>
</div>
""", unsafe_allow_html=True)