import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans, MiniBatchKMeans
from collections import Counter
import colorsys
import json
import time
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy.cluster.hierarchy as sch

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
        font-size: 16px;
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
    .algorithm-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .tips-section {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
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

def rgb_to_lab(rgb):
    """Convert RGB to LAB color space for better perceptual distance"""
    rgb = np.array(rgb).reshape(-1, 3) / 255.0

    # Convert RGB to XYZ
    rgb = np.where(rgb > 0.04045, np.power((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)
    xyz = rgb @ np.array([[0.4124564, 0.3575761, 0.1804375],
                          [0.2126729, 0.7151522, 0.0721750],
                          [0.0193339, 0.1191920, 0.9503041]])

    # Convert XYZ to LAB
    xyz = xyz / np.array([0.95047, 1.00000, 1.08883])
    xyz = np.where(xyz > 0.008856, np.power(xyz, 1/3), (7.787 * xyz + 16/116))

    lab = np.zeros_like(xyz)
    lab[:, 0] = 116 * xyz[:, 1] - 16  # L
    lab[:, 1] = 500 * (xyz[:, 0] - xyz[:, 1])  # A
    lab[:, 2] = 200 * (xyz[:, 1] - xyz[:, 2])  # B

    return lab.reshape(-1, 3)

def calculate_color_distance(color1, color2, method='lab'):
    """Calculate perceptual distance between two colors"""
    if method == 'lab':
        lab1 = rgb_to_lab(color1)
        lab2 = rgb_to_lab(color2)
        return np.sqrt(np.sum((lab1 - lab2) ** 2))
    elif method == 'hsv':
        hsv1 = colorsys.rgb_to_hsv(color1[0]/255, color1[1]/255, color1[2]/255)
        hsv2 = colorsys.rgb_to_hsv(color2[0]/255, color2[1]/255, color2[2]/255)
        # Weight hue difference more heavily
        h_diff = min(abs(hsv1[0] - hsv2[0]), 1 - abs(hsv1[0] - hsv2[0])) * 2
        s_diff = abs(hsv1[1] - hsv2[1])
        v_diff = abs(hsv1[2] - hsv2[2])
        return np.sqrt(h_diff**2 + s_diff**2 + v_diff**2)
    else:  # euclidean
        return np.sqrt(np.sum((np.array(color1) - np.array(color2)) ** 2))

def filter_similar_colors_improved(colors, min_distance=30, method='lab'):
    """Improved color filtering using perceptual distance"""
    if len(colors) <= 1:
        return colors

    filtered_colors = [colors[0]]

    for color in colors[1:]:
        is_different = True
        for existing_color in filtered_colors:
            distance = calculate_color_distance(color, existing_color, method)
            if distance < min_distance:
                is_different = False
                break

        if is_different:
            filtered_colors.append(color)

    return np.array(filtered_colors)

def get_color_name(rgb):
    """Generate a descriptive name for a color based on its RGB values"""
    r, g, b = rgb

    # Calculate hue, saturation, and lightness
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    diff = max_val - min_val

    # Lightness
    lightness = (max_val + min_val) / 2

    # Basic color names based on dominant channel and lightness
    if diff < 30:  # Low saturation (grayscale)
        if lightness < 50:
            return "Charcoal"
        elif lightness < 100:
            return "Dark gray"
        elif lightness < 150:
            return "Medium gray"
        elif lightness < 200:
            return "Light gray"
        else:
            return "Off white"

    # Determine dominant hue
    if r >= g and r >= b:  # Red dominant
        if g > b + 50:  # More yellow
            if lightness > 150:
                return "Peach" if lightness > 200 else "Orange"
            else:
                return "Orange peel" if g > 100 else "Persimmon"
        elif b > g + 30:  # More blue (purple)
            return "Magenta" if lightness > 100 else "Maroon"
        else:  # Pure red
            if lightness > 150:
                return "Salmon" if lightness > 200 else "Coral"
            else:
                return "Sinopia" if lightness < 100 else "Red"

    elif g >= r and g >= b:  # Green dominant
        if r > b + 30:  # Yellow-green
            return "Lime" if lightness > 150 else "Olive"
        elif b > r + 30:  # Blue-green
            return "Teal" if lightness > 100 else "Dark teal"
        else:  # Pure green
            return "Light green" if lightness > 150 else "Forest green"

    else:  # Blue dominant
        if r > g + 30:  # Purple-blue
            return "Lavender" if lightness > 150 else "Purple"
        elif g > r + 30:  # Cyan-blue
            return "Sky blue" if lightness > 150 else "Steel blue"
        else:  # Pure blue
            return "Light blue" if lightness > 150 else "Navy blue"

def extract_colors_kmeans_improved(image, n_colors, sample_fraction=0.1):
    """Improved K-means with better initialization and sampling"""
    img_array = np.array(image)
    pixels = img_array.reshape(-1, 3)

    # Sample pixels for faster processing on large images
    if len(pixels) > 10000:
        sample_size = max(int(len(pixels) * sample_fraction), n_colors * 100)
        indices = np.random.choice(len(pixels), sample_size, replace=False)
        pixels_sample = pixels[indices]
    else:
        pixels_sample = pixels

    # Use MiniBatchKMeans for better performance
    kmeans = MiniBatchKMeans(
        n_clusters=n_colors,
        random_state=42,
        batch_size=1000,
        n_init=10,
        init='k-means++'
    )

    kmeans.fit(pixels_sample)
    colors = kmeans.cluster_centers_

    # Get the frequency of each cluster
    labels = kmeans.predict(pixels_sample)
    frequencies = Counter(labels)

    # Sort colors by frequency
    color_freq_pairs = [(colors[i], frequencies[i]) for i in range(len(colors))]
    color_freq_pairs.sort(key=lambda x: x[1], reverse=True)

    return np.array([pair[0] for pair in color_freq_pairs]).astype(int)

def extract_colors_histogram(image, n_colors, bins_per_channel=8):
    """Extract colors using 3D color histogram"""
    img_array = np.array(image)

    # Create 3D histogram
    hist, edges = np.histogramdd(
        img_array.reshape(-1, 3),
        bins=bins_per_channel,
        range=[(0, 256), (0, 256), (0, 256)]
    )

    # Find the most frequent color combinations
    flat_hist = hist.flatten()
    top_indices = np.argpartition(flat_hist, -n_colors*3)[-n_colors*3:]
    top_indices = top_indices[np.argsort(flat_hist[top_indices])][::-1]

    # Convert indices back to RGB values
    colors = []
    for idx in top_indices:
        if flat_hist[idx] > 0:  # Only consider bins with actual pixels
            # Convert flat index to 3D coordinates
            coords = np.unravel_index(idx, hist.shape)
            # Convert bin coordinates to RGB values (take bin center)
            r = int((edges[0][coords[0]] + edges[0][coords[0]+1]) / 2)
            g = int((edges[1][coords[1]] + edges[1][coords[1]+1]) / 2)
            b = int((edges[2][coords[2]] + edges[2][coords[2]+1]) / 2)
            colors.append([r, g, b])

            if len(colors) >= n_colors:
                break

    return np.array(colors) if colors else np.array([[0, 0, 0]])

def extract_colors_median_cut(image, n_colors):
    """Extract colors using median cut algorithm"""
    img_array = np.array(image)
    pixels = img_array.reshape(-1, 3)

    def median_cut_recursive(pixels, depth):
        if depth == 0 or len(pixels) == 0:
            return [np.mean(pixels, axis=0)] if len(pixels) > 0 else []

        # Find the channel with the greatest range
        ranges = np.max(pixels, axis=0) - np.min(pixels, axis=0)
        channel = np.argmax(ranges)

        # Sort pixels by the selected channel
        pixels_sorted = pixels[pixels[:, channel].argsort()]

        # Split at median
        median_idx = len(pixels_sorted) // 2
        left_pixels = pixels_sorted[:median_idx]
        right_pixels = pixels_sorted[median_idx:]

        # Recursively process both halves
        left_colors = median_cut_recursive(left_pixels, depth - 1)
        right_colors = median_cut_recursive(right_pixels, depth - 1)

        return left_colors + right_colors

    # Calculate required depth
    depth = int(np.ceil(np.log2(n_colors)))
    colors = median_cut_recursive(pixels, depth)

    return np.array(colors[:n_colors]).astype(int) if colors else np.array([[0, 0, 0]])

def extract_colors_hierarchical(image, n_colors, linkage_method='ward'):
    """Extract colors using hierarchical clustering"""
    img_array = np.array(image)
    pixels = img_array.reshape(-1, 3)

    # Sample for performance
    if len(pixels) > 5000:
        indices = np.random.choice(len(pixels), 5000, replace=False)
        pixels_sample = pixels[indices]
    else:
        pixels_sample = pixels

    try:
        # Perform hierarchical clustering
        linkage_matrix = linkage(pixels_sample, method=linkage_method)
        cluster_labels = fcluster(linkage_matrix, n_colors, criterion='maxclust')

        # Calculate cluster centers
        colors = []
        for i in range(1, n_colors + 1):
            cluster_pixels = pixels_sample[cluster_labels == i]
            if len(cluster_pixels) > 0:
                colors.append(np.mean(cluster_pixels, axis=0))

        return np.array(colors).astype(int) if colors else np.array([[0, 0, 0]])
    except Exception as e:
        # Fallback to kmeans if hierarchical clustering fails
        st.warning(f"Hierarchical clustering failed, falling back to k-means: {str(e)}")
        return extract_colors_kmeans_improved(image, n_colors, 0.2)

def extract_colors_dominant_sampling(image, n_colors, sample_method='grid'):
    """Extract colors by sampling dominant regions"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]

    if sample_method == 'grid':
        # Sample from a grid
        grid_size = int(np.sqrt(n_colors * 20))
        y_coords = np.linspace(0, h-1, grid_size).astype(int)
        x_coords = np.linspace(0, w-1, grid_size).astype(int)

        sampled_colors = []
        for y in y_coords:
            for x in x_coords:
                sampled_colors.append(img_array[y, x])

    elif sample_method == 'random':
        # Random sampling
        num_samples = n_colors * 50
        y_coords = np.random.randint(0, h, num_samples)
        x_coords = np.random.randint(0, w, num_samples)
        sampled_colors = [img_array[y, x] for y, x in zip(y_coords, x_coords)]

    # Cluster the sampled colors
    if len(sampled_colors) > n_colors:
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(sampled_colors)
        colors = kmeans.cluster_centers_
    else:
        colors = sampled_colors

    return np.array(colors).astype(int)

def extract_colors_quantization_improved(image, n_colors, quant_level=8):
    """Improved quantization method"""
    img_array = np.array(image)
    pixels = img_array.reshape(-1, 3)

    # Reduce color space
    div = 256 // quant_level
    quantized_pixels = (pixels // div) * div

    # Get unique colors and their counts
    unique, counts = np.unique(quantized_pixels, axis=0, return_counts=True)

    # Sort by frequency and take top colors
    top_colors = unique[np.argsort(-counts)[:n_colors]]

    return top_colors.astype(int)

@st.cache_data
def extract_colors_main(image_array, n_colors, method='kmeans_improved', min_distance=30, **kwargs):
    """Main color extraction function with caching"""
    # Reconstruct PIL image from array
    image = Image.fromarray(image_array)

    try:
        if method == 'kmeans_improved':
            colors = extract_colors_kmeans_improved(image, n_colors, kwargs.get('sample_fraction', 0.1))
        elif method == 'histogram':
            colors = extract_colors_histogram(image, n_colors, kwargs.get('bins_per_channel', 8))
        elif method == 'median_cut':
            colors = extract_colors_median_cut(image, n_colors)
        elif method == 'hierarchical':
            colors = extract_colors_hierarchical(image, n_colors, kwargs.get('linkage_method', 'ward'))
        elif method == 'dominant_sampling':
            colors = extract_colors_dominant_sampling(image, n_colors, kwargs.get('sample_method', 'grid'))
        elif method == 'quantization_improved':
            colors = extract_colors_quantization_improved(image, n_colors, kwargs.get('quant_level', 8))
        elif method == 'kmeans_basic':
            # Basic kmeans implementation
            img_array = np.array(image)
            pixels = img_array.reshape(-1, 3)
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_.astype(int)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Apply similarity filtering if requested
        if min_distance > 0 and len(colors) > 1:
            colors = filter_similar_colors_improved(colors, min_distance, method='lab')

        return colors

    except Exception as e:
        st.error(f"Error in {method}: {str(e)}")
        # Fallback to basic kmeans
        img_array = np.array(image)
        pixels = img_array.reshape(-1, 3)
        n_clusters = min(n_colors, len(np.unique(pixels.view(np.dtype((np.void, pixels.dtype.itemsize*pixels.shape[1]))).view(pixels.dtype).reshape(-1, pixels.shape[1]))))
        if n_clusters > 0:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(pixels)
            return kmeans.cluster_centers_.astype(int)
        else:
            return np.array([[0, 0, 0]])

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
    # Create a more interesting sample image
    width, height = 400, 200
    image = Image.new('RGB', (width, height))
    pixels = []

    for y in range(height):
        for x in range(width):
            # Create a more varied color pattern
            r = int(255 * (x / width) * (1 - y / height))
            g = int(255 * (y / height))
            b = int(255 * ((width - x) / width) * (y / height))
            # Add some purple areas
            if x > width * 0.7 and y > height * 0.7:
                r = int(r * 0.8 + 128 * 0.2)
                b = int(b * 0.8 + 200 * 0.2)
            pixels.append((r, g, b))

    image.putdata(pixels)
    return image

def display_color_results():
    """Display the extracted color results from session state"""
    # Check if extracted_colors exists and is not None
    if ('extracted_colors' not in st.session_state or
        st.session_state.extracted_colors is None or
        len(st.session_state.extracted_colors) == 0):
        return

    # Check if all required session state variables exist and are not None
    required_vars = ['color_list', 'output_format', 'extraction_method', 'requested_colors']
    for var in required_vars:
        if var not in st.session_state or st.session_state[var] is None:
            return

    # Now we can safely access the variables
    colors = st.session_state.extracted_colors
    color_list = st.session_state.color_list
    output_format = st.session_state.output_format
    method = st.session_state.extraction_method
    n_colors = st.session_state.requested_colors

    # Additional safety check - make sure colors is not None and has content
    if colors is None or len(colors) == 0:
        st.warning("⚠️ No colors were extracted. Please try again with different settings.")
        return

    # Display success message
    st.success(f"🎉 Successfully extracted {len(colors)} colors using {method.replace('_', ' ').title()} algorithm!")

    # Rest of the function continues as before...
    if len(colors) < n_colors:
        st.warning(f"⚠️ Requested {n_colors} colors, but only found {len(colors)} distinct colors with current similarity settings.")
        st.info("💡 Try reducing the similarity threshold or using a different algorithm to get more colors.")

    # Color harmony analysis
    harmony_info = get_color_harmony_info(colors)
    if harmony_info:
        st.markdown(harmony_info)

    # Display color swatches
    if len(colors) > 0:
        st.header("🎨 Extracted Color Palette")

        # Create columns for color swatches
        cols = st.columns(min(len(colors), 5))  # Max 5 columns

        for idx, color in enumerate(colors):
            col_idx = idx % 5  # Wrap to new row if more than 5 colors
            if idx > 0 and col_idx == 0:
                # Create new row of columns if needed
                cols = st.columns(min(len(colors) - idx, 5))

            with cols[col_idx]:
                hex_color = rgb_to_hex(color)
                # Determine text color based on background brightness
                brightness = (0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])
                text_color = 'white' if brightness < 128 else 'black'

                st.markdown(
                    f'''
                    <div class="color-swatch" style="
                        background-color: {hex_color};
                        color: {text_color};
                    ">
                        {color_list[idx]}
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

                # Add color info below swatch
                st.markdown(f"""
                <div style="text-align: center; font-size: 12px; color: #666; margin-bottom: 15px;">
                    Color #{idx + 1}<br>
                    RGB: {color[0]}, {color[1]}, {color[2]}<br>
                    HEX: {hex_color}
                </div>
                """, unsafe_allow_html=True)

        # Display color values for easy copying
        st.subheader("📋 Copy Color Values")

        # Create tabs for different formats
        tab1, tab2, tab3, tab4 = st.tabs(["📝 List Format", "🔗 Comma Separated", "📊 JSON Format", "🗃️ XML Format"])

        with tab1:
            color_text = '\n'.join([str(color) for color in color_list])
            st.text_area(
                "Color list (one per line):",
                color_text,
                height=120,
                help="Each color on a separate line - perfect for spreadsheets"
            )

        with tab2:
            comma_separated = ', '.join([str(color) for color in color_list])
            st.text_area(
                "Comma-separated values:",
                comma_separated,
                height=80,
                help="All colors in one line, separated by commas"
            )

        with tab3:
            json_colors = json.dumps(color_list, indent=2)
            st.text_area(
                "JSON format:",
                json_colors,
                height=120,
                help="Structured JSON format for programming use"
            )

        with tab4:
            # Generate XML content for display
            xml_lines = ['<palette>']
            for i, color in enumerate(colors):
                hex_color = rgb_to_hex(color)
                r, g, b = color[0], color[1], color[2]
                xml_lines.append(f'  <color name="Color {i+1}" hex="{hex_color}" r="{r}" g="{g}" b="{b}" />')
            xml_lines.append('</palette>')
            xml_content = '\n'.join(xml_lines)

            st.text_area(
                "XML format:",
                xml_content,
                height=120,
                help="XML format with color attributes"
            )

        # Export functionality
        st.subheader("📥 Download Palette")
        st.markdown("Save your color palette in various formats:")

        col_export1, col_export2, col_export3, col_export4 = st.columns(4)

        with col_export1:
            st.markdown("**📄 JSON Data**")
            st.markdown("*Complete extraction info*")
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
                    'palette_name': f'Extracted_Palette_{int(time.time())}',
                    'colors': serializable_colors,
                    'color_count': len(colors),
                    'format': output_format,
                    'extraction_method': method,
                    'parameters': {
                        'requested_colors': int(n_colors),
                        'actual_colors': len(colors),
                        'similarity_threshold': int(st.session_state.get('min_distance', 30)),
                        **{k: convert_numpy_types(v) for k, v in st.session_state.get('method_params', {}).items()}
                    },
                    'image_info': st.session_state.get('image_info', {}),
                    'extracted_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }

                json_data = json.dumps(palette_data, indent=2)
                st.download_button(
                    label="📄 Download JSON",
                    data=json_data,
                    file_name=f"color_palette_{int(time.time())}.json",
                    mime="application/json",
                    help="Complete data with extraction settings and metadata",
                    key="download_json"
                )
            except Exception as e:
                st.error(f"Error creating JSON: {str(e)}")

        with col_export2:
            st.markdown("**🎨 CSS Variables**")
            st.markdown("*Ready for web development*")
            try:
                # Generate CSS variables
                css_vars = []
                for i, color in enumerate(color_list):
                    css_vars.append(f'  --color-{i+1}: {color};')
                    # Simple color naming
                    color_name = f'color-{i+1}'
                    css_vars.append(f'  --{color_name}: {color};')

                css_content = f':root {{\n{chr(10).join(css_vars)}\n}}\n\n/* Usage examples */\n.primary {{ color: var(--color-1); }}\n.secondary {{ background-color: var(--color-2); }}'

                st.download_button(
                    label="🎨 Download CSS",
                    data=css_content,
                    file_name=f"palette_{int(time.time())}.css",
                    mime="text/css",
                    help="CSS custom properties ready to use in your stylesheets",
                    key="download_css"
                )

            except Exception as e:
                st.error(f"Error creating CSS: {str(e)}")

        with col_export3:
            st.markdown("**📄 TXT File**")
            st.markdown("*Simple text format*")
            try:
                # Generate TXT content
                txt_lines = []
                txt_lines.append(f"Color Palette - Extracted {time.strftime('%Y-%m-%d %H:%M:%S')}")
                txt_lines.append(f"Method: {method.replace('_', ' ').title()}")
                txt_lines.append(f"Colors: {len(colors)}")
                txt_lines.append("-" * 40)

                for i, color in enumerate(colors):
                    hex_color = rgb_to_hex(color)
                    txt_lines.append(f"Color {i+1}")
                    txt_lines.append(f"  HEX: {hex_color}")
                    txt_lines.append(f"  RGB: {color[0]}, {color[1]}, {color[2]}")
                    txt_lines.append("")

                txt_content = '\n'.join(txt_lines)

                st.download_button(
                    label="📄 Download TXT",
                    data=txt_content,
                    file_name=f"palette_{int(time.time())}.txt",
                    mime="text/plain",
                    help="Simple text format with color names and values",
                    key="download_txt"
                )

            except Exception as e:
                st.error(f"Error creating TXT: {str(e)}")

        with col_export4:
            st.markdown("**🗃️ XML Palette**")
            st.markdown("*Structured with color names*")
            try:
                # Generate XML content
                xml_lines = ['<palette>']

                for i, color in enumerate(colors):
                    hex_color = rgb_to_hex(color)
                    r, g, b = color[0], color[1], color[2]

                    xml_lines.append(f'  <color name="Color {i+1}" hex="{hex_color}" r="{r}" g="{g}" b="{b}" />')

                xml_lines.append('</palette>')
                xml_content = '\n'.join(xml_lines)

                st.download_button(
                    label="🗃️ Download XML",
                    data=xml_content,
                    file_name=f"palette_{int(time.time())}.xml",
                    mime="application/xml",
                    help="XML format with descriptive color names and RGB/HEX values",
                    key="download_xml"
                )

            except Exception as e:
                st.error(f"Error creating XML: {str(e)}")

        # Clear results button
        st.markdown("---")
        col_clear1, col_clear2, col_clear3 = st.columns([1, 1, 1])

        with col_clear2:
            if st.button("🗑️ Clear Results", help="Clear extracted colors to start fresh"):
                # Clear all extraction-related session state
                keys_to_clear = [
                    'extracted_colors', 'color_list', 'output_format',
                    'extraction_method', 'requested_colors', 'min_distance',
                    'method_params', 'image_info', 'extraction_time'
                ]

                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]

                st.rerun()

# Main UI
st.markdown('<h1 class="main-header">🎨 Color Palette Extractor</h1>', unsafe_allow_html=True)

st.markdown("""
Extract dominant colors from any image using multiple advanced algorithms including K-means clustering,
color histograms, median cut, and hierarchical clustering. Upload an image or try the sample to get started!
""")

# Initialize session state - COMPLETE INITIALIZATION
if 'extracted_colors' not in st.session_state:
    st.session_state.extracted_colors = None
if 'color_list' not in st.session_state:
    st.session_state.color_list = None
if 'output_format' not in st.session_state:
    st.session_state.output_format = 'HEX'
if 'extraction_method' not in st.session_state:
    st.session_state.extraction_method = None
if 'requested_colors' not in st.session_state:
    st.session_state.requested_colors = 5
if 'min_distance' not in st.session_state:
    st.session_state.min_distance = 30
if 'method_params' not in st.session_state:
    st.session_state.method_params = {}
if 'image_info' not in st.session_state:
    st.session_state.image_info = {}

# Enhanced expandable help section with all tips combined
with st.expander("ℹ️ Complete Guide: Algorithms, Tips & Color Theory", expanded=False):
    st.markdown("""
    ### 🧠 Available Algorithms

    **K-means (Improved)** - Best overall performance
    - Uses smart pixel sampling and frequency-based sorting
    - Excellent for photographs and complex images
    - Fast processing with high accuracy

    **Color Histogram** - Best for graphics and illustrations
    - Analyzes color frequency distribution in 3D space
    - Great for images with distinct color regions
    - Very fast processing

    **Median Cut** - Classic algorithm used by many image editors
    - Recursively divides color space at median points
    - Produces well-balanced color palettes
    - Good for all image types

    **Hierarchical Clustering** - Best for color relationships
    - Creates tree-like color groupings
    - Excellent for analyzing color harmony
    - Slower but very accurate

    **Dominant Sampling** - Best for quick previews
    - Samples key regions of the image
    - Very fast processing
    - Good for real-time applications

    ### 💡 Tips for Best Results
    - **High contrast images** produce more distinct palettes
    - **Well-lit photos** give accurate color representation
    - **Different algorithms** work better for different image types
    - **Lower similarity thresholds** preserve more color variations
    - **Higher similarity thresholds** create more cohesive palettes

    ### 🎯 Algorithm Recommendations by Image Type
    - **Photographs:** K-means (Improved) or Hierarchical
    - **Logos/Graphics:** Histogram or Median Cut
    - **Artwork/Paintings:** Median Cut or K-means (Improved)
    - **Quick Analysis:** Dominant Sampling or Histogram

    ### 🔥 Quick Performance Tips
    - **Photographs:** Use K-means Improved
    - **Graphics/Logos:** Try Histogram or Median Cut
    - **Artistic Images:** Hierarchical works great
    - **Quick Preview:** Use Dominant Sampling
    - **Need exact colors:** Set similarity to 0
    - **Want cohesive palette:** Increase similarity threshold

    ### ⚡ Technical Performance Tips
    - Large images are auto-resized for speed
    - Sampling reduces processing time
    - Try different algorithms for best results
    - Lower similarity = more colors

    ### 🎨 Color Theory Basics
    - **Complementary:** Colors opposite on color wheel - create high contrast
    - **Analogous:** Colors next to each other - create harmony
    - **Triadic:** Three evenly spaced colors - balanced and vibrant
    - **Monochromatic:** Shades of single hue - elegant and cohesive
    - **Warm Colors:** Reds, oranges, yellows - energetic and bold
    - **Cool Colors:** Blues, greens, purples - calming and professional

    ### 🎯 Use Cases & Applications
    - **Web Design:** Create CSS color variables from design mockups
    - **Brand Identity:** Extract official colors from logos and marketing materials
    - **Interior Design:** Match paint colors to inspiration photos
    - **Art Analysis:** Study color relationships in paintings and artwork
    - **Fashion:** Extract color palettes from clothing and accessories
    - **Photography:** Analyze dominant tones for editing decisions

    ### 🔧 Advanced Settings Guide

    **Similarity Filtering:**
    - 0: No filtering - keep all extracted colors
    - 1-20: Low filtering - remove very similar colors
    - 20-50: Medium filtering - remove moderately similar colors
    - 50+: High filtering - keep only very distinct colors

    **Sampling (K-means Improved):**
    - 0.05-0.1: Fast processing, good for large images
    - 0.1-0.3: Balanced speed and accuracy
    - 0.3-0.5: Slower but more accurate results

    **Histogram Resolution:**
    - 4-8 bins: Broad color groupings, fewer distinct colors
    - 8-12 bins: Balanced detail and simplicity
    - 12-16 bins: High detail, more nuanced colors

    ### 📁 Export Formats
    - **JSON:** Complete extraction data with metadata
    - **CSS:** Ready-to-use CSS custom properties
    - **TXT:** Simple text format for any application
    - **XML:** Structured format with color names and values
    """)

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("⚙️ Extraction Settings")

    # Number of colors
    n_colors = st.slider(
        "Number of colors to extract",
        2, 15, 5,
        help="More colors create a detailed palette, fewer colors create a simplified color scheme"
    )

    # Enhanced method selection with descriptions
    st.markdown("**Extraction Algorithm**")
    method = st.selectbox(
        "Choose algorithm",
        [
            'kmeans_improved',
            'histogram',
            'median_cut',
            'hierarchical',
            'dominant_sampling',
            'quantization_improved',
            'kmeans_basic'
        ],
        format_func=lambda x: {
            'kmeans_improved': '🧠 K-means (Improved)',
            'histogram': '📊 Color Histogram',
            'median_cut': '✂️ Median Cut',
            'hierarchical': '🌳 Hierarchical Clustering',
            'dominant_sampling': '🎯 Dominant Sampling',
            'quantization_improved': '⚡ Quantization (Improved)',
            'kmeans_basic': '⚙️ K-means (Basic)'
        }[x],
        help="Each algorithm has different strengths - see the guide above for recommendations"
    )

    # Show algorithm description
    algorithm_descriptions = {
        'kmeans_improved': "🧠 **Smart K-means** - Uses sampling and frequency sorting for optimal results",
        'histogram': "📊 **Color Histogram** - Analyzes color frequency in 3D space",
        'median_cut': "✂️ **Median Cut** - Classic algorithm that divides color space recursively",
        'hierarchical': "🌳 **Hierarchical** - Creates tree-based color groupings",
        'dominant_sampling': "🎯 **Smart Sampling** - Samples key image regions",
        'quantization_improved': "⚡ **Quantization** - Reduces color precision to find dominant colors",
        'kmeans_basic': "⚙️ **Basic K-means** - Standard clustering approach"
    }

    st.markdown(f"""
    <div class="algorithm-card">
    {algorithm_descriptions[method]}
    </div>
    """, unsafe_allow_html=True)

    # Method-specific parameters
    method_params = {}

    if method == 'kmeans_improved':
        st.markdown("**Sampling Settings**")
        method_params['sample_fraction'] = st.slider(
            "Sample fraction", 0.05, 0.5, 0.1,
            help="Fraction of pixels to sample (lower = faster, higher = more accurate)"
        )
    elif method == 'histogram':
        method_params['bins_per_channel'] = st.slider(
            "Histogram resolution", 4, 16, 8,
            help="Higher values = more color precision, lower = broader groupings"
        )
    elif method == 'hierarchical':
        method_params['linkage_method'] = st.selectbox(
            "Clustering method",
            ['ward', 'complete', 'average', 'single'],
            help="Ward generally produces the best results for color clustering"
        )
    elif method == 'dominant_sampling':
        method_params['sample_method'] = st.selectbox(
            "Sampling pattern",
            ['grid', 'random'],
            help="Grid sampling is more systematic, random sampling covers more variation"
        )
    elif method == 'quantization_improved':
        method_params['quant_level'] = st.slider(
            "Quantization level", 4, 32, 8,
            help="Higher values preserve more color nuances"
        )

    # Similarity filtering
    st.markdown("**Color Similarity Filter**")
    min_distance = st.slider(
        "Minimum color distance", 0, 100, 30,
        help="Higher values remove more similar colors. Based on perceptual LAB color space."
    )

    if min_distance == 0:
        st.info("🔄 No filtering - all extracted colors will be kept")
    elif min_distance < 20:
        st.info("🎨 Low filtering - very similar colors will be removed")
    elif min_distance < 50:
        st.info("⚖️ Medium filtering - moderately similar colors will be removed")
    else:
        st.info("🎯 High filtering - only very distinct colors will be kept")

    # Output format
    st.markdown("**Output Format**")
    output_format = st.selectbox(
        "Color format for export",
        ['HEX', 'RGB'],
        help="HEX for web/CSS use, RGB for design software"
    )

    # Sample image option
    st.subheader("🖼️ Try Sample")
    if st.button("🌈 Use Sample Image", help="Try the app with a colorful test image"):
        st.session_state.use_sample = True

with col1:
    # File uploader
    st.markdown("### 📁 Upload Your Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'],
        help="Supported: PNG, JPG, JPEG, WEBP, BMP, TIFF. Best results with high-quality, well-lit images."
    )

    # Handle sample image
    if st.session_state.get('use_sample', False) and uploaded_file is None:
        image = create_sample_image()
        st.image(image, caption='🌈 Sample Test Image - Perfect for Algorithm Testing!', use_container_width=True)
        st.markdown("""
        <div class="help-box">
        This test image contains gradients and distinct color regions.
        Perfect for comparing different extraction algorithms and settings!
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
            st.image(image, caption=f'📁 Uploaded: {uploaded_file.name} ({image.size[0]}×{image.size[1]} pixels)',
                    use_container_width=True)

            # Image analysis info
            total_pixels = image.size[0] * image.size[1]
            file_size = len(uploaded_file.getvalue()) / 1024  # KB
            st.markdown(f"""
            <div class="help-box">
            <strong>📊 Image Analysis:</strong><br>
            • Resolution: {image.size[0]} × {image.size[1]} pixels<br>
            • Total pixels: {total_pixels:,}<br>
            • File size: {file_size:.1f} KB<br>
            • Color mode: {image.mode}<br>
            {f'<em>⚡ Large image - processing will use smart sampling for speed</em>' if total_pixels > 100000 else '<em>✅ Good size for quick processing</em>'}
            </div>
            """, unsafe_allow_html=True)

            process_image = True

            # Reset sample flag
            if 'use_sample' in st.session_state:
                del st.session_state.use_sample

        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")
            st.markdown("""
            **Troubleshooting Tips:**
            - Ensure the file is a valid image format
            - Try a different image if this one might be corrupted
            - Check that the file isn't too large (recommended < 50MB)
            - Some exotic image formats might not be supported
            """)
            st.stop()
    else:
        process_image = False
        st.info("👆 Upload an image above or try the sample image to get started!")
        st.markdown("""
        **💡 For best results, use images with:**
        - Good lighting and contrast
        - Multiple distinct colors
        - High resolution (but not too large for web upload)
        - Minimal noise or compression artifacts
        """)

# Process image if available
if process_image:
    # Resize for processing to improve performance
    processing_image = resize_image_for_processing(image.copy())

    if st.button("🎨 Extract Color Palette", type="primary", help="Analyze your image and extract the dominant colors"):
        try:
            with st.spinner('🔍 Analyzing image colors...'):
                progress_bar = st.progress(0)
                progress_text = st.empty()

                progress_text.text("📊 Preparing image data...")
                progress_bar.progress(20)

                progress_text.text(f"🧠 Applying {method.replace('_', ' ').title()} algorithm...")
                progress_bar.progress(40)

                # Convert image to array for caching
                img_array = np.array(processing_image)

                # Extract colors using cached function
                colors = extract_colors_main(
                    img_array,
                    n_colors,
                    method,
                    min_distance,
                    **method_params
                )

                progress_text.text("🎨 Processing color palette...")
                progress_bar.progress(70)

                # Prepare color output
                if output_format == 'HEX':
                    color_list = [rgb_to_hex(color) for color in colors]
                else:  # RGB
                    color_list = [tuple(int(c) for c in color) for color in colors]

                progress_text.text("✅ Color extraction complete!")
                progress_bar.progress(100)
                time.sleep(0.5)  # Brief pause to show completion
                progress_bar.empty()
                progress_text.empty()

                # Store results in session state
                st.session_state.extracted_colors = colors
                st.session_state.color_list = color_list
                st.session_state.output_format = output_format
                st.session_state.extraction_method = method
                st.session_state.requested_colors = n_colors
                st.session_state.min_distance = min_distance
                st.session_state.method_params = method_params
                st.session_state.image_info = {
                    'original_size': list(image.size),
                    'processed_size': list(processing_image.size) if processing_image.size != image.size else None,
                    'total_pixels': int(np.prod(image.size))
                }

        except Exception as e:
            st.error(f"❌ Error extracting colors: {str(e)}")
            st.markdown("""
            **Troubleshooting:**
            - Try reducing the image size or using a different image
            - Switch between different extraction algorithms
            - Reduce the number of colors to extract
            - Check that the image file is not corrupted
            - Try with a different file format
            """)

# Display results if they exist in session state
display_color_results()

# Footer with additional information
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🎨 <strong>Color Palette Extractor</strong> | Built with Streamlit & scikit-learn</p>
    <p><small>
    💡 <strong>Pro Tips:</strong> For brand color extraction, use high-resolution logos with clean backgrounds.
    For photography, well-lit images with good contrast produce the most accurate palettes.
    Try different algorithms to see which works best for your specific image type!
    </small></p>
    <p><small>
    <strong>Algorithms:</strong> K-means Clustering • Color Histograms • Median Cut • Hierarchical Clustering • Smart Sampling
    </small></p>
    <p><small>
    <strong>Export Formats:</strong> JSON • CSS Variables • TXT • XML Palette
    </small></p>
</div>
""", unsafe_allow_html=True)

# Additional tips section for users without extracted colors
if 'extracted_colors' not in st.session_state or st.session_state.extracted_colors is None:
    st.markdown("---")
    st.markdown("### 💡 Getting Started Tips")

    col_tip1, col_tip2, col_tip3 = st.columns(3)

    with col_tip1:
        st.markdown("""
        **🎯 Best Image Types:**
        - High-resolution photos
        - Good lighting and contrast
        - Multiple distinct colors
        - Clean, uncompressed images
        """)

    with col_tip2:
        st.markdown("""
        **⚙️ Algorithm Guide:**
        - **Photos**: K-means Improved
        - **Graphics**: Color Histogram
        - **Art**: Median Cut
        - **Quick**: Dominant Sampling
        """)

    with col_tip3:
        st.markdown("""
        **📊 Export Options:**
        - **JSON**: Complete data
        - **CSS**: Web development
        - **TXT**: Simple text
        - **XML**: Named colors
        """)



# Version and credits
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px; padding: 10px;'>
    <p><strong>Color Palette Extractor</strong></p>
    <p>Powered by scikit-learn, PIL, and Streamlit</p>
    <p>Supports multiple extraction algorithms with perceptual color filtering</p>
</div>
""", unsafe_allow_html=True)
