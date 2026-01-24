import base64
import uuid
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans, MiniBatchKMeans
from collections import Counter
import colorsys
import json
import time
import urllib.parse
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy.cluster.hierarchy as sch
from skimage import data
from skimage.color import rgb2lab

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    from streamlit.web.server.server import Server
except (ImportError, ModuleNotFoundError):
    get_script_run_ctx = None
    Server = None

# =============================================================================
# Constants
# =============================================================================

# Image processing
MAX_IMAGE_SIZE = 800  # Maximum dimension for processing (pixels)
LARGE_IMAGE_THRESHOLD = 100000  # Pixels above which smart sampling is recommended

# Color extraction
DEFAULT_SAMPLE_FRACTION = 0.1  # Fraction of pixels to sample in K-means
MIN_SAMPLE_SIZE_MULTIPLIER = 100  # Minimum samples = n_colors * this value
LARGE_PIXEL_THRESHOLD = 10000  # Above this, sampling is used
DEFAULT_HISTOGRAM_BINS = 8  # Bins per channel for histogram method
DEFAULT_QUANT_LEVEL = 8  # Default quantization level

# Hierarchical clustering
HIERARCHICAL_SAMPLE_MULTIPLIER = 100  # Max samples = n_colors * this
HIERARCHICAL_MIN_SAMPLES = 500  # Minimum samples for hierarchical clustering

# Similarity filtering
DEFAULT_MIN_DISTANCE = 30  # Default LAB color distance threshold

# UI layout
MAX_SWATCH_COLUMNS = 5  # Maximum color swatches per row

# Configure page
st.set_page_config(
    page_title="Colour Palette Extractor",
    page_icon="🎨",
    layout="wide"
)

# Add custom CSS (cached to avoid re-injection on every rerun)
@st.cache_resource
def inject_custom_css():
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

    /* Mobile responsive layout */
    @media (max-width: 768px) {
        /* Stack columns vertically on mobile */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }

        /* Ensure proper spacing between stacked columns */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
        }

        /* Adjust main header for mobile */
        .main-header {
            font-size: 1.5rem;
            margin-bottom: 20px;
        }

        /* Make color swatches smaller on mobile */
        .color-swatch {
            height: 60px;
            font-size: 14px;
        }

        /* Adjust help boxes for mobile */
        .help-box {
            padding: 10px;
            font-size: 14px;
        }

        /* Ensure buttons are touch-friendly (min 44px) */
        .stButton > button {
            min-height: 44px;
            padding: 0.5rem 1rem;
        }

        /* Adjust algorithm cards */
        .algorithm-card {
            padding: 10px;
            margin: 6px 0;
        }
    }

    /* Extra small screens */
    @media (max-width: 480px) {
        .main-header {
            font-size: 1.25rem;
        }

        .color-swatch {
            height: 50px;
            font-size: 12px;
        }
    }
</style>
""", unsafe_allow_html=True)

inject_custom_css()


def get_app_base_url():
    if get_script_run_ctx is None or Server is None:
        return ""

    try:
        ctx = get_script_run_ctx()
        if ctx is None:
            return ""

        server = Server.get_current()
        if server is None:
            return ""

        session_info = server._get_session_info(ctx.session_id)
        if session_info is None:
            return ""

        request = session_info.client.request
        if request is None:
            return ""

        headers = getattr(request, 'headers', {}) or {}

        protocol = (
            getattr(request, 'protocol', '')
            or headers.get('X-Forwarded-Proto', '')
            or ('https' if headers.get('X-Forwarded-Proto', '').startswith('https') else '')
        )

        host = getattr(request, 'host', '') or headers.get('Host', '')
        if not host:
            client = getattr(session_info, 'client', None)
            client_host = getattr(client, 'host', '') if client else ''
            client_port = getattr(client, 'port', None) if client else None
            if client_host:
                if client_port and client_port not in (80, 443) and ':' not in client_host:
                    host = f"{client_host}:{client_port}"
                else:
                    host = client_host

        raw_path = getattr(request, 'path', '') or getattr(request, 'uri', '') or '/'
        split_path = urllib.parse.urlsplit(raw_path)
        path = split_path.path or '/'

        if not protocol:
            if host and host.startswith(('localhost', '127.', '0.0.0.0', '[::1]')):
                protocol = 'http'
            else:
                protocol = 'https'

        if host.startswith(('http://', 'https://')):
            parsed_host = urllib.parse.urlsplit(host)
            protocol = parsed_host.scheme or protocol
            host = parsed_host.netloc or parsed_host.path

        if not protocol or not host:
            return ""

        return urllib.parse.urlunsplit((protocol, host, path, '', ''))
    except Exception:
        return ""


def encode_palette_payload(payload):
    payload_clean = convert_numpy_types(payload)
    data = json.dumps(payload_clean, separators=(',', ':')).encode('utf-8')
    return base64.urlsafe_b64encode(data).decode('utf-8').rstrip('=')


def decode_palette_payload(token):
    padding = '=' * (-len(token) % 4)
    decoded = base64.urlsafe_b64decode(token + padding)
    return json.loads(decoded.decode('utf-8'))

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


def update_share_link(persist_query=None):
    colors = st.session_state.get('extracted_colors')
    if colors is None or len(colors) == 0:
        clear_share_link()
        return

    payload = {
        'colors': np.asarray(colors, dtype=int),
        'output_format': st.session_state.get('output_format', 'HEX'),
        'extraction_method': st.session_state.get('extraction_method'),
        'requested_colors': st.session_state.get('requested_colors'),
        'min_distance': st.session_state.get('min_distance'),
        'method_params': st.session_state.get('method_params', {}),
        'image_info': st.session_state.get('image_info', {}),
    }

    token = encode_palette_payload(payload)

    should_persist = persist_query if persist_query is not None else st.session_state.get('_palette_from_query', False)

    if should_persist:
        st.query_params["palette"] = token
    else:
        try:
            del st.query_params["palette"]
        except KeyError:
            pass

    base_url = get_app_base_url() or st.session_state.get('_base_url_cached', '')
    share_url = f"?palette={token}"

    if base_url:
        parsed = urllib.parse.urlsplit(base_url)
        sanitized_path = parsed.path or '/'
        base = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, sanitized_path, '', ''))
        share_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, sanitized_path, f"palette={token}", ''))
        st.session_state['_base_url_cached'] = base

    if not share_url.startswith(('http://', 'https://')) and st.session_state.get('_base_url_cached'):
        parsed_cached = urllib.parse.urlsplit(st.session_state['_base_url_cached'])
        sanitized_path = parsed_cached.path or '/'
        share_url = urllib.parse.urlunsplit((parsed_cached.scheme, parsed_cached.netloc, sanitized_path, f"palette={token}", ''))

    st.session_state.share_token = token
    st.session_state.share_link = share_url
    st.session_state['_palette_from_query'] = should_persist
    st.session_state['_last_loaded_palette_token'] = token


def clear_share_link():
    try:
        del st.query_params["palette"]
    except KeyError:
        pass

    st.session_state.pop('share_link', None)
    st.session_state.pop('share_token', None)
    st.session_state['_palette_from_query'] = False
    st.session_state['_last_loaded_palette_token'] = None
    st.session_state['_active_image_signature'] = None


def load_palette_from_query():
    params = st.query_params
    token_value = params.get('palette')
    if isinstance(token_value, list):
        token = token_value[0] if token_value else None
    else:
        token = token_value

    if not token:
        st.session_state['_palette_from_query'] = False
        st.session_state['_last_loaded_palette_token'] = None
        clear_share_link()
        return

    if token == st.session_state.get('_last_loaded_palette_token'):
        return

    try:
        payload = decode_palette_payload(token)
    except Exception:
        st.warning("⚠️ Unable to load the shared palette. The link may be invalid or corrupted.")
        st.session_state['_palette_from_query'] = False
        st.session_state['_last_loaded_palette_token'] = None
        clear_share_link()
        return

    colors = np.array(payload.get('colors', []), dtype=int)
    if colors.size == 0:
        st.session_state['_palette_from_query'] = False
        st.session_state['_last_loaded_palette_token'] = None
        clear_share_link()
        return

    output_format = payload.get('output_format', 'HEX')
    method = payload.get('extraction_method')
    requested_colors = payload.get('requested_colors', len(colors))
    min_distance = payload.get('min_distance', st.session_state.get('min_distance', 30))
    method_params = payload.get('method_params', {})
    image_info = payload.get('image_info', {})

    st.session_state.extracted_colors = colors
    # Note: output_format from shared link is used for color_list generation,
    # but we use current widget value since output_format is managed by selectbox
    st.session_state.extraction_method = method
    st.session_state.requested_colors = requested_colors
    st.session_state.min_distance = min_distance
    st.session_state.method_params = method_params
    st.session_state.image_info = image_info

    # Use the format from the shared link for initial display
    if output_format == 'HEX':
        color_list = [rgb_to_hex(color) for color in colors]
    else:
        color_list = [tuple(int(c) for c in color) for color in colors]

    st.session_state.color_list = color_list
    update_share_link(persist_query=True)
    st.session_state['_palette_from_query'] = True
    st.session_state['_last_loaded_palette_token'] = token
    st.session_state['_active_image_signature'] = None


def backup_palette_state():
    """Backup current palette state before clearing (for undo functionality)"""
    if st.session_state.extracted_colors is not None and len(st.session_state.extracted_colors) > 0:
        st.session_state._palette_backup = {
            'extracted_colors': st.session_state.extracted_colors.copy() if hasattr(st.session_state.extracted_colors, 'copy') else st.session_state.extracted_colors,
            'color_list': st.session_state.color_list.copy() if st.session_state.color_list else None,
            'output_format': st.session_state.output_format,
            'extraction_method': st.session_state.extraction_method,
            'requested_colors': st.session_state.get('requested_colors'),
            'min_distance': st.session_state.get('min_distance'),
            'method_params': st.session_state.method_params.copy() if st.session_state.method_params else {},
            'image_info': st.session_state.image_info.copy() if st.session_state.image_info else {},
        }


def restore_palette_state():
    """Restore palette state from backup (undo clear)"""
    backup = st.session_state.get('_palette_backup')
    if backup:
        st.session_state.extracted_colors = backup['extracted_colors']
        # Regenerate color_list based on current format (output_format is managed by widget)
        colors = backup['extracted_colors']
        current_format = st.session_state.get('output_format', 'HEX')
        if current_format == 'HEX':
            st.session_state.color_list = [rgb_to_hex(color) for color in colors]
        else:
            st.session_state.color_list = [tuple(int(c) for c in color) for color in colors]
        st.session_state.extraction_method = backup['extraction_method']
        st.session_state.requested_colors = backup['requested_colors']
        st.session_state.min_distance = backup['min_distance']
        st.session_state.method_params = backup['method_params']
        st.session_state.image_info = backup['image_info']
        # Clear the backup after restoring
        st.session_state._palette_backup = None
        # Regenerate share link
        update_share_link(persist_query=False)


def clear_backup():
    """Clear the palette backup (called when new extraction occurs)"""
    st.session_state._palette_backup = None


def clear_palette_state():
    st.session_state.extracted_colors = None
    st.session_state.color_list = None
    st.session_state.extraction_method = None
    st.session_state.method_params = {}
    st.session_state.image_info = {}
    st.session_state.pop('extraction_time', None)

    # Clear algorithm recommendation state
    st.session_state.pop('recommended_algorithm', None)
    st.session_state.pop('recommendation_text', None)
    st.session_state.pop('current_image_array', None)
    st.session_state.pop('_recommendation_image_sig', None)

    st.session_state['_palette_from_query'] = False
    st.session_state['_last_loaded_palette_token'] = None
    st.session_state['_active_image_signature'] = None
    clear_share_link()


def sync_palette_with_image(new_signature):
    previous_signature = st.session_state.get('_active_image_signature')
    if new_signature == previous_signature:
        return

    should_clear = new_signature is not None or not st.session_state.get('_palette_from_query', False)
    if should_clear:
        clear_palette_state()

    st.session_state['_active_image_signature'] = new_signature


def on_format_change():
    """Regenerate color_list when output format changes"""
    if st.session_state.extracted_colors is not None and len(st.session_state.extracted_colors) > 0:
        colors = st.session_state.extracted_colors
        if st.session_state.output_format == 'HEX':
            st.session_state.color_list = [rgb_to_hex(color) for color in colors]
        else:
            st.session_state.color_list = [tuple(int(c) for c in color) for color in colors]
        # Update share link to reflect new format
        update_share_link(persist_query=st.session_state.get('_palette_from_query', False))


def render_copy_link_button(share_url):
    button_id = f"copy-btn-{uuid.uuid4().hex}"
    status_id = f"copy-status-{uuid.uuid4().hex}"
    escaped_url = json.dumps(share_url)

    components.html(
        f"""
        <div style="display:flex;align-items:center;gap:0.5rem;">
            <button id="{button_id}" style="background-color:#1f77b4;color:white;border:none;padding:0.5rem 1rem;border-radius:4px;cursor:pointer;">
                Copy the link to this palette
            </button>
            <span id="{status_id}" style="font-size:0.85rem;color:#4caf50;"></span>
        </div>
        <script>
        const rawLink = {escaped_url};

        function getHostContext() {{
            try {{
                if (window.parent && window.parent !== window && window.parent.location) {{
                    return {{
                        origin: window.parent.location.origin,
                        pathname: window.parent.location.pathname
                    }};
                }}
            }} catch (err) {{
                // Accessing parent can throw in cross-origin scenarios; ignore.
            }}
            return {{ origin: window.location.origin || '', pathname: window.location.pathname || '' }};
        }}

        function normaliseLink(link) {{
            if (/^https?:\\/\\//i.test(link)) {{
                return link;
            }}
            const context = getHostContext();
            const origin = context.origin || '';
            const path = context.pathname || '';
            if (!origin) {{
                return link;
            }}
            if (link.startsWith('?')) {{
                return origin + path + link;
            }}
            if (link.startsWith('/')) {{
                return origin + link;
            }}
            return origin + '/' + link;
        }}

        const copyBtn = document.getElementById('{button_id}');
        const status = document.getElementById('{status_id}');
        if (copyBtn) {{
            copyBtn.addEventListener('click', async () => {{
                try {{
                    const linkToCopy = normaliseLink(rawLink);
                    await navigator.clipboard.writeText(linkToCopy);
                    if (status) {{
                        status.textContent = 'Copied!';
                        status.style.color = '#4caf50';
                        setTimeout(() => status.textContent = '', 2000);
                    }}
                }} catch (err) {{
                    if (status) {{
                        status.textContent = 'Press Ctrl+C / ⌘+C to copy';
                        status.style.color = '#e63946';
                        setTimeout(() => {{
                            status.textContent = '';
                            status.style.color = '#4caf50';
                        }}, 4000);
                    }}
                }}
            }});
        }}
        </script>
        """,
        height=60,
    )

def rgb_to_hex(rgb):
    """Convert RGB values to HEX format"""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def resize_image_for_processing(image, max_size=MAX_IMAGE_SIZE):
    """Resise large images for faster processing"""
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image

def rgb_to_lab(rgb):
    """Convert RGB to LAB colour space for better perceptual distance"""
    # Reshape and normalize
    rgb = np.array(rgb).reshape(-1, 3) / 255.0
    # Use the optimized scikit-image function
    return rgb2lab(rgb)

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
    """Improved colour filtering using perceptual distance"""
    if len(colors) <= 1:
        return np.array(colors)

    colors_array = np.asarray(colors, dtype=float)

    if method == 'lab':
        # Convert once to LAB space for perceptual distance calculations
        lab_colors = rgb2lab(colors_array[np.newaxis, ...] / 255.0).reshape(len(colors_array), 3)
        distance_matrix = cdist(lab_colors, lab_colors)

        selected_indices = []
        for idx in range(len(colors_array)):
            if all(distance_matrix[idx, chosen] >= min_distance for chosen in selected_indices):
                selected_indices.append(idx)

        return colors_array[selected_indices].astype(int)

    # Fallback to pairwise comparisons for other distance metrics
    filtered_colors = [colors_array[0]]

    for color in colors_array[1:]:
        is_different = True
        for existing_color in filtered_colors:
            distance = calculate_color_distance(color, existing_color, method)
            if distance < min_distance:
                is_different = False
                break

        if is_different:
            filtered_colors.append(color)

    return np.array(filtered_colors).astype(int)

def get_color_name(rgb):
    """Generate a descriptive name for a colour based on its RGB values"""
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

def extract_colors_kmeans_improved(image_array, n_colors, sample_fraction=DEFAULT_SAMPLE_FRACTION):
    """Improved K-means with better initialization and sampling"""
    pixels = image_array.reshape(-1, 3)

    # Sample pixels for faster processing on large images
    if len(pixels) > LARGE_PIXEL_THRESHOLD:
        sample_size = max(int(len(pixels) * sample_fraction), n_colors * MIN_SAMPLE_SIZE_MULTIPLIER)
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

def extract_colors_histogram(image_array, n_colors, bins_per_channel=DEFAULT_HISTOGRAM_BINS):
    """Extract colors using 3D colour histogram"""

    # Create 3D histogram
    hist, edges = np.histogramdd(
        image_array.reshape(-1, 3),
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

def extract_colors_median_cut(image_array, n_colors):
    """Extract colors using median cut algorithm"""
    pixels = image_array.reshape(-1, 3)

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

def extract_colors_hierarchical(image_array, n_colors, linkage_method='ward'):
    """Extract colors using hierarchical clustering"""
    pixels = image_array.reshape(-1, 3)

    # Sample for performance
    max_samples = min(len(pixels), max(HIERARCHICAL_SAMPLE_MULTIPLIER * n_colors, HIERARCHICAL_MIN_SAMPLES))
    if len(pixels) > max_samples:
        indices = np.random.choice(len(pixels), max_samples, replace=False)
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
        st.warning(f"⚠️ Hierarchical clustering failed, falling back to k-means: {str(e)}")
        return extract_colors_kmeans_improved(image_array, n_colors, 0.2)

def extract_colors_dominant_sampling(image_array, n_colors, sample_method='grid'):
    """Extract colors by sampling dominant regions"""
    h, w = image_array.shape[:2]
    pixels = image_array.reshape(-1, 3)

    if sample_method == 'grid':
        # Sample from a grid
        grid_size = int(np.sqrt(n_colors * 20))
        y_coords = np.linspace(0, h-1, grid_size).astype(int)
        x_coords = np.linspace(0, w-1, grid_size).astype(int)

        sampled_colors = image_array[np.ix_(y_coords, x_coords)].reshape(-1, 3)

    elif sample_method == 'random':
        # Random sampling
        num_samples = n_colors * 50
        num_samples = min(num_samples, len(pixels))
        indices = np.random.choice(len(pixels), num_samples, replace=False)
        sampled_colors = pixels[indices]
    else:
        sampled_colors = pixels

    # Cluster the sampled colors
    if len(sampled_colors) > n_colors:
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(sampled_colors)
        colors = kmeans.cluster_centers_
    else:
        colors = sampled_colors

    return np.array(colors).astype(int)

def extract_colors_quantization_improved(image_array, n_colors, quant_level=DEFAULT_QUANT_LEVEL):
    """Improved quantization method"""
    pixels = image_array.reshape(-1, 3)

    # Reduce color space
    div = 256 // quant_level
    quantized_pixels = (pixels // div) * div

    # Get unique colors and their counts
    unique, counts = np.unique(quantized_pixels, axis=0, return_counts=True)

    # Sort by frequency and take top colors
    top_colors = unique[np.argsort(-counts)[:n_colors]]

    return top_colors.astype(int)

@st.cache_data
def extract_colors_main(image_array, n_colors, method='kmeans_improved', min_distance=DEFAULT_MIN_DISTANCE, **kwargs):
    """Main colour extraction function with caching"""
    # Reconstruct PIL image from array
    image_array = np.asarray(image_array, dtype=np.uint8)

    try:
        if method == 'kmeans_improved':
            colors = extract_colors_kmeans_improved(image_array, n_colors, kwargs.get('sample_fraction', 0.1))
        elif method == 'histogram':
            colors = extract_colors_histogram(image_array, n_colors, kwargs.get('bins_per_channel', 8))
        elif method == 'median_cut':
            colors = extract_colors_median_cut(image_array, n_colors)
        elif method == 'hierarchical':
            colors = extract_colors_hierarchical(image_array, n_colors, kwargs.get('linkage_method', 'ward'))
        elif method == 'dominant_sampling':
            colors = extract_colors_dominant_sampling(image_array, n_colors, kwargs.get('sample_method', 'grid'))
        elif method == 'quantization_improved':
            colors = extract_colors_quantization_improved(image_array, n_colors, kwargs.get('quant_level', 8))
        elif method == 'kmeans_basic':
            # Basic kmeans implementation
            pixels = image_array.reshape(-1, 3)
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
        st.error(f"❌ Error in {method}: {str(e)}")
        # Fallback to basic kmeans
        pixels = image_array.reshape(-1, 3)
        n_clusters = min(n_colors, len(np.unique(pixels.view(np.dtype((np.void, pixels.dtype.itemsize*pixels.shape[1]))).view(pixels.dtype).reshape(-1, pixels.shape[1]))))
        if n_clusters > 0:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(pixels)
            return kmeans.cluster_centers_.astype(int)
        else:
            return np.array([[0, 0, 0]])

def get_color_harmony_info(colors):
    """Provide information about colour harmony"""
    if len(colors) >= 2:
        warm_colors = sum(1 for color in colors if color[0] > color[2])  # More red than blue
        cool_colors = len(colors) - warm_colors

        # Calculate average brightness
        avg_brightness = np.mean([sum(color) for color in colors]) / 3
        brightness_desc = "bright" if avg_brightness > 127 else "dark"

        return f"🎨 **Palette Analysis:** {warm_colors} warm, {cool_colors} cool colors. Overall tone: {brightness_desc}"
    return ""

def get_algorithm_recommendation(image_array):
    """Analyze image and recommend the best extraction algorithm"""
    pixels = image_array.reshape(-1, 3)

    # Sample pixels for performance (max 10000)
    if len(pixels) > 10000:
        indices = np.random.choice(len(pixels), 10000, replace=False)
        sample = pixels[indices]
    else:
        sample = pixels

    # Calculate color variance (standard deviation across channels)
    color_std = np.std(sample, axis=0).mean()

    # Estimate unique colors (using quantized colors for speed)
    quantized = (sample // 32) * 32
    unique_colors = len(np.unique(quantized, axis=0))

    # Get image dimensions
    total_pixels = len(pixels)

    # Decision logic
    if unique_colors < 50:
        # Few distinct colors - likely a logo or graphic
        return "histogram", "This image has few distinct colours - Histogram works best for logos and graphics"
    elif color_std > 60:
        # High color variance - complex photograph
        return "kmeans_improved", "This image has high colour variety - K-means (Improved) works best for photographs"
    elif total_pixels < 50000:
        # Small image - any algorithm works
        return "median_cut", "This is a smaller image - Median Cut provides well-balanced results"
    else:
        # Default recommendation
        return "kmeans_improved", "K-means (Improved) is recommended as a good all-rounder"

def create_sample_image():
    """Create a sample gradient image for demo purposes using vectorized NumPy operations"""
    width, height = 400, 200

    # Create coordinate grids using array indexing
    y_coords = np.arange(height)[:, np.newaxis]  # Shape: (height, 1)
    x_coords = np.arange(width)[np.newaxis, :]   # Shape: (1, width)

    # Compute normalized coordinates (broadcasting creates (height, width) arrays)
    x_norm = x_coords / width
    y_norm = y_coords / height

    # Compute R, G, B channels as vectorized array operations
    r = 255 * x_norm * (1 - y_norm)
    g = 255 * y_norm * np.ones_like(x_norm)
    b = 255 * (1 - x_norm) * y_norm

    # Apply the purple corner modification using boolean masking
    purple_mask = (x_coords > width * 0.7) & (y_coords > height * 0.7)
    r = np.where(purple_mask, r * 0.8 + 128 * 0.2, r)
    b = np.where(purple_mask, b * 0.8 + 200 * 0.2, b)

    # Stack channels and convert to uint8
    rgb_array = np.stack([r, g, b], axis=-1).astype(np.uint8)

    # Create PIL Image from NumPy array
    return Image.fromarray(rgb_array, mode='RGB')

def _generate_xml_content(colors):
    """Generate XML content for color palette"""
    xml_lines = ['<palette>']
    for i, color in enumerate(colors):
        hex_color = rgb_to_hex(color)
        r, g, b = color[0], color[1], color[2]
        xml_lines.append(f'  <color name="Colour {i+1}" hex="{hex_color}" r="{r}" g="{g}" b="{b}" />')
    xml_lines.append('</palette>')
    return '\n'.join(xml_lines)


def _render_color_swatches(colors, color_list):
    """Render color swatch cards in a grid layout"""
    st.header("🎨 Extracted Colour Palette")

    cols = st.columns(min(len(colors), MAX_SWATCH_COLUMNS))

    for idx, color in enumerate(colors):
        col_idx = idx % MAX_SWATCH_COLUMNS
        if idx > 0 and col_idx == 0:
            cols = st.columns(min(len(colors) - idx, MAX_SWATCH_COLUMNS))

        with cols[col_idx]:
            hex_color = rgb_to_hex(color)
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

            st.markdown(f"""
            <div style="text-align: center; font-size: 12px; color: #666; margin-bottom: 15px;">
                Colour #{idx + 1}<br>
                RGB: {color[0]}, {color[1]}, {color[2]}<br>
                HEX: {hex_color}
            </div>
            """, unsafe_allow_html=True)


def _render_copy_tabs(colors, color_list):
    """Render tabs with copyable color values in different formats"""
    st.subheader("📋 Copy Colour Values")

    tab1, tab2, tab3, tab4 = st.tabs(["📝 List Format", "🔗 Comma Separated", "📊 JSON Format", "🗃️ XML Format"])

    with tab1:
        color_text = '\n'.join([str(color) for color in color_list])
        st.text_area(
            "Colour list (one per line):",
            color_text,
            height=120,
            help="Select all (Ctrl+A / Cmd+A) and copy (Ctrl+C / Cmd+C)"
        )

    with tab2:
        comma_separated = ', '.join([str(color) for color in color_list])
        st.text_area(
            "Comma-separated values:",
            comma_separated,
            height=80,
            help="Click inside, select all, and copy to clipboard"
        )

    with tab3:
        json_colors = json.dumps(color_list, indent=2)
        st.text_area(
            "JSON format:",
            json_colors,
            height=120,
            help="Select all and copy - ready for your code"
        )

    with tab4:
        xml_content = _generate_xml_content(colors)
        st.text_area(
            "XML format:",
            xml_content,
            height=120,
            help="Select all and copy - includes HEX and RGB values"
        )


def _render_share_section():
    """Render the share palette section if a share URL exists"""
    share_url = st.session_state.get('share_link')
    if share_url:
        st.subheader("🔗 Share Palette")
        st.markdown("Get the link to share the palette with others. The base image is not going to be shared.")
        render_copy_link_button(share_url)


def _render_export_buttons(colors, color_list, output_format, method, n_colors, timestamp):
    """Render download buttons for various export formats"""
    st.subheader("📥 Download Palette")
    st.markdown("Save your colour palette in various formats:")

    col_export1, col_export2, col_export3, col_export4 = st.columns(4)

    with col_export1:
        st.markdown("**JSON Data**")
        st.markdown("*Complete extraction info*")
        try:
            if output_format == 'HEX':
                serializable_colors = [str(color) for color in color_list]
            else:
                serializable_colors = [
                    [int(c) for c in color] if isinstance(color, (tuple, list, np.ndarray))
                    else convert_numpy_types(color)
                    for color in color_list
                ]

            palette_data = {
                'palette_name': f'Extracted_Palette_{timestamp}',
                'colors': serializable_colors,
                'color_count': len(colors),
                'format': output_format,
                'extraction_method': method,
                'parameters': {
                    'requested_colors': int(n_colors),
                    'actual_colors': len(colors),
                    'similarity_threshold': int(st.session_state.get('min_distance', DEFAULT_MIN_DISTANCE)),
                    **{k: convert_numpy_types(v) for k, v in st.session_state.get('method_params', {}).items()}
                },
                'image_info': st.session_state.get('image_info', {}),
                'extracted_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            json_data = json.dumps(palette_data, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"color_palette_{timestamp}.json",
                mime="application/json",
                help="Complete data with extraction settings and metadata",
                key="download_json"
            )
        except Exception as e:
            st.error(f"❌ Error creating JSON: {str(e)}")

    with col_export2:
        st.markdown("**CSS Variables**")
        st.markdown("*Ready for web development*")
        try:
            css_vars = []
            for i, color in enumerate(color_list):
                css_vars.append(f'  --color-{i+1}: {color};')

            css_content = f':root {{\n{chr(10).join(css_vars)}\n}}\n\n/* Usage examples */\n.primary {{ color: var(--color-1); }}\n.secondary {{ background-color: var(--color-2); }}'

            st.download_button(
                label="Download CSS",
                data=css_content,
                file_name=f"palette_{timestamp}.css",
                mime="text/css",
                help="CSS custom properties ready to use in your stylesheets",
                key="download_css"
            )
        except Exception as e:
            st.error(f"❌ Error creating CSS: {str(e)}")

    with col_export3:
        st.markdown("**TXT File**")
        st.markdown("*Simple text format*")
        try:
            txt_lines = [
                f"Colour Palette - Extracted {time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"Method: {method.replace('_', ' ').title()}",
                f"Colours: {len(colors)}",
                "-" * 40
            ]

            for i, color in enumerate(colors):
                hex_color = rgb_to_hex(color)
                txt_lines.append(f"Colour {i+1}")
                txt_lines.append(f"  HEX: {hex_color}")
                txt_lines.append(f"  RGB: {color[0]}, {color[1]}, {color[2]}")
                txt_lines.append("")

            txt_content = '\n'.join(txt_lines)

            st.download_button(
                label="Download TXT",
                data=txt_content,
                file_name=f"palette_{timestamp}.txt",
                mime="text/plain",
                help="Simple text format with colour names and values",
                key="download_txt"
            )
        except Exception as e:
            st.error(f"❌ Error creating TXT: {str(e)}")

    with col_export4:
        st.markdown("**XML Palette**")
        st.markdown("*Structured with colour names*")
        try:
            xml_content = _generate_xml_content(colors)

            st.download_button(
                label="Download XML",
                data=xml_content,
                file_name=f"palette_{timestamp}.xml",
                mime="application/xml",
                help="XML format with descriptive colour names and RGB/HEX values",
                key="download_xml"
            )
        except Exception as e:
            st.error(f"❌ Error creating XML: {str(e)}")


def _render_clear_button():
    """Render the clear results button with undo functionality"""
    st.markdown("---")

    col_clear, col_undo = st.columns([1, 1])

    with col_clear:
        if st.button("🗑️ Clear Results", help="Clear extracted colours to start fresh"):
            backup_palette_state()
            clear_palette_state()
            st.rerun()

    with col_undo:
        if st.session_state.get('_palette_backup'):
            if st.button("↩️ Undo Clear", help="Restore the previously cleared palette"):
                restore_palette_state()
                st.rerun()


def display_color_results():
    """Display the extracted colour results from session state"""
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

    # Extract variables from session state
    colors = st.session_state.extracted_colors
    color_list = st.session_state.color_list
    output_format = st.session_state.output_format
    method = st.session_state.extraction_method
    n_colors = st.session_state.requested_colors

    # Additional safety check
    if colors is None or len(colors) == 0:
        st.warning("⚠️ No colors were extracted. Please try again with different settings.")
        return

    # Generate timestamp once for consistent filenames
    timestamp = int(time.time())

    # Display success message
    st.success(f"🎉 Successfully extracted {len(colors)} colors using {method.replace('_', ' ').title()} algorithm!")

    if len(colors) < n_colors:
        st.warning(f"⚠️ Requested {n_colors} colors, but only found {len(colors)} distinct colors with current similarity settings.")
        st.info("💡 Try reducing the similarity threshold or using a different algorithm to get more colors.")

    # Colour harmony analysis
    harmony_info = get_color_harmony_info(colors)
    if harmony_info:
        st.markdown(harmony_info)

    # Render UI sections
    if len(colors) > 0:
        _render_color_swatches(colors, color_list)
        _render_copy_tabs(colors, color_list)
        _render_share_section()
        _render_export_buttons(colors, color_list, output_format, method, n_colors, timestamp)
        _render_clear_button()

# Main UI
st.markdown('<h1 class="main-header">🎨 Colour Palette Extractor</h1>', unsafe_allow_html=True)

st.markdown("""
Extract dominant colors from any image using multiple algorithms including K-means clustering,
color histograms, median cut, and hierarchical clustering. Upload an image or try the sample to get started, then use
**Share Palette** to copy a link to the extracted colours (the original image stays private).
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
if '_palette_from_query' not in st.session_state:
    st.session_state._palette_from_query = False
if '_last_loaded_palette_token' not in st.session_state:
    st.session_state._last_loaded_palette_token = None
if '_active_image_signature' not in st.session_state:
    st.session_state._active_image_signature = None
if '_base_url_cached' not in st.session_state:
    st.session_state._base_url_cached = ''
if '_palette_backup' not in st.session_state:
    st.session_state._palette_backup = None

load_palette_from_query()

# Enhanced expandable help section with all tips combined
with st.expander("ℹ️ App guide: Algorithms, Tips & Colour Theory", expanded=False):
    st.markdown("### 🧠 Available algorithms")

    # Create 3 columns for algorithms
    algo_col1, algo_col2, algo_col3 = st.columns(3)

    with algo_col1:
        st.markdown("""
        **🧠 K-means**
        - Smart pixel sampling
        - Frequency-based sorting
        - Best for photographs

        **📊 Colour Histogram**
        - 3D colour frequency analysis
        - Great for graphics
        - Very fast processing
        """)

    with algo_col2:
        st.markdown("""
        **✂️ Median Cut**
        - Divides colour space recursively
        - Well-balanced palettes
        - Classic algorithm

        **🌳 Hierarchical Clustering**
        - Tree-like colour groupings
        - Excellent for colour harmony
        - More accurate results
        """)

    with algo_col3:
        st.markdown("""
        **🎯 Dominant Sampling**
        - Samples key image regions
        - Very fast processing
        - Good for previews

        **⚡ Quantization**
        - Reduces colour precision
        - Finds dominant colors
        - Memory efficient
        """)

    st.markdown("---")
    st.markdown("### 💡 Usage tips")

    # Create 2 columns for tips
    tip_col1, tip_col2 = st.columns(2)

    with tip_col1:
        st.markdown("""
        **🎯 Algorithm by image type:**
        - **Photographs:** K-means (Improved) or Hierarchical
        - **Logos/Graphics:** Histogram or Median Cut
        - **Artwork/Paintings:** Median Cut or K-means
        - **Quick Analysis:** Dominant Sampling

        **🔥 Performance tips:**
        - **Need exact colors:** Set similarity to 0
        - **Want cohesive palette:** Increase similarity
        - **Large images:** Use sampling algorithms
        - **Quick preview:** Try Dominant Sampling
        """)

    with tip_col2:
        st.markdown("""
        **💡 Best Results Tips:**
        - **High contrast images** → more distinct palettes
        - **Well-lit photos** → accurate colour representation
        - **Lower similarity** → preserve colour variations
        - **Higher similarity** → create cohesive palettes
        """)

    st.markdown("---")
    st.markdown("### 🎨 Colour theory basics")

    # Create 2 columns for color theory
    theory_col1, theory_col2 = st.columns(2)

    with theory_col1:
        st.markdown("""
        - **Complementary:** Opposite colors - high contrast
        - **Analogous:** Adjacent colors - harmony
        - **Triadic:** Three evenly spaced - balanced
        - **Monochromatic:** Single hue shades - elegant
        - **Warm Colours:** Reds, oranges, yellows - energetic
        - **Cool Colours:** Blues, greens, purples - calming
        """)

    # with theory_col2:
    #     st.markdown("""
    #     **🎯 Use Cases & Applications:**
    #     - **Web Design:** CSS color variables from mockups
    #     - **Brand Identity:** Extract official brand colors
    #     - **Interior Design:** Match paint to inspiration photos
    #     - **Art Analysis:** Study color relationships
    #     - **Fashion:** Extract clothing color palettes
    #     - **Photography:** Analyze dominant tones for editing
    #     """)

    st.markdown("---")
    st.markdown("### 🔧 Advanced settings")

    # Create 2 columns for advanced settings
    settings_col1, settings_col2 = st.columns(2)

    with settings_col1:
        st.markdown("""
        **Similarity filtering:**
        - **0:** No filtering - keep all colors
        - **1-20:** Low filtering - remove very similar
        - **20-50:** Medium filtering - moderately similar
        - **50+:** High filtering - only distinct colors

        **Sampling (K-means Improved):**
        - **0.05-0.1:** Fast processing, large images
        - **0.1-0.3:** Balanced speed and accuracy
        - **0.3-0.5:** Slower but more accurate
        """)

    with settings_col2:
        st.markdown("""
        **Histogram resolution:**
        - **4-8 bins:** Broad groupings, fewer colors
        - **8-12 bins:** Balanced detail and simplicity
        - **12-16 bins:** High detail, nuanced colors

        **Export formats:**
        - **JSON:** Complete extraction data + metadata
        - **CSS:** Ready-to-use CSS custom properties
        - **TXT:** Simple text format for any application
        - **XML:** Structured format with colour names
        """)

    st.markdown("---")
    st.markdown("### 🔗 Sharing palettes")
    st.markdown("""
    - Use the **Share Palette** button beneath the results to copy a link to your colours
    - Shared links only include palette data – the original image never leaves your device
    - Perfect for sending palettes to teammates or dropping them into design docs
    """)

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("⚙️ Extraction Settings")

    # Number of colors
    n_colors = st.slider(
        "Number of colors to extract",
        2, 15, 5,
        help="More colors create a detailed palette, fewer colors create a simplified colour scheme"
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
            'histogram': '📊 Colour Histogram',
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
        'kmeans_improved': "🧠 Smart K-means - Uses sampling and frequency sorting for optimal results",
        'histogram': "📊 Colour Histogram - Analyzes colour frequency in 3D space",
        'median_cut': "✂️ Median Cut - Classic algorithm that divides colour space recursively",
        'hierarchical': "🌳 Hierarchical - Creates tree-based colour groupings",
        'dominant_sampling': "🎯 Smart Sampling - Samples key image regions",
        'quantization_improved': "⚡ Quantization - Reduces colour precision to find dominant colors",
        'kmeans_basic': "⚙️ Basic K-means - Standard clustering approach"
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
            help="Higher values = more colour precision, lower = broader groupings"
        )
    elif method == 'hierarchical':
        method_params['linkage_method'] = st.selectbox(
            "Clustering method",
            ['ward', 'complete', 'average', 'single'],
            help="Ward generally produces the best results for colour clustering"
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
            help="Higher values preserve more colour nuances"
        )

    # Similarity filtering
    st.markdown("**Colour Similarity Filter**")
    min_distance = st.slider(
        "Minimum colour distance", 0, 100, 30,
        help="Higher values = fewer, more distinct colours. Lower values = more colour variations."
    )

    if min_distance == 0:
        st.info("🔄 No filtering - keeping all extracted colours, including very similar ones")
    elif min_distance < 20:
        st.info("🎨 Subtle filtering - removing nearly identical colours")
    elif min_distance < 50:
        st.info("⚖️ Balanced filtering - keeping colours that look noticeably different")
    else:
        st.info("🎯 Strong filtering - only keeping colours that are clearly distinct")

    # Output format
    st.markdown("**Output Format**")
    output_format = st.selectbox(
        "Colour format for export",
        ['HEX', 'RGB'],
        key='output_format',
        on_change=on_format_change,
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
        sync_palette_with_image('sample')
        image = create_sample_image()
        st.image(image, caption='🌈 Sample Test Image - Perfect for algorithm testing', use_container_width=True)
        st.markdown("""
        <div class="help-box">
        This test image contains gradients and distinct colour regions.
        Perfect for comparing different extraction algorithms and settings.
        </div>
        """, unsafe_allow_html=True)

        # Generate algorithm recommendation for sample image
        current_sig = 'sample'
        if st.session_state.get('_recommendation_image_sig') != current_sig:
            st.session_state.current_image_array = np.array(image)
            recommended_algo, recommendation_text = get_algorithm_recommendation(st.session_state.current_image_array)
            st.session_state.recommended_algorithm = recommended_algo
            st.session_state.recommendation_text = recommendation_text
            st.session_state['_recommendation_image_sig'] = current_sig

        if st.session_state.get('recommendation_text'):
            st.info(f"💡 **Suggested algorithm:** {st.session_state.recommendation_text}")

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
            file_size_bytes = getattr(uploaded_file, 'size', len(uploaded_file.getvalue()))
            file_size = file_size_bytes / 1024  # KB
            st.markdown(f"""
            <div class="help-box">
            <strong>📊 Image Analysis:</strong><br>
            • Resolution: {image.size[0]} × {image.size[1]} pixels<br>
            • Total pixels: {total_pixels:,}<br>
            • File size: {file_size:.1f} KB<br>
            • Colour mode: {image.mode}<br>
            {f'<em>⚡ Large image - processing will use smart sampling for speed</em>' if total_pixels > LARGE_IMAGE_THRESHOLD else '<em>✅ Good size for quick processing</em>'}
            </div>
            """, unsafe_allow_html=True)

            upload_signature = f"upload:{uploaded_file.name}:{file_size_bytes}"
            sync_palette_with_image(upload_signature)

            # Generate algorithm recommendation
            if st.session_state.get('_recommendation_image_sig') != upload_signature:
                st.session_state.current_image_array = np.array(image)
                recommended_algo, recommendation_text = get_algorithm_recommendation(st.session_state.current_image_array)
                st.session_state.recommended_algorithm = recommended_algo
                st.session_state.recommendation_text = recommendation_text
                st.session_state['_recommendation_image_sig'] = upload_signature

            if st.session_state.get('recommendation_text'):
                st.info(f"💡 **Suggested algorithm:** {st.session_state.recommendation_text}")

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
        # st.info("👆 Upload an image above or try the sample image to get started!")
        st.markdown("""
        **💡 For best results, use images with:**
        - Good lighting and contrast
        - Multiple distinct colors
        - High resolution (but not too large for web upload)
        - Minimal noise or compression artifacts
        """)

        sync_palette_with_image(None)

# Process image if available
if process_image:
    # Resize for processing to improve performance
    processing_image = resize_image_for_processing(image.copy())

    if st.button("🎨 Extract Colour Palette", type="primary", help="Analyze your image and extract the dominant colors"):
        try:
            with st.spinner('Extracting colours from image...'):
                img_array = np.array(processing_image)
                colors = extract_colors_main(
                    img_array,
                    n_colors,
                    method,
                    min_distance,
                    **method_params
                )

                if output_format == 'HEX':
                    color_list = [rgb_to_hex(color) for color in colors]
                else:
                    color_list = [tuple(int(c) for c in color) for color in colors]

            # Store results in session state (clear any stale backup first)
            clear_backup()
            st.session_state.extracted_colors = colors
            st.session_state.color_list = color_list
            # Note: output_format is managed by the selectbox widget via key='output_format'
            st.session_state.extraction_method = method
            st.session_state.requested_colors = n_colors
            st.session_state.min_distance = min_distance
            st.session_state.method_params = method_params
            st.session_state.image_info = {
                'original_size': list(image.size),
                'processed_size': list(processing_image.size) if processing_image.size != image.size else None,
                'total_pixels': int(np.prod(image.size))
            }

            update_share_link(persist_query=False)

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
🎨 Colour palette extractor | Built with Streamlit & scikit-learn
            
Designed by: Anya Prosvetova, [anyalitica.dev](https://www.anyalitica.dev/)
""", unsafe_allow_html=True)
