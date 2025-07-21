# Color Palette Extractor

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://color-extractor-app.streamlit.app/)

## Overview

This Streamlit app helps designers, developers, and creatives easily extract dominant color palettes from any image. Using advanced machine learning algorithms like K-means clustering and color quantization, the app analyzes uploaded images to identify the most prominent colors and presents them in a clean, usable format.

![App Screenshot](https://github.com/yourusername/color-extractor-app/blob/main/assets/app_screenshot.png)

## Features

- **Multiple Extraction Methods**: Choose between K-means clustering for better color grouping or quantization for faster processing
- **Smart Hue Filtering**: Remove similar colors based on configurable hue difference thresholds
- **Color Harmony Analysis**: Get insights about warm/cool color balance and overall palette brightness
- **Multiple Output Formats**: Export colors in HEX or RGB formats
- **Professional Color Swatches**: View colors with automatic contrast-optimized text labels
- **Export Options**: Download palettes as JSON for data use or CSS for web development
- **Performance Optimized**: Automatic image resizing and caching for fast processing
- **Sample Images**: Try the app instantly with built-in sample images

## Why Extract Color Palettes?

Color palette extraction has numerous practical applications:
- **Web Design**: Create cohesive color schemes for websites and applications
- **Brand Development**: Extract brand colors from logos and marketing materials
- **Art & Design**: Analyze color relationships in artwork and photography
- **Interior Design**: Match colors from inspiration images
- **Data Visualization**: Generate color schemes for charts and graphs
- **Fashion**: Coordinate colors from fabric swatches or inspiration photos

## Getting Started

1. Visit the [live app](https://color-extractor-app.streamlit.app/)
2. Upload an image (PNG, JPG, JPEG, WEBP) or try the sample gradient
3. Adjust extraction parameters:
   - **Number of colors**: 2-10 colors in your palette
   - **Extraction method**: K-means or quantization
   - **Hue threshold**: Filter out similar colors
   - **Output format**: HEX or RGB values
4. Click "Extract Colors" to generate your palette
5. Copy color values or download as JSON/CSS files

## Extraction Methods

### K-means Clustering
- **Best for**: Complex images with varied colors
- **Advantages**: Better color grouping, more representative colors
- **Use when**: Quality is more important than speed

### Color Quantization
- **Best for**: Simple images or quick extraction
- **Advantages**: Faster processing, good for real-time applications
- **Use when**: Speed is important or working with simple color schemes

## Parameters Guide

- **Number of Colors (2-10)**: More colors = more detailed palette, fewer colors = simplified scheme
- **Hue Threshold (0-90°)**: Higher values remove more similar colors, 0 keeps all extracted colors
- **Quantization Level (2-32)**: Only for quantization mode - higher values = more color precision

## Export Formats

### JSON Export
Perfect for developers and data applications:
```json
{
  "colors": ["#ff5733", "#33ff57", "#3357ff"],
  "format": "HEX",
  "extraction_method": "kmeans",
  "parameters": {
    "n_colors": 3,
    "hue_threshold": 30
  }
}
```

### CSS Export
Ready-to-use CSS custom properties:
```css
:root {
  --color-1: #ff5733;
  --color-2: #33ff57;
  --color-3: #3357ff;
}
```

## Local Development

To run the app locally:

```bash
# Clone the repository
git clone https://github.com/anyalitica/color-extractor-app.git
cd color-extractor-app/streamlit

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Requirements

- Python 3.8+
- Streamlit
- NumPy
- Pillow (PIL)
- scikit-learn

See `requirements.txt` for specific versions.

## Technical Details

### Image Processing
- Automatic RGB conversion for all image formats
- Smart resizing for large images (max 800px) to optimize performance
- Pixel-level analysis using NumPy arrays

### Color Analysis
- HSV color space conversion for accurate hue filtering
- Brightness analysis for color harmony insights
- Automatic text color selection for optimal contrast on color swatches

### Performance Features
- Streamlit caching for faster repeated operations
- Progress indicators for better user experience
- Error handling for unsupported files and processing issues

## Use Cases

### For Designers
- Extract color schemes from mood boards and inspiration images
- Create brand guideline color palettes
- Analyze competitor color schemes

### For Developers
- Generate CSS color variables from design mockups
- Create programmatic color schemes for applications
- Extract colors for data visualization themes

### For Artists
- Analyze color relationships in reference images
- Study color harmony in famous artworks
- Plan color schemes for new projects

## Tips for Best Results

- **Use high-contrast images** for more distinct color extraction
- **Upload high-quality images** for better color accuracy
- **Adjust hue threshold** to remove muddy or similar colors
- **Try different extraction methods** - K-means for complex images, quantization for simple ones
- **Consider image content** - photos work better than graphics with gradients

## Troubleshooting

**Colors look muddy or similar?**
- Increase the hue threshold to filter similar colors
- Try the K-means method for better clustering

**Not enough colors extracted?**
- Decrease the hue threshold
- Use an image with more color variety
- Increase the quantization level (if using quantization mode)

**App running slowly?**
- The app automatically resizes large images, but very high-resolution images may still take time
- Try using a smaller image file

## Contributions

Contributions are welcome! Please submit any issues, suggestions, or improvements on the [GitHub issues page](https://github.com/anyalitica/color-extractor-app/issues).

### Contributing Guidelines
- Fork the repository and create a feature branch
- Test your changes thoroughly
- Update documentation as needed
- Submit a pull request with a clear description

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- Uses [scikit-learn](https://scikit-learn.org/) for K-means clustering
- Color processing powered by [Pillow](https://pillow.readthedocs.io/)

## Credits

Designed by: Anya Prosvetova, [anyalitica.dev](https://anyalitica.dev)