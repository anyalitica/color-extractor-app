# Color Palette Extractor

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://color-extractor-app.streamlit.app/)

## Overview

This advanced Streamlit app helps designers, developers, and creatives easily extract dominant color palettes from any image using multiple sophisticated algorithms. Choose from 7 different extraction methods including K-means clustering, color histograms, median cut, and hierarchical clustering to find the perfect color palette for your project.

![App Screenshot](https://github.com/anyalitica/color-extractor-app/blob/main/assets/app_screenshot.png)

## ✨ Key Features

### 🧠 Multiple Extraction Algorithms
- **K-means (Improved)** - Smart pixel sampling with frequency-based sorting
- **Color Histogram** - 3D color frequency analysis for graphics
- **Median Cut** - Classic algorithm that divides color space recursively
- **Hierarchical Clustering** - Tree-like color groupings for harmony analysis
- **Dominant Sampling** - Fast sampling of key image regions
- **Quantization (Improved)** - Memory-efficient color precision reduction
- **K-means (Basic)** - Standard clustering approach

### 🎨 Advanced Color Processing
- **Perceptual Color Filtering** - LAB color space similarity filtering
- **Smart Color Naming** - Automatic descriptive color names
- **Color Harmony Analysis** - Warm/cool balance and brightness insights
- **Automatic Contrast Optimization** - Perfect text colors on color swatches

### 📁 Multiple Export Formats
- **JSON** - Complete extraction data with metadata
- **CSS** - Ready-to-use CSS custom properties
- **TXT** - Simple text format with color names and values
- **XML** - Structured palette format with attributes

### 🔧 Performance Optimized
- **Automatic Image Resizing** - Smart resizing for faster processing
- **Intelligent Sampling** - Configurable pixel sampling for large images
- **Streamlit Caching** - Fast repeated operations
- **Progress Indicators** - Real-time processing feedback

## 🎯 Why Extract Color Palettes?

Color palette extraction has numerous practical applications:

- **🌐 Web Design** - Create cohesive color schemes for websites and applications
- **🏢 Brand Development** - Extract official colors from logos and marketing materials
- **🎨 Art & Design** - Analyze color relationships in artwork and photography
- **🏠 Interior Design** - Match paint colors to inspiration photos
- **📊 Data Visualization** - Generate color schemes for charts and graphs
- **👗 Fashion** - Coordinate colors from fabric swatches and inspiration images

## 🚀 Getting Started

1. **Visit the App** - Go to the [live app](https://color-extractor-app.streamlit.app/)
2. **Upload or Sample** - Upload an image (PNG, JPG, JPEG, WEBP, BMP, TIFF) or try the sample gradient
3. **Configure Settings**:
   - **Number of colors** (2-15) - Size of your palette
   - **Extraction algorithm** - Choose from 7 different methods
   - **Similarity filtering** (0-100) - Remove similar colors
   - **Output format** - HEX or RGB values
   - **Algorithm-specific parameters** - Fine-tune for optimal results
4. **Extract Colors** - Click the extraction button to generate your palette
5. **Copy & Export** - Use the copy tabs or download in multiple formats

## 🧠 Algorithm Guide

### When to Use Each Algorithm

| Algorithm | Best For | Speed | Accuracy | Use Case |
|-----------|----------|--------|----------|----------|
| **K-means (Improved)** | Photographs | Medium | High | Complex images, best overall |
| **Color Histogram** | Graphics/Logos | Fast | High | Images with distinct regions |
| **Median Cut** | All image types | Medium | High | Classic, well-balanced results |
| **Hierarchical** | Color harmony | Slow | Very High | Artistic analysis |
| **Dominant Sampling** | Quick previews | Very Fast | Medium | Real-time applications |
| **Quantization** | Simple images | Fast | Medium | Memory-constrained environments |

### Algorithm-Specific Parameters

#### K-means (Improved)
- **Sample Fraction** (0.05-0.5) - Lower = faster, higher = more accurate

#### Color Histogram
- **Histogram Resolution** (4-16 bins) - Higher = more color precision

#### Hierarchical Clustering
- **Clustering Method** - Ward (recommended), Complete, Average, Single

#### Dominant Sampling
- **Sampling Pattern** - Grid (systematic) or Random (varied coverage)

#### Quantization
- **Quantization Level** (4-32) - Higher values preserve more color nuances

## 📋 Export Formats

### 🔗 Copy Color Values
Quick access to colors in multiple formats:
- **List Format** - One color per line for spreadsheets
- **Comma Separated** - All colors in one line
- **JSON Format** - Structured data for programming
- **XML Format** - Structured palette with attributes

### 📥 Download Options

#### JSON Export
Complete extraction data with metadata:
```json
{
  "palette_name": "Extracted_Palette_1234567890",
  "colors": ["#FF5733", "#33FF57", "#3357FF"],
  "color_count": 3,
  "format": "HEX",
  "extraction_method": "kmeans_improved",
  "parameters": {
    "requested_colors": 3,
    "similarity_threshold": 30,
    "sample_fraction": 0.1
  },
  "extracted_at": "2024-01-15 10:30:45"
}
```

#### CSS Variables
Ready-to-use CSS custom properties:
```css
:root {
  --color-1: #FF5733;
  --color-2: #33FF57;
  --color-3: #3357FF;
  --color-1: #FF5733;
}

/* Usage examples */
.primary { color: var(--color-1); }
.secondary { background-color: var(--color-2); }
```

#### TXT Format
Simple text format with color information:
```
Color Palette - Extracted 2024-01-15 10:30:45
Method: K-means (Improved)
Colors: 3
----------------------------------------

Color 1
  HEX: #FF5733
  RGB: 255, 87, 51

Color 2
  HEX: #33FF57
  RGB: 51, 255, 87
```

#### XML Palette
Structured format with color attributes:
```xml
<palette>
  <color name="Color 1" hex="#FF5733" r="255" g="87" b="51" />
  <color name="Color 2" hex="#33FF57" r="51" g="255" b="87" />
  <color name="Color 3" hex="#3357FF" r="51" g="87" b="255" />
</palette>
```

## 🛠️ Local Development

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/anyalitica/color-extractor-app.git
cd color-extractor-app/streamlit

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Requirements
```
streamlit
numpy
Pillow
scikit-learn
scipy
```

## 🔧 Technical Details

### Image Processing
- **Automatic Format Conversion** - All images converted to RGB
- **Smart Resizing** - Large images resized to 800px max for performance
- **Pixel Analysis** - NumPy array processing for efficiency
- **Memory Management** - Optimized for large image handling

### Color Analysis
- **LAB Color Space** - Perceptual color distance calculations
- **HSV Analysis** - Hue, saturation, and brightness evaluation
- **Statistical Methods** - Frequency analysis and clustering
- **Color Theory** - Automatic warm/cool color detection

### Performance Features
- **Caching** - Streamlit cache for repeated operations
- **Progress Tracking** - Real-time processing updates
- **Error Handling** - Comprehensive error management
- **Responsive UI** - Adaptive layout for different screen sizes

## 📚 Complete User Guide

### 💡 Tips for Best Results

#### Image Selection
- **High contrast images** produce more distinct palettes
- **Well-lit photos** give accurate color representation
- **Multiple distinct colors** work better than monochrome images
- **Clean, uncompressed images** provide better accuracy

#### Algorithm Selection
- **Photographs** → K-means (Improved) or Hierarchical
- **Logos/Graphics** → Histogram or Median Cut
- **Artwork/Paintings** → Median Cut or K-means (Improved)
- **Quick Analysis** → Dominant Sampling or Histogram

#### Parameter Tuning
- **Lower similarity thresholds** preserve more color variations
- **Higher similarity thresholds** create more cohesive palettes
- **More colors** = detailed palette, fewer colors = simplified scheme
- **Higher sampling** = more accuracy, lower sampling = faster processing

### 🎨 Color Theory Integration

#### Color Relationships
- **Complementary** - Colors opposite on color wheel (high contrast)
- **Analogous** - Adjacent colors (harmonious)
- **Triadic** - Three evenly spaced colors (balanced)
- **Monochromatic** - Shades of single hue (elegant)

#### Color Psychology
- **Warm Colors** (reds, oranges, yellows) - Energetic and bold
- **Cool Colors** (blues, greens, purples) - Calming and professional
- **Neutral Colors** (grays, browns, beiges) - Balanced and sophisticated

### 🚀 Advanced Settings Guide

#### Similarity Filtering
- **0** - No filtering, keep all extracted colors
- **1-20** - Low filtering, remove very similar colors
- **20-50** - Medium filtering, remove moderately similar colors
- **50+** - High filtering, keep only very distinct colors

#### Processing Optimization
- **Large images** - Automatically resized for speed
- **Sampling** - Reduces processing time while maintaining accuracy
- **Algorithm choice** - Different methods excel with different image types
- **Memory usage** - Optimized for both small and large images

## 🎯 Use Cases & Applications

### For Designers
- **Brand Identity** - Extract official colors from logos and materials
- **Mood Boards** - Create cohesive color schemes from inspiration images
- **Color Analysis** - Study competitor color schemes and trends
- **Palette Creation** - Generate harmonious color combinations

### For Developers
- **CSS Generation** - Automatic CSS custom properties from mockups
- **Theme Creation** - Dynamic color schemes for applications
- **Data Visualization** - Color palettes for charts and graphs
- **Brand Compliance** - Ensure consistent brand color usage

### For Artists & Creatives
- **Reference Analysis** - Study color relationships in master artworks
- **Digital Art** - Extract color inspiration from photographs
- **Fashion Design** - Coordinate colors from fabric and inspiration
- **Photography** - Analyze dominant tones for post-processing

## 🔍 Troubleshooting

### Common Issues & Solutions

**Colors appear muddy or too similar?**
- Increase the similarity threshold to filter similar colors
- Try K-means (Improved) for better color separation
- Use images with higher contrast

**Not extracting enough distinct colors?**
- Decrease the similarity threshold
- Use images with more color variety
- Try Hierarchical clustering for complex color relationships

**Processing taking too long?**
- App automatically optimizes large images
- Try Dominant Sampling for faster results
- Use smaller image files when possible

**Colors don't match expectations?**
- Ensure image is well-lit and high quality
- Try different algorithms - each excels with different image types
- Adjust algorithm-specific parameters for fine-tuning

### Performance Tips
- **Optimal image size** - 400-1200px width for best balance of quality/speed
- **File formats** - PNG and JPG work best
- **Internet connection** - Faster uploads improve experience
- **Browser compatibility** - Modern browsers recommended

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test thoroughly
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Contribution Guidelines
- **Code Quality** - Follow Python PEP 8 style guidelines
- **Testing** - Test with various image types and parameters
- **Documentation** - Update README and code comments
- **Performance** - Ensure changes don't degrade performance

### Areas for Contribution
- **New Algorithms** - Additional color extraction methods
- **Export Formats** - Support for more output formats
- **UI Improvements** - Enhanced user experience features
- **Performance** - Optimization and speed improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

### Technologies
- **[Streamlit](https://streamlit.io/)** - Web interface framework
- **[scikit-learn](https://scikit-learn.org/)** - Machine learning algorithms
- **[Pillow](https://pillow.readthedocs.io/)** - Image processing
- **[NumPy](https://numpy.org/)** - Numerical computing
- **[SciPy](https://scipy.org/)** - Scientific computing

### Inspiration
- Color theory principles from design and art communities
- Feedback from designers, developers, and creative professionals
- Open source community contributions and suggestions

## 📞 Contact & Support

**Created by:** Anya Prosvetova  
**Website:** [anyalitica.dev](https://anyalitica.dev)  
**GitHub:** [@anyalitica](https://github.com/anyalitica)

### Support
- **Issues** - [GitHub Issues](https://github.com/anyalitica/color-extractor-app/issues)
- **Discussions** - [GitHub Discussions](https://github.com/anyalitica/color-extractor-app/discussions)
- **Email** - For direct support and collaboration

---

⭐ **Star this repository** if you find it useful!  
🔗 **Share with others** who might benefit from color palette extraction!