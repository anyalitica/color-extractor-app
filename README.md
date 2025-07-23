# Colour Palette Extractor

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://colour-extractor-app.streamlit.app/)

This Streamlit app will help you to extract colour palettes from any image. Designed for developers, designers, and creatives, it uses multiple clustering and analysis algorithms to provide accurate and useful colour schemes that can be exported in various formats.

![App Screenshot](https://github.com/anyalitica/colour-extractor-app/blob/main/assets/app_screenshot.png)

## Key features

- Extraction & analysis
    - 7 extraction algorithms
    - Warm/Cool tone detection
- Customization & filtering
    - Adjustable palette size
    - Similarity filtering
    - Algorithm-specific tuning
    - HEX & RGB formats
- Export & integration
    - CSS variables
    - JSON
    - TXT & XML

## Live Demo & usage

**[Try the live app here!](https://colour-extractor-app.streamlit.app/)**

1.  **Upload an image**: Upload your own image or use the provided sample
2.  **Configure the algorithm**: Adjust the number of colours, choose an extraction algorithm, and set the similarity filter
3.  **Extract**: Click the "Extract colour palette" button.
4.  **Export**: Copy the colour values or download the complete palette as a JSON, CSS, TXT, or XML file.

## Algorithm guide

Choose the best algorithm for your image type for optimal results.

| Algorithm | Best For | Speed | Accuracy | Use Case |
|:---|:---|:---:|:---:|:---|
| **K-means (Improved)** | Photographs | Medium | High | Best all-rounder for complex images |
| **Colour Histogram** | Graphics/Logos | Fast | High | Images with large, distinct colour regions |
| **Median Cut** | All image types | Medium | High | Classic algorithm, gives well-balanced results |
| **Hierarchical** | Art/Complex Scenes | Slow | Very High | Excellent for analyzing colour harmony |
| **Dominant Sampling** | Quick Previews | Very Fast | Medium | Ideal for a fast first look |
| **Quantization** | Simple Graphics | Fast | Medium | Memory-efficient and great for simple images |

## Future improvements

Potential features on the roadmap include:

-   **Accessibility checker**: A tool to calculate the contrast ratio between palette colours to ensure they meet standards for web accessibility
<!-- -   **Manual colour adjustment**: An option to manually fine-tune an extracted colour using a colour picker before exporting the palette -->
-   **Transparent background support**: A feature to correctly handle transparent pixels in PNG images, either by ignoring them or treating them as a background colour
-   **Advanced colour naming**: Integration with a colour-name library to provide more accurate and standardized names than the current rule-based system

## Local development

To run the app on your local machine:

```bash
# Clone the repository
git clone https://github.com/anyalitica/colour-extractor-app.git
cd colour-extractor-app/streamlit

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```
**Dependencies:** `streamlit`, `numpy`, `Pillow`, `scikit-learn`, `scipy`.

## Contributing

Contributions, issues, and feature requests are welcome! Please feel free to fork the repository, make changes, and open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Created by:** [@anyalitica](https://github.com/anyalitica)
**Website:** [anyalitica.dev](https://anyalitica.dev)  
