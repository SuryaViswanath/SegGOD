# SegGOD: Segmentation Guided Object Detection

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Kaggle-Ready-20BEFF?logo=kaggle)](https://www.kaggle.com/)

**A training-free object detection framework that combines SAM segmentation with Vision-Language Models (VLMs) to detect objects through natural language queries.**

## Quick Start (For Examiners)

** Total setup time: < 2 minutes**

### Option 1: Run on Kaggle (Recommended)
1. **Open the Kaggle Notebook**: [SegGOD Testing Notebook](https://www.kaggle.com/your-username/seggod-testing) 
2. **Click "Copy and Edit"** to create your own version
3. **Click "Run All"** - all dependencies will install automatically
4. **Wait for the interactive prompt** and start testing!

### Option 2: Use Local Files
1. **Clone this repository**:
   ```bash
   git clone https://github.com/your-username/seggod.git
   cd seggod
   ```
2. **Upload `seggod-testing.ipynb` to Kaggle** or Google Colab
3. **Upload the test dataset** from `/test-images/` folder
4. **Run all cells**

## 📖 How It Works

SegGOD uses a **6-stage pipeline** to detect objects without training:

```
Image + "text query" → SAM → Object Proposals → Content-Aware Interpolation → SigLIP → Detections
```

### Architecture Overview
1. **Stage 1**: SAM generates 256+ segment masks
2. **Stage 2**: Morphological operations enhance segments  
3. **Stage 3**: Content-aware interpolation preserves spatial relationships
4. **Stage 4**: SigLIP scores segments against text query
5. **Stage 5**: Threshold filtering removes low-confidence detections
6. **Stage 6**: Visualization and result saving

## Interactive Demo

Once you run the notebook, the system will:

1. **Show you each test image**
2. **Prompt for text input**: `"Enter the text query:"`
3. **Examples of queries**:
   - `"person"`
   - `"car"`
   - `"table"`
   - `"red car"`
   - `"person wearing blue shirt"`

4. **Display results instantly**:
   - Detection result with bounding boxes
   - Enhanced grid showing top proposals with confidence scores
   - Analysis summary with statistics

5. **Save outputs automatically**:
   - `{N}_seggod_detection_result.jpg` - Image with bounding boxes
   - `{N}_seggod_enhanced_grid.jpg` - Visualization grid

## Repository Structure

```
seggod/
├── README.md                    # This file
├── seggod-testing.ipynb         # Main implementation notebook
├── test-images/                 # Sample test dataset
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── docs/                        # Documentation
│   ├── architecture.md
│   └── examples/
```

## Dependencies (Auto-Installed)

The notebook installs all required packages automatically:

- **Core**: `torch`, `torchvision`, `opencv-python`, `numpy`, `matplotlib`
- **Models**: `transformers`, `segment-anything`, `open_clip_torch`
- **Utils**: `Pillow`, `scipy`, `timm`
- **Pre-trained Models**: SAM checkpoints downloaded automatically

## 💡 Key Features

- **Training-Free**: No model training required
- **Natural Language**: Query objects using text descriptions
- **Content-Aware**: Novel interpolation technique preserves object relationships
- **Visual Analysis**: Comprehensive visualization and debugging tools
- **Ready-to-Run**: Complete pipeline in a single notebook

## Expected Results

### Sample Outputs

**Text Query**: `"person"`
- **Detection**: Red bounding boxes around detected people
- **Confidence**: Scores typically range 0.1-0.8
- **Grid**: Top 6 proposals with scores and query text

**Text Query**: `"red car"`
- **Detection**: More specific detection focusing on color attributes
- **Performance**: Demonstrates fine-grained understanding

## Customization Options

Modify these parameters in the notebook:

```python
# Detection sensitivity
threshold = 0.08  # Lower = more detections, Higher = fewer but confident


# VLM model (change if needed)
vlm_model = "google/siglip-base-patch16-224"
```

## Evaluation Metrics

The system outputs:
- **Detection Count**: Number of objects found
- **Confidence Statistics**: Min, Max, Mean, Standard deviation
- **Visual Quality**: Bounding box accuracy and proposal quality

## Troubleshooting

### Common Issues:
1. **"No detections found"**: Try lowering the threshold (0.05-0.1)
2. **Image loading issues**: Ensure images are in standard formats (JPG, PNG)

### Performance Tips:
- Use clear, descriptive text queries
- Test with different threshold values
- Higher resolution images generally work better

## Academic Context

This implementation accompanies the master's dissertation:
**"SegGOD: Training-Free Object Detection through Segmentation and Vision-Language Models"**

### Key Contributions:
1. **Novel Architecture**: First segmentation-guided training-free detection framework
2. **Content-Aware Interpolation**: Aspect-ratio aware preprocessing technique
3. **Practical Framework**: Ready-to-use implementation with extensive validation

## Support

For questions about this implementation:
1. Check the notebook comments for detailed explanations
2. Review the visualization outputs for debugging
3. Modify threshold values if results are unexpected


---

**Ready to test? Just click "Run All" in the Kaggle notebook and start detecting objects with natural language!**
