# Waste Classification using YOLO

This project implements an automated waste segregation system using YOLO (You Only Look Once) for object detection and classification of waste materials into different categories.

## Features
- Uses YOLO for real-time waste detection
- Classifies waste into predefined categories (e.g., plastic, paper, metal, etc.)
- Can be optimized for better accuracy and efficiency

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Waste_classification/waste-classification.git
   cd waste-classification
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Application
   ```bash
   streamlit run app.py
   ```
4. Ensure you have a compatible version of Python (e.g., Python 3.7+).

## Dataset
The model is trained on a dataset of waste images categorized into different types. You can use an existing dataset or create your own by collecting images and labeling them using annotation tools like LabelImg.

Dataset Link: https://universe.roboflow.com/ai-project-i3wje/waste-detection-vqkjo/model/3

## Usage

1. Run the notebook:
   ```bash
   jupyter notebook waste_classification.ipynb
   ```
2. Train the model with your dataset.
3. Use the trained model to classify waste images.

## Future Improvements
- Optimization for faster inference
- Data augmentation to improve accuracy
- Integration with an edge device for real-time waste sorting

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

