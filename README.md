# TensorFlow Augmentation Visualizer

Tool for visualizing TensorFlow Object Detection API's data augmentation techniques on images with bounding boxes. This tool solves that problem by allowing you to:

1. Visualize TensorFlow's augmentation functions on your images
2. See how augmentations affect bounding box coordinates
3. Use your existing Pascal VOC format annotations
4. Customize augmentation parameters via a YAML config file
5. Generate visualization images for evaluation and documentation

## Dependencies

- TensorFlow (1.x compatible)
- OpenCV
- PIL/Pillow
- NumPy
- PyYAML
- TensorFlow Object Detection API

## Usage

Configure your file directory path:

```python
# Configuration
    CONFIG_PATH = "home/aug_parameter.yaml"
    IMAGE_PATH = "home/data/car.png"
    OUTPUT_DIR = "home/output_image"

# XML annotation path
    XML_PATH = "home/data/car.xml"  # Optional: Path to XML file for bounding boxes
```

Then customize augmentation parameters inside `aug_parameter.yaml` file:

```bash
...
# Color augmentations
random_adjust_brightness:
  max_delta: 0.1

random_adjust_contrast:
  min_delta: 0.8
  max_delta: 1.1

random_adjust_hue:
  max_delta: 0.1

random_adjust_saturation:
  min_delta: 0.8
  max_delta: 1.1

random_distort_color:
  color_ordering: 1
...
```

Run the python script:

```bash
python visualize_augmentation_tfod.py
```

Augmented images will be saved inside `output_img` folder.
