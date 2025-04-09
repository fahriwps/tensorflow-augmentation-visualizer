"""
Augmentation Visualization Tool for TensorFlow Object Detection API

This script allows visualization of various image augmentation techniques from the
TensorFlow Object Detection API, helping users to select appropriate augmentations
for their training pipeline.
"""

import os
import functools
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml
import numpy as np
import cv2
from PIL import Image
import tensorflow.compat.v1 as tf
from object_detection.core import preprocessor
from object_detection import inputs
from object_detection.core import standard_fields as fields

# Disable eager execution for compatibility with TF Object Detection API
tf.disable_eager_execution()

def load_image_into_numpy_array(image):
    """Convert PIL image to numpy array."""
    width, height = image.size
    return np.array(image.getdata()).reshape(
        (height, width, 3)).astype(np.float32)

def draw_bounding_boxes_with_labels(image, bboxes, class_ids, class_names=None, color=(0, 255, 0)):
    """Draw bounding boxes and class labels on an image."""
    image_copy = image.copy()
    
    # Ensure bboxes is properly shaped
    bboxes = bboxes[:, :4].reshape(-1, 4)
    
    for i, (bbox, class_id) in enumerate(zip(bboxes, class_ids)):
        # Convert coordinates to integers
        x_min, y_min = int(bbox[0]), int(bbox[1])
        x_max, y_max = int(bbox[2]), int(bbox[3])
        
        # Draw rectangle
        line_thickness = int(max(image_copy.shape[:2]) / 200)
        cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), color, line_thickness)
        
        # Add class label
        if class_names and i < len(class_names):
            label_text = class_names[i]
        else:
            label_text = f"Class {int(class_id)}"
            
        ((text_width, text_height), _) = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.putText(
            image_copy, 
            label_text, 
            (x_min, y_min - int(0.3 * text_height)),
            cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=1, 
            color=(255, 255, 255), 
            lineType=cv2.LINE_AA
        )
    
    return image_copy

def parse_xml_annotation(xml_path):
    """
    Parse Pascal VOC format XML annotation file.
    
    Args:
        xml_path: Path to XML annotation file
        
    Returns:
        tuple: (bounding_boxes, class_names, image_width, image_height)
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image size
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        bounding_boxes = []
        class_names = []
        
        # Extract bounding box information
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            bbox = obj.find('bndbox')
            
            x_min = int(float(bbox.find('xmin').text))
            y_min = int(float(bbox.find('ymin').text))
            x_max = int(float(bbox.find('xmax').text))
            y_max = int(float(bbox.find('ymax').text))
            
            bounding_boxes.append([x_min, y_min, x_max, y_max])
            class_names.append(class_name)
            
        return np.array(bounding_boxes, dtype=np.float32), class_names, width, height
        
    except Exception as e:
        print(f"Error parsing XML file {xml_path}: {e}")
        return np.array([]), [], 0, 0

class AugmentationVisualizer:
    def __init__(self, config_path, image_path, output_dir, xml_path=None):
        """
        Initialize the augmentation visualizer.
        
        Args:
            config_path: Path to the augmentation configuration YAML file
            image_path: Path to the image to augment
            output_dir: Directory to save augmented images
            xml_path: Path to the Pascal VOC format XML annotation file (optional)
        """
        self.config = self._load_config(config_path)
        self.image_path = image_path
        self.output_dir = output_dir
        self.original_image = self._load_image()
        self.image_np = load_image_into_numpy_array(self.original_image)
        self.image_height, self.image_width = self.image_np.shape[:2]
        
        # Convert image from RGB to BGR for OpenCV compatibility
        self.image_np = cv2.cvtColor(self.image_np, cv2.COLOR_RGB2BGR)
        
        # Initialize ground truth boxes
        self.ground_truth_boxes = np.array([], dtype=np.float32)
        self.class_names = []
        
        # Load annotations from XML if provided
        if xml_path:
            self._load_annotations_from_xml(xml_path)
            
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Define augmentation techniques to visualize
        self.augmentation_techniques = self._define_augmentation_techniques()
        
    def _load_config(self, config_path):
        """Load the augmentation configuration from a YAML file."""
        try:
            return yaml.safe_load(Path(config_path).read_text())
        except Exception as e:
            print(f"Error loading config file: {e}")
            return {}
            
    def _load_image(self):
        """Load the image to augment."""
        try:
            return Image.open(self.image_path)
        except Exception as e:
            print(f"Error loading image: {e}")
            raise
            
    def _load_annotations_from_xml(self, xml_path):
        """Load ground truth boxes from XML annotation file."""
        boxes, class_names, width, height = parse_xml_annotation(xml_path)
        if len(boxes) > 0:
            self.ground_truth_boxes = boxes
            self.class_names = class_names
            
            # Check if image dimensions match
            if width != self.image_width or height != self.image_height:
                print(f"Warning: XML dimensions ({width}x{height}) do not match image dimensions "
                      f"({self.image_width}x{self.image_height})")
            
            # Normalize box coordinates to [0, 1]
            if np.max(self.ground_truth_boxes) > 1.0:
                self.ground_truth_boxes = self.ground_truth_boxes / np.array([self.image_width, self.image_height, 
                                                                             self.image_width, self.image_height])
                
            print(f"Loaded {len(self.ground_truth_boxes)} bounding boxes from XML")
        else:
            print("No bounding boxes found in XML file")
            
    def _define_augmentation_techniques(self):
        """Define the augmentation techniques to visualize."""
        return [
            None,  # Original image without augmentation
            (preprocessor.random_horizontal_flip, {}),
            (preprocessor.random_vertical_flip, {}),
            (preprocessor.random_rotation90, {}),
            (preprocessor.random_pixel_value_scale, {}),
            (preprocessor.random_image_scale, {}),
            (preprocessor.random_rgb_to_gray, {}),
            (preprocessor.random_adjust_brightness, {}),
            (preprocessor.random_adjust_contrast, {}),
            (preprocessor.random_adjust_hue, {}),
            (preprocessor.random_adjust_saturation, {}),
            (preprocessor.random_distort_color, {}),
            (preprocessor.random_jitter_boxes, {}),
            (preprocessor.random_crop_image, {}),
            (preprocessor.random_pad_image, {}),
            (preprocessor.random_crop_pad_image, {}),
            (preprocessor.random_pad_to_aspect_ratio, {}),
            (preprocessor.random_black_patches, {}),
            (preprocessor.random_resize_method, {}),
            (preprocessor.random_patch_gaussian, {}),
            (preprocessor.subtract_channel_mean, {})
        ]
    
    def set_ground_truth_boxes(self, bounding_boxes, class_names=None):
        """
        Set ground truth bounding boxes for the image.
        
        Args:
            bounding_boxes: List of bounding boxes in format [x_min, y_min, x_max, y_max]
            class_names: List of class names for each bounding box (optional)
        """
        self.ground_truth_boxes = np.array(bounding_boxes, dtype=np.float32)
        if class_names:
            self.class_names = class_names
        
        # Normalize box coordinates to [0, 1]
        if np.max(self.ground_truth_boxes) > 1.0:
            self.ground_truth_boxes = self.ground_truth_boxes / np.array([self.image_width, self.image_height, 
                                                                         self.image_width, self.image_height])
    
    def visualize_augmentations(self, num_repetitions=1):
        """
        Apply and visualize each augmentation technique.
        
        Args:
            num_repetitions: Number of times to repeat each augmentation
        """
        # Check if ground truth boxes are available
        if len(self.ground_truth_boxes) == 0:
            print("Warning: No ground truth boxes available. Augmentations that modify boxes may not work properly.")
            
        for augmentation in self.augmentation_techniques:
            for i in range(num_repetitions):
                if augmentation is None:
                    print("\nImage without augmentation")
                    # Save original image
                    original_path = os.path.join(self.output_dir, 'original.png')
                    cv2.imwrite(original_path, self.image_np)
                    
                    # If we have ground truth boxes, save image with boxes
                    if len(self.ground_truth_boxes) > 0:
                        # Scale boxes back to image dimensions
                        scaled_boxes = self.ground_truth_boxes * np.array([self.image_width, self.image_height, 
                                                                          self.image_width, self.image_height])
                        image_with_boxes = draw_bounding_boxes_with_labels(
                            self.image_np,
                            scaled_boxes,
                            np.ones(len(scaled_boxes)),  # Assuming all boxes have class 1
                            self.class_names
                        )
                        boxes_path = os.path.join(self.output_dir, 'original_boxes.png')
                        cv2.imwrite(boxes_path, image_with_boxes)
                        print(f"Saved original image with boxes to {boxes_path}")
                    continue
                
                # Reset TensorFlow graph for each augmentation
                tf.reset_default_graph()
                
                augmentation_name = augmentation[0].__name__
                print(f"\nApplying {augmentation_name}")
                
                # Check if augmentation exists in config and update parameters
                augmentation_params = self._get_augmentation_params(augmentation_name, augmentation[1])
                current_augmentation = (augmentation[0], augmentation_params)
                
                # Apply augmentation
                augmented_image, augmented_boxes, augmented_classes = self._apply_augmentation(current_augmentation)
                
                if augmented_image is not None:
                    # Save augmented image
                    output_filename = f'aug_{augmentation_name}_{i}.png'
                    output_path = os.path.join(self.output_dir, output_filename)
                    cv2.imwrite(output_path, augmented_image)
                    
                    # Save image with bounding boxes
                    if augmented_boxes is not None and len(augmented_boxes) > 0:
                        # Scale boxes back to image dimensions
                        scaled_boxes = augmented_boxes * np.array([self.image_width, self.image_height, 
                                                                 self.image_width, self.image_height])
                        image_with_boxes = draw_bounding_boxes_with_labels(
                            augmented_image,
                            scaled_boxes,
                            augmented_classes,
                            self.class_names if len(self.class_names) == len(augmented_boxes) else None
                        )
                        boxes_output_path = output_path.replace('.png', '_boxes.png')
                        cv2.imwrite(boxes_output_path, image_with_boxes)
                        
                    print(f"Saved to {output_path}")
                    print(f"Image shape: {augmented_image.shape}")
                    if augmented_boxes is not None and len(augmented_boxes) > 0:
                        print(f"Bounding boxes: {augmented_boxes}")
    
    def _get_augmentation_params(self, augmentation_name, default_params):
        """Get augmentation parameters from config or use defaults."""
        if augmentation_name in self.config:
            params = self.config[augmentation_name]
            if params is None:
                print(f"DEBUG: No parameters in config for {augmentation_name}")
                return default_params
            else:
                print(f"Using config parameters for {augmentation_name}: {params}")
                return params
        else:
            print(f"WARNING: {augmentation_name} not found in config")
            return default_params
    
    def _apply_augmentation(self, augmentation):
        """
        Apply an augmentation to the image.
        
        Args:
            augmentation: Tuple (augmentation_function, params)
            
        Returns:
            Tuple (augmented_image, augmented_boxes, augmented_classes)
        """
        try:
            # Create tensor dictionary
            tensor_dict = {
                fields.InputDataFields.image: tf.constant(self.image_np.astype(np.float32)),
            }
            
            # Add ground truth boxes and classes if available
            if len(self.ground_truth_boxes) > 0:
                tensor_dict[fields.InputDataFields.groundtruth_boxes] = tf.constant(self.ground_truth_boxes)
                tensor_dict[fields.InputDataFields.groundtruth_classes] = tf.constant(
                    np.ones(len(self.ground_truth_boxes), dtype=np.float32)
                )
            
            # Create augmentation function
            augmentation_fn = functools.partial(
                inputs.augment_input_data,
                data_augmentation_options=[augmentation]
            )
            
            # Apply augmentation
            augmented_tensor_dict = augmentation_fn(tensor_dict=tensor_dict)
            
            # Run session to get output
            with tf.Session() as sess:
                augmented_tensor_dict_out = sess.run(augmented_tensor_dict)
            
            # Extract augmented image and boxes
            augmented_image = augmented_tensor_dict_out[fields.InputDataFields.image]
            
            augmented_boxes = None
            augmented_classes = None
            
            if fields.InputDataFields.groundtruth_boxes in augmented_tensor_dict_out:
                augmented_boxes = augmented_tensor_dict_out[fields.InputDataFields.groundtruth_boxes]
                augmented_classes = augmented_tensor_dict_out[fields.InputDataFields.groundtruth_classes]
            
            return augmented_image, augmented_boxes, augmented_classes
            
        except Exception as e:
            print(f"Error applying {augmentation[0].__name__}: {e}")
            return None, None, None


def main():
    # Configuration
    CONFIG_PATH = "home/aug_parameter.yaml"
    IMAGE_PATH = "home/data/car.png"
    OUTPUT_DIR = "home/output_image"
    
    # XML annotation path
    XML_PATH = "home/data/car.xml"  # Optional: Path to XML file for bounding boxes
    
    # Create visualizer with XML annotations
    visualizer = AugmentationVisualizer(CONFIG_PATH, IMAGE_PATH, OUTPUT_DIR, xml_path=XML_PATH)
    
    # Alternatively, you can set ground truth boxes manually
    # ground_truth_boxes = [[145, 122, 546, 456]]
    # visualizer.set_ground_truth_boxes(ground_truth_boxes, class_names=["car"])
    
    # Visualize augmentations
    visualizer.visualize_augmentations(num_repetitions=1)  # Adjust number of repetitions as needed


if __name__ == "__main__":
    main()