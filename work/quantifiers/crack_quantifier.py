import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.morphology import skeletonize, medial_axis
from skimage.measure import regionprops, label
import math
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime
import argparse


class CrackQuantifier:
    """
    A class to quantify cracks in images after detection by YOLOv8.
    
    This class provides methods to:
    1. Extract crack masks from YOLOv8 detections
    2. Compute metrics like length, width, and orientation
    3. Classify cracks by pattern and severity
    4. Visualize the results
    """
    
    def __init__(self, severity_thresholds=None):
        """
        Initialize the CrackQuantifier with optional severity thresholds.
        
        Args:
            severity_thresholds (dict): Thresholds for classifying crack severity
                Default is {'minor': 0.5, 'moderate': 2.0, 'severe': 5.0} (in mm)
        """
        self.severity_thresholds = severity_thresholds or {'minor': 0.5, 'moderate': 2.0, 'severe': 5.0}
        self.pixel_to_mm_ratio = 1.0  # Default value, should be calibrated for your specific setup
    
    def set_calibration(self, pixel_to_mm_ratio):
        """
        Set the pixel to millimeter ratio for accurate physical measurements.
        
        Args:
            pixel_to_mm_ratio (float): Number of pixels per millimeter
        """
        self.pixel_to_mm_ratio = pixel_to_mm_ratio
    
    def extract_crack_mask(self, image, detection, conf_threshold=0.25):
        """
        Extract a binary mask of the crack from a YOLOv8 detection.
        
        Args:
            image (numpy.ndarray): Original image
            detection: YOLOv8 detection result containing bounding boxes and segmentation masks
            conf_threshold (float): Confidence threshold for filtering detections
            
        Returns:
            numpy.ndarray: Binary mask of the crack
        """
        # Initialize an empty mask with the same shape as the image
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Check if there are any detections
        if len(detection) == 0:
            return mask
        
        # Process each detection
        for i, det in enumerate(detection.boxes.data):
            # Extract confidence score
            conf = det[4].item()
            
            # Filter based on confidence
            if conf < conf_threshold:
                continue
            
            # Get class
            cls = int(det[5].item())
            
            # Check if the class corresponds to crack (assuming class index 0 is crack)
            if cls == 0:  # Modify this according to your model's class mapping
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, det[:4])
                
                # If segmentation mask is available
                if hasattr(detection, 'masks') and detection.masks is not None:
                    segment = detection.masks.data[i].cpu().numpy()
                    # Resize the mask to the original image dimensions
                    segment = cv2.resize(segment, (width, height))
                    # Add this segment to the mask
                    mask = np.logical_or(mask, segment > 0.5).astype(np.uint8) * 255
                else:
                    # If no segmentation mask, use the bounding box
                    mask[y1:y2, x1:x2] = 255
        
        # Clean the mask (fill small holes, remove small objects)
        mask = self._clean_mask(mask)
        
        return mask
    
    def _clean_mask(self, mask, min_size=50):
        """
        Clean the binary mask by filling holes and removing small objects.
        
        Args:
            mask (numpy.ndarray): Binary mask
            min_size (int): Minimum size of objects to keep
            
        Returns:
            numpy.ndarray: Cleaned binary mask
        """
        # Normalize mask to binary (0 and 1)
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)
        
        # Fill holes
        mask = binary_fill_holes(mask).astype(np.uint8)
        
        # Label connected components
        labeled_mask = label(mask)
        
        # Remove small objects
        for region in regionprops(labeled_mask):
            if region.area < min_size:
                mask[labeled_mask == region.label] = 0
        
        return mask * 255
    
    def compute_skeleton(self, mask):
        """
        Compute the skeleton (medial axis) of the crack.
        
        Args:
            mask (numpy.ndarray): Binary mask of the crack
            
        Returns:
            numpy.ndarray: Skeleton image
            numpy.ndarray: Distance transform result
        """
        # Normalize mask to binary (0 and 1)
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)
        
        # Compute the medial axis
        skel, distance = medial_axis(mask, return_distance=True)
        
        return skel, distance
    
    def measure_crack_length(self, skeleton):
        """
        Measure the length of the crack.
        
        Args:
            skeleton (numpy.ndarray): Skeleton of the crack
            
        Returns:
            float: Length of the crack in pixels
        """
        # Count the number of pixels in the skeleton
        pixel_count = np.sum(skeleton)
        
        # Convert to physical units if calibration is set
        length_mm = pixel_count / self.pixel_to_mm_ratio
        
        return length_mm
    
    def measure_crack_width(self, mask, skeleton, distance):
        """
        Measure the average and maximum width of the crack.
        
        Args:
            mask (numpy.ndarray): Binary mask of the crack
            skeleton (numpy.ndarray): Skeleton of the crack
            distance (numpy.ndarray): Distance transform result
            
        Returns:
            tuple: (average_width, max_width) in pixels
        """
        # Get distance values along the skeleton
        skeleton_distances = distance * skeleton
        
        # Calculate the average width (multiply by 2 because distance is to the nearest edge)
        if np.sum(skeleton) > 0:
            avg_width = 2 * np.sum(skeleton_distances) / np.sum(skeleton)
        else:
            avg_width = 0
        
        # Calculate the maximum width
        max_width = 2 * np.max(skeleton_distances) if np.max(skeleton_distances) > 0 else 0
        
        # Convert to physical units if calibration is set
        avg_width_mm = avg_width / self.pixel_to_mm_ratio
        max_width_mm = max_width / self.pixel_to_mm_ratio
        
        return avg_width_mm, max_width_mm
    
    def determine_crack_orientation(self, skeleton):
        """
        Determine the predominant orientation of the crack.
        
        Args:
            skeleton (numpy.ndarray): Skeleton of the crack
            
        Returns:
            tuple: (angle_degrees, 'horizontal'|'vertical'|'diagonal')
        """
        # Find coordinates of skeleton pixels
        y_coords, x_coords = np.where(skeleton > 0)
        
        if len(x_coords) < 2:
            return 0, 'unknown'
        
        # Calculate the covariance matrix
        coords = np.column_stack((x_coords, y_coords))
        cov_matrix = np.cov(coords, rowvar=False)
        
        # Calculate eigenvalues and eigenvectors
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            # The eigenvector corresponding to the largest eigenvalue gives the main direction
            largest_idx = np.argmax(eigenvalues)
            main_direction = eigenvectors[:, largest_idx]
            
            # Calculate the angle in degrees (arctan2 returns result in radians)
            angle = math.degrees(math.atan2(main_direction[1], main_direction[0]))
            
            # Normalize angle to be between 0 and 180 degrees
            angle = angle % 180
            
            # Classify the orientation
            if 0 <= angle < 30 or 150 <= angle < 180:
                orientation = 'horizontal'
            elif 60 <= angle < 120:
                orientation = 'vertical'
            else:
                orientation = 'diagonal'
                
            return angle, orientation
        except:
            # In case of error (e.g., singular covariance matrix)
            return 0, 'unknown'
    
    def classify_crack_pattern(self, skeleton):
        """
        Classify the crack pattern as linear, branched, or mesh.
        
        Args:
            skeleton (numpy.ndarray): Skeleton of the crack
            
        Returns:
            str: 'linear', 'branched', or 'mesh'
        """
        # Find junction points (pixels with more than 2 neighbors)
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0  # Don't count the center pixel
        
        # Count neighbors for each pixel
        neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        
        # Junction points have more than 2 neighbors
        junction_points = np.logical_and(skeleton, neighbors > 2)
        num_junctions = np.sum(junction_points)
        
        # Classification based on number of junctions
        if num_junctions == 0:
            return 'linear'
        elif num_junctions < 5:
            return 'branched'
        else:
            return 'mesh'
    
    def classify_crack_severity(self, avg_width_mm):
        """
        Classify the crack severity based on its width.
        
        Args:
            avg_width_mm (float): Average width of the crack in millimeters
            
        Returns:
            str: 'minor', 'moderate', or 'severe'
        """
        if avg_width_mm < self.severity_thresholds['minor']:
            return 'minor'
        elif avg_width_mm < self.severity_thresholds['moderate']:
            return 'severe'
        else:
            return 'critical'
    
    def analyze_crack(self, image, detection, conf_threshold=0.25):
        """
        Analyze detected cracks and return quantification metrics.
        
        Args:
            image (numpy.ndarray): Original image
            detection: YOLOv8 detection result
            conf_threshold (float): Confidence threshold for filtering detections
            
        Returns:
            dict: Dictionary containing crack metrics and classifications
        """
        # Extract crack mask
        mask = self.extract_crack_mask(image, detection, conf_threshold)
        
        # Check if any crack was detected
        if np.sum(mask) == 0:
            return {
                'detected': False,
                'message': 'No cracks detected or confidence below threshold'
            }
        
        # Compute skeleton
        skeleton, distance = self.compute_skeleton(mask)
        
        # Measure length
        length_mm = self.measure_crack_length(skeleton)
        
        # Measure width
        avg_width_mm, max_width_mm = self.measure_crack_width(mask, skeleton, distance)
        
        # Determine orientation
        angle, orientation = self.determine_crack_orientation(skeleton)
        
        # Classify pattern
        pattern = self.classify_crack_pattern(skeleton)
        
        # Classify severity
        severity = self.classify_crack_severity(avg_width_mm)
        
        # Compile results
        result = {
            'detected': True,
            'length_mm': round(length_mm, 2),
            'avg_width_mm': round(avg_width_mm, 2),
            'max_width_mm': round(max_width_mm, 2),
            'orientation': {
                'angle_degrees': round(angle, 2),
                'type': orientation
            },
            'pattern': pattern,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'mask': mask,
            'skeleton': skeleton
        }
        
        return result
    
    def visualize_results(self, image, result, save_path=None):
        """
        Visualize the crack analysis results.
        
        Args:
            image (numpy.ndarray): Original image
            result (dict): Analysis result from analyze_crack
            save_path (str, optional): Path to save the visualization
            
        Returns:
            numpy.ndarray: Visualization image
        """
        if not result['detected']:
            return image.copy()
        
        # Create a color visualization
        vis_image = image.copy()
        
        # Convert grayscale to RGB if needed
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        # Create a color overlay for the mask
        mask_color = np.zeros_like(vis_image)
        mask_color[result['mask'] > 0] = [0, 255, 0]  # Green for the mask
        
        # Create an overlay for the skeleton
        skeleton_color = np.zeros_like(vis_image)
        skeleton_color[result['skeleton'] > 0] = [255, 0, 0]  # Red for the skeleton
        
        # Blend the overlays with the original image
        alpha = 0.5
        vis_image = cv2.addWeighted(vis_image, 1, mask_color, alpha, 0)
        vis_image = cv2.addWeighted(vis_image, 1, skeleton_color, alpha, 0)
        
        # Add text with quantification results
        metrics_text = [
            f"Length: {result['length_mm']:.2f} mm",
            f"Avg Width: {result['avg_width_mm']:.2f} mm",
            f"Max Width: {result['max_width_mm']:.2f} mm",
            f"Orientation: {result['orientation']['type']} ({result['orientation']['angle_degrees']:.1f} deg)",
            f"Pattern: {result['pattern']}",
            f"Severity: {result['severity']}"
        ]
        
        y_offset = 30
        for i, text in enumerate(metrics_text):
            cv2.putText(vis_image, text, (10, y_offset + i * 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image


def process_image_with_crack_quantification(model_path, image_path, output_path=None, conf_threshold=0.25, pixel_to_mm_ratio=10.0):
    """
    Process an image with YOLOv8 and crack quantification.
    
    Args:
        model_path (str): Path to the YOLOv8 model
        image_path (str): Path to the input image
        output_path (str, optional): Path to save the visualization
        conf_threshold (float): Confidence threshold for filtering detections
        pixel_to_mm_ratio (float): Number of pixels per millimeter
        
    Returns:
        dict: Analysis results
    """
    # Load the model
    model = YOLO(model_path)
    
    # Load the image
    image = cv2.imread(image_path)
    
    # Run detection
    results = model(image)[0]
    
    # Initialize crack quantifier
    quantifier = CrackQuantifier()
    quantifier.set_calibration(pixel_to_mm_ratio)
    
    # Analyze cracks
    crack_result = quantifier.analyze_crack(image, results, conf_threshold)
    
    # Visualize if requested
    if output_path and crack_result['detected']:
        vis_image = quantifier.visualize_results(image, crack_result, output_path)
        
        # Create a figure with two subplots
        plt.figure(figsize=(15, 10))
        
        # Original image with detection
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        # Visualization with quantification
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title('Crack Quantification')
        plt.axis('off')
        
        # Add summary text
        plt.figtext(0.5, 0.01, 
                   f"Crack Length: {crack_result['length_mm']:.2f} mm | " + 
                   f"Avg Width: {crack_result['avg_width_mm']:.2f} mm | " + 
                   f"Severity: {crack_result['severity']} | " + 
                   f"Pattern: {crack_result['pattern']}", 
                   ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    return crack_result


def parse_args():
    parser = argparse.ArgumentParser(description='Crack detection and quantification using YOLOv8')

    # Required parameters
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the YOLOv8 model')
    parser.add_argument('--image-path', type=str, required=True,
                        help='Path to the input image')

    # Optional parameters
    parser.add_argument('--output-path', type=str, default=None,
                        help='Path to save the visualization (default: None)')
    parser.add_argument('--pixel-to-mm-ratio', type=float, default=10.0,
                        help='Number of pixels per millimeter (default: 10.0)')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='Confidence threshold for filtering detections (default: 0.25)')

    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_args()

    # Process the image
    results = process_image_with_crack_quantification(
        model_path=args.model_path,
        image_path=args.image_path,
        output_path=args.output_path,
        conf_threshold=args.conf_threshold,
        pixel_to_mm_ratio=args.pixel_to_mm_ratio
    )

    # Print results
    print(f"Crack Analysis Results:")
    for key, value in results.items():
        if key not in ['mask', 'skeleton']:  # Skip binary images
            print(f"{key}: {value}")


# Example usage when run directly
if __name__ == "__main__":
    main()

# Example usage:
