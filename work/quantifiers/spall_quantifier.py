import cv2
import numpy as np
from skimage.measure import regionprops, label
from sklearn.cluster import DBSCAN
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime
import argparse


class SpallationQuantifier:
    """
    A class to quantify spallation defects in images after detection by YOLOv8.

    This class provides methods to:
    1. Extract spallation masks from YOLOv8 detections
    2. Compute metrics like area, perimeter, and depth estimation
    3. Classify spallation by severity
    4. Visualize the results
    """

    def __init__(self, severity_thresholds=None):
        """
        Initialize the SpallationQuantifier with optional severity thresholds.

        Args:
            severity_thresholds (dict): Thresholds for classifying spallation severity
                Default is {'minor': 100, 'moderate': 500, 'severe': 2000} (in cm²)
        """
        self.severity_thresholds = severity_thresholds or {'minor': 100, 'moderate': 500, 'severe': 2000}
        self.pixel_to_cm_ratio = 1.0  # Default value, should be calibrated for your specific setup

    def set_calibration(self, pixel_to_cm_ratio):
        """
        Set the pixel to centimeter ratio for accurate physical measurements.

        Args:
            pixel_to_cm_ratio (float): Number of pixels per centimeter
        """
        self.pixel_to_cm_ratio = pixel_to_cm_ratio

    def extract_spallation_mask(self, image, detection, conf_threshold=0.25, class_id=2):
        """
        Extract a binary mask of the spallation from a YOLOv8 detection.

        Args:
            image (numpy.ndarray): Original image
            detection: YOLOv8 detection result containing bounding boxes and segmentation masks
            conf_threshold (float): Confidence threshold for filtering detections
            class_id (int): Class ID for spallation in your model

        Returns:
            numpy.ndarray: Binary mask of the spallation
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

            # Check if the class corresponds to spallation
            if cls == class_id:  # Modify this according to your model's class mapping
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

        # Clean the mask (remove small objects)
        mask = self._clean_mask(mask)

        return mask

    def _clean_mask(self, mask, min_size=50):
        """
        Clean the binary mask by removing small objects.

        Args:
            mask (numpy.ndarray): Binary mask
            min_size (int): Minimum size of objects to keep

        Returns:
            numpy.ndarray: Cleaned binary mask
        """
        # Normalize mask to binary (0 and 1)
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)

        # Label connected components
        labeled_mask = label(mask)

        # Remove small objects
        for region in regionprops(labeled_mask):
            if region.area < min_size:
                mask[labeled_mask == region.label] = 0

        return mask * 255

    def measure_area(self, mask):
        """
        Measure the area of the spallation.

        Args:
            mask (numpy.ndarray): Binary mask of the spallation

        Returns:
            float: Area of the spallation in square centimeters
        """
        # Count the number of pixels in the mask
        pixel_count = np.sum(mask > 0)

        # Convert to physical units if calibration is set
        area_cm2 = pixel_count / (self.pixel_to_cm_ratio ** 2)

        return area_cm2

    def measure_perimeter(self, mask):
        """
        Measure the perimeter of the spallation.

        Args:
            mask (numpy.ndarray): Binary mask of the spallation

        Returns:
            float: Perimeter of the spallation in centimeters
        """
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate total perimeter
        perimeter_pixels = sum(cv2.arcLength(contour, True) for contour in contours)

        # Convert to physical units if calibration is set
        perimeter_cm = perimeter_pixels / self.pixel_to_cm_ratio

        return perimeter_cm

    def estimate_depth(self, image, mask):
        """
        Estimate the depth of the spallation using texture and shadow analysis.

        Args:
            image (numpy.ndarray): Original image
            mask (numpy.ndarray): Binary mask of the spallation

        Returns:
            float: Estimated depth of the spallation in centimeters (very approximate)
        """
        # Convert image to grayscale if it's not already
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply the mask to the grayscale image
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask.astype(np.uint8))

        # Get pixel values within the mask
        pixels = masked_gray[mask > 0]

        if len(pixels) == 0:
            return 0.0

        # Calculate statistics for depth estimation
        mean_intensity = np.mean(pixels)
        std_intensity = np.std(pixels)

        # A very simple heuristic for depth estimation based on intensity variation
        # This is just a placeholder - in a real application, you'd need to calibrate
        # this based on known depth measurements
        estimated_depth_cm = std_intensity / 25  # This is an arbitrary scaling factor

        return estimated_depth_cm

    def analyze_texture_complexity(self, mask, image):
        """
        Analyze the texture complexity of the spallation area.

        Args:
            mask (numpy.ndarray): Binary mask of the spallation
            image (numpy.ndarray): Original image

        Returns:
            float: Texture complexity score (higher means more complex)
        """
        # Convert image to grayscale if it's not already
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply the mask to the grayscale image
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask.astype(np.uint8))

        # Calculate texture complexity using gradient information
        sobelx = cv2.Sobel(masked_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(masked_gray, cv2.CV_64F, 0, 1, ksize=3)

        # Get gradient magnitude
        gradient_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

        # Calculate mean gradient magnitude within the mask
        if np.sum(mask > 0) > 0:
            texture_complexity = np.sum(gradient_mag * (mask > 0)) / np.sum(mask > 0)
        else:
            texture_complexity = 0

        return texture_complexity

    def classify_shape_complexity(self, mask):
        """
        Classify the shape complexity of the spallation.

        Args:
            mask (numpy.ndarray): Binary mask of the spallation

        Returns:
            str: 'simple', 'moderate', or 'complex'
        """
        # Find contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 'unknown'

        # Calculate shape complexity metrics for the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)

        # Calculate circularity (4π × area / perimeter²)
        # A perfect circle has circularity of 1, more complex shapes have lower values
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0

        # Classify based on circularity
        if circularity > 0.7:
            return 'simple'
        elif circularity > 0.3:
            return 'moderate'
        else:
            return 'complex'

    def classify_severity(self, area_cm2, depth_cm=None, proximity_to_rebar=None):
        """
        Classify the severity of the spallation.

        Args:
            area_cm2 (float): Area of the spallation in square centimeters
            depth_cm (float, optional): Estimated depth in centimeters
            proximity_to_rebar (bool, optional): Whether the spallation is near rebar

        Returns:
            str: 'minor', 'moderate', 'severe', or 'critical'
        """
        # Base classification on area
        if area_cm2 < self.severity_thresholds['minor']:
            severity = 'minor'
        elif area_cm2 < self.severity_thresholds['moderate']:
            severity = 'moderate'
        elif area_cm2 < self.severity_thresholds['severe']:
            severity = 'severe'
        else:
            severity = 'critical'

        # Adjust based on depth if available
        if depth_cm is not None and depth_cm > 2.0:
            # Increase severity by one level if depth is significant
            severity_levels = ['minor', 'moderate', 'severe', 'critical']
            current_index = severity_levels.index(severity)
            if current_index < len(severity_levels) - 1:
                severity = severity_levels[current_index + 1]

        # Adjust based on proximity to rebar if available
        if proximity_to_rebar:
            # Always classify as critical if near rebar
            severity = 'critical'

        return severity

    def detect_proximity_to_rebar(self, image, spallation_mask, detections, rebar_class_id=1, distance_threshold=50):
        """
        Detect if the spallation is in proximity to exposed rebar.

        Args:
            image (numpy.ndarray): Original image
            spallation_mask (numpy.ndarray): Binary mask of the spallation
            detections: YOLOv8 detection results
            rebar_class_id (int): Class ID for exposed rebar
            distance_threshold (int): Maximum distance in pixels to consider proximity

        Returns:
            bool: True if spallation is near rebar, False otherwise
        """
        # Check if there are any detections
        if len(detections) == 0:
            return False

        # Find spallation contours
        spallation_contours, _ = cv2.findContours(
            spallation_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not spallation_contours:
            return False

        # Extract points from spallation contours
        spallation_points = np.vstack([contour.reshape(-1, 2) for contour in spallation_contours])

        # Initialize rebar points
        rebar_points = []

        # Process each detection to find rebar
        for i, det in enumerate(detections.boxes.data):
            # Get class
            cls = int(det[5].item())

            # Check if the class corresponds to exposed rebar
            if cls == rebar_class_id:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, det[:4])

                # Create a mask for this rebar detection
                height, width = image.shape[:2]
                rebar_mask = np.zeros((height, width), dtype=np.uint8)

                # If segmentation mask is available
                if hasattr(detections, 'masks') and detections.masks is not None:
                    segment = detections.masks.data[i].cpu().numpy()
                    # Resize the mask to the original image dimensions
                    segment = cv2.resize(segment, (width, height))
                    rebar_mask[segment > 0.5] = 255
                else:
                    # If no segmentation mask, use the bounding box
                    rebar_mask[y1:y2, x1:x2] = 255

                # Find rebar contours
                rebar_contours, _ = cv2.findContours(
                    rebar_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                # Extract points from rebar contours
                for contour in rebar_contours:
                    points = contour.reshape(-1, 2)
                    rebar_points.extend(points)

        if not rebar_points:
            return False

        # Convert to numpy array
        rebar_points = np.array(rebar_points)

        # Calculate minimum distance between spallation and rebar points
        min_distance = float('inf')

        # Use DBSCAN to cluster and reduce computational complexity
        # This is more efficient than calculating all pairwise distances
        for sp_point in spallation_points[::10]:  # Sample every 10th point to reduce computation
            for rb_point in rebar_points[::10]:  # Sample every 10th point
                dist = np.sqrt(np.sum((sp_point - rb_point) ** 2))
                min_distance = min(min_distance, dist)

                # Early termination if we find a distance below threshold
                if min_distance < distance_threshold:
                    return True

        return min_distance < distance_threshold

    def analyze_spallation(self, image, detection, conf_threshold=0.25, spallation_class_id=2, rebar_class_id=1):
        """
        Analyze detected spallation and return quantification metrics.

        Args:
            image (numpy.ndarray): Original image
            detection: YOLOv8 detection result
            conf_threshold (float): Confidence threshold for filtering detections
            spallation_class_id (int): Class ID for spallation
            rebar_class_id (int): Class ID for exposed rebar

        Returns:
            dict: Dictionary containing spallation metrics and classifications
        """
        # Extract spallation mask
        mask = self.extract_spallation_mask(image, detection, conf_threshold, spallation_class_id)

        # Check if any spallation was detected
        if np.sum(mask) == 0:
            return {
                'detected': False,
                'message': 'No spallation detected or confidence below threshold'
            }

        # Measure area
        area_cm2 = self.measure_area(mask)

        # Measure perimeter
        perimeter_cm = self.measure_perimeter(mask)

        # Estimate depth
        depth_cm = self.estimate_depth(image, mask)

        # Analyze texture complexity
        texture_complexity = self.analyze_texture_complexity(mask, image)

        # Classify shape complexity
        shape_complexity = self.classify_shape_complexity(mask)

        # Detect proximity to rebar
        near_rebar = self.detect_proximity_to_rebar(image, mask, detection, rebar_class_id)

        # Classify severity
        severity = self.classify_severity(area_cm2, depth_cm, near_rebar)

        # Compile results
        result = {
            'detected': True,
            'area_cm2': round(area_cm2, 2),
            'perimeter_cm': round(perimeter_cm, 2),
            'estimated_depth_cm': round(depth_cm, 2),
            'texture_complexity': round(texture_complexity, 2),
            'shape_complexity': shape_complexity,
            'near_rebar': near_rebar,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'mask': mask
        }

        return result

    def visualize_results(self, image, result, save_path=None):
        """
        Visualize the spallation analysis results.

        Args:
            image (numpy.ndarray): Original image
            result (dict): Analysis result from analyze_spallation
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

        # Color based on severity
        if result['severity'] == 'minor':
            color = [255, 255, 0]  # Yellow
        elif result['severity'] == 'moderate':
            color = [0, 165, 255]  # Orange
        elif result['severity'] == 'severe':
            color = [0, 0, 255]  # Red
        else:  # critical
            color = [128, 0, 128]  # Purple

        mask_color[result['mask'] > 0] = color

        # Find contours for highlighting borders
        contours, _ = cv2.findContours(
            result['mask'].astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Blend the overlay with the original image
        alpha = 0.5
        vis_image = cv2.addWeighted(vis_image, 1, mask_color, alpha, 0)

        # Draw contours
        cv2.drawContours(vis_image, contours, -1, color, 2)

        # Add text with quantification results
        metrics_text = [
            f"Area: {result['area_cm2']:.2f} cm²",
            f"Perimeter: {result['perimeter_cm']:.2f} cm",
            f"Est. Depth: {result['estimated_depth_cm']:.2f} cm",
            f"Shape: {result['shape_complexity']}",
            f"Near Rebar: {'Yes' if result['near_rebar'] else 'No'}",
            f"Severity: {result['severity']}"
        ]

        y_offset = 30
        for i, text in enumerate(metrics_text):
            cv2.putText(vis_image, text, (10, y_offset + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if save_path:
            cv2.imwrite(save_path, vis_image)

        return vis_image


def process_image_with_spallation_quantification(model_path, image_path, output_path=None,
                                                 conf_threshold=0.25, pixel_to_cm_ratio=10.0,
                                                 spallation_class_id=2, rebar_class_id=1):
    """
    Process an image with YOLOv8 and spallation quantification.

    Args:
        model_path (str): Path to the YOLOv8 model
        image_path (str): Path to the input image
        output_path (str, optional): Path to save the visualization
        conf_threshold (float): Confidence threshold for filtering detections
        pixel_to_cm_ratio (float): Number of pixels per centimeter
        spallation_class_id (int): Class ID for spallation in your model
        rebar_class_id (int): Class ID for exposed rebar in your model

    Returns:
        dict: Analysis results
    """
    # Load the model
    model = YOLO(model_path)

    # Load the image
    image = cv2.imread(image_path)

    # Run detection
    results = model(image)[0]

    # Initialize spallation quantifier
    quantifier = SpallationQuantifier()
    quantifier.set_calibration(pixel_to_cm_ratio)

    # Analyze spallation
    spallation_result = quantifier.analyze_spallation(
        image, results, conf_threshold, spallation_class_id, rebar_class_id
    )

    # Visualize if requested
    if output_path and spallation_result['detected']:
        vis_image = quantifier.visualize_results(image, spallation_result, output_path)

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
        plt.title('Spallation Quantification')
        plt.axis('off')

        # Add summary text
        plt.figtext(0.5, 0.01,
                    f"Area: {spallation_result['area_cm2']:.2f} cm² | " +
                    f"Depth: {spallation_result['estimated_depth_cm']:.2f} cm | " +
                    f"Severity: {spallation_result['severity']} | " +
                    f"Near Rebar: {'Yes' if spallation_result['near_rebar'] else 'No'}",
                    ha="center", fontsize=12, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    return spallation_result


def parse_args():
    parser = argparse.ArgumentParser(description='Spall detection and quantification using YOLOv8')

    # Required parameters
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the YOLOv8 model')
    parser.add_argument('--image-path', type=str, required=True,
                        help='Path to the input image')

    # Optional parameters
    parser.add_argument('--output-path', type=str, default=None,
                        help='Path to save the visualization (default: None)')
    parser.add_argument('--pixel-to-cm-ratio', type=float, default=10.0,
                        help='Number of pixels per millimeter (default: 10.0)')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='Confidence threshold for filtering detections (default: 0.25)')

    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_args()

    # Process the image
    results = process_image_with_spallation_quantification(
        model_path=args.model_path,
        image_path=args.image_path,
        output_path=args.output_path,
        conf_threshold=args.conf_threshold,
        pixel_to_cm_ratio=args.pixel_to_cm_ratio
    )

    # Print results
    print(f"Crack Analysis Results:")
    for key, value in results.items():
        if key not in ['mask', 'skeleton']:  # Skip binary images
            print(f"{key}: {value}")


# Example usage when run directly
if __name__ == "__main__":
    main()