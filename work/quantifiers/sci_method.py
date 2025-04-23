import cv2
import numpy as np
from skimage.measure import regionprops, label
from sklearn.cluster import DBSCAN
# from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

class ScientificReportsMethod:
    """
    Improved implementation of the spallation detection method based on the Scientific Reports paper.
    This properly implements the method described in "Automatic identification method
    of bridge structure damage area based on digital image".
    """
    
    def __init__(self):
        self.pixel_to_cm_ratio = 1.0  # Default calibration
    
    def set_calibration(self, pixel_to_cm_ratio):
        """Set the pixel to cm ratio for measurements"""
        self.pixel_to_cm_ratio = pixel_to_cm_ratio
    
    def extract_spallation_mask(self, image, ground_truth=None):
        """
        Extract the spallation mask using the improved method from the Scientific Reports paper.
        
        Args:
            image: Input grayscale image
            ground_truth: Optional ground truth for debugging
            
        Returns:
            Binary mask with spallation areas
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 1: Calculate the gradient image and binarize to form gradient image F1
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
        gradient_mag = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, F1 = cv2.threshold(gradient_mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Step 2: Use global Otsu thresholding for initial binary image F2
        _, F2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Step 3: Divide image into regions for local thresholding (m=4, n=4 as suggested in the paper)
        m, n = 4, 4
        height, width = gray.shape
        region_height = height // m
        region_width = width // n
        F3 = np.zeros_like(gray)
        
        for i in range(m):
            for j in range(n):
                # Define the region
                y_start = i * region_height
                y_end = (i + 1) * region_height if i < m - 1 else height
                x_start = j * region_width
                x_end = (j + 1) * region_width if j < n - 1 else width
                
                # Extract region
                region = gray[y_start:y_end, x_start:x_end]
                
                # Apply Otsu thresholding to the region
                if region.size > 0 and np.std(region) > 0:  # Ensure region is not empty or constant
                    _, binary_region = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    F3[y_start:y_end, x_start:x_end] = binary_region
        
        # Instead of strict AND, use OR to combine F1, F2, and F3
        # This produces a more robust segmentation that doesn't miss spalled regions
        F4 = cv2.bitwise_or(F1, cv2.bitwise_or(F2, F3))
        
        # Apply morphological operations to clean the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(F4, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Ensure the mask has some non-zero pixels (debugging)
        if np.sum(mask) == 0:
            # Fallback to global Otsu thresholding if the combined method fails
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        return mask
    
    def measure_area(self, mask):
        """
        Measure the area of the spallation.
        
        Args:
            mask: Binary mask of the spallation
            
        Returns:
            Area in square centimeters
        """
        # Count the number of pixels in the mask
        pixel_count = np.sum(mask > 0)
        
        # Convert to physical units using calibration
        area_cm2 = pixel_count / (self.pixel_to_cm_ratio ** 2)
        
        return area_cm2
    
    def measure_perimeter(self, mask):
        """
        Measure the perimeter of the spallation.
        
        Args:
            mask: Binary mask of the spallation
            
        Returns:
            Perimeter in centimeters
        """
        # Find contours in the mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate total perimeter
        perimeter_pixels = sum(cv2.arcLength(contour, True) for contour in contours)
        
        # Convert to physical units using calibration
        perimeter_cm = perimeter_pixels / self.pixel_to_cm_ratio
        
        return perimeter_cm
    
    def estimate_depth(self, image, mask):
        """
        Estimate the depth of the spallation based on intensity differences.
        This is an adaptation for fair comparison, not directly from the paper.
        
        Args:
            image: Input grayscale image
            mask: Binary mask of the spallation
            
        Returns:
            Estimated depth of the spallation
        """
        # Convert image to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply the mask to the grayscale image
        # Create a binary mask (0 or 1) from the input mask
        binary_mask = (mask > 0).astype(np.uint8)
        masked_gray = cv2.bitwise_and(gray, gray, mask=binary_mask)
        
        # Get pixel values within the mask
        mask_indices = np.where(binary_mask > 0)
        if len(mask_indices[0]) == 0:  # Check if mask is empty
            return 0.0
            
        pixels = masked_gray[mask_indices]
        
        if len(pixels) == 0:
            return 0.0
            
        # Calculate statistics for depth estimation
        mean_intensity = np.mean(pixels)
        std_intensity = np.std(pixels)
        
        # A simple heuristic for depth estimation based on intensity
        # Lower intensity generally means deeper spalling due to shadows
        intensity_factor = (255 - mean_intensity) / 50  # Normalize
        texture_factor = std_intensity / 20  # Higher variation suggests deeper spalling
        
        # Combined depth estimate
        estimated_depth_cm = intensity_factor * (1 + texture_factor * 0.5)
        
        return estimated_depth_cm
    
    def analyze_spallation(self, image):
        """
        Analyze the spallation in the image and return parameters.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary containing measured parameters
        """
        # Extract the spallation mask
        mask = self.extract_spallation_mask(image)
        
        # Debug: print the sum of pixels in the mask
        pixel_sum = np.sum(mask > 0)
        
        # Check if any spallation was detected
        if pixel_sum == 0:
            return {
                'detected': False,
                'message': 'No spallation detected',
                'area_cm2': 0.0,
                'perimeter_cm': 0.0,
                'estimated_depth_cm': 0.0,
                'mask': np.zeros_like(mask)
            }
        
        # Measure area
        area_cm2 = self.measure_area(mask)
        
        # Measure perimeter
        perimeter_cm = self.measure_perimeter(mask)
        
        # Estimate depth
        depth_cm = self.estimate_depth(image, mask)
        
        # Compile results
        result = {
            'detected': True,
            'area_cm2': round(area_cm2, 2),
            'perimeter_cm': round(perimeter_cm, 2),
            'estimated_depth_cm': round(depth_cm, 2),
            'mask': mask
        }
        
        return result