import numpy as np
import cv2
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

def make_realistic_mask(binary_mask, random_state=None):
    """
    Convert a perfect binary mask to a more realistic detection mask with deterministic behavior.
    
    Parameters:
    -----------
    binary_mask : ndarray
        Binary mask input
    random_state : int or np.random.RandomState
        Random state for reproducibility
        
    Returns:
    --------
    realistic_mask : ndarray
        Mask with realistic noise and artifacts
    """
    # Set up random state for reproducibility
    if random_state is None:
        rng = np.random.RandomState(42)  # Fixed seed
    elif isinstance(random_state, int):
        rng = np.random.RandomState(random_state)
    else:
        rng = random_state
    
    # Add blur to simulate soft boundaries
    blurred = cv2.GaussianBlur(binary_mask, (5, 5), 2)
    
    # Add some noise (reproducible)
    noise = rng.normal(0, 10, binary_mask.shape).astype(np.int16)
    noisy = np.clip(blurred.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Randomly add small artifacts (reproducible)
    if rng.random() < 0.3:
        # Add some small false positives
        for _ in range(rng.randint(1, 5)):
            x = rng.randint(0, binary_mask.shape[1])
            y = rng.randint(0, binary_mask.shape[0])
            size = rng.randint(3, 10)
            cv2.circle(noisy, (x, y), size, rng.randint(100, 255), -1)
    
    # Randomly remove small regions (reproducible)
    if rng.random() < 0.3:
        kernel = np.ones((3, 3), np.uint8)
        noisy = cv2.erode(noisy, kernel, iterations=1)
    
    return noisy

def deterministic_cross_validate(dataset, methods_list, output_dir=None, random_seed=42):
    """
    Deterministic cross-validation of spallation quantification methods.
    
    Parameters:
    -----------
    dataset : dict
        Dictionary containing images, masks, and ground truth
    methods_list : list of tuples
        List of (method_name, method_instance) tuples
    output_dir : str
        Directory to save validation results
    random_seed : int
        Seed for random number generator
        
    Returns:
    --------
    results : dict
        Dictionary containing validation results
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    random_state = np.random.RandomState(random_seed)
    
    results = {
        "methods": {},
        "metrics": {},
        "time": {}
    }
    
    # Pre-generate realistic masks for all samples - this is key for reproducibility
    realistic_masks = []
    for i, mask in enumerate(dataset["masks"]):
        realistic_mask = make_realistic_mask(mask, random_state=random_state)
        realistic_masks.append(realistic_mask)
        
        if output_dir:
            cv2.imwrite(os.path.join(output_dir, f'realistic_mask_{i:03d}.png'), realistic_mask)
    
    # Process each method
    for method_name, method_instance in methods_list:
        print(f"Evaluating {method_name}...")
        method_results = []
        method_times = []
        
        # Process each image
        for i, (img, mask, ground_truth, realistic_mask) in enumerate(zip(
                dataset["images"], dataset["masks"], dataset["ground_truth"], realistic_masks)):
            
            # Set calibration based on ground truth
            if hasattr(method_instance, 'set_calibration'):
                method_instance.set_calibration(ground_truth['pixel_to_cm_ratio'])
            
            # Create a binary version of the mask (ensure it's 0 or 255)
            binary_mask = np.zeros_like(realistic_mask)
            binary_mask[realistic_mask > 128] = 255
            
            # Time the method
            start_time = time.time()
            
            # Direct control of the area calculation
            if method_name == "Custom Method":
                # Create mock detection for SpallationQuantifier
                class MockDetection:
                    def __init__(self, mask):
                        self.boxes = MockBoxes()
                        self.masks = MockMasks(mask)
                    
                    def __len__(self):
                        return 1
                
                class MockBoxes:
                    def __init__(self):
                        self.data = np.array([[0, 0, mask.shape[1], mask.shape[0], 1.0, 2]])
                    
                    def __len__(self):
                        return len(self.data)
                
                class MockMasks:
                    def __init__(self, mask):
                        class MockTensor:
                            def __init__(self, data):
                                self.data = data
                            
                            def cpu(self):
                                return self
                            
                            def numpy(self):
                                return self.data
                        
                        normalized_mask = mask.astype(np.float32) / 255.0
                        self.data = [MockTensor(normalized_mask)]
                
                mock_detection = MockDetection(binary_mask)
                result = method_instance.analyze_spallation(img, mock_detection)
            
            elif method_name == "Scientific Reports Method" or method_name == "German Method":
                # Calculate area directly using the method's measure_area function
                measured_area_cm2 = method_instance.measure_area(binary_mask)
                measured_perimeter_cm = method_instance.measure_perimeter(binary_mask)
                
                if method_name == "German Method":
                    # German method includes depth estimation
                    reinforcement_mask = method_instance.detect_longitudinal_reinforcement(img, binary_mask)
                    depth_category, depth_cm = method_instance.estimate_depth(img, binary_mask, reinforcement_mask)
                    
                    # Create a result dict with depth measurement
                    result = {
                        'detected': True,
                        'area_cm2': measured_area_cm2,
                        'perimeter_cm': measured_perimeter_cm,
                        'length_cm': method_instance.measure_extent(binary_mask),
                        'estimated_depth_cm': depth_cm,
                        'depth_category': depth_category,
                        'reinforcement_exposed': np.sum(reinforcement_mask) > 0,
                        'mask': binary_mask,
                        'reinforcement_mask': reinforcement_mask
                    }
                else:  # Scientific Reports method
                    # Create result without depth estimation
                    result = {
                        'detected': True,
                        'area_cm2': measured_area_cm2,
                        'perimeter_cm': measured_perimeter_cm,
                        'mask': binary_mask
                    }
                            
            end_time = time.time()
            
            # Record processing time
            processing_time = end_time - start_time
            method_times.append(processing_time)
            
            # Store result and ground truth
            result["ground_truth"] = ground_truth
            method_results.append(result)
            
            # Print first few results for debugging
            if i < 3:
                print(f"  Sample {i}:")
                print(f"    Ground Truth - Area: {ground_truth['area_cm2']:.2f} cm², Depth: {ground_truth['depth_cm']:.2f} cm")
                print(f"    {method_name} - Area: {result.get('area_cm2', 0):.2f} cm², Depth: {result.get('estimated_depth_cm', 0):.2f} cm")
        
        # Store results for this method
        results["methods"][method_name] = method_results
        
        # Calculate average processing time
        if method_times:
            results["time"][method_name] = {
                "mean": np.mean(method_times),
                "std": np.std(method_times),
                "min": np.min(method_times),
                "max": np.max(method_times)
            }
        else:
            results["time"][method_name] = {
                "mean": 0,
                "std": 0,
                "min": 0,
                "max": 0
            }
    
    # Calculate metrics
    metrics = {}
    
    # Area accuracy comparison
    area_percent_errors = {}
    area_mse = {}
    area_mae = {}
    
    # Depth accuracy comparison
    depth_percent_errors = {}
    depth_mse = {}
    depth_mae = {}
    
    for method_name, method_results in results["methods"].items():
        # Area evaluation
        true_areas = []
        predicted_areas = []
        area_percent_errors_list = []
        
        # Depth evaluation
        true_depths = []
        predicted_depths = []
        depth_percent_errors_list = []
        
        for result in method_results:
            if "ground_truth" not in result:
                continue
                
            gt = result["ground_truth"]
            
            # Area evaluation
            if "area_cm2" in result and result.get("detected", False):
                true_area = gt["area_cm2"]
                predicted_area = result["area_cm2"]
                
                # Store values for calculating metrics
                true_areas.append(true_area)
                predicted_areas.append(predicted_area)
                
                # Calculate percent error
                percent_error = abs(predicted_area - true_area) / true_area if true_area > 0 else 0
                area_percent_errors_list.append(percent_error)
            
            # Depth evaluation
            if "estimated_depth_cm" in result and result.get("detected", False):
                true_depth = gt["depth_cm"]
                predicted_depth = result["estimated_depth_cm"]
                
                # Store values for calculating metrics
                true_depths.append(true_depth)
                predicted_depths.append(predicted_depth)
                
                # Calculate percent error
                percent_error = abs(predicted_depth - true_depth) / true_depth if true_depth > 0 else 0
                depth_percent_errors_list.append(percent_error)
        
        # Print comparison of true vs predicted
        print(f"  {method_name} true vs predicted areas (first 3 samples):")
        for i in range(min(3, len(true_areas))):
            print(f"    Sample {i}: True={true_areas[i]:.2f}, Predicted={predicted_areas[i]:.2f}, " +
                  f"Error={abs(true_areas[i]-predicted_areas[i]):.2f} cm², " +
                  f"Percent Error={abs(true_areas[i]-predicted_areas[i])/true_areas[i]*100:.2f}%")
        
        # Calculate area metrics
        if true_areas and predicted_areas:
            # Calculate MSE and MAE
            area_mse_value = mean_squared_error(true_areas, predicted_areas)
            area_mae_value = mean_absolute_error(true_areas, predicted_areas)
            
            # Calculate percent error statistics
            area_percent_errors[method_name] = {
                "mean": np.mean(area_percent_errors_list) * 100 if area_percent_errors_list else 0,
                "std": np.std(area_percent_errors_list) * 100 if area_percent_errors_list else 0,
                "min": np.min(area_percent_errors_list) * 100 if area_percent_errors_list else 0,
                "max": np.max(area_percent_errors_list) * 100 if area_percent_errors_list else 0
            }
            
            # Store MSE and MAE
            area_mse[method_name] = area_mse_value
            area_mae[method_name] = area_mae_value
        
        # Calculate depth metrics
        if true_depths and predicted_depths:
            # Calculate MSE and MAE
            depth_mse_value = mean_squared_error(true_depths, predicted_depths)
            depth_mae_value = mean_absolute_error(true_depths, predicted_depths)
            
            # Calculate percent error statistics
            depth_percent_errors[method_name] = {
                "mean": np.mean(depth_percent_errors_list) * 100 if depth_percent_errors_list else 0,
                "std": np.std(depth_percent_errors_list) * 100 if depth_percent_errors_list else 0,
                "min": np.min(depth_percent_errors_list) * 100 if depth_percent_errors_list else 0,
                "max": np.max(depth_percent_errors_list) * 100 if depth_percent_errors_list else 0
            }
            
            # Store MSE and MAE
            depth_mse[method_name] = depth_mse_value
            depth_mae[method_name] = depth_mae_value
    
    # Store all metrics
    metrics["area_error_percent"] = area_percent_errors
    metrics["area_mse"] = area_mse
    metrics["area_mae"] = area_mae
    metrics["depth_error_percent"] = depth_percent_errors
    metrics["depth_mse"] = depth_mse
    metrics["depth_mae"] = depth_mae
    results["metrics"] = metrics
    
    # Generate visualization charts if needed
    if output_dir:
        # Time comparison
        plt.figure(figsize=(10, 6))
        method_names = list(results["time"].keys())
        mean_times = [results["time"][name]["mean"] for name in method_names]
        std_times = [results["time"][name]["std"] for name in method_names]
        
        plt.bar(method_names, mean_times, yerr=std_times, capsize=5)
        plt.xlabel('Method')
        plt.ylabel('Processing Time (seconds)')
        plt.title('Method Processing Time Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_comparison.png'))
        plt.close()
        
        # Area error comparison (percent)
        plt.figure(figsize=(10, 6))
        method_names = list(area_percent_errors.keys())
        mean_errors = [area_percent_errors[name]["mean"] for name in method_names]
        std_errors = [area_percent_errors[name]["std"] for name in method_names]
        
        plt.bar(method_names, mean_errors, yerr=std_errors, capsize=5)
        plt.xlabel('Method')
        plt.ylabel('Area Error (%)')
        plt.title('Area Measurement Percent Error Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'area_error_percent_comparison.png'))
        plt.close()

    # Display final results
    print("\n===== FINAL RESULTS: MEAN ABSOLUTE ERROR =====")
    print("Area MAE (cm²):")
    for method_name in sorted(area_mae.keys()):
        print(f"  {method_name}: {area_mae[method_name]:.4f}")

    print("\nDepth MAE (cm): (Custom Method and German Method only)")
    for method_name in ["Custom Method", "German Method"]:
        if method_name in depth_mae and depth_mae[method_name] > 0:
            print(f"  {method_name}: {depth_mae[method_name]:.4f}")

    print("\nArea Percent Error (%):")
    for method_name in sorted(area_percent_errors.keys()):
        print(f"  {method_name}: {area_percent_errors[method_name]['mean']:.2f}% ± {area_percent_errors[method_name]['std']:.2f}%")

    print("\nDepth Percent Error (%): (Custom Method and German Method only)")
    for method_name in ["Custom Method", "German Method"]:
        if method_name in depth_percent_errors and depth_percent_errors[method_name]["mean"] > 0:
            print(f"  {method_name}: {depth_percent_errors[method_name]['mean']:.2f}% ± {depth_percent_errors[method_name]['std']:.2f}%")
    
    return results

def save_validation_results(results, filename='validation_results.pkl'):
    """Save validation results to a file"""
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"Validation results saved to {filename}")

def load_validation_results(filename='validation_results.pkl'):
    """Load validation results from a file"""
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    print(f"Validation results loaded from {filename}")
    return results

# You should add this function to your main script
def run_deterministic_validation(dataset_file="saved_dataset_use.pkl", 
                               output_dir="spallation_validation_results",
                               random_seed=42):
    """
    Run a deterministic validation with the saved dataset.
    
    Parameters:
    -----------
    dataset_file : str
        Path to the saved dataset file
    output_dir : str
        Directory to save validation results
    random_seed : int
        Random seed for reproducibility
    """
    from spall_quantifier import SpallationQuantifier
    from sci_method import ScientificReportsMethod
    
    # Load the dataset
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)
    print(f"Loaded dataset with {len(dataset['images'])} synthetic spallation images")
    
    # Create instances of all methods
    custom_method = SpallationQuantifier()
    scientific_reports_method = ScientificReportsMethod()
    
    # Define methods to compare
    methods = [
        ("Custom Method", custom_method),
        ("Scientific Reports Method", scientific_reports_method),
    ]
    
    # Run deterministic cross-validation
    print(f"\nRunning deterministic cross-validation with random seed {random_seed}...")
    results = deterministic_cross_validate(
        dataset, 
        methods, 
        output_dir=output_dir,
        random_seed=random_seed
    )
    
    # Save results for future reference
    save_validation_results(results, os.path.join(output_dir, f'validation_results_seed{random_seed}.pkl'))
    
    return results

# Use this in the main script
if __name__ == "__main__":
    results = run_deterministic_validation()