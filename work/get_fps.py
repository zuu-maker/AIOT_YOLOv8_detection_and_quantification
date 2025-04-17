import time
import numpy as np
from ultralytics import YOLO, RTDETR
import torch
import cv2
import argparse


def benchmark_inference_speed(model_path, img_size=640, warmup=10, num_iterations=100, device=None):
    """
    Benchmark the inference speed of a YOLO model.

    Args:
        model_path (str): Path to the YOLO model (.pt file)
        img_size (int): Input image size
        warmup (int): Number of warmup iterations
        num_iterations (int): Number of iterations to average over
        device (str): Device to run inference on ('cpu', 'cuda', 'mps', None=auto)

    Returns:
        dict: Dictionary containing benchmark results
    """
    # Load the model
    model = YOLO(model_path)
    # model = RTDETR(model_path)

    # Set device if specified
    if device:
        model.to(device)

    # Create a dummy input (random noise)
    dummy_input = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

    # Warmup
    print(f"Warming up for {warmup} iterations...")
    for _ in range(warmup):
        _ = model(dummy_input)

    # Benchmark
    print(f"Running benchmark for {num_iterations} iterations...")
    inference_times = []

    for i in range(num_iterations):
        start_time = time.time()
        _ = model(dummy_input)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_iterations} iterations")

    # Calculate statistics
    inference_times = np.array(inference_times)
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)

    # Calculate FPS and ms
    fps = 1.0 / avg_time
    ms_per_frame = avg_time * 1000

    # Prepare results
    results = {
        "model": model_path,
        "device": model.device,
        "img_size": img_size,
        "iterations": num_iterations,
        "avg_time_s": avg_time,
        "std_time_s": std_time,
        "min_time_s": min_time,
        "max_time_s": max_time,
        "ms_per_frame": ms_per_frame,
        "fps": fps
    }

    return results



def print_results(results):
    """Print the benchmark results in a readable format."""
    print("\n" + "=" * 50)
    print(f"YOLO Model Inference Speed Benchmark")
    print("=" * 50)
    print(f"Model: {results['model']}")
    print(f"Device: {results['device']}")
    print(f"Image Size: {results['img_size']}x{results['img_size']}")
    print(f"Iterations: {results['iterations']}")
    print("-" * 50)
    print(f"Average Inference Time: {results['avg_time_s']:.4f} seconds")
    print(f"Standard Deviation: {results['std_time_s']:.4f} seconds")
    print(f"Minimum Time: {results['min_time_s']:.4f} seconds")
    print(f"Maximum Time: {results['max_time_s']:.4f} seconds")
    print("-" * 50)
    print(f"Inference Speed: {results['ms_per_frame']:.2f} ms/frame")
    print(f"Throughput: {results['fps']:.2f} FPS")
    print("=" * 50)


def benchmark_with_video(model_path, video_path=None, duration=10, device=None):
    """
    Benchmark the inference speed of a YOLO model using a real video.

    Args:
        model_path (str): Path to the YOLO model (.pt file)
        video_path (str): Path to video file (if None, use webcam)
        duration (int): Duration of benchmark in seconds
        device (str): Device to run inference on ('cpu', 'cuda', 'mps', None=auto)

    Returns:
        dict: Dictionary containing benchmark results
    """
    # Load the model
    model = YOLO(model_path)
    # model = RTDETR(model_path)

    # Set device if specified
    if device:
        model.to(device)

    # Open video source
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)  # Use webcam

    if not cap.isOpened():
        raise ValueError("Error opening video source")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    inference_times = []
    start_benchmark = time.time()
    frames_processed = 0

    print(f"Running video benchmark for {duration} seconds...")

    while (time.time() - start_benchmark) < duration:
        ret, frame = cap.read()
        if not ret:
            if video_path:  # If using a file and reached the end, loop back
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        # Run inference and time it
        start_time = time.time()
        _ = model(frame)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        frames_processed += 1

        # Simple progress indicator
        if frames_processed % 10 == 0:
            elapsed = time.time() - start_benchmark
            print(f"Processed {frames_processed} frames in {elapsed:.2f} seconds")

    # Release resources
    cap.release()

    # Calculate statistics
    if not inference_times:
        return {"error": "No frames were processed"}

    inference_times = np.array(inference_times)
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)

    # Calculate FPS and ms
    fps = 1.0 / avg_time
    ms_per_frame = avg_time * 1000

    results = {
        "model": model_path,
        "device": model.device,
        "frame_size": f"{frame_width}x{frame_height}",
        "frames_processed": frames_processed,
        "benchmark_duration": time.time() - start_benchmark,
        "avg_time_s": avg_time,
        "std_time_s": std_time,
        "min_time_s": min_time,
        "max_time_s": max_time,
        "ms_per_frame": ms_per_frame,
        "fps": fps
    }

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="fps parser")

    parser.add_argument("--model-path", type=str, default="pretraind_models/pm.pt")
    parser.add_argument("--model-type", type=str, default="YOLO", choices=["YOLO", "RTDETR"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--img-size", type=int, default=640)
    return parser.parse_args()

def main():
    args = parse_args()
    # model_path = "/home/chengjie/Desktop/mk/yolos/yolov8/final_models/yolov8-att-p23/weights/best.pt"  # Change to your model path
    results = benchmark_inference_speed(
        model_path=args.model_path,
        img_size=args.img_size,
        warmup=10,
        num_iterations=100,
        device='cuda'  # Set to 'cpu', 'cuda', or 'mps' if needed
    )
    print_results(results)

if __name__ == "__main__":
    main()

    # Choose one of these benchmark methods:
    # 1. Basic benchmark with random data
    # model_path = "/home/chengjie/Desktop/mk/yolos/yolov8/final_models/yolov8-att-p23/weights/best.pt"  # Change to your model path
    # results = benchmark_inference_speed(
    #     model_path=model_path,
    #     img_size=640,
    #     warmup=10,
    #     num_iterations=100,
    #     device='cuda'  # Set to 'cpu', 'cuda', or 'mps' if needed
    # )
    # print_results(results)

    # 2. Benchmark with video
    # video_results = benchmark_with_video(
    #     model_path=model_path,
    #     video_path=None,  # Set to video path or None for webcam
    #     duration=10,
    #     device=None  # Set to 'cpu', 'cuda', or 'mps' if needed
    # )
    # print_results(video_results)