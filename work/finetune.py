import argparse
from ultralytics import YOLO, RTDETR


def parse_args():
    parser = argparse.ArgumentParser(description='Train a YOLO or RT-DETR model')

    # Model parameters
    parser.add_argument('--model-type', type=str, default='YOLO', choices=['YOLO', 'RTDETR'],
                        help='Model type to train (YOLO or RTDETR)')
    parser.add_argument('--model-cfg', type=str, default='models_cfg/yolov8-p2_1_att.yaml',
                        help='Path to model configuration file (.yaml)')
    parser.add_argument('--model-weights', type=str, default='pretraind_models/pm.pt',
                        help='Path to pretrained weights (.pt)')

    # Training parameters
    parser.add_argument('--data', type=str, default='dataset.yaml',
                        help='Path to dataset configuration file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--name', type=str, default='exp',
                        help='Experiment name')
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Project name')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        help='Optimizer (AdamW, SGD, etc.)')
    parser.add_argument('--lr0', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01,
                        help='Final learning rate factor')

    # Additional options
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--plots', action='store_true',
                        help='Generate plots after training')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize model features during training')

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize model based on type
    if args.model_type == 'YOLO':
        if args.model_cfg:
            model = YOLO(args.model_cfg, verbose=args.verbose)
        else:
            model = YOLO(args.model_weights, verbose=args.verbose)
    else:  # RTDETR
        model = RTDETR(args.model_weights)

    # Load weights if specified and model_cfg was used
    if args.model_weights and args.model_cfg:
        model.load(args.model_weights)

    # Train the model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project=args.project,
        verbose=args.verbose,
        plots=args.plots,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        visualize=args.visualize
    )

    print("-------------------------------------------------------")
    print(f"{args.model_type} training complete")


if __name__ == "__main__":
    main()
