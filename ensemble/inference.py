import os
import argparse
import cv2
import glob
from pathlib import Path
import json
import time
from wbf_ensemble import WBFEnsemble

def run_inference_on_image(ensemble, image_path, output_dir, visualize=True):
    print(f"\nProcessing: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    # Run prediction
    start_time = time.time()
    predictions = ensemble.predict(image)
    inference_time = time.time() - start_time
    
    print(f"Inference time: {inference_time:.3f}s")
    print(f"Detections: {len(predictions['boxes'])}")
    
    # Print detections
    for i, (box, label, score, class_name) in enumerate(zip(
        predictions['boxes'],
        predictions['labels'],
        predictions['scores'],
        predictions['class_names']
    )):
        print(f"  {i+1}. {class_name}: {score:.3f} at {box.astype(int).tolist()}")
    
    images_dir = os.path.join(output_dir, 'images')
    predictions_dir = os.path.join(output_dir, 'img_predictions')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Save visualization
    if visualize:
        image_name = Path(image_path).stem
        vis_path = os.path.join(images_dir, f"{image_name}_ensemble.jpg")
        ensemble.visualize_predictions(image, predictions, save_path=vis_path)
    
    # Save predictions as JSON
    image_name = Path(image_path).stem
    json_path = os.path.join(predictions_dir, f"{image_name}_predictions.json")
    
    # Convert numpy arrays to lists for JSON serialization
    json_data = {
        'image_path': image_path,
        'inference_time': inference_time,
        'detections': [
            {
                'class_name': class_name,
                'class_id': int(label),
                'confidence': float(score),
                'bbox': box.tolist()
            }
            for box, label, score, class_name in zip(
                predictions['boxes'],
                predictions['labels'],
                predictions['scores'],
                predictions['class_names']
            )
        ]
    }
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Results saved to {json_path}")
    
    return predictions


def run_inference_on_directory(ensemble, input_dir, output_dir, visualize=True):
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Process each image
    all_results = []
    total_time = 0
    
    for image_path in image_paths:
        result = run_inference_on_image(ensemble, image_path, output_dir, visualize)
        if result:
            all_results.append({
                'image': os.path.basename(image_path),
                'num_detections': len(result['boxes'])
            })
    
    # Save summary
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'total_images': len(image_paths),
            'results': all_results
        }, f, indent=2)
    
    print(f"\nProcessing complete! Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Run ensemble inference on images')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input image or directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Directory to save results (default: outputs)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Confidence threshold (default: 0.5)'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.5,
        help='IoU threshold for WBF (default: 0.5)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run inference on (default: cuda)'
    )
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Skip visualization generation'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize ensemble
    print("Initializing ensemble model...")
    ensemble = WBFEnsemble(
        device=args.device,
        confidence_threshold=args.confidence
    )
    
    # Run inference
    if os.path.isfile(args.input):
        # Single image
        run_inference_on_image(
            ensemble,
            args.input,
            args.output,
            visualize=not args.no_visualize
        )
    elif os.path.isdir(args.input):
        # Directory of images
        run_inference_on_directory(
            ensemble,
            args.input,
            args.output,
            visualize=not args.no_visualize
        )
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return
    
    print("\nDone!")


if __name__ == "__main__":
    main()