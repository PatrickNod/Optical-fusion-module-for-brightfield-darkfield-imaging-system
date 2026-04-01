import cv2
import time
import concurrent.futures
import numpy as np
import argparse
import json
import logging
from pathlib import Path
from core_processor import DualModeCoreProcessor
from config import SystemConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def process_single_task(task_id: int, bright_path: str, dark_path: str, transform_matrix: np.ndarray) -> tuple:
    try:
        bright_img = cv2.imread(bright_path, cv2.IMREAD_GRAYSCALE)
        dark_img = cv2.imread(dark_path, cv2.IMREAD_GRAYSCALE)
        
        if bright_img is None or dark_img is None:
            raise FileNotFoundError(f"Missing image files for task {task_id}")
        
        processor = DualModeCoreProcessor(transform_matrix)
        results = processor.process_image_pair(bright_img, dark_img)
        
        return task_id, len(results), results
    except Exception as e:
        logging.error(f"Task {task_id} failed: {str(e)}")
        return task_id, 0, []

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=SystemConfig.MAX_WORKERS)
    parser.add_argument("--output", type=str, default=SystemConfig.OUTPUT_DIR)
    args = parser.parse_args()

    logging.info("=== Dual-Mode Line-Scanning Dark-Field Scattering Imaging System Started ===")
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    global_transform_matrix = np.eye(3, dtype=np.float32)
    
    simulated_image_queue = [
        {"id": i, "bright": f"data/bright_chunk_{i}.png", "dark": f"data/dark_chunk_{i}.png"} 
        for i in range(10)
    ]
    
    start_time = time.time()
    total_particles_found = 0
    all_results = {}
    
    logging.info(f"Starting high-throughput parallel processing pool, worker threads: {args.workers}...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {
            executor.submit(process_single_task, item["id"], item["bright"], item["dark"], global_transform_matrix): item 
            for item in simulated_image_queue
        }
        
        for future in concurrent.futures.as_completed(future_to_task):
            task_id, result_count, details = future.result()
            total_particles_found += result_count
            all_results[task_id] = details
            logging.info(f"FOV chunk {task_id} processing completed, targets identified: {result_count}")
            
    report_file = output_path / "final_report.json"
    with open(report_file, "w") as f:
        json.dump(all_results, f, indent=4)
        
    elapsed_time = time.time() - start_time
    logging.info("======================================")
    logging.info(f"Analysis complete! Processed {len(simulated_image_queue)} image pairs.")
    logging.info(f"Total precipitate targets identified across all FOVs: {total_particles_found}")
    logging.info(f"Results exported to: {report_file}")
    logging.info(f"Total time elapsed: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()