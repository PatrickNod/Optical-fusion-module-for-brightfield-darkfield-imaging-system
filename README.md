# Optical-fusion-module-for-brightfield-darkfield-imaging-system
The implementation code of the optical processing fusion module in the patent "A Dark Field Scattering Imaging System and Its Immunoassay Method".

## Project Structure and File Descriptions

* **config.py**: Stores all system configurations and optical hyperparameters. This includes the binarization thresholds for bright-field images, the Signal-to-Noise Ratio (SNR) thresholds for dark-field image signal extraction, the Intersection over Union (IoU) thresholds for fusion decision-making, and the multi-processing worker allocations.
* **core_processor.py**: Contains the core image processing and optical fusion algorithms. It implements the sub-pixel spatial transformation matrix, bright-field background illumination normalization, dark-field morphological enhancement, depth-first search (DFS) connected component analysis, and the dynamic quantification switching logic between bright-field grayscale data and dark-field particle counting.
* **main.py**: Serves as the high-throughput execution entry point for the system. It utilizes a multiprocessing pool architecture to handle continuous line-scanning data streams, dispatching image pairs to multiple CPU cores for parallel processing, and aggregates the final quantitative detection reports.

## License and Usage Rights

This open-source code belongs to the patented content and is strictly protected by patent law. Please contact the author to obtain formal authorization prior to any commercial use.
