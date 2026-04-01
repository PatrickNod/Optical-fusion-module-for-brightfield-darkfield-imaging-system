import os
from dataclasses import dataclass

@dataclass
class SystemConfig:
    BRIGHT_THRESHOLD_S1: int = 30
    MIN_BRIGHT_PIXELS: int = 10
    DARK_SNR_THRESHOLD: float = 10.0
    MIN_DARK_PIXELS: int = 1
    IOU_THRESHOLD: float = 0.65
    MAX_WORKERS: int = 8

    BLUR_KERNEL_SIZE: tuple = (5, 5)
    TOPHAT_KERNEL_SIZE: tuple = (50, 50)
    DARK_ELLIPSE_KERNEL: tuple = (5, 5)
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_GRID: tuple = (8, 8)
    LOCAL_MEAN_KERNEL: tuple = (15, 15)

    OUTPUT_DIR: str = "results"
    LOG_LEVEL: str = "INFO"