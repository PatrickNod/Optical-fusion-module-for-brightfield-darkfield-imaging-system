import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from config import SystemConfig

class DualModeCoreProcessor:
    def __init__(self, transform_matrix: Optional[np.ndarray] = None) -> None:
        self.transform_matrix = transform_matrix
        self.config = SystemConfig()

    def preprocess_bright_field(self, bright_img: np.ndarray) -> np.ndarray:
        smoothed = cv2.GaussianBlur(bright_img, self.config.BLUR_KERNEL_SIZE, 0)
        background_illumination = cv2.GaussianBlur(smoothed, (99, 99), 0)
        uniform_img = cv2.divide(smoothed, background_illumination, scale=255.0)
        return cv2.convertScaleAbs(uniform_img)

    def preprocess_dark_field(self, dark_img: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.config.DARK_ELLIPSE_KERNEL)
        tophat = cv2.morphologyEx(dark_img, cv2.MORPH_TOPHAT, kernel)
        clahe = cv2.createCLAHE(clipLimit=self.config.CLAHE_CLIP_LIMIT, tileGridSize=self.config.CLAHE_TILE_GRID)
        return clahe.apply(tophat)

    def _dfs_connected_components(self, binary_mask: np.ndarray, min_pixels: int) -> List[Dict[str, Tuple[int, int]]]:
        h, w = binary_mask.shape
        visited = np.zeros_like(binary_mask, dtype=bool)
        bounding_boxes = []
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for i in range(h):
            for j in range(w):
                if binary_mask[i, j] > 0 and not visited[i, j]:
                    stack = [(i, j)]
                    visited[i, j] = True
                    pixels_count = 0
                    min_x, min_y = j, i
                    max_x, max_y = j, i

                    while stack:
                        cy, cx = stack.pop()
                        pixels_count += 1
                        min_x, max_x = min(min_x, cx), max(max_x, cx)
                        min_y, max_y = min(min_y, cy), max(max_y, cy)

                        for dy, dx in directions:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if binary_mask[ny, nx] > 0 and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    stack.append((ny, nx))
                    
                    if pixels_count >= min_pixels:
                        bounding_boxes.append({
                            'top_left': (min_x, min_y),
                            'bottom_right': (max_x, max_y)
                        })
        return bounding_boxes

    def detect_features(self, bright_img: np.ndarray, dark_img: np.ndarray) -> Tuple[List[Dict[str, Tuple[int, int]]], List[Dict[str, Tuple[int, int]]]]:
        pre_bright = self.preprocess_bright_field(bright_img)
        _, binary_bright = cv2.threshold(pre_bright, self.config.BRIGHT_THRESHOLD_S1, 255, cv2.THRESH_TOZERO)
        _, binary_bright = cv2.threshold(binary_bright, 0, 255, cv2.THRESH_BINARY)
        bright_boxes = self._dfs_connected_components(binary_bright, self.config.MIN_BRIGHT_PIXELS)

        pre_dark = self.preprocess_dark_field(dark_img)
        original_float = dark_img.astype(np.float32)
        local_mean = cv2.blur(original_float, self.config.LOCAL_MEAN_KERNEL)
        local_sq_mean = cv2.blur(original_float ** 2, self.config.LOCAL_MEAN_KERNEL)
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))
        
        snr_map = (pre_dark.astype(np.float32) - local_mean) / (local_std + 1e-5)
        signal_mask = np.zeros_like(pre_dark, dtype=np.uint8)
        signal_mask[snr_map >= self.config.DARK_SNR_THRESHOLD] = 255
        dark_boxes = self._dfs_connected_components(signal_mask, self.config.MIN_DARK_PIXELS)

        return dark_boxes, bright_boxes

    def calculate_iou(self, boxA: Dict[str, Tuple[int, int]], boxB: Dict[str, Tuple[int, int]]) -> float:
        xA = max(boxA['top_left'][0], boxB['top_left'][0])
        yA = max(boxA['top_left'][1], boxB['top_left'][1])
        xB = min(boxA['bottom_right'][0], boxB['bottom_right'][0])
        yB = min(boxA['bottom_right'][1], boxB['bottom_right'][1])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA['bottom_right'][0] - boxA['top_left'][0] + 1) * (boxA['bottom_right'][1] - boxA['top_left'][1] + 1)
        boxBArea = (boxB['bottom_right'][0] - boxB['top_left'][0] + 1) * (boxB['bottom_right'][1] - boxB['top_left'][1] + 1)

        denominator = float(boxAArea + boxBArea - interArea)
        if denominator == 0.0:
            return 0.0
        return interArea / denominator

    def process_image_pair(self, bright_img: np.ndarray, dark_img: np.ndarray) -> List[Dict[str, Any]]:
        dark_boxes, bright_boxes = self.detect_features(bright_img, dark_img)
        
        quantification_results = []
        for d_box in dark_boxes:
            if self.transform_matrix is not None:
                tl = np.array([[[d_box['top_left'][0], d_box['top_left'][1]]]], dtype=np.float32)
                br = np.array([[[d_box['bottom_right'][0], d_box['bottom_right'][1]]]], dtype=np.float32)
                
                try:
                    mapped_tl = cv2.perspectiveTransform(tl, self.transform_matrix)[0][0]
                    mapped_br = cv2.perspectiveTransform(br, self.transform_matrix)[0][0]
                    mapped_box = {
                        'top_left': (int(mapped_tl[0]), int(mapped_tl[1])), 
                        'bottom_right': (int(mapped_br[0]), int(mapped_br[1]))
                    }
                except Exception:
                    mapped_box = d_box
            else:
                mapped_box = d_box

            max_iou = 0.0
            for b_box in bright_boxes:
                iou = self.calculate_iou(mapped_box, b_box)
                if iou > max_iou:
                    max_iou = iou

            if max_iou > self.config.IOU_THRESHOLD:
                quantification_results.append({
                    'target': mapped_box, 
                    'mode': 'Bright-Field Grayscale',
                    'confidence': float(max_iou)
                })
            else:
                quantification_results.append({
                    'target': mapped_box, 
                    'mode': 'Dark-Field Count',
                    'confidence': float(1.0 - max_iou)
                })

        return quantification_results