import cv2
from ultralytics import YOLO
import numpy as np

class BatSegmentation:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.baseball_bat_class_id = 34  # COCO dataset index for baseball bat

    def segment_bat(self, image):
        results = self.model(image, task='segment')

        # Get the first result (assuming batch size of 1)
        result = results[0]
        
        # Check if masks are present in the results
        if result.masks is None:
            print("No masks found in the result.")
            return None

        masks = result.masks.data.cpu().numpy()  # Get masks
        print(f"Total masks found: {len(masks)}")

        for i, (mask, cls) in enumerate(zip(masks, result.boxes.cls)):
            print(f"Processing mask {i+1}/{len(masks)}, Class: {int(cls)}")
            if int(cls) == self.baseball_bat_class_id:
                mask = mask.astype(np.uint8)
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                colored_mask = np.zeros_like(image)
                colored_mask[mask_resized == 1] = [0, 255, 0]  # Green mask
                image = cv2.addWeighted(image, 1, colored_mask, 0.5, 0)
                print("Baseball bat mask applied.")

        return image
