import os
import cv2
import numpy as np
import torch
import re
import random
from collections import defaultdict

class ColormaskNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "folder": ("STRING", {"widget": "folder"}),
            "index": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            "random_index": ("BOOLEAN", {"default": False}),
            "resize_mode": (["keep original", "resize", "resize and crop"], {"default": "keep original"}),
            "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
            "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
        }}

    RETURN_TYPES = (
        "MASK", "MASK", "MASK", "MASK", "MASK", "MASK",
        "MASK", "MASK", "MASK", "MASK", "MASK", "MASK",
        "MASK", "MASK", "MASK", "IMAGE", "IMAGE", "IMAGE","STRING"
    )
    RETURN_NAMES = (
        "female_body", "female_face", "female_hair","female_breast", "female_hands", "female_vag",
        "male_body", "male_face",  "male_hair", "male_hands", "male_pen", "male_pubic",
        "female_full", "male_full", "all_gens", "color", "depth", "segmentation", "prompt"
    )
    CATEGORY = "McKlinton/masking"
    FUNCTION = "process_folder"

    # Constants for the colors in RGB format
    FEMALE_BODY_RGB = (0, 0, 255)
    FEMALE_FACE_RGB = (0, 255, 0)
    FEMALE_BREAST_RGB = (255, 0, 255)
    FEMALE_HANDS_RGB = (255, 255, 0)
    FEMALE_VAG_RGB = (255, 0, 0)
    FEMALE_HAIR_RGB = (0, 128, 128)

    MALE_BODY_RGB = (128, 0, 255)
    MALE_FACE_RGB = (255, 0, 128)
    MALE_HANDS_RGB = (255, 128, 0)
    MALE_PEN_RGB = (128, 255, 0)
    MALE_PUBIC_RGB = (128, 255, 255)
    MALE_HAIR_RGB = (128, 128, 0)

    def resize_image(self, image, resize_mode, width, height):
        if image is None or resize_mode == "keep original":
            return image
            
        h, w = image.shape[:2]
        
        if resize_mode == "resize":
            # Simple resize without preserving aspect ratio
            return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        
        elif resize_mode == "resize and crop":
            # Resize preserving aspect ratio and then crop center
            scale = max(width / w, height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Crop center
            start_x = max(0, new_w // 2 - width // 2)
            start_y = max(0, new_h // 2 - height // 2)
            cropped = resized[start_y:start_y + height, start_x:start_x + width]
            
            # Handle the case where the cropped image is smaller than the target size
            if cropped.shape[0] < height or cropped.shape[1] < width:
                result = np.zeros((height, width, 3), dtype=cropped.dtype) if len(image.shape) == 3 else np.zeros((height, width), dtype=cropped.dtype)
                y_offset = (height - cropped.shape[0]) // 2
                x_offset = (width - cropped.shape[1]) // 2
                result[y_offset:y_offset+cropped.shape[0], x_offset:x_offset+cropped.shape[1]] = cropped
                return result
            
            return cropped
            
        return image

    def process_folder(self, folder, index, random_index, resize_mode="keep original", width=512, height=512):
        if not os.path.exists(folder):
            raise ValueError("Folder does not exist: " + folder)

        files = os.listdir(folder)
        prompt_files = [f for f in files if f.lower().endswith(".txt")]
        image_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        if not image_files:
            raise ValueError("No image files found in folder")

        file_groups = defaultdict(dict)
        for filename in image_files:
            match = re.match(r'^(masks|color|depth)_(.+)\.', filename)
            if match:
                file_type, timestamp = match.groups()
                file_groups[timestamp][file_type] = filename


        prompts_by_timestamp = {}
        for filename in prompt_files:
            match = re.match(r'^(prompt|PROMPT)_(.+)\.', filename)
            if match:
                timestamp = match.group(2)
                with open(os.path.join(folder, filename), 'r') as f:
                    prompt_text = f.read()
                    prompts_by_timestamp[timestamp] = prompt_text

        timestamps = sorted(file_groups.keys())
        if not timestamps:
            raise ValueError("No valid image sets found in folder")
        
        # If random_index is True, randomly select an index
        if random_index:
            index = random.randint(0, len(timestamps) - 1)
        
        if index >= len(timestamps):
            raise ValueError(f"Index {index} is out of range. Only {len(timestamps)} sets found.")

        selected_timestamp = timestamps[index]
        selected_files = file_groups[selected_timestamp]
        selected_prompt = prompts_by_timestamp.get(selected_timestamp, "")

        # Process masks
        if 'masks' in selected_files:
            mask_path = os.path.join(folder, selected_files['masks'])
            colored_mask_image = cv2.imread(mask_path)
            if colored_mask_image is None:
                raise ValueError("Failed to read mask image: " + mask_path)
            
            # Apply resizing if needed
            colored_mask_image = self.resize_image(colored_mask_image, resize_mode, width, height)
            
            female_body_mask = self.extract_mask_by_color(colored_mask_image, self.FEMALE_BODY_RGB)
            female_face_mask = self.extract_mask_by_color(colored_mask_image, self.FEMALE_FACE_RGB)
            female_hair_mask = self.extract_mask_by_color(colored_mask_image, self.FEMALE_HAIR_RGB)
            female_breast_mask = self.extract_mask_by_color(colored_mask_image, self.FEMALE_BREAST_RGB)
            female_hands_mask = self.extract_mask_by_color(colored_mask_image, self.FEMALE_HANDS_RGB)
            female_vag_mask = self.extract_mask_by_color(colored_mask_image, self.FEMALE_VAG_RGB)

            male_body_mask = self.extract_mask_by_color(colored_mask_image, self.MALE_BODY_RGB)
            male_hair_mask = self.extract_mask_by_color(colored_mask_image, self.MALE_HAIR_RGB)
            male_face_mask = self.extract_mask_by_color(colored_mask_image, self.MALE_FACE_RGB)
            male_hands_mask = self.extract_mask_by_color(colored_mask_image, self.MALE_HANDS_RGB)
            male_pen_mask = self.extract_mask_by_color(colored_mask_image, self.MALE_PEN_RGB)
            male_pubic_mask = self.extract_mask_by_color(colored_mask_image, self.MALE_PUBIC_RGB)

            female_full_mask = torch.max(
                torch.max(torch.max(torch.max(torch.max(female_body_mask, female_face_mask), female_breast_mask), female_hands_mask), female_hair_mask),
                female_vag_mask
            )
            male_full_mask = torch.max(
                torch.max(torch.max(torch.max(torch.max(male_body_mask, male_face_mask), male_hands_mask), male_pen_mask),male_hair_mask),
                male_pubic_mask
            )
            all_gens_mask = torch.max(torch.max(female_vag_mask, male_pen_mask), male_pubic_mask)
        else:
            empty_mask = torch.zeros((1, height, width), dtype=torch.float32)
            female_body_mask = female_face_mask = female_breast_mask = female_hands_mask = female_vag_mask = empty_mask
            male_body_mask = male_face_mask = male_hands_mask = male_pen_mask = male_pubic_mask = empty_mask
            female_full_mask = male_full_mask = all_gens_mask = empty_mask

        # Process color image
        if 'color' in selected_files:
            color_path = os.path.join(folder, selected_files['color'])
            color_image = cv2.imread(color_path)
            if color_image is None:
                color_image_tensor = torch.zeros((height, width, 3), dtype=torch.float32).unsqueeze(0)
            else:
                # Apply resizing if needed
                color_image = self.resize_image(color_image, resize_mode, width, height)
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                color_image_tensor = torch.from_numpy(color_image).float() / 255.0
                color_image_tensor = color_image_tensor.unsqueeze(0)
        else:
            color_image_tensor = torch.zeros((height, width, 3), dtype=torch.float32).unsqueeze(0)

        # Process depth image
        if 'depth' in selected_files:
            depth_path = os.path.join(folder, selected_files['depth'])
            depth_image = cv2.imread(depth_path)
            if depth_image is None:
                depth_image_tensor = torch.zeros((height, width, 3), dtype=torch.float32).unsqueeze(0)
            else:
                # Apply resizing if needed
                depth_image = self.resize_image(depth_image, resize_mode, width, height)
                depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)
                depth_image_tensor = torch.from_numpy(depth_image).float() / 255.0
                depth_image_tensor = depth_image_tensor.unsqueeze(0)
        else:
            depth_image_tensor = torch.zeros((height, width, 3), dtype=torch.float32).unsqueeze(0)

        # Process segmentation image
        if 'masks' in selected_files:
            masks_path = os.path.join(folder, selected_files['masks'])
            masks_image = cv2.imread(masks_path)
            if masks_image is None:
                masks_image_tensor = torch.zeros((height, width, 3), dtype=torch.float32).unsqueeze(0)
            else:
                # Apply resizing if needed (same as above since this uses the same file)
                masks_image = self.resize_image(masks_image, resize_mode, width, height)
                masks_image = cv2.cvtColor(masks_image, cv2.COLOR_BGR2RGB)
                masks_image_tensor = torch.from_numpy(masks_image).float() / 255.0
                masks_image_tensor = masks_image_tensor.unsqueeze(0)
        else:
            masks_image_tensor = torch.zeros((height, width, 3), dtype=torch.float32).unsqueeze(0)        
            
        return (
            female_body_mask,
            female_face_mask,
            female_hair_mask,
            female_breast_mask,
            female_hands_mask,
            female_vag_mask,
            male_body_mask,
            male_face_mask,
            male_hair_mask,
            male_hands_mask,
            male_pen_mask,
            male_pubic_mask,
            female_full_mask,
            male_full_mask,
            all_gens_mask,
            color_image_tensor,
            depth_image_tensor,
            masks_image_tensor,            
            selected_prompt
        )

    def extract_mask_by_color(self, image, color_rgb):
        color_hsv = cv2.cvtColor(np.uint8([[color_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, saturation, value = color_hsv
        color_lower = np.array([hue, saturation, value])
        color_upper = np.array([hue, saturation, value])
        mask = cv2.inRange(hsv_image, color_lower, color_upper)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0
        return mask_tensor

NODE_CLASS_MAPPINGS = {"ColormaskNode": ColormaskNode}
NODE_DISPLAY_NAME_MAPPINGS = {"ColormaskNode": "Mask Node"}