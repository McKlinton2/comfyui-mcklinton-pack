import os
import re
import torch
import numpy as np
from PIL import Image, ImageOps
from collections import defaultdict

class LoadFilteredImageBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "folder": ("STRING", {"widget": "folder"}),
            "filter": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "root_dir", "filenames")
    OUTPUT_IS_LIST = (False, False, True)
    FUNCTION = "load_filtered_images"
    CATEGORY = "McKlinton/load"

    def load_filtered_images(self, folder, filter):
        if not os.path.exists(folder):
            raise ValueError(f"Folder does not exist: {folder}")
            
        # Get list of image files in the directory
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
        base_filenames = []

        # Filter files by extension and optional filter text
        files = []
        for filename in os.listdir(folder):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                if not filter or filter.strip() == "" or re.search(filter, filename):

                    files.append(filename)

                    # Take only the base name e.g. "anything_1231231314512.png" -> "1231231314512".
                    # Remove the extension and any leading path before underscore.
                    filename = os.path.splitext(filename)[0]
                    filename = filename.split("_")[-1]

                    base_filenames.append(filename)

        
        if not files:
            raise ValueError(f"No matching image files found in folder: {folder}")
            
        # Sort the files for consistent ordering
        files = sorted(files)
        
        # Load all images into a batch
        images = []
        
        for filename in files:
            filepath = os.path.join(folder, filename)
            try:
                image = Image.open(filepath)
                image = ImageOps.exif_transpose(image)  # Handle EXIF orientation
                image = image.convert("RGB")
                
                # Convert to the tensor format expected by ComfyUI
                image_np = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None,]
                images.append(image_tensor)
            except Exception as e:
                print(f"Error loading image {filepath}: {e}")
                
        if not images:
            raise ValueError(f"Failed to load any valid images from folder: {folder}")
            
        # Stack all images into a single batch tensor
        image_batch = torch.cat(images, dim=0)
        
        # Return the image batch, root directory, and list of filenames
        return (image_batch, folder, base_filenames)

# Register the node
NODE_CLASS_MAPPINGS = {
    "LoadFilteredImageBatch": LoadFilteredImageBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadFilteredImageBatch": "Load Filtered Image Batch"
}