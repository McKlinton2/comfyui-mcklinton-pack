import os
import folder_paths
from PIL import Image
import numpy as np
import torch

class SaveTextArrayToFiles:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "root_directory": ("STRING", {"widget": "folder"}),
            "prefix": ("STRING", {"default": ""}),
        },
        "optional": {
            "texts": ("STRING", {"forceInput": True, "multiline": True}),
            "filenames": ("STRING", {"forceInput": True}),
            "images": ("IMAGE", ),
            "image_format": (["png", "jpg", "jpeg", "webp"], {"default": "png"}),
            "quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
        }}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_files"
    CATEGORY = "McKlinton/save"

    def save_files(self, root_directory, prefix="", texts=None, filenames=None, images=None, image_format="png", quality=95):
        # Ensure root directory exists
        if not os.path.exists(root_directory):
            os.makedirs(root_directory, exist_ok=True)
        
        saved_files = []
        
        # Handle text files if provided
        if texts is not None and filenames is not None:
            # Ensure texts and filenames are lists
            if not isinstance(texts, list):
                texts = [texts]
            if not isinstance(filenames, list):
                filenames = [filenames]
            
            # Ensure lists have the same length
            if len(texts) != len(filenames):
                raise ValueError(f"Length of texts ({len(texts)}) and filenames ({len(filenames)}) must match")
            
            # Save each text file
            for i, (text, filename) in enumerate(zip(texts, filenames)):
                # Clean filename, remove path components for security
                clean_filename = os.path.basename(filename)
                
                # Add prefix if provided
                if prefix:
                    clean_filename = f"{prefix}{clean_filename}"
                
                # If filename doesn't have a .txt extension, add it
                if not clean_filename.lower().endswith('.txt'):
                    clean_filename += '.txt'
                
                # Full path to save
                filepath = os.path.join(root_directory, clean_filename)
                
                try:
                    # Create directories if they don't exist
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    
                    # Write the text to file
                    with open(filepath, 'w', encoding='utf-8') as file:
                        file.write(text)
                    
                    saved_files.append(filepath)
                except Exception as e:
                    print(f"Error saving file {filepath}: {str(e)}")
            
            print(f"Saved {len(saved_files)} text files to {root_directory}")
        
        # Handle image files if provided
        if images is not None and filenames is not None:
            # Get batch size (number of images)
            image_count = images.shape[0]
            
            if not isinstance(filenames, list):
                filenames = [filenames]
            
            # If filenames count doesn't match images count, use filenames in a loop
            if len(filenames) < image_count:
                filenames = [filenames[i % len(filenames)] for i in range(image_count)]
            elif len(filenames) > image_count:
                filenames = filenames[:image_count]
            
            image_saved_files = []
            
            # Debug info
            print(f"Processing {image_count} images with {len(filenames)} filenames")
            
            # Convert entire batch to CPU if it's a tensor
            if isinstance(images, torch.Tensor):
                images = images.cpu()
            
            for i in range(image_count):
                # Get current image from batch and filename
                img = images[i]  # Direct indexing to get image i
                clean_filename = os.path.basename(filenames[i])
                
                # Add prefix if provided
                if prefix:
                    clean_filename = f"{prefix}{clean_filename}"
                
                # Remove any existing extension and add the chosen format
                clean_filename = os.path.splitext(clean_filename)[0] + f".{image_format}"
                
                # Full path to save
                filepath = os.path.join(root_directory, clean_filename)
                
                try:
                    # Create directories if they don't exist
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    
                    # Convert tensor to numpy array if necessary
                    if isinstance(img, torch.Tensor):
                        img_numpy = img.numpy()  # Already on CPU from earlier conversion
                    else:
                        img_numpy = img
                    
                    # Convert to PIL Image (ensure proper scaling)
                    pil_img = Image.fromarray((img_numpy * 255).astype(np.uint8))
                    
                    # Save the image
                    if image_format.lower() in ["jpg", "jpeg"]:
                        pil_img.save(filepath, quality=quality)
                    elif image_format.lower() == "webp":
                        pil_img.save(filepath, quality=quality)
                    else:  # PNG
                        pil_img.save(filepath)
                    
                    image_saved_files.append(filepath)
                    print(f"Saved image {i+1}/{image_count}: {filepath}")
                        
                except Exception as e:
                    print(f"Error saving image {filepath}: {str(e)}")
                    print(f"Image shape: {img.shape if hasattr(img, 'shape') else 'unknown'}")
            
            saved_files.extend(image_saved_files)
            print(f"Saved {len(image_saved_files)} images to {root_directory}")
        
        return {}

# Register the node
NODE_CLASS_MAPPINGS = {
    "SaveTextArrayToFiles": SaveTextArrayToFiles
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveTextArrayToFiles": "Save Text/Image Arrays to Files"
}