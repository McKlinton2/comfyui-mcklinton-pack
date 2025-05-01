from PIL import Image
import os
import argparse

def resize_and_crop(input_folder, output_folder, target_width, target_height):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            img_path = os.path.join(input_folder, filename)
            with Image.open(img_path) as img:
                # Calculate aspect ratios
                img_ratio = img.width / img.height
                target_ratio = target_width / target_height

                if img_ratio > target_ratio:
                    # Image is wider than target: resize based on height, then crop width
                    new_height = target_height
                    new_width = int(new_height * img_ratio)
                else:
                    # Image is taller than target: resize based on width, then crop height
                    new_width = target_width
                    new_height = int(new_width / img_ratio)

                img = img.resize((new_width, new_height))

                # Calculate coordinates for cropping to center
                left = (new_width - target_width) / 2
                top = (new_height - target_height) / 2
                right = (new_width + target_width) / 2
                bottom = (new_height + target_height) / 2

                img = img.crop((left, top, right, bottom))

                output_path = os.path.join(output_folder, filename)
                img.save(output_path)
                print(f"Processed {filename} and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize and crop images to a target size.")
    parser.add_argument("input_folder", type=str, help="Path to the input folder containing images.")
    parser.add_argument("output_folder", type=str, help="Path to the output folder to save processed images.")
    parser.add_argument("target_width", type=int, help="Target width for the resized images.")
    parser.add_argument("target_height", type=int, help="Target height for the resized images.")

    args = parser.parse_args()

    resize_and_crop(args.input_folder, args.output_folder, args.target_width, args.target_height)