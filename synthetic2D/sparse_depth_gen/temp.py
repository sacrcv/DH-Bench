import PIL
import os

from PIL import Image

def crop_to_square(image_path):
    # Open the image file
    img = Image.open(image_path)

    # Get the dimensions of the image
    width, height = img.size

    # Calculate the cropping parameters
    left = (width - height) / 2
    top = 0
    right = (width + height) / 2
    bottom = height

    # Crop the image to a square
    cropped_img = img.crop((left, top, right, bottom))

    return cropped_img

# Example usage
input_image_path = "../real_world_bg/clouds.jpg"  # Replace with the path to your image
output_image_path = "cropped_image.jpg"  # Replace with the desired output path

cropped_image = crop_to_square(input_image_path)
cropped_image.resize((1024, 1024))
cropped_image.save(output_image_path)
