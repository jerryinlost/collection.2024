import os
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

def extract_images_from_pptx(input_file, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the presentation
    prs = Presentation(input_file)

    # Initialize an image counter
    image_counter = 0

    # Iterate through all slides
    for slide_num, slide in enumerate(prs.slides):
        # Iterate through all shapes in the slide
        for shape_num, shape in enumerate(slide.shapes):
            # Check if the shape is a picture
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                # Get the image
                image = shape.image
                image_bytes = image.blob

                # Generate a filename for the image
                image_filename = f"slide_{slide_num+1}_shape_{shape_num+1}_image_{image_counter+1}.{image.ext}"
                image_path = os.path.join(output_dir, image_filename)

                # Save the image to disk
                with open(image_path, 'wb') as image_file:
                    image_file.write(image_bytes)

                print(f"Extracted image: {image_path}")
                image_counter += 1

    print(f"Total {image_counter} images extracted.")

# Example usage
input_pptx = 'path/to/your/input_presentation.pptx'
output_dir = 'path/to/your/output_directory'
extract_images_from_pptx(input_pptx, output_dir)