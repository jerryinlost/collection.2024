from pptx import Presentation

def remove_hyperlinks_from_pptx(input_file, output_file):
    # Open the presentation
    prs = Presentation(input_file)

    # Iterate through all slides
    for slide in prs.slides:
        # Iterate through all shapes in the slide
        for shape in slide.shapes:
            # Check if the shape has a hyperlink
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    for run in paragraph.runs:
                        if run.hyperlink:
                            run.hyperlink.address = None  # Remove the hyperlink

    # Save the modified presentation
    prs.save(output_file)
    print(f"Hyperlinks removed and saved to {output_file}")

# Example usage
input_pptx = 'f:\\Report.pptx'
output_pptx = 'f:\\Report2.pptx'
remove_hyperlinks_from_pptx(input_pptx, output_pptx)