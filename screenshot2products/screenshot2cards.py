import cv2
import numpy as np
import pytesseract
import os

# If you don't have tesseract executable in your PATH, include the following:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return thresh

def find_contours(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def extract_text_from_image(image):
    config = '--psm 6'  # Assume a single uniform block of text.
    text = pytesseract.image_to_string(image, config=config)
    return text

def save_image(image, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(os.path.join(directory, filename), image)

def process_screenshot(screenshot_path):
    if not os.path.exists(screenshot_path):
        print("File not found!");return
    image = cv2.imread(screenshot_path)
    thresh = preprocess_image(image)
    contours = find_contours(thresh)

    card_index = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        card = image[y:y+h, x:x+w]

        # Save the card image
        card_filename = f"card_{card_index}.png"
        save_image(card, 'cards', card_filename)
        
        # Assume product image is at the top of the card
        product_image = card[0:int(h/2), 0:w]

        # Extract text (assuming text is at the bottom of the card)
        text_block = card[int(h/2):h, 0:w]
        text = extract_text_from_image(text_block)

        # Extract product name and price (you might need to refine this part depending on the text structure)
        lines = text.split('\n')
        product_name = lines[0] if lines else 'unknown_product'
        price = lines[1] if len(lines) > 1 else 'unknown_price'

        # Save the product image with a unique name
        product_image_filename = f"{product_name}_{price}.png".replace(' ', '_')
        save_image(product_image, 'product_images', product_image_filename)

        card_index += 1

if __name__ == "__main__":
    # screenshot_path = input("Enter the path to the screenshot: ")
    process_screenshot('screenshot.alibaba.png')#screenshot_path)
    print("Processing complete. Check 'cards' and 'product_images' directories for results.")