import cv2
import pytesseract
import re

# Load the screenshot image
image = cv2.imread('screenshot.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use edge detection to find contours
edges = cv2.Canny(gray, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours to find product cards
product_cards = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 50 and h > 50:  # Adjust the size filter as needed
        product_cards.append((x, y, w, h))

# Helper function to sanitize filenames
def sanitize_filename(text):
    return re.sub(r'[^A-Za-z0-9 ]+', '', text).replace(' ', '_')

# Extract product info from each card and save images with appropriate names
for i, (x, y, w, h) in enumerate(product_cards):
    card_image = image[y:y+h, x:x+w]
    text = pytesseract.image_to_string(card_image)

    # Extract product name and price using simple regex (adjust as needed)
    product_name = "unknown_product"
    price = "unknown_price"

    lines = text.split('\n')
    for line in lines:
        if re.search(r'\$', line):  # Look for price pattern, e.g., "$19.99"
            price = line.strip()
        elif len(line.strip()) > 0:  # Assuming non-empty line is part of product name
            product_name = line.strip()
            break  # Assume first non-empty line is the product name

    # Create a sanitized filename
    filename = f"{sanitize_filename(product_name)}_{sanitize_filename(price)}.png"
    cv2.imwrite(filename, card_image)

    print(f"Saved: {filename}")