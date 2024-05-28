import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os

class CardExtractor:
    def __init__(self, image):
        self.image = image
        self.start_point = None
        self.card_width = None
        self.card_height = None
        self.figure, self.ax = plt.subplots(figsize=(15, 10))  # Set larger figure size
        self.ax.imshow(self.image)
        self.cid = self.figure.canvas.mpl_connect('button_press_event', self.onclick)
        self.rect = None

    def onclick(self, event):
        if self.start_point is None:
            self.start_point = (int(event.xdata), int(event.ydata))
            print(f"Start Point Set: {self.start_point}")
        elif self.card_width is None and self.card_height is None:
            self.card_width = abs(int(event.xdata) - self.start_point[0])
            self.card_height = abs(int(event.ydata) - self.start_point[1])
            print(f"Card Width: {self.card_width}, Card Height: {self.card_height}")
            self.draw_rectangle()

    def draw_rectangle(self):
        if self.rect:
            self.rect.remove()
        self.rect = plt.Rectangle(self.start_point, self.card_width, self.card_height,
                                  linewidth=1, edgecolor='r', facecolor='none')
        self.ax.add_patch(self.rect)
        self.figure.canvas.draw()

    def extract_cards(self):
        if self.start_point is None or self.card_width is None or self.card_height is None:
            print("Please set start point and card dimensions first.")
            return []
        x_start, y_start = self.start_point
        card_images = []
        y = y_start
        while y + self.card_height <= self.image.shape[0]:
            x = x_start
            while x + self.card_width <= self.image.shape[1]:
                card = self.image[y:y + self.card_height, x:x + self.card_width]
                card_images.append(card)
                x += self.card_width
            y += self.card_height
        return card_images

    def save_cards(self, card_images, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for i, card in enumerate(card_images):
            output_path = os.path.join(output_dir, f'card_{i+1}.png')
            cv2.imwrite(output_path, cv2.cvtColor(card, cv2.COLOR_RGB2BGR))
            print(f'Saved: {output_path}')

def on_button_click(event):
    card_images = extractor.extract_cards()
    if card_images:
        output_dir = 'extracted_cards'
        extractor.save_cards(card_images, output_dir)
        for i, card in enumerate(card_images):
            plt.figure()
            plt.imshow(card)
            plt.axis('off')
            plt.title(f'Card {i+1}')
        plt.show()
    else:
        print("No cards extracted. Please set the start point and card dimensions first.")

# Load the screenshot image
screenshot_path = './samples/screenshot.alibaba.png'
screenshot = cv2.imread(screenshot_path)
screenshot_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)

# Initialize the card extractor with the screenshot
extractor = CardExtractor(screenshot_rgb)

# Add a button to extract and save cards
ax_button = plt.axes([0.8, 0.02, 0.1, 0.05])
button = Button(ax_button, 'Extract & Save Cards')
button.on_clicked(on_button_click)

plt.show()