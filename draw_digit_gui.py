import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage import center_of_mass, shift

WIDTH, HEIGHT = 280, 280  # Large canvas
WHITE = 255
BLACK = 0

def center_image(img):
    cy, cx = center_of_mass(img)
    shift_y = img.shape[0] // 2 - cy
    shift_x = img.shape[1] // 2 - cx
    return shift(img, shift=[shift_y, shift_x], mode='constant')

class DrawApp:
    def __init__(self, root):
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="black")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)
        self.button = tk.Button(root, text="Save (28x28) and Exit", command=self.save)
        self.button.pack()

        self.image = Image.new("L", (WIDTH, HEIGHT), color=BLACK)
        self.drawn = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        r = 12
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.drawn.ellipse([x-r, y-r, x+r, y+r], fill=WHITE)

    def save(self):
        img_resized = self.image.resize((28, 28), Image.LANCZOS)
        arr = np.array(img_resized) / 255.0
        arr = center_image(arr)  # ✅ Center the digit before saving
        np.save("sample_input.npy", arr)
        print("✅ Saved as sample_input.npy")
        root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop()
