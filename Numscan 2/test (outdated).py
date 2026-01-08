
# This file used to be the editor for Numscan 2, use editor.py instead

import os
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageTk

from model import MODEL_PATH, load_mnist_data, load_model_pickle

CANVAS_SIZE = 280
BRUSH_SIZE = 18
RESAMPLING = getattr(Image, "Resampling", None)
RESAMPLE_LANCZOS = RESAMPLING.LANCZOS if RESAMPLING else Image.LANCZOS
RESAMPLE_NEAREST = RESAMPLING.NEAREST if RESAMPLING else Image.NEAREST


class DigitTester(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.title("Numscan 2 editor")
        self.resizable(False, False)

        self.model = model
        (_, _), (self.x_test, self.y_test) = load_mnist_data()
        self.demo_index = 0

        self.prediction_var = tk.StringVar(value="Prediction: -")
        self.demo_prediction_var = tk.StringVar(value="Demo prediction: -")
        self.upload_prediction_var = tk.StringVar(value="Uploaded prediction: -")

        self._create_canvas_section()
        self._create_demo_section()
        self._create_upload_section()

        self.show_demo_image()

    def _create_canvas_section(self):
        frame = tk.LabelFrame(self, text="Draw a digit")
        frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        self.canvas = tk.Canvas(
            frame, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white", cursor="cross"
        )
        self.canvas.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        self.canvas.bind("<ButtonRelease-1>", self.reset_draw)

        tk.Button(frame, text="Predict drawing", command=self.predict_canvas).grid(
            row=1, column=0, sticky="ew", padx=5, pady=(0, 5)
        )
        tk.Button(frame, text="Clear", command=self.clear_canvas).grid(
            row=1, column=1, sticky="ew", padx=5, pady=(0, 5)
        )
        tk.Label(frame, textvariable=self.prediction_var).grid(
            row=2, column=0, columnspan=2, pady=(0, 5)
        )

        self.drawing_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), "white")
        self.drawing_draw = ImageDraw.Draw(self.drawing_image)
        self.last_x = None
        self.last_y = None

    def _create_demo_section(self):
        frame = tk.LabelFrame(self, text="MNIST demo samples")
        frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        self.demo_panel = tk.Label(frame)
        self.demo_panel.grid(row=0, column=0, columnspan=2, padx=5, pady=5)

        tk.Button(frame, text="Previous", command=self.prev_demo).grid(
            row=1, column=0, sticky="ew", padx=5
        )
        tk.Button(frame, text="Next", command=self.next_demo).grid(
            row=1, column=1, sticky="ew", padx=5
        )
        tk.Button(frame, text="Predict demo digit", command=self.predict_demo).grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=(5, 0)
        )
        tk.Label(frame, textvariable=self.demo_prediction_var).grid(
            row=3, column=0, columnspan=2, pady=(5, 0)
        )

    def _create_upload_section(self):
        frame = tk.LabelFrame(self, text="Upload an image")
        frame.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="n")

        tk.Button(frame, text="Choose image", command=self.load_image).grid(
            row=0, column=0, sticky="ew", padx=5, pady=5
        )
        self.upload_panel = tk.Label(frame)
        self.upload_panel.grid(row=1, column=0, padx=5, pady=5)
        tk.Label(frame, textvariable=self.upload_prediction_var).grid(
            row=2, column=0, padx=5, pady=(0, 5)
        )
        self.upload_photo = None

    def start_draw(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def draw_lines(self, event):
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(
                self.last_x,
                self.last_y,
                event.x,
                event.y,
                fill="black",
                width=BRUSH_SIZE,
                capstyle=tk.ROUND,
                smooth=True,
            )
            self.drawing_draw.line(
                (self.last_x, self.last_y, event.x, event.y),
                fill="black",
                width=BRUSH_SIZE,
            )
        self.last_x = event.x
        self.last_y = event.y

    def reset_draw(self, _event):
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.drawing_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), "white")
        self.drawing_draw = ImageDraw.Draw(self.drawing_image)
        self.prediction_var.set("Prediction: -")

    def predict_canvas(self):
        image = self.drawing_image.resize((28, 28), RESAMPLE_LANCZOS)
        image = ImageOps.invert(image)
        array = np.array(image, dtype=np.float32) / 255.0
        digit, confidence = self.predict_array(array)
        self.prediction_var.set(f"Prediction: {digit} ({confidence:.2%})")

    def predict_demo(self):
        demo = self.x_test[self.demo_index]
        digit, confidence = self.predict_array(demo)
        self.demo_prediction_var.set(
            f"Demo prediction: {digit} ({confidence:.2%}) | Label: {self.y_test[self.demo_index]}"
        )

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*")],
        )
        if not file_path:
            return
        try:
            image = Image.open(file_path).convert("L")
        except Exception as exc:
            messagebox.showerror("Open image failed", f"Could not open the image:\n{exc}")
            return

        preview = image.resize((CANVAS_SIZE, CANVAS_SIZE), RESAMPLE_LANCZOS)
        self.upload_photo = ImageTk.PhotoImage(preview)
        self.upload_panel.configure(image=self.upload_photo)

        processed = ImageOps.invert(image).resize((28, 28), RESAMPLE_LANCZOS)
        array = np.array(processed, dtype=np.float32) / 255.0
        digit, confidence = self.predict_array(array)
        self.upload_prediction_var.set(f"Uploaded prediction: {digit} ({confidence:.2%})")

    def next_demo(self):
        self.demo_index = (self.demo_index + 1) % len(self.x_test)
        self.show_demo_image()

    def prev_demo(self):
        self.demo_index = (self.demo_index - 1) % len(self.x_test)
        self.show_demo_image()

    def show_demo_image(self):
        img = (self.x_test[self.demo_index].squeeze() * 255).astype(np.uint8)
        image = Image.fromarray(img, mode="L").resize(
            (CANVAS_SIZE, CANVAS_SIZE), RESAMPLE_NEAREST
        )
        self.demo_photo = ImageTk.PhotoImage(image)
        self.demo_panel.configure(image=self.demo_photo)
        self.demo_prediction_var.set("Demo prediction: -")

    def predict_array(self, array):
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 2:
            array = array[..., np.newaxis]
        batch = np.expand_dims(array, axis=0)
        probabilities = self.model.predict(batch, verbose=0)[0]
        digit = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))
        return digit, confidence


def main():
    if not os.path.exists(MODEL_PATH):
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Missing model",
            f"Could not find '{MODEL_PATH}'. Train the model first by running model.py.",
        )
        root.destroy()
        return

    model = load_model_pickle(MODEL_PATH)
    app = DigitTester(model)
    app.mainloop()


if __name__ == "__main__":
    main()
