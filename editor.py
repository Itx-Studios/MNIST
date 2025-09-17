import os
import random
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageDraw, ImageTk, ImageOps

from predict import feed_forward, exec as nn_exec, soft_max
from network import nn


CANVAS_SIZE = 280  # Displayed drawing area (pixels)
GRID_SIZE = 28     # Model input size (MNIST)
STROKE_WIDTH = 18  # Drawing stroke width on the high-res canvas


class EditorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MNIST Editor – Draw or Load and Predict")

        # State
        self.prev_x = None
        self.prev_y = None
        self.bg_image_id = None
        self.tk_bg = None  # Keep reference to PhotoImage

        # Off-screen PIL image where we keep the current drawing/content
        self.img_hi = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)  # black background
        self.draw_hi = ImageDraw.Draw(self.img_hi)

        # UI Layout
        self._build_ui()

    def _build_ui(self):
        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        # Left: Canvas
        canvas_frame = tk.Frame(main)
        canvas_frame.pack(side=tk.LEFT, padx=8, pady=8)

        self.canvas = tk.Canvas(canvas_frame, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black", highlightthickness=1, highlightbackground="#888")
        self.canvas.pack()

        # Bind drawing events
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        # Right: Controls
        ctrl = tk.Frame(main)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        # Prediction result
        self.result_var = tk.StringVar(value="Prediction: –")
        tk.Label(ctrl, textvariable=self.result_var, font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 8))

        # Options
        self.auto_invert_var = tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl, text="Auto invert colors", variable=self.auto_invert_var).pack(anchor="w")

        # Buttons
        tk.Button(ctrl, text="Predict Drawing", command=self.predict).pack(fill=tk.X, pady=(8, 2))
        tk.Button(ctrl, text="Clear", command=self.clear_canvas).pack(fill=tk.X, pady=2)
        tk.Button(ctrl, text="Load Image…", command=self.load_image_dialog).pack(fill=tk.X, pady=(8, 2))
        tk.Button(ctrl, text="Random Demo", command=self.load_random_demo).pack(fill=tk.X, pady=2)
        tk.Button(ctrl, text="Load Model…", command=self.load_model_dialog).pack(fill=tk.X, pady=(8, 2))

        # Info
        info = (
            "How to use:\n"
            " - Draw a digit with the mouse (white on black).\n"
            " - Or load a PNG image (28x28 preferred; others are resized).\n"
            " - Click Predict to run the model.\n"
            " - 'Auto invert colors' helps when images are black-on-white."
        )
        tk.Label(ctrl, text=info, justify=tk.LEFT, fg="#444").pack(anchor="w", pady=(8, 0))

    # ---------- Drawing Handlers ----------
    def _on_mouse_down(self, event):
        self.prev_x, self.prev_y = event.x, event.y

    def _on_mouse_move(self, event):
        if self.prev_x is None or self.prev_y is None:
            return
        x, y = event.x, event.y
        # Draw on the visible canvas
        self.canvas.create_line(self.prev_x, self.prev_y, x, y, fill="white", width=STROKE_WIDTH, capstyle=tk.ROUND, smooth=True)
        # Draw on the off-screen PIL image
        self.draw_hi.line([(self.prev_x, self.prev_y), (x, y)], fill=255, width=STROKE_WIDTH)
        self.prev_x, self.prev_y = x, y

    def _on_mouse_up(self, _event):
        self.prev_x, self.prev_y = None, None

    # ---------- Canvas and Images ----------
    def clear_canvas(self):
        self.canvas.delete("all")
        self.img_hi = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw_hi = ImageDraw.Draw(self.img_hi)
        self.tk_bg = None
        self.bg_image_id = None
        self.result_var.set("Prediction: –")

    def _set_canvas_background_from_pil(self):
        # Place the PIL image as a background on the canvas
        self.tk_bg = ImageTk.PhotoImage(self.img_hi.convert("RGB"))
        if self.bg_image_id is None:
            self.bg_image_id = self.canvas.create_image(0, 0, image=self.tk_bg, anchor=tk.NW)
        else:
            self.canvas.itemconfig(self.bg_image_id, image=self.tk_bg)

    def load_image_dialog(self):
        path = filedialog.askopenfilename(title="Open image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*")])
        if not path:
            return
        try:
            self._load_image(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")

    def _load_image(self, path: str):
        img = Image.open(path).convert("L")
        # Resize for display/drawing canvas
        img = img.resize((CANVAS_SIZE, CANVAS_SIZE), Image.LANCZOS)
        self.img_hi = img
        self.draw_hi = ImageDraw.Draw(self.img_hi)
        self.canvas.delete("all")
        self._set_canvas_background_from_pil()
        self.result_var.set("Prediction: –")

    def load_random_demo(self):
        # Try sampling from mnist-png/train/<digit>/...
        base = os.path.join("mnist-png", "train")
        if not os.path.isdir(base):
            messagebox.showinfo("Info", "Demo dataset not found at mnist-png/train. Use 'Load Image…' instead.")
            return

        # Collect all image paths
        candidates = []
        for d in range(10):
            d_path = os.path.join(base, str(d))
            if os.path.isdir(d_path):
                for name in os.listdir(d_path):
                    if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        candidates.append(os.path.join(d_path, name))

        if not candidates:
            messagebox.showinfo("Info", "No images found in mnist-png/train/*.")
            return

        path = random.choice(candidates)
        try:
            self._load_image(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load demo image:\n{e}")

    # ---------- Prediction ----------
    def _prepare_input_vector(self) -> list:
        # Downscale high-res canvas image to 28x28 for the model
        small = self.img_hi.resize((GRID_SIZE, GRID_SIZE), Image.LANCZOS)
        # Auto-invert if enabled and image seems black-on-white
        if self.auto_invert_var.get():
            # Compute mean to infer polarity (white background -> high mean)
            mean_val = sum(small.getdata()) / (GRID_SIZE * GRID_SIZE * 255.0)
            if mean_val > 0.5:
                small = ImageOps.invert(small)
        # Normalize to [0, 1]
        pixels = [p / 255.0 for p in small.getdata()]
        return pixels

    def predict(self):
        try:
            X = self._prepare_input_vector()
            # Get logits/probabilities for better feedback
            _, z = nn_exec(X)
            logits = z[-1]
            probs = soft_max(logits)
            pred = max(range(10), key=lambda i: probs[i])

            # Top-3 probabilities
            top3 = sorted(((i, p) for i, p in enumerate(probs)), key=lambda t: t[1], reverse=True)[:3]
            top3_txt = ", ".join(f"{i}:{p:.2f}" for i, p in top3)
            self.result_var.set(f"Prediction: {pred}  (Top-3: {top3_txt})")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed:\n{e}")

    # ---------- Model Management ----------
    def load_model_dialog(self):
        path = filedialog.askopenfilename(title="Open model pickle", filetypes=[("Pickle files", "*.pickle;*.pkl"), ("All files", "*.*")])
        if not path:
            return
        try:
            nn.load_from_pickle(path)
            messagebox.showinfo("Model Loaded", f"Loaded parameters from:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{e}")


def main():
    root = tk.Tk()
    app = EditorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

