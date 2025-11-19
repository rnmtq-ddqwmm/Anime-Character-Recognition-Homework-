import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# ---------------- 模型加载 ----------------

MODEL_PATH = r"C:\Users\User\Downloads\UI\models\final_two_charate.h5"
model = load_model(MODEL_PATH)

IMG_WIDTH, IMG_HEIGHT = 128, 128
class_names = ["Hatsune Miku", "Yoisaki Kanade"]

# ---------------- GUI ----------------
root = tk.Tk()
root.title("Character Recognizer")
root.geometry("500x500")

label_image = tk.Label(root)
label_image.pack(pady=20)

def predict_image(img_path):
    img = load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    preds = model.predict(img_array)[0]
    
    class_idx = np.argmax(preds)
    confidence = preds[class_idx]
    
    messagebox.showinfo("Prediction Result", f"Character: {class_names[class_idx]}\nConfidence: {confidence:.2f}")

def load_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.webp")]
    )
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        label_image.config(image=img_tk)
        label_image.image = img_tk
        predict_image(file_path)

btn_load = tk.Button(root, text="Load Image", command=load_image, font=("Arial", 14))
btn_load.pack(pady=10)

root.mainloop()
