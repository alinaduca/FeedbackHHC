import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
import os


class ImageDisplayApp:
    def __init__(self, master, image_paths, default_width=300, default_height=300):
        self.master = master
        self.master.title("FeedbackHHC")
        self.image_paths = image_paths
        self.default_width = default_width
        self.default_height = default_height
        self.image_var = tk.StringVar(self.master)
        self.image_var.set(self.image_paths[0])  # Set the default image
        self.image_dropdown = tk.OptionMenu(self.master, self.image_var, *self.image_paths, command=self.load_image)
        self.image_dropdown.pack()
        self.label = tk.Label(self.master)
        self.label.pack()
        self.load_image(self.image_var.get())

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
        if file_path:
            self.image_paths.append(file_path)
            self.image_dropdown['menu'].delete(0, 'end')  # Clear existing menu options
            for path in self.image_paths:
                self.image_dropdown['menu'].add_command(label=os.path.basename(path), command=tk._setit(self.image_var, path))
            self.load_image(self.image_var.get(), self.default_width, self.default_height)

    def load_image(self, selected_path, width=None, height=None):
        image = Image.open(selected_path)
        if width and height:
            image = image.resize((width, height), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        self.label.config(image=photo)
        self.label.image = photo


if __name__ == "__main__":
    image_paths = [
        'corr-based/ROC_categorical.png',
        'corr-based/ROC_per_label.png',
        'all-data/ROC_all-data_categorical.png',
        'all-data/ROC_all-data_per_label.png',
        'corr-based/confusion_matrix_NeuralNetwork_sklearn.png',
        'corr-based/confusion_matrix_RandomForest_sklearn.png',
        'all-data/confusion_matrix_all-data_NeuralNetwork_sklearn.png',
        'all-data/confusion_matrix_all-data_RandomForest_sklearn.png',
        'corr-based/visualize_misclassified_pointsNN.png',
        'corr-based/visualize_misclassified_pointsRF.png',
        'all-data/visualize_misclassified_pointsNN_all-data.png',
        'all-data/visualize_misclassified_pointsRF_all-data.png',
        '../Tema2/confusion_matrix_NeuralNetwork.png',
        '../Tema2/confusion_matrix_RandomForest50_100.png',
        '../Tema2/confusion_matrix_RandomForest100_100.png',
    ]
    root = tk.Tk()
    app = ImageDisplayApp(root, image_paths, default_width=300, default_height=300)
    root.mainloop()
