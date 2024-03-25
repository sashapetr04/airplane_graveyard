import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk


class ImageSegmentationApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Segmentation App")

        # Создаем кнопку "Load Image"
        self.load_button = tk.Button(master, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=0)
        self.exit_button = tk.Button(master, text="Exit", command=self.exit_app)
        self.exit_button.pack(pady=0)

        # Переменные
        self.image = None
        self.segmented_image = None

        # Создаем фрейм для отображения изображения
        self.image_frame = tk.Frame(master)
        self.image_frame.pack(pady=0)

        # Поле для отображения изображения
        self.image_label = tk.Label(self.image_frame, width=900, height=700)
        self.image_label.pack(pady=0)

        # Создаем фрейм для выбора фильтра
        self.filter_frame = tk.Frame(master)
        self.filter_frame.pack(pady=0)

        # Создаем выпадающий список для выбора типа фильтра
        self.filter_type_label = tk.Label(self.filter_frame, text="Select Filter:")
        self.filter_type_label.grid(row=0, column=0, pady=0)
        self.filter_type_var = tk.StringVar()
        self.filter_type_combobox = ttk.Combobox(self.filter_frame, textvariable=self.filter_type_var)
        self.filter_type_combobox['values'] = ('Gaussian', 'Bilateral', 'Median', 'Minimum', 'Maximum')
        self.filter_type_combobox.current(0)
        self.filter_type_combobox.grid(row=0, column=1, pady=0)

        # Создаем поле ввода для размера ядра
        self.kernel_size_label = tk.Label(self.filter_frame, text="Kernel Size:")
        self.kernel_size_label.grid(row=1, column=0, pady=0)
        self.kernel_size_entry = tk.Entry(self.filter_frame)
        self.kernel_size_entry.insert(0, '3')  # Значение по умолчанию
        self.kernel_size_entry.grid(row=1, column=1, pady=0)

        # Создаем кнопку "Применить фильтр"
        self.apply_filter_button = tk.Button(self.filter_frame, text="Apply Filter", command=self.apply_filter)
        self.apply_filter_button.grid(row=2, columnspan=2, pady=0)

        # Создаем кнопки
        self.binary_button = tk.Button(master, text="Binary Image", command=self.binary_image)
        self.segment_button = tk.Button(master, text="Segment Image", command=self.segment_image)
        self.detect_planes_button = tk.Button(master, text="Detect Planes", command=self.detect_planes)

        self.binary_button.pack()
        self.segment_button.pack()
        self.detect_planes_button.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.display_image(self.image)

    def display_image(self, image):
        # Создаем копию изображения для отображения
        display_image = image.copy()

        # Масштабирование изображения для отображения на экране
        max_height = 700
        max_width = 900
        if image.shape[0] > max_height or image.shape[1] > max_width:
            scale = min(max_height / image.shape[0], max_width / image.shape[1])
            display_image = cv2.resize(display_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        display_image = Image.fromarray(display_image)
        image_tk = ImageTk.PhotoImage(display_image)

        # Отображение изображения
        self.image_label.configure(image=image_tk)
        self.image_label.image = image_tk

    def apply_filter(self):
        if self.image is None:
            messagebox.showerror("Error", "No image loaded!")
            return

        filter_type = self.filter_type_var.get()
        kernel_size = int(self.kernel_size_entry.get())

        if filter_type == 'Gaussian':
            filtered_image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)
        elif filter_type == 'Bilateral':
            filtered_image = cv2.bilateralFilter(self.image, kernel_size, sigmaColor=75, sigmaSpace=75)
        elif filter_type == 'Median':
            filtered_image = cv2.medianBlur(self.image, kernel_size)
        elif filter_type == 'Minimum':
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            filtered_image = cv2.erode(self.image, kernel, iterations=1)
        elif filter_type == 'Maximum':
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            filtered_image = cv2.dilate(self.image, kernel, iterations=1)
        else:
            filtered_image = self.image

        self.display_image(filtered_image)
        self.image = filtered_image


    def binary_image(self):
        if self.image is None:
            messagebox.showerror("Error", "No image loaded!")
            return

        binary_img = self.binary_image_process(self.image)
        self.display_image(binary_img)

    def binary_image_process(self, image, percentile_value=94):
        segmented_image = image.copy()
        b = image[:, :, 0]
        g = image[:, :, 1]
        r = image[:, :, 2]
        br = 0.299 * r + 0.587 * g + 0.114 * b
        mask = np.logical_not(np.logical_or.reduce((r < b - 15,
                                                    np.logical_and(r < b - 5, br > 180),
                                                    np.logical_and(g < b, np.logical_and(r < b, b > 210)))))
        segmented_image[mask] = [0, 0, 0]  # Черный цвет (фон)

        gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
        threshold = np.percentile(gray, percentile_value)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return binary

    def segment_image(self):
        if self.image is None:
            messagebox.showerror("Error", "No image loaded!")
            return

        self.segmented_image = self.segmented_image_process(self.image)
        self.display_image(self.segmented_image)

    def segmented_image_process(self, image):
        segmented_image = self.binary_image_process(image, 70)
        segmented_image = self.split_and_remove(self.fill_holes(segmented_image), 30)
        return segmented_image

    def fill_holes(self, binary, kernel_size=(2, 2)):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return closing

    def split_and_remove(self, img, percentile=10):
        image = img.copy()
        height, width = image.shape[:2]
        h_coef = height // 300 + 1
        w_coef = width // 400 + 1
        # Разделение изображения части
        cell_height = height // h_coef
        cell_width = width // w_coef

        for i in range(w_coef):
            for j in range(h_coef):
                x1 = i * cell_width
                y1 = j * cell_height
                x2 = (i + 1) * cell_width
                y2 = (j + 1) * cell_height
                cell = image[y1:y2, x1:x2]
                # Нахождение контуров в текущей ячейке
                contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue

                areas = [cv2.contourArea(cnt) for cnt in contours]
                min_area = np.percentile(np.unique(areas), percentile)

                for contour in contours:
                    if cv2.contourArea(contour) < min_area:
                        cv2.drawContours(cell, [contour], -1, 0, thickness=cv2.FILLED)

                image[y1:y2, x1:x2] = cell

        return image

    def detect_planes(self):
        if self.segmented_image is None:
            messagebox.showerror("Error", "No segmented image available!")
            return

        planes_count = self.count_planes(self.segmented_image)
        messagebox.showinfo("Planes Detected", f"Number of planes detected: {planes_count}")

    def count_planes(self, binary):
        connectivity = 4
        output = cv2.connectedComponentsWithStats(binary, connectivity, cv2.CV_32S)
        return output[0]

    def exit_app(self):
        self.master.destroy()


def main():
    root = tk.Tk()
    app = ImageSegmentationApp(root)
    root.attributes('-fullscreen', True)
    root.mainloop()


if __name__ == "__main__":
    main()
