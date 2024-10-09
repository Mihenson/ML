import numpy as np
import tkinter as tk
from tkinter import Canvas, Button, Label
from PIL import Image, ImageDraw
from tensorflow import keras

# Загружаем предварительно обученную модель
model = keras.models.load_model('fashion_mnist_model.h5')

# Имена классов
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Функция для предсказания
def predict(image):
    image = image.resize((28, 28)).convert('L') #
    image = np.array(image)
    image = 1 - image / 255.0  # Инвертируем цвета и нормализуем
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction

# Класс для приложения
class DrawApp:
    def __init__(self, root):
        self.root = root
        self.canvas = Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack()

        self.image = Image.new('L', (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind('<B1-Motion>', self.paint)
        
        self.predict_button = Button(root, text='Найти', command=self.predict_image)
        self.predict_button.pack()

        self.clear_button = Button(root, text='Сброс', command=self.clear_canvas)
        self.clear_button.pack()

        self.label = Label(root, text='Нарисуйте что-нибудь!')
        self.label.pack()

        self.probabilities_label = Label(root, text='')
        self.probabilities_label.pack()

    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-2, y-2, x+2, y+2, fill='black', width=10)
        self.draw.ellipse([x-2, y-2, x+2, y+2], fill='black')

    def predict_image(self):
        prediction = predict(self.image)
        predicted_class = class_names[np.argmax(prediction)]
        probabilities = "\n".join([f'{class_names[i]}: {prediction[0][i]*100:.2f}%' for i in range(len(class_names))])
        self.label.config(text=f'Предсказание: {predicted_class}')
        self.probabilities_label.config(text=f'Вероятности:\n{probabilities}')

    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.label.config(text='Нарисуйте что-нибудь!')
        self.probabilities_label.config(text='')

# Создаём и запускаем приложение
root = tk.Tk()
root.title("Fashion MNIST Predictor")
app = DrawApp(root)
root.mainloop()
