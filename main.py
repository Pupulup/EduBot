import numpy as np
import telebot
import random
import tensorflow as tf
from config import TOKEN

bot = telebot.TeleBot(TOKEN)

# Создаем модель нейронной сети
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])
model.compile(optimizer='sgd', loss='mean_squared_error')

# Генерируем обучающие данные
X_train = np.array([[i / 100] for i in range(1, 1001)])
y_train = np.array([[i / 100] for i in range(1, 1001)])

# Обучаем нейронную сеть
model.fit(X_train, y_train, epochs=200)

# Глобальные переменные для хранения чисел и операции
num1 = None
num2 = None
operation = None


@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.send_message(message.chat.id, "Привет! Я бот-помощник в обучении. Как я могу помочь вам?")


@bot.message_handler(commands=['math'])
def handle_math_exercise(message):
    global num1, num2, operation
    # Генерация математического упражнения
    num1 = random.randint(1, 1000)
    num2 = random.randint(1, 1000)
    operation = random.choice(['+', '-', '*'])
    question = f"Решите пример: {num1} {operation} {num2}"
    # Отправка упражнения пользователю
    bot.send_message(message.chat.id, question)


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    global num1, num2, operation
    try:
        # Пробуем преобразовать ответ пользователя в число
        user_answer = float(message.text)
        # Проверяем ответ пользователя с помощью нейронной сети
        predicted_answer = model.predict(np.array([[user_answer]]))[0][0]
        correct_answer = eval(f"{num1} {operation} {num2}")
        difference = abs(predicted_answer - correct_answer)
        print("Предсказанный ответ:", predicted_answer)
        print("Правильный ответ:", correct_answer)
        print("Разница:", difference)
        # Установим порог для сравнения
        threshold = 0.15  # Увеличим порог для сравнения
        if difference < threshold:
            bot.send_message(message.chat.id, "Правильно!")
        else:
            bot.send_message(message.chat.id, "Неправильно! Попробуйте еще раз.")
    except ValueError:
        bot.send_message(message.chat.id, "Введите число.")


bot.polling()
