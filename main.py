from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
import telebot
import random
import tensorflow as tf
from tensorflow.keras.activations import softplus
from tensorflow.keras.activations import elu
from tensorflow.keras.layers import LeakyReLU

# токен бота
from config import TOKEN

bot = telebot.TeleBot(TOKEN)

# модель для сложения / больше не треш
model_addition = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=[2]),
    tf.keras.layers.Dense(units=2)
])
model_addition.compile(optimizer='adam', loss='mean_absolute_error')

# модель для вычитания / не норм
model_subtraction = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=[2]),
    tf.keras.layers.Dense(units=3)
])
model_subtraction.compile(optimizer='adam', loss='mean_absolute_error')

# модель для умножения / ульратреш
model_multiplication = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation=LeakyReLU(alpha=0.5), input_shape=[2]),
    tf.keras.layers.Dense(units=2)
])
model_multiplication.compile(optimizer='adam', loss='mean_absolute_error')

# модель для деления / не трогать
model_division = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=[2]),
    tf.keras.layers.Dense(units=1)
])
model_division.compile(optimizer='sgd', loss='mean_absolute_error')

# обучающие данные для сложения
X_train_addition = np.array([[i / 10, j / 10] for i in range(1, 11) for j in range(1, 11)])
y_train_addition = np.array([[i / 10 + j / 10] for i in range(1, 11) for j in range(1, 11)])

# обучающие данные для вычитания
X_train_subtraction = np.array([[i / 10, j / 10] for i in range(1, 11) for j in range(1, 11)])
y_train_subtraction = np.array([[i / 10 - j / 10] for i in range(1, 11) for j in range(1, 11)])

# обучающие данные для умножения
X_train_multiplication = np.array([[i / 10, j / 10] for i in range(1, 11) for j in range(1, 11)])
y_train_multiplication = np.array([[i / 10 * j / 10] for i in range(1, 11) for j in range(1, 11)])

# обучающие данные для деления
X_train_division = np.array([[i, j] for i in range(1, 11) for j in range(1, 11) if j != 0])
y_train_division = np.array([[i / j] for i in range(1, 11) for j in range(1, 11) if j != 0])

epochs = 3000

# обучение моделей
model_addition.fit(X_train_addition, y_train_addition, epochs=epochs)
model_subtraction.fit(X_train_subtraction, y_train_subtraction, epochs=epochs)
model_multiplication.fit(X_train_multiplication, y_train_multiplication, epochs=epochs)
model_division.fit(X_train_division, y_train_division, epochs=epochs)

user_difficulty = {}
user_results = {}


# дальше без комментов я ебал
def generate_math_image(num1, num2, operation):
    img = Image.new('RGB', (150, 100), color='white')
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 36)
    if operation == '+':
        d.text((20, 30), f"{num1} + {num2} =", fill=(0, 0, 0), font=font)
    elif operation == '-':
        d.text((20, 30), f"{num1} - {num2} =", fill=(0, 0, 0), font=font)
    elif operation == '*':
        d.text((20, 30), f"{num1} * {num2} =", fill=(0, 0, 0), font=font)
    elif operation == '/':
        d.text((20, 30), f"{num1} / {num2} =", fill=(0, 0, 0), font=font)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


@bot.message_handler(commands=['start'])
def handle_start(message):
    keyboard = telebot.types.InlineKeyboardMarkup()
    keyboard.row(
        telebot.types.InlineKeyboardButton('Легко', callback_data='easy'),
        telebot.types.InlineKeyboardButton('Средне', callback_data='medium'),
        telebot.types.InlineKeyboardButton('Сложно', callback_data='hard')
    )
    bot.send_message(message.chat.id, "Привет! Я бот-помощник в обучении. Выберите уровень сложности:",
                     reply_markup=keyboard)


@bot.callback_query_handler(func=lambda call: True)
def handle_difficulty_callback(call):
    user_difficulty[call.message.chat.id] = call.data
    bot.send_message(call.message.chat.id, f"Вы выбрали уровень сложности: {call.data}")


@bot.message_handler(commands=['math'])
def handle_math_exercise(message):
    if message.chat.id not in user_difficulty:
        bot.send_message(message.chat.id, "Выберите уровень сложности сначала.")
        return

    difficulty = user_difficulty[message.chat.id]
    num1, num2 = generate_numbers(difficulty)
    operation = random.choice(['+', '-', '*', '/'])

    image_buffer = generate_math_image(num1, num2, operation)
    bot.send_photo(message.chat.id, photo=image_buffer)

    bot.num1 = num1
    bot.num2 = num2
    bot.operation = operation

    if operation == '+':
        bot.current_model = model_addition
    elif operation == '-':
        bot.current_model = model_subtraction
    elif operation == '*':
        bot.current_model = model_multiplication
    elif operation == '/':
        bot.current_model = model_division


@bot.message_handler(commands=['help'])
def handle_help(message):
    help_text = "Доступные команды:\n"
    help_text += "/start - Начать общение с ботом\n"
    help_text += "/math - Решить математическое упражнение\n"
    help_text += "/stats - Посмотреть статистику результатов\n"
    help_text += "/difficulty - Выбрать уровень сложности\n"
    bot.send_message(message.chat.id, help_text)


@bot.message_handler(commands=['stats'])
def handle_stats(message):
    if message.chat.id in user_results:
        correct_answers = user_results[message.chat.id]['correct']
        total_answers = user_results[message.chat.id]['total']
        if total_answers > 0:
            accuracy = correct_answers / total_answers * 100
            bot.send_message(message.chat.id, f"Статистика ваших ответов:\n"
                                              f"Правильные ответы: {correct_answers}\n"
                                              f"Всего ответов: {total_answers}\n"
                                              f"Точность: {accuracy:.2f}%")
        else:
            bot.send_message(message.chat.id, "Вы еще не решали ни одного упражнения.")
    else:
        bot.send_message(message.chat.id, "Вы еще не решали ни одного упражнения.")


def generate_numbers(difficulty):
    if difficulty == 'easy':
        return random.randint(1, 10), random.randint(1, 10)
    elif difficulty == 'medium':
        return random.randint(1, 100), random.randint(1, 100)
    elif difficulty == 'hard':
        return random.randint(1, 1000), random.randint(1, 1000)


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    if message.text.isdigit():
        try:
            user_answer = float(message.text)
            predicted_answer = bot.current_model.predict(np.array([[bot.num1, bot.num2]]))[0][0]
            correct_answer = eval(f"{bot.num1} {bot.operation} {bot.num2}")
            difference = abs(predicted_answer - user_answer)
            print("Предсказанный ответ:", predicted_answer)
            print("Данный ответ:", user_answer)
            print("Правильный ответ:", correct_answer)
            print("Разница:", difference)
            threshold = 0.2
            if difference < threshold:
                bot.send_message(message.chat.id, "Правильно!")
                if message.chat.id not in user_results:
                    user_results[message.chat.id] = {'correct': 0, 'total': 0}
                user_results[message.chat.id]['correct'] += 1
            else:
                bot.send_message(message.chat.id, "Неправильно! Правильный ответ: {}".format(correct_answer))
            if message.chat.id not in user_results:
                user_results[message.chat.id] = {'correct': 0, 'total': 0}
            user_results[message.chat.id]['total'] += 1
        except ValueError:
            bot.send_message(message.chat.id, "Введите число.")


bot.polling()
