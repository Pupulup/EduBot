from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
import telebot
import random
import tensorflow as tf
from config import TOKEN

bot = telebot.TeleBot(TOKEN)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=[2]),
    tf.keras.layers.Dense(units=1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

X_train = np.array([[i / 1000, j / 1000] for i in range(1, 1001) for j in range(1, 1001)])
y_train = np.array([[i / 1000 + j / 1000] for i in range(1, 1001) for j in range(1, 1001)])

model.fit(X_train, y_train, epochs=3)

user_difficulty = {}
user_results = {}


def generate_math_image(num1, num2, operation):
    img = Image.new('RGB', (300, 100), color='white')
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 36)
    d.text((20, 30), f"{num1} {operation} {num2} =", fill=(0, 0, 0), font=font)
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
    operation = random.choice(['+'])

    image_buffer = generate_math_image(num1, num2, operation)
    bot.send_photo(message.chat.id, photo=image_buffer)

    bot.num1 = num1
    bot.num2 = num2
    bot.operation = operation


@bot.message_handler(commands=['help'])
def handle_help(message):
    help_text = "Доступные команды:\n"
    help_text += "/start - Начать общение с ботом\n"
    help_text += "/math - Решить математическое упражнение\n"
    help_text += "/stats - Посмотреть статистику результатов\n"
    help_text += "/difficulty - Выбрать уровень сложности\n"
    bot.send_message(message.chat.id, help_text)


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
            predicted_answer = model.predict(np.array([[bot.num1, bot.num2]]))[0][0]
            correct_answer = eval(f"{bot.num1} {bot.operation} {bot.num2}")
            difference = abs(predicted_answer - user_answer)
            print("Предсказанный ответ:", predicted_answer)
            print("Данный ответ:", correct_answer)
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
