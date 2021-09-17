import numpy as np
import telebot
import cv2
from keras.models import load_model

bot = telebot.TeleBot("Your Token")

model = load_model("model.h5")

width = 224
height = 224
@bot.message_handler(commands=['start'])
def start(messages):
    bot.send_message(messages.chat.id, f'welcome {messages.from_user.first_name}')
    bot.send_message(messages.chat.id, f'***Normal person or Sheikh***')
    bot.send_message(messages.chat.id, f'Please send me a photo😊')

@bot.message_handler(content_types=['photo'])
def photo(message):
    img_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(img_info.file_path)
    src = img_info.file_path
    with open(src, 'wb') as img_file:
        img_file.write(downloaded_file)

    img = cv2.imread(src)
    img = cv2.resize(img, (width, height))
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_np = np.array(img1)
    img_np = img_np / 255.0
    img_np = img_np.reshape(1, width, height, 3)

    y_pred = model.predict(img_np)
    prediction = np.argmax(y_pred)

    if prediction == 1:
      bot.reply_to(message, 'sheikh')
    else:
      bot.reply_to(message, 'normal person')


bot.polling()
