import telebot
from datetime import datetime
from predict import predict_image
from keyboa import Keyboa


translate = {'shirts': 'рубашка',
             'sweaters': 'кофта',
             'T-shirts': 'футболка',
             'skirts': 'юбка',
             'pants': 'штаны',
             'dresses': 'платье',
             'shorts': 'шорты',
             'other': 'другое',
             'OK': 'ТЫ ПРАВ!',}


def get_keyboard(ans):
    return Keyboa(items=[translate[x] for x in translate if x != ans], copy_text_to_callback=True, items_in_row=2).keyboard


def get_token():
    with open('config.txt', 'r') as f:
        result = f.readline().strip()
        return result


token = get_token()
bot = telebot.TeleBot(token)

questions = dict()


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(message.from_user.id, 'Привет! Ты не знаешь, что за предмет одежды '
                                           'сфотографировал? Смело присылай его мне!')


@bot.message_handler(content_types=['photo'])
def handle_docs_photo(message):
    try:
        user_id = message.from_user.id
        file_info_1 = bot.get_file(message.photo[0].file_id)
        downloaded_file = bot.download_file(file_info_1.file_path)
        src = f"Photos/photo_{datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}.jpg"
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
        questions[user_id] = src
        send_answer(user_id)

    except Exception as e:
        bot.reply_to(message, str(e.__class__))


def send_answer(user_id):
    filename = questions[user_id]
    res = predict_image(filename)
    bot.send_message(user_id, f'Я думаю, это {translate[res]}')
    with open('log.txt', 'a+') as f:
        f.write(f'{filename} {res}\n')
    bot.send_message(user_id, 'Если я ошибся, напиши ты мне, что это такое!', reply_markup=get_keyboard(res))


@bot.callback_query_handler(func=lambda call: True)
def callback(call):
    user_id = call.message.chat.id
    with open('log.txt', 'a+') as f:
        f.write('answer ' + questions[user_id] + ' ' + call.data + '\n')
    if call.data == 'ТЫ ПРАВ!':
        bot.send_message(call.message.chat.id, 'Да, я молодец!')
    else:
        bot.send_message(call.message.chat.id, 'Я постараюсь исправиться, спасибо за помощь!')


if __name__ == '__main__':
    try:
        bot.polling(none_stop=True)
    except Exception as e:
        pass
