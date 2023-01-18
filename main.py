import string
import joblib

import nltk
from aiogram.types import ContentTypes

nltk.download('stopwords')
from nltk.corpus import stopwords

import pandas as pd
import numpy as np

from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.pipeline import Pipeline

from config import TOKEN


bot = Bot(token=TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await message.reply("Hello! It's bot for remove spam messages from public chats.\n"
                        "Only add this bot to your chat and get rid of spam messages.")



@dp.message_handler(commands=['train_model'])
async def train_model(message: types.Message):
    sms_data = pd.read_csv(
        "spam.csv", encoding='latin-1'
    )

    sms_data = sms_data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
    sms_data = sms_data.rename(columns={"v1": "label", "v2": "text"})

    sms_data["label_num"] = sms_data.label.map({"ham": 0, "spam": 1})
    sms_data["text_cleaned"] = sms_data["text"].apply(remove_punctuation_and_stopwords)

    ham_words = sms_data[sms_data.label == "ham"].text_cleaned.apply(lambda x: [word for word in x.split()])
    spam_words = sms_data[sms_data.label == "spam"].text_cleaned.apply(lambda x: [word for word in x.split()])

    ham_words_counter = Counter()
    for msg in ham_words:
        ham_words_counter.update(msg)

    spam_words_counter = Counter()
    for msg in spam_words:
        spam_words_counter.update(msg)

    X = sms_data.text_cleaned
    y = sms_data.label_num

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    vect = CountVectorizer()
    vect.fit(X_train)

    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)

    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit(X_train_dtm)
    tfidf_transformer.transform(X_train_dtm)

    nb = MultinomialNB()
    nb.fit(X_train_dtm, y_train)
    y_pred_class = nb.predict(X_test_dtm)
    metrics.accuracy_score(y_test, y_pred_class)
    metrics.confusion_matrix(y_test, y_pred_class)

    y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
    metrics.roc_auc_score(y_test, y_pred_prob)

    pipe = Pipeline([('bow', CountVectorizer()),
                     ('tfid', TfidfTransformer()),
                     ('model', MultinomialNB())])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metrics.accuracy_score(y_test, y_pred)
    metrics.confusion_matrix(y_test, y_pred)

    joblib.dump(pipe, 'injection_model.pkl')


@dp.message_handler(content_types=[types.ContentType.TEXT],
                    chat_type=[types.ChatType.SUPERGROUP, types.ChatType.GROUP])
async def message_handler(message: types.Message):
    clf = joblib.load('injection_model.pkl')
    if clf.predict([str(message.text)])[0]:
        if await bot.delete_message(message.chat.id, message.message_id):
            if not message.from_user.username:
                await bot.send_message(message.chat.id, f"Spam from **{message.from_user.first_name}** removed.")
            else:
                await bot.send_message(message.chat.id, f"Spam from @{message.from_user.username} removed.")


@dp.my_chat_member_handler()
async def bot_added_to_chat(my_chat_member: types.ChatMemberUpdated):
    await bot.send_message(my_chat_member.chat.id,
                           "Hi to all!\nI will fight spam in your chat room.\n\n"
                           "But, in order to do that, give me admin rights :3")


def remove_punctuation_and_stopwords(sms):
    nltk.download('stopwords')

    STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
    sms_no_punctuation = [ch for ch in sms if ch not in string.punctuation]
    sms_no_punctuation = "".join(sms_no_punctuation).split()

    sms_no_punctuation_no_stopwords = [word.lower() for word in sms_no_punctuation if word.lower() not in STOPWORDS]

    return ' '.join(sms_no_punctuation_no_stopwords)


if __name__ == '__main__':
    executor.start_polling(dp)
