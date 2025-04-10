#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "run bilingual quart application (en, de) providing Four Shades of Life Sciences Service"
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2025 by Eva Seidlmayer"
__license__ = "ISC license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1 "

from quart import Quart, Response, request, render_template, render_template_string, session,  redirect, url_for
import joblib
import json
from transformers import BertTokenizer, AutoModelForSequenceClassification
import torch
import aiohttp
from quart_babel import Babel, _
from quart_session import Session
import os
import redis.asyncio as redis
import logging
import  redis


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Quart(__name__)
app.secret_key = "perasperaadastra"

# config session
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'session:'
app.config['SESSION_REDIS'] = redis.Redis(host='localhost', port=6379, db=0)

#app.config['SESSION_REDIS'] = 'redis://localhost:5000'  # Adjust the Redis URL as needed


async def init_redis_pool():
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    app.config['SESSION_REDIS'] = redis_client

Session(app)

@app.before_serving
async def startup():
    await init_redis_pool()


# languages
LANGUAGES = ['en', 'de']
babel = Babel(app)

app.config['BABEL_TRANSLATION_DIRECTORIES'] = os.path.join(os.path.dirname(__file__), 'translations')


async def init_redis_pool():
    try:
        # Create a Redis client to connect to the Redis server
        redis_client = redis.Redis(host='localhost', port=6379, db=0)

        # Try to ping the Redis server to verify the connection
        redis_client.ping()  # This sends a PING command to Redis
        logger.info("Redis connection established successfully.")
        print('the app is running')
        # If no error occurs, set the Redis client in the app configuration
        app.config['SESSION_REDIS'] = redis_client
    except redis.ConnectionError as e:
        # If Redis connection fails, log an error
        logger.error(f"Redis connection failed: {e}")
        raise Exception("Failed to connect to Redis server.")


def get_locale():
    """Manually set locale based on session or request."""
    print('get locale is running')
    return session.get("language") or request.accept_languages.best_match(["en", "de"])

babel.init_app(app, locale_selector=get_locale)
logger.info(f"DEBUG mode is {'on' if app.debug else 'off'}")

@app.route("/")
async def home():
    print('home is running')
    predictions_1 = []
    predictions_2 = []
    num_predictions = []

    return await render_template("index.html", title=_("Welcome"), num_predictions=num_predictions, predictions_1=predictions_1, predictions_2=predictions_2)

@app.route("/index")
def index():
    return f"Current language: {get_locale()}"


@app.route('/language')
async def language():
    return await render_template('language.html')  # Language selection page

@app.route('/set_language/<lang>')
async def set_language(lang):
    try:
        logger.info(f'Attempting to set language to: {lang}')
        if lang in LANGUAGES:
            session['language'] = lang
            logger.info(f'Language set to: {lang}')
        else:
            logger.warning(f'Invalid language: {lang}')
        return redirect(url_for('home'))
    except Exception as e:
        logger.error(f'Error setting language: {e}')
        print('language:', lang)
        return "Internal Server Error", 500

async def Call_AQUAS_RandomForest(text, classifier, vectorizer):
    text_list = [text]
    # vectorize!
    text_vectorized = vectorizer.transform(text_list)
    # predict!
    predictions = classifier.predict(text_vectorized)
    predictions_int = predictions.astype(int).tolist()

    return {'Probabilities': predictions_int}


async def Call_AQUAS_Bert(text, model, Bert_tokenizer):
    tokens = Bert_tokenizer(text, max_length=512, padding="max_length", truncation=True, return_tensors='pt')
    input_ids = tokens['input_ids']
    attn_mask = tokens['attention_mask']

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = output['logits']
        sigmoid_output = torch.sigmoid(logits)

        soft_output = torch.softmax(logits, -1)
        print('DAS IST DER SOFT OUTPUT', soft_output)
        # sigmoid = sigmoid_output.tolist()
        # pred_sci, pred_pop, pred_dis, pred_alt = sigmoid[0][0], sigmoid[0][1], sigmoid[0][2], sigmoid[0][3]
        print('DAS IST DER sigmoid OUTPUT', sigmoid_output)

    predicted_probabilities = sigmoid_output[0].tolist()

    return {
        'calculated probabilities': predicted_probabilities,
        'Probabilities': sigmoid_output.tolist()
    }

    # return pred_sci, pred_pop, pred_dis, pred_alt


async def CallWikifier(text, lang="en", threshold=0.8):
    # Prepare the URL.
    url = "http://www.wikifier.org/annotate-article"
    data = {
        "text": text, "lang": lang,
        "userKey": "amfkqziidvlmqytlboyaoxrysxgvgw",
        "pageRankSqThreshold": "%g" % threshold, "applyPageRankSqThreshold": "true",
        "nTopDfValuesToIgnore": "200", "nWordsToIgnoreFromList": "200",
        "wikiDataClasses": "true", "wikiDataClassIds": "false",
        "support": "true", "ranges": "false", "minLinkFrequency": "2",
        "includeCosines": "false", "maxMentionEntropy": "3"
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data) as resp:
            response = await resp.json()
            return response.get("annotations", [])


# Load the trained classifier from the file
RF_classifier = joblib.load(
    '/vol/2025-01-06_FSoLF-25-v5_random_forest_classifier.pkl')

# Load the vectorizer from the file
RF_vectorizer = joblib.load(
    '/vol/2025-01-06_FSoLF-25-v5_vectorizer.pkl')

SVM_classifier = joblib.load(
    '/vol/2025-01-10_FSoLF-25-v5_SVM_classifier.pkl')
SVM_vectorizer = joblib.load(
    '/vol/2025-01-10_FSoLF-25-v5_SVM_vectorizer.pkl')

LRG_classifier = joblib.load(
    '/vol/2025-01-10_FSoLF-25-v5_LRG_classifier.pkl')
LRG_vectorizer = joblib.load(
    '/vol/2025-01-10_FSoLF-25-v5_LRG_vectorizer.pkl')
Bertbase_model = AutoModelForSequenceClassification.from_pretrained(
    '/vol/FSoLS-24-v5_Bertbase_e1_lr3e-5_mlclass', num_labels=4)
Scibert_model = AutoModelForSequenceClassification.from_pretrained(
    '/vol/FSoLS-24-v5_SciBert_e3_lr3e-5_mlclass', num_labels=4)
SPECTER_model = AutoModelForSequenceClassification.from_pretrained(
    '/vol/FSoLS-24-v5_Specter_e3_lr3e-5_mlclass', num_labels=4)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


@app.route('/', methods=['GET', 'POST'])
async def input():
    response_1 = _('Please submit text for Wikifier Annotation')
    response_2 = _('Please submit text for text style classification')
    predictions_1 = []
    predictions_2 = []
    num_predictions = []
    user_text_A = ""
    user_text_B = ""

    if request.method == "POST":
        # Get user input from form
        form_data = await request.form
        user_text_A = form_data.get('user_text_A', '')
        user_text_A = user_text_A[:1000]
        user_text_B = form_data.get('user_text_B', '')
        option = form_data.get(_('option'))

        if user_text_A:
            annotations = await CallWikifier(user_text_A)
            response_1 = json.dumps(annotations, indent=2)

        if user_text_B:
            # Handle the selected option
            if option == 'option_1':
                results = await Call_AQUAS_RandomForest(user_text_B, RF_classifier, RF_vectorizer)
                predictions_1 = results['Probabilities'][0]
                response_2 = json.dumps(results, indent=2)
            if option == 'option_2':
                results = await Call_AQUAS_RandomForest(user_text_B, SVM_classifier, SVM_vectorizer)
                predictions_1 = results['Probabilities'][0]
                response_2 = json.dumps(results, indent=2)
            if option == 'option_3':
                results = await Call_AQUAS_RandomForest(user_text_B, LRG_classifier, LRG_vectorizer)
                predictions_1 = results['Probabilities'][0]
                response_2 = json.dumps(results, indent=2)
            elif option == 'option_4':
                results = await Call_AQUAS_Bert(user_text_B, Bertbase_model, bert_tokenizer)
                print(results)
                predictions_2 = results['Probabilities'][0]
                response_2 = json.dumps(results, indent=2)
            elif option == 'option_5':
                results = await Call_AQUAS_Bert(user_text_B, Scibert_model, bert_tokenizer)
                predictions_2 = results['Probabilities'][0]
                response_2 = json.dumps(results, indent=2)
            elif option == 'option_6':
                results = await Call_AQUAS_Bert(user_text_B, SPECTER_model, bert_tokenizer)
                predictions_2 = results['Probabilities'][0]
                response_2 = json.dumps(results, indent=2)
        if predictions_1:
            num_predictions = predictions_1
        elif predictions_2:
            num_predictions = predictions_2
        else:
            num_predictions = []
    return await render_template('index.html', user_text_A=user_text_A, user_text_B=user_text_B,
                                        response_1=response_1, response_2=response_2,
                                        predictions_1=predictions_1, predictions_2=predictions_2,
                                        num_predictions=json.dumps(num_predictions))


if __name__ == '__main__':
    app.run(debug=True)

