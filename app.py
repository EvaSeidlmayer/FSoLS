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
from quart_session.sessions import RedisSessionInterface
import os
import redis.asyncio as redis
import logging
from dotenv import load_dotenv
import uuid

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
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
app.config['BABEL_SUPPORTED_LOCALES'] = ['en', 'de']



if os.path.exists(".env"):
    load_dotenv()

@app.before_serving
async def setup():
    app.jinja_env.globals.update(_=_)

# Read environment variables
env = os.getenv("APP_ENV", "development")
path = os.getenv("APP_DATA_PATH", "./static" if env == "development" else "/vol")

print(f"Running in {env} mode. Data path: {path}")
print(f"APP_ENV: {os.getenv('APP_ENV')}")
print(f"APP_DATA_PATH: {os.getenv('APP_DATA_PATH')}")

Session(app)

@app.before_serving
async def startup():
    await init_redis_pool()

_original_save_session = RedisSessionInterface.save_session

async def patched_save_session(self, app, session, response):
    session_id = session.sid
    if isinstance(session_id, bytes):
        session_id = session_id.decode('utf-8')

    # call response.set_cookie manually with the corrected session_id
    response.set_cookie(
        self.session_cookie_name,
        session_id,
        max_age=self.permanent_session_lifetime.total_seconds() if session.permanent else None,
        expires=None,
        path=self.get_cookie_path(app),
        domain=self.get_cookie_domain(app),
        secure=app.config.get("SESSION_COOKIE_SECURE", False),
        httponly=app.config.get("SESSION_COOKIE_HTTPONLY", True),
        samesite=app.config.get("SESSION_COOKIE_SAMESITE", "Lax"),
    )

    await self.store_session(app, session.sid, session)

@app.before_request
async def debug_session():
    sid = getattr(session, 'sid', None)
    if isinstance(sid, bytes):
        sid = sid.decode('utf-8')
    print(f"[SESSION DEBUG] Session ID: {sid}")

# languages
LANGUAGES = ['en', 'de']
babel = Babel(app)

app.config['BABEL_TRANSLATION_DIRECTORIES'] = os.path.join(os.path.dirname(__file__), 'translations')


async def init_redis_pool():
    try:
        # Create a Redis client to connect to the Redis server
        redis_client = redis.Redis(host='localhost', port=6379, db=0)

        # Try to ping the Redis server to verify the connection
        await redis_client.ping()  # This sends a PING command to Redis
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

@app.before_request
async def log_language():
    current_language = session.get('language', 'default not set')
    logger.info(f"Current language in session: {current_language}")

@app.before_request
async def log_language_id():
    session_id = session.get('session_id')
    if isinstance(session_id, bytes):
        try:
            session['session_id'] = session_id.decode('utf-8')
        except UnicodeDecodeError:
            session['session_id'] = session_id.hex()
    print(f"Session ID: {session.get('session_id')}, Type: {type(session.get('session_id'))}")

@app.route('/set_language/<lang>')
async def set_language(lang):
    print(f"Setting language to: {lang}")
    if lang not in app.config['BABEL_SUPPORTED_LOCALES']:
        print(f"Unsupported language: {lang}")
        return 'Unsupported language', 400
    session['language'] = lang
    print(f"Language set to: {session['language']}")
    return redirect(request.referrer or url_for('index'))



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
RF_classifier = joblib.load(f'{path}/2025-01-06_FSoLF-25-v5_random_forest_classifier.pkl')

# Load the vectorizer from the file
RF_vectorizer = joblib.load(f'{path}/2025-01-06_FSoLF-25-v5_vectorizer.pkl')

SVM_classifier = joblib.load(f'{path}/2025-01-10_FSoLF-25-v5_SVM_classifier.pkl')
SVM_vectorizer = joblib.load(f'{path}/2025-01-10_FSoLF-25-v5_SVM_vectorizer.pkl')

LRG_classifier = joblib.load(f'{path}/2025-01-10_FSoLF-25-v5_LRG_classifier.pkl')
LRG_vectorizer = joblib.load(f'{path}/2025-01-10_FSoLF-25-v5_LRG_vectorizer.pkl')
Bertbase_model = AutoModelForSequenceClassification.from_pretrained(f'{path}/FSoLS-24-v5_Bertbase_e1_lr3e-5_mlclass', num_labels=4)
Scibert_model = AutoModelForSequenceClassification.from_pretrained(f'{path}/FSoLS-24-v5_SciBert_e3_lr3e-5_mlclass', num_labels=4)
SPECTER_model = AutoModelForSequenceClassification.from_pretrained(f'{path}/FSoLS-24-v5_Specter_e3_lr3e-5_mlclass', num_labels=4)
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
                print('typ',type(results))
                predictions_2 = results['Probabilities'][0]
                result = ('Probability for scientific text class: ', results['Probabilities'][0][0],
                          'Probability for vernacular text class: ', results['Probabilities'][0][1],
                          'Probability for disinformative text class: ', results['Probabilities'][0][2],
                          'Probability for alternative scientific text class: ', results['Probabilities'][0][3])
                response_2 = json.dumps(result, indent=2)
            elif option == 'option_5':
                results = await Call_AQUAS_Bert(user_text_B, Scibert_model, bert_tokenizer)
                predictions_2 = results['Probabilities'][0]
                response_2 = json.dumps(results, indent=2)
            elif option == 'option_6':
                results = await Call_AQUAS_Bert(user_text_B, SPECTER_model, bert_tokenizer)
                predictions_2 = results['Probabilities'][0]
                response_2 = json.dumps(results, indent=2)

        num_predictions = predictions_1 or predictions_2 or []
        print("num_predictions:", num_predictions)  # Debugging statement
    return await render_template('index.html', user_text_A=user_text_A, user_text_B=user_text_B,
                                        response_1=response_1, response_2=response_2,
                                        predictions_1=predictions_1, predictions_2=predictions_2,
                                        num_predictions=json.dumps(num_predictions))


if __name__ == '__main__':
    app.run(debug=True )

