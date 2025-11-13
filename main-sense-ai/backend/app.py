import os
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import pdfplumber
import joblib
import tempfile
from utils import preprocess_text
from dotenv import load_dotenv

# Load environment
load_dotenv()

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

app = Flask(__name__, static_folder='../frontend', static_url_path='/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = 'models/classifier.joblib'
classifier = None
vectorizer = None
pipeline = None

# Tentativa de carregar pipeline completo (mais simples)
if os.path.exists('models/pipeline_full.joblib'):
    pipeline = joblib.load('models/pipeline_full.joblib')
    print('Pipeline carregado (pipeline_full.joblib)')
elif os.path.exists(MODEL_PATH):
    data = joblib.load(MODEL_PATH)
    classifier = data.get('model')
    vectorizer = data.get('vectorizer')
    print('Modelo e vectorizer carregados')
else:
    print('Nenhum modelo encontrado. Rode train_classifier.py para treinar.')

# Helpers
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

import re

def extract_text_from_pdf(file_path):
    text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)
    return "\n".join(text)

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/classify', methods=['POST'])
def classify_email():
    text = request.form.get('text', '').strip()

    if 'file' in request.files and request.files['file'].filename != '':
        f = request.files['file']
        if allowed_file(f.filename):
            filename = secure_filename(f.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(tmp_path)
            ext = filename.rsplit('.', 1)[1].lower()
            if ext == 'pdf':
                text_from_file = extract_text_from_pdf(tmp_path)
            else:
                with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as fh:
                    text_from_file = fh.read()
            text = (text + "\n" + text_from_file).strip() if text else text_from_file
        else:
            return jsonify({'error': 'Formato de arquivo não permitido'}), 400

    if not text:
        return jsonify({'error': 'Nenhum texto ou arquivo enviado'}), 400

    cleaned = preprocess_text(text)

    # Inferência
    if pipeline is not None:
        probs = pipeline.predict_proba([cleaned])[0]
        classes = pipeline.classes_
        # alcança a classe com maior prob.
        idx = probs.argmax()
        category = classes[idx]
        score = float(probs[idx])
    elif classifier is not None and vectorizer is not None:
        X = vectorizer.transform([cleaned])
        probs = classifier.predict_proba(X)[0]
        labels = classifier.classes_
        # pega prob da classe 'Produtivo' se existir
        try:
            idxp = list(labels).index('Produtivo')
            prod_score = float(probs[idxp])
            category = 'Produtivo' if prod_score >= 0.5 else 'Improdutivo'
            score = prod_score if category == 'Produtivo' else 1 - prod_score
        except ValueError:
            category = labels[probs.argmax()]
            score = float(probs.max())
    else:
        # Heurística simples (fallback)
        produtivo_keywords = ['solicit', 'problema', 'erro', 'ajuda', 'suporte', 'status', 'anexo', 'documento', 'urgente', 'vencimento', 'pagamento', 'reclama']
        lower = cleaned.lower()
        is_prod = any(k in lower for k in produtivo_keywords)
        category = 'Produtivo' if is_prod else 'Improdutivo'
        score = 0.7 if is_prod else 0.4

    suggestion = generate_suggestion(category, text)

    return jsonify({'category': category, 'score': round(score, 3), 'suggestion': suggestion})


def generate_suggestion(category, original_text):
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if OPENAI_API_KEY:
        try:
            import openai
            openai.api_key = OPENAI_API_KEY
            prompt = build_prompt_for_response(category, original_text)
            resp = openai.ChatCompletion.create(
                model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                messages=[{'role':'system','content':'Você é um assistente que escreve respostas profissionais e concisas para e-mails corporativos.'},
                          {'role':'user','content':prompt}],
                max_tokens=200,
                temperature=0.2
            )
            suggestion_text = resp.choices[0].message.content.strip()
            return suggestion_text
        except Exception as e:
            print('OpenAI error:', e)

    # templates
    if category == 'Produtivo':
        first_line = original_text.strip().splitlines()[0][:200]
        return (f"Olá,\n\nObrigado pelo contato. Recebemos sua solicitação: \"{first_line}...\". "
                "Estamos verificando e retornaremos com uma atualização em até 24 horas. "
                "Caso seja urgente, por favor responda indicando 'URGENTE'.\n\nAtenciosamente,\nEquipe de Suporte")
    else:
        return ("Olá,\n\nAgradecemos a mensagem! Não há ações pendentes relacionadas a este e-mail. "
                "Caso precise de suporte ou queira abrir uma solicitação, por favor nos informe com detalhes.\n\nAtenciosamente,\nEquipe")


def build_prompt_for_response(category, original_text):
    if category == 'Produtivo':
        prompt = (
            "O e-mail abaixo foi classificado como 'Produtivo' (requer ação). Gere uma resposta profissional e concisa (máx 120 palavras) confirmando o recebimento, informando prazo padrão (24h) e pedindo informações adicionais se necessário.\n\nE-mail:\n\n" + original_text)
    else:
        prompt = (
            "O e-mail abaixo foi classificado como 'Improdutivo' (não requer ação). Gere uma resposta educada e curta (máx 60 palavras) agradecendo e explicando que não há ações pendentes.\n\nE-mail:\n\n" + original_text)
    return prompt

if __name__ == '__main__':
    app.run(debug=True, port=int(os.getenv('PORT', 5000)))