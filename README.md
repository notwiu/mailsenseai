# MailSense AI
Resumo: aplicação web que classifica e-mails em **Produtivo** ou **Improdutivo** e gera respostas automáticas.

## Rodando localmente

1. Clone:
   git clone <repo_url>
2. Backend:
   cd backend
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
3. Treinar modelo (opcional, usa data/sample_emails.csv):
   python train_classifier.py
4. Rodar servidor:
   python app.py
5. Acesse: http://localhost:5000

## Deploy
- Plataformas sugeridas: Render, Heroku, Vercel (frontend) + Render/Heroku para backend ou Hugging Face Spaces (se migrar para Gradio/Streamlit).
- Certifique-se de: configurar variáveis de ambiente (OPENAI_API_KEY se usar OpenAI).

## Observações técnicas
- Pré-processamento: NLTK (tokenize, stopwords, lemmatize)
- Classifier: TF-IDF + LogisticRegression (treinável)
- Geração de respostas: OpenAI ChatCompletion (opcional) + templates de fallback
