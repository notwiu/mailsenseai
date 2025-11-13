import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

try:
  nltk.data.find('corpora/stopwords')
except:
  nltk.download('stopwords')
try:
  nltk.data.find('tokenizers/punkt')
except:
  nltk.download('punkt')

STOPWORDS = set(stopwords.words('portuguese'))
stemmer = RSLPStemmer()

def strip_accents(text):
  text = unicodedata.normalize('NFKD', text)
  return ''.join([c for c in text if not unicodedata.combining(c)])

def preprocess_text(text: str) -> str:
  if not isinstance(text, str):
    text = str(text)
  text = text.lower()
  text = re.sub(r"(^ | \n)(from|to|subject|sent|date):.*", " ", text)
  text = re.sub(r'\S+@\S+', ' ', text)
  text = re.sub(r'http\S+', ' ', text)
  text = re.sub(r'[^0-9a-zà-ú\s]', ' ', text)
  text = re.sub(r'\s+', ' ', text).strip()

  tokens = nltk.word_tokenize(text, language='portuguese')
  tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

  tokens = [stemmer.stem(t) for t in tokens]
  return ' '.join(tokens)
