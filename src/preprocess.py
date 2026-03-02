import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ensure nltk data is available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

_stopwords = set(stopwords.words('english'))
_lemmatizer = WordNetLemmatizer()

URL_PATTERN = re.compile(r'http\S+|www\.\S+')
HTML_PATTERN = re.compile(r'<.*?>')
NON_ALPHABETIC = re.compile(r'[^a-z\s]')

def clean_text(text: str) -> str:
    """
    Basic cleaning pipeline returning a cleaned string.
    Steps: lower, remove urls, remove html, keep letters only, tokenize,
    remove stopwords, lemmatize.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = URL_PATTERN.sub(' ', text)
    text = HTML_PATTERN.sub(' ', text)
    text = NON_ALPHABETIC.sub(' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in _stopwords and len(t) > 2]
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def concat_and_clean(title: str, content: str) -> str:
    """Concatenate title and content, then clean."""
    title = title or ''
    content = content or ''
    return clean_text(f"{title} {content}")
