import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess (text):
    text=text.lower()
    tokens=word_tokenize(text)
    
    tokens=[stemmer.stem(t) for t in tokens]

    return tokens

if __name__ == "__main__":
    tests = [
        "i need  some assistance",
        "Tell me a funny joke please",
        "I need some motivation",
        "What can you do?"
    ]
    for t in tests:
        print(f"Input:  {t}")
        print(f"Output: {preprocess(t)}")
        print()
