import streamlit as st
# pip install micropip
# import micropip
# async def install_packages():
#     await micropip.install('scikit-learn')
#     await micropip.install('sastrawi')
#     await micropip.install('nltk')

# Call the async function to install the packages
# await install_packages()
import re
import sklearn
import string
import pickle
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory #stopword remover
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory #stemming

# Load model dan objek CountVectorizer dari file pickle
with open('trained_model.pkl', 'rb') as file:
    model_data = pickle.load(file)

model = model_data['model']
vectorizer = model_data['vectorizer']
stemmer = StemmerFactory().create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
# Fungsi untuk melakukan prediksi sentimen
def preprocess_tweet2(tweet):
    EMOJI_PATTERN = re.compile(
    "(["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "])")
    tweet = re.sub(r'[0-9]+','', str(tweet))
    tweet = tweet.lower()  # convert to lower case
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)  # remove URLs
    tweet = re.sub(r'\@\w+|\#\w+', '', tweet)  # remove mentions and hashtags
    tweet = re.sub(r'\d+', '', tweet) # remove numbers
    tweet = re.sub(r'\.',' ', tweet) #
    tweet = tweet.translate(str.maketrans("", "", string.punctuation))  # remove punctuations
    tweet = tweet.strip()
    tweet = re.sub(EMOJI_PATTERN, r'', tweet)
    tweet = re.sub(r'\n+', '', tweet)
    tweet = re.sub(r'^\s+', '', tweet)
    tweet = re.sub(r'\brt', '', tweet)
    return tweet
#stemming
def stemming(text):
        # Menerapkan stemming pada setiap teks dalam batch
    unique_token = "aasdpemilu"  # Token unik untuk kata "pemilu"
    text = text.replace("pemilu", unique_token)  # Mengganti kata "pemilu" dengan token unik
    stem_text = stemmer.stem(text) if text.lower() not in ["pemilu", "pilpres"] else text  # Melakukan proses stemming
    stem_text = stem_text.replace(unique_token, "pemilu")  # Mengembalikan token unik menjadi kata "pemilu"
    
    return stem_text
# Stopword
def stopword(text):
        # Melakukan penghapusan stopwords pada setiap teks dalam batch
    stopwords = ['yg', 'dgn', 'kl', 'spt', 'pk', 'tp', 'krn', 'dr', 'utk', 'lg', 'gw', 'si', 'jg', 'jd', 'shg', 'sbg']
    stopwords = set(stopwords)
    words = text.split()
    cleaned_words = [word for word in words if word.lower() not in stopwords]
    cleaned_text = ''.join(cleaned_words)
    return cleaned_text 
    
def predict_sentiment(text):
    clean = preprocess_tweet2(text)
    stem_text = stemming(clean)
    clean_text = stopword(stem_text)
    text_vectorized = vectorizer.transform([clean_text])
    prediction = model.predict(text_vectorized)
    menarik = model.predict_proba(text_vectorized)
    return prediction[0], menarik, clean_text

# Tampilan aplikasi dengan Streamlit
st.title("Sentiment Analysis Pilpres 2024")

# Input teks
text = st.text_input("Input Kalimat")
kolom = ['Neg','Net','Pos']
# Tombol untuk melakukan prediksi
if st.button("Predict"):
    if text:
        sentiment = predict_sentiment(text)
        st.write("kalimat yang diuji:", sentiment[2])
        df = pd.DataFrame(sentiment[1], columns=kolom)
        if sentiment[0] == 'Positive':
            st.write("Sentiment:", "<span style='color:green;'>Positive 😆</span>", unsafe_allow_html=True)
            st.write(df)
        elif sentiment[0] == 'Negative':
            st.write("Sentiment:", "<span style='color:red;'>Negative 😭</span>", unsafe_allow_html=True)
            st.write(df)
        else:
            st.write("Sentiment:", "<span style='color:yellow;'>Neutral 😅</span>", unsafe_allow_html=True)
            st.write(df)
    else:
        st.write("Please input text.")
