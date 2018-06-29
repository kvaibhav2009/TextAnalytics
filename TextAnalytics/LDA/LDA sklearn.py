
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from tika import parser
from textblob import TextBlob


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic %d:" % (topic_idx)
        print " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])

#dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
#documents = dataset.data


def filteringText(sentence,stem=False,lemmatize=True,clean_special_chars_numbers=True,remove_stopwords=True,stops=set(stopwords.words("english"))):
    filteredText=list()
    if sentence.startswith('==') == False:
        sentence_text = sentence

        # Optionally remove non-letters (true by default)
        if clean_special_chars_numbers:
            sentence_text = re.sub("[^a-zA-Z]", " ", sentence_text)

        # Convert words to lower case and split them
        words = sentence_text.lower().split()

        # Optional stemmer
        if stem:
            stemmer = PorterStemmer()
            words = [stemmer.stem(w) for w in words]

        if lemmatize:
            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(w) for w in words]

        # Optionally remove stop words (false by default)
        if remove_stopwords:
            words = [w for w in words if not w in stops]

        filteredText.append(words)
    return filteredText

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

ppt_filename="../SOW 96 Extension - AI sample 1.pptx"

raw=parser.from_file(ppt_filename)
text=raw['content']

blob=TextBlob(text)

st=list()
zd=list()
for sentence in blob.sentences:
    print sentence
    sent=filteringText(str(sentence))
    st.append(sent)
    zd.append(str(" ".join(sent[0])))

documents=zd



no_features = 1000

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 5

# Run NMF
#nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

# Run LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

no_top_words = 10
#display_topics(nmf, tfidf_feature_names, no_top_words)
display_topics(lda, tf_feature_names, no_top_words)