import collections
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint

def tokenizer(text):
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]
    return tokens



def cluster_sentences(sentences , nb_clusters = 2):
    tfidf_vector = TfidfVectorizer(tokenizer = tokenizer , stop_words = stopwords.words('english') , lowercase = True)
    tfidf_matrix = tfidf_vector.fit_transform(sentences)
    kmeans = KMeans(n_clusters = nb_clusters)
    kmeans.fit(tfidf_matrix)
    
    clusters = collections.defaultdict(list)
    for i,label in enumerate(kmeans.labels_):
        clusters[label].append(i)
        
    return dict(clusters)


if __name__== "__main__":
    sentences = ["Quantum physics is quite important Nowadays.",
                 "Astro physics loses popularity",
                 "Napa is a popular drug for fever",
                 "Square hospital is the most popular hospital in our countr",
                 "The Owner of Beximco CEO is a very cunning person.He know better where to investate money"]
    
    nclusters = 2
    clusters = cluster_sentences(sentences , nclusters)
    for cluster in range(nclusters):
        print("CLUSTERS",cluster,":")
        for i,sentence in enumerate(clusters[cluster]):
            print("\tSENTENCES",i,":",sentences[sentence])