from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from ufal.udpipe import Model, Pipeline
import preproc

udpipe_model_url = 'https://rusvectores.org/static/models/udpipe_syntagrus.model'
udpipe_filename = udpipe_model_url.split('/')[-1]

modell = Model.load(udpipe_filename)
process_pipeline = Pipeline(modell, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')

def proctext(text):
    return preproc.process(process_pipeline, text)

import zipfile
from gensim import models

model_file = '/Users/tikhon/Desktop/CHOK/prog-Y/data/182.zip'

with zipfile.ZipFile(model_file, 'r') as archive:
    stream = archive.open('model.bin')
    model = models.KeyedVectors.load_word2vec_format(stream, binary=True)


def read_article(text):
    article = text.split(". ")
    sentences = []

    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()

    return sentences

def count2doc(doc, model):
    documents = [" ".join(doc)]
    mydict = corpora.Dictionary([simple_preprocess(line) for line in documents])
    corpus = [mydict.doc2bow(simple_preprocess(line)) for line in documents]

    tfidf = models.TfidfModel(corpus, smartirs='ntc')

    tfidf_dict = {}

    for doc in tfidf[corpus]:
        for id, freq in doc:
            w = mydict[id].split(sep='_')
            try:
                ww = w[0] + '_' + w[1].upper()
            except IndexError:
                pass
                continue
            tfidf_dict[ww]=freq


    w2v_dict = {}


    for word in tfidf_dict:
        try:
            vec = model[word]
        except KeyError:
            pass
            continue
        newvec = vec*tfidf_dict[word]
        w2v_dict[word] = newvec

    w2v_list = np.zeros(300,)

    for word in w2v_dict:
        w2v_list = w2v_list + w2v_dict[word]


    return w2v_list


def cosndist(vec1, vec2):
    return ds.cosine(vec1, vec2)

def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

def sentence_similarity(s1, s2):
    s1, s2 = " ".join(proctext(s1)), " ".join(proctext(s2))
    vec1 = count2doc(s1.split(), model)
    vec2 = count2doc(s2.split(), model)
    return 1 - cosndist(vec1, vec2)

def generate_summary(text, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    sentences =  read_article(text)

    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))

    return ". ".join(summarize_text);
