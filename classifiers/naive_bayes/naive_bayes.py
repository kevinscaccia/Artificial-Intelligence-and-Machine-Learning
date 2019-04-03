import numpy as np
from sklearn import datasets


newsgroups_train = datasets.fetch_20newsgroups(subset='train')

print(newsgroups_train)
print(list(newsgroups_train.target_names))




targets = newsgroups_train.target[:10]

# creates some example data to experiment
def load_data():
    data = [
            "vai se ferrar".split(' '),
            "vamos todos nos ferrar".split(' '),
            "biscoitos vao se ferrar".split(' '),
            "biscoitos sao legais".split(' '),
    ]
    labels = ['ofencivo','ofencivo','ofencivo', 'naoOfencivo']  # ofensivo
    return data, labels


# creates a set of words
def create_vocabulary(dataset):
    vocabulary = set([])
    for document in dataset:
        vocabulary = vocabulary.union(set(document))
    return list(vocabulary)

# list of vocabulary words and if a word appears in the document
def list_of_occurrences(vocabulary, document):
    occurrences = [0]*len(vocabulary)  # fill with zeros
    for word in document:
        if word in vocabulary:  # only for words in our vocabulary
            occurrences[vocabulary.index(word)] += 1 ## or += 1 to count occurrences
        else:
            print("The word {} is not in the vocabulary", word)
    return occurrences

# train (calculate posteriori)
def train_naive_bayes(vocabulary, train_documents, train_classes):
    list_of_classes = list( set(train_classes) )
    prob_of_classes = {c: 0.0 for c in list_of_classes}
    
    for clss in list_of_classes:
        prob_of_classes[clss] = train_classes.count(clss) / float( len(train_classes) )
        #print("Probabilidade da classe <{}>: {:.4}%".format(list_of_classes[i], prob_of_classes[i]*100))
    
    prob_each_word = {c: np.zeros(len(vocabulary)) for c in list_of_classes}
    words_per_class = {c: 0 for c in list_of_classes}
    
    #print(prob_each_word)
    # para cada classe, contar a ocorrencia das 
    # palavras do vocabulario, nessa classe
    for document_index in range( len(train_documents) ):
        document_class = train_classes[document_index]
        # soma de vetores
        prob_each_word[document_class] = prob_each_word[document_class] + train_documents[document_index]
        words_per_class[document_class] += sum( train_documents[document_index] )   
    
    print("number of words in the class {} is: {}".format( list_of_classes[0], words_per_class[list_of_classes[0]] ))
    for clss in list_of_classes:
        prob_each_word[clss] = prob_each_word[clss] / float( words_per_class[list_of_classes[0]] )
    
    print("probab. of word <{}> in class <{}> is {:.4}%".format( vocabulary[0], list_of_classes[0], 100*prob_each_word[list_of_classes[0]][0]))
    
    # for each class, what is the occourrences of a word(a word in vocabulary)
    #print(prob_each_word['classeB'])
    return list_of_classes, prob_of_classes, prob_each_word


def naive_bayes_classify(document, prob_of_classes, prob_of_words, list_of_classes):
    #
    prob = np.zeros( len(prob_of_classes) )
    for i in range( len(list_of_classes) ):
        clss = list_of_classes[i]
        prob[i] = sum(prob_of_words[clss] * document) * prob_of_classes[clss] 
        #print("probabilidade de ser da classe <{}>: {:.4}%".format(clss, 100*prob[i]))
    max_index = np.argmax(prob)
    
    return list_of_classes[max_index], prob[max_index]
    

'''
data, train_classes = load_data()
vocabulary = create_vocabulary(data)
train_documents = []
for document in data:
    train_documents.append(list_of_occurrences(vocabulary, document))

list_of_classes, prob_of_classes, prob_of_words = train_naive_bayes(vocabulary, train_documents, train_classes)

dt = ['biscoitos','gostam','de','q']
instance = list_of_occurrences(vocabulary, dt)

clss, prob = naive_bayes_classify(instance, prob_of_classes, prob_of_words, list_of_classes)

print("Classe: {} com {:.4}% de probabilidade".format( clss, 100*prob ))
'''
