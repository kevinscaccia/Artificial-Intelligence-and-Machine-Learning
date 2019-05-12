import numpy as np


# creates some example data to experiment
def load_data():
    data = [
        ['kevin','é','um','cara','filho','da','puta'],
        ['eu','te','amo','coisa','fofa'],
        ['essa','coisa','de','corno','arrombado','puta'],
        ['seu','filho','da','puta','coisa','puta'],
        ['coisa','linda','minha','te','amo'],
    ]
    labels = [1, 0, 1, 1, 0]  # ofensivo
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
            occurrences[vocabulary.index(word)] = 1 ## or += 1 to count occurrences
        else:
            print("The word {} is not in the vocabulary", word)
    return occurrences

# train (calculate posteriori)
def train_naive_bayes(train_matrix, classes):
    #count the number of documents in each class
    num_documents = len(train_matrix)  # total of documents
    num_words = len(train_matrix[0])


    pAbusive = sum(classes)/float(len(classes))  # abusive = 1(count abusives)
    pnoAbusive = 1 - pAbusive

    abusive_numerador = np.zeros(num_words)
    noAbusive_numerador = np.zeros(num_words)
    abusive_denominador = 0.0
    noAbusive_denominador = 0.0
    # for each training document
    for i in range(num_documents):
        if classes[i] == 1:  # solve abusive
            abusive_numerador += train_matrix[i]  # soma de vetores
            abusive_denominador += sum(train_matrix[i])
        else:
            noAbusive_numerador += train_matrix[i]
            noAbusive_denominador += sum(train_matrix[i])
    # divide every element by the total number of words for that class
    abusive_vector = abusive_numerador/abusive_denominador
    noAbusive_vector = noAbusive_numerador / noAbusive_denominador
    return abusive_vector, noAbusive_vector, pAbusive



data, labels = load_data()
vocabulary = create_vocabulary(data)
occurrences = list_of_occurrences(vocabulary, data[0])
print(vocabulary)
train_matrix = []
for document in data:
    train_matrix.append(list_of_occurrences(vocabulary, document))
abusive_vector, noAbusive_vector, pAbusive = train_naive_bayes(train_matrix, labels)
print(pAbusive)
#print(abusive_vector)
#for i in range(len(vocabulary)):
#    print("{} abusive in {:.5}% cases".format(vocabulary[i], abusive_vector[i]*100))

#print(data[0])
#print(vocabulary)
#print(occurrences)

