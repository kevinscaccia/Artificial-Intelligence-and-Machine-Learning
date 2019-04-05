import numpy as np
import json
import re


########################################
# Carrega Base de Dados de Tweets
########################################
def load_tweets(N):
    tweets = []
    for line in open('data.json', 'r'):
        tweets.append(json.loads(line))
    data = []
    labels = []
    if N > len(tweets): N = len(tweets)  # pega tudo caso queira mais que o que tem
    # Classe 1: Cyber-Troll
    # Classe 2: Normal-User
    for i in range(0, N):
        rand = np.random.randint(0, len(tweets))  # get a random tweet
        data.append(tweets[rand]['content'])
        labels.append(tweets[rand]['annotation']['label'][0])
    print("> Carregados {} Tweets!".format(len(data)))
    return data, labels


########################################
# Pre-processa os Tweets
########################################
def preprocessar(document):
    doc = re.sub(r'[^\w\s]', '', document)  # remove pontuacao
    doc = re.sub(r'\b\w{1,3}\b', '', doc)  # somente palavras de len(word) > 3
    #print("Tweet carregado: {}".format(doc.split()))
    return doc.split()  # vetor de palavras


########################################
# Com base nos Tweets, cria um vocabulario
########################################
def create_vocabulary(dataset):
    vocabulary = set([]) # set com todas as palavras encontradas nos documentos
    print("> Preprocessando palavras")
    for document in dataset:
        document_preprocessado = preprocessar(document)
        vocabulary = vocabulary.union( set(document_preprocessado) ) # adiciona as do doc atual
    print(">>> Vocabulario de {} palavras gerado!".format(len(vocabulary)))
    return list(vocabulary) # lista ordenada de palavras(vocabulario do problema)


########################################
# Lista de ocorrencia(numero de) de cada palavra em um documento
########################################
def list_of_occurrences(vocabulary, document):
    occurrences = [0] * len(vocabulary)  # cada palavra com zero ocorrencias
    doc = preprocessar(document)  # preprocessa documento
    for word in doc:  # para cada palavra no documento preprocessado
        if word in vocabulary:  # se a palavra pertence ao vocabulario
            occurrences[vocabulary.index(word)] += 1 # soma mais uma ocorrencia na posicao dela do vetor

    return occurrences  # retorna vetor de ocorrencias (mesma ordem do vocabulario: importante!)


########################################
# Treina Bayes: Calcula Posteriories
########################################
def train_naive_bayes(vocabulary, train_documents, train_classes):
    print("> Treinando Classificador..")
    list_of_classes = list(set(train_classes))  # lista de classes
    prob_of_classes = {c: 0.0 for c in list_of_classes}  # probabilidade de cada classe
    # calcula probabilidade de cada classe p(c)
    for clss in list_of_classes:
        prob_of_classes[clss] = train_classes.count(clss) / float(len(train_classes))

    # probabilidade de cada palavra pertencer a cada classe p(w|c)
    prob_each_word = {c: np.ones(len(vocabulary)) for c in list_of_classes}  # pelo menos uma ocorrencia
    words_per_class = {c: len(list_of_classes) for c in list_of_classes} # numero palavras por classe (1)
    # para cada classe, contar a ocorrencia das
    # palavras do vocabulario, nessa classe
    for document_index in range( len(train_documents) ):
        document_class = train_classes[document_index]  # classe do documento
        # soma ocorrencias das palavras em cada classe soma de vetores
        prob_each_word[document_class] += train_documents[document_index]
        words_per_class[document_class] += sum(train_documents[document_index])
    #print("palavras na classe 0: {} ".format(words_per_class['0']))
    # calc probability of each word in each class
    for clss in list_of_classes:
        prob_each_word[clss] = np.log( prob_each_word[clss] / float(words_per_class[list_of_classes[0]]) )

    # Printa a probabilidade a priori P(palavra | classe)
    #for clss in list_of_classes:  # para cada classe
    #    print("----------------\nClasse {}:".format(clss))
    #    for word in range(len(vocabulary)):  # para cada palavra
    #        print(">>>> P({} | {}) = {:.4}%".format(vocabulary[word], clss, 100*prob_each_word[clss][word]))
    #    print("----------------")

    return list_of_classes, prob_of_classes, prob_each_word

########################################
# Classificador em Si
########################################
def naive_bayes_classify(document, prob_of_classes, prob_of_words, list_of_classes):
    #print("> Classificando..")
    prob = np.zeros(len(prob_of_classes))  # probabilidade P(classe|documento)
    for i in range(len(list_of_classes)):
        clss = list_of_classes[i]  # classe do documento
        # logaritmo para evitar o underflow
        prob[i] = sum(prob_of_words[clss] * document) + np.log(prob_of_classes[clss])

    max_index = np.argmax(prob)  # pega a classe com maior probabilidade
    return list_of_classes[max_index]


########################################
# Testa o Classificador
########################################
N = 1000  # pega aleatoriamente 1.000 tweets do banco (MAXIMO 20.001)
data, train_classes = load_tweets(N)
vocabulary = create_vocabulary(data)  # create vocabulary
# gera documentos de treino
print("> Gerando dados para Treino..")
train_documents = []
#for i in range(0, len(data)):
#    print("OCORRENCIAS: {}".format(list_of_occurrences(vocabulary, data[i])) )
#print(vocabulary)
for i in range(0, len(data)):
    train_documents.append(list_of_occurrences(vocabulary, data[i]))
print(">>> Dados de Treino Gerados..")
# TREINA O CLASSIFICADOR
list_of_classes, prob_of_classes, prob_of_words = train_naive_bayes(vocabulary, train_documents, train_classes)


########################################
# Matriz de confusao:
########################################
print("> Calculando Matriz de Confusao..")
import pandas
target_names = ['Troll', 'Normal']
classes_count = len(target_names)
# rows: actual class, cols: predicted class
confusion_matrix = np.zeros((classes_count, classes_count), dtype=int)

test_N = 50 # Calcula acuracia com 50 exemplos
print(">> Testando com {} Tweets".format(test_N))
accuracy = 0.0

# for each example in test data
for i in range(0, test_N):
    index = np.random.randint(0, len(data))
    instance = list_of_occurrences(vocabulary, data[index])
    predicted_class = naive_bayes_classify(instance, prob_of_classes, prob_of_words, list_of_classes)
    actual = train_classes[index]
    if (predicted_class == actual):
        accuracy += 1
    predicted = predicted_class
    confusion_matrix[int(actual)][int(predicted)] += 1  # sum 1

# Print Confusion Matrix
print("--- Confusion Matrix:")
print(pandas.DataFrame(confusion_matrix, target_names, target_names))
print("Accuracy: {:.4}%".format(100*(accuracy/test_N)))
