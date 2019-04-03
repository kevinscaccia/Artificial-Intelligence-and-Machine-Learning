from sklearn import datasets

########################################
# Load and organize data into different arrays
########################################
iris_dataset = datasets.load_wine()  # load
data = iris_dataset['data']  # data values
target = iris_dataset['target']  # its targets
target_names = iris_dataset['target_names']
# split data index
split_80 = int(len(data)*0.8)  # *0.8 = get 80%
# 80% for train
train_data = data[0:split_80]
train_target = target[0:split_80]
# 20% for test
test_data = data[split_80:]
test_target = target[split_80:]


########################################
# The KNN Classifier it-self
########################################
from classifiers.knn import knn

k = 3
success = 0
fail = 0

# for each example in test data
for i in range(0, len(test_data)):
    clss = knn.classify(test_data[i], train_data, train_target, k)
    if clss == test_target[i]:
        success += 1
        print("Classified with success!!, its a {}".format(target_names[clss]))
    else:
        fail += 1
        print("Sorry, its a {}, not {}".format(target_names[test_target[i]], target_names[clss]))


print("Success rate: {}%".format(100*success/float(len(test_data))))





'''
########################################
# Testa Classificador
########################################
# Gera data e a classe de cada exemplo
dataSet, labels = gerar_data()

# Plota data de treino
plt.scatter(dataSet[:,0], dataSet[:,1], c=labels)
plt.title("Data")
plt.xlabel("Variavel X")
plt.ylabel("Variavel Y")

# cria um ponto aleatorio entre 0 e 100
novo_ponto = np.random.randint(100, size=2)# [inx, iny]
# plota e classifica o ponto
knn_classifier(novo_ponto, dataSet, labels, 3)
plt.show()
########################################
'''
