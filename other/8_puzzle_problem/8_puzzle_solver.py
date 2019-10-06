import numpy as np  # math
import random  # random
import time  # calculate runtime

###################################################
### Class representing a Board State
###################################################
class State:
    correct_positions = {0:[0,0], 1:[0,1], 2:[0,2],
                     3:[1,0], 4:[1,1], 5:[1,2],
                     6:[2,0], 7:[2,1], 8:[2,2]}
    board = []
    def __init__(self, board=[]):
        if len(board) == 0: # gera um board aleatorio
            self.board = np.arange(0,9)  # vetor com 9 posicoes
            random.shuffle(self.board)  # ordena aleatoriamente
            self.board = self.board.reshape(3,3)  # transforma em matrix 2D
        else:
            self.board = board  # board passado por parâmetro

    def neighbors(self):
        neighbors = []
        zero_row = 0
        zero_column = 0
        # encontra indice do elemento 0
        for i in range(0,3):
            for j in range(0,3):
                if self.board[i][j] == 0 :
                    zero_row = i
                    zero_col = j
                    break
        # movimenta zero para baixo
        if zero_row == 0 or zero_row == 1:
            neighbor = self.board.copy()  # copia State atual
            # faz o movimento
            neighbor[zero_row][zero_col] = neighbor[zero_row + 1][zero_col] # sobe o numero
            neighbor[zero_row + 1][zero_col] = 0# desce o zero
            neighbors.append(State(neighbor))
        # movimenta zero para cima
        if zero_row == 1 or zero_row == 2:
            neighbor = self.board.copy()  # copia State atual
            # faz o movimento
            neighbor[zero_row][zero_col] = neighbor[zero_row - 1][zero_col]  # desce o numero
            neighbor[zero_row - 1][zero_col] = 0  # sobe o zero
            neighbors.append(State(neighbor))
        # movimenta zero para direita
        if zero_col == 0 or zero_col == 1:
            neighbor = self.board.copy()  # copia State atual
            # faz o movimento
            neighbor[zero_row][zero_col] = neighbor[zero_row][zero_col + 1]  # move o numero
            neighbor[zero_row][zero_col + 1] = 0  # move o zero
            neighbors.append(State(neighbor))
        # movimenta zero para esquerda
        if zero_col == 1 or zero_col == 2:
            neighbor = self.board.copy()  # copia State atual
            # faz o movimento
            neighbor[zero_row][zero_col] = neighbor[zero_row][zero_col - 1]  # move o numero
            neighbor[zero_row][zero_col - 1] = 0  # move o zero
            neighbors.append(State(neighbor))
        return neighbors

    def distance_heuristic(self):
        h = 0
        for line in range(0, 3):
            for column in range(0, 3):
                num = self.board[line,column]
                atual = [line,column]  # posicao  do numero
                final_state = State.correct_positions[num]  # posicao corretaa
                h += abs(final_state[0] - atual[0])  # distancia em x
                h += abs(final_state[1] - atual[1])  # distancia em y
        return h

    def heuristica_fora_lugar(self):
        h = 0
        for line in range(0, 3):
            for column in range(0, 3):
                num = self.board[line,column]
                atual = [line,column]  # posicao  do numero
                final_state = State.correct_positions[num]  # posicao correta
                if atual != final_state:
                    h = h + 1
        return h


###################################################
### Hill Climbing Algorithm
###################################################
def hill_climbing(start_state):
    current_state = start_state  # inicia o State atual
    movement_list = [start_state]  # coloca o State inicial na lista
    while True:
        neighbors = current_state.neighbors()  # get neighbors
        best_neighbor = neighbors[0]  # começa com o primeiro neighbor
        for neighbor in neighbors:
            if neighbor.distance_heuristic() < best_neighbor.distance_heuristic():
                best_neighbor = neighbor  # atualiza o melhor neighbor
        # caso não houver melhoria
        if best_neighbor.distance_heuristic() >= current_state.distance_heuristic():
            break
        #
        current_state = best_neighbor
        movement_list.append(best_neighbor)
    return movement_list


###################################################
### Simulated Annealing Algorithm
###################################################
def simulated_annealing(start_state, taxa_escalonamento):
    current_state = start_state  # inicia o State atual
    movement_list = [start_state]  # coloca o State inicial na lista
    T = np.arange(1, 0, -taxa_escalonamento)  # 'temperatura' inicial
    for t in T: # até a temperatura decair à 0
        neighbors = current_state.neighbors()
        neighbor = neighbors[np.random.randint(len(neighbors))]  # escolhe neighbor aleatoriamente
        delta_e = current_state.distance_heuristic() - neighbor.distance_heuristic()
        if delta_e > 0:  # melhor que State atual, portanto aceitar
                current_state = neighbor
                movement_list.append(neighbor)  # adiciona na lista
        elif current_state.distance_heuristic() == 0:
            break  # atingiu máximo global
        else:
            delta_e = float(abs(delta_e))
            probabilidade_aceitar = np.e ** (-delta_e/t)
            r = np.random.random() # gera numero aleatorio em [0,1]
            if r <= probabilidade_aceitar:
                current_state = neighbor  # aceita filho 'ruim'
                movement_list.append(neighbor)

    return movement_list


###################################################
### Auxiliary procedure to execute some tests and view statistics
###################################################
def statistics(n):
    path_best_progress = []
    best_progress = 0
    melhor_movimentacao_h = 1e+10
    heuristicas, n_movements = np.zeros(n), np.zeros(n)
    for i in range(n):
        start_state = State()  # gera State aleatoriamente
        start = time.time()
        path = simulated_annealing(start_state, .001) # roda a busca
        end = time.time()
        # calcula estatisticas
        tempo_gasto = end - start # tempo gasto
        h_inicial = path[0].distance_heuristic()  # heuristica no primeiro State
        h_final = path[-1].distance_heuristic()  # heuristica no ultimo State
        n_movements[i] = len(path) - 1  # quantidade de movimentos(fora o State inicial)
        evolucao_apos_busca = h_final - h_inicial  # melhoria que a busca proporcionou
        heuristicas[i] = h_final # usadao para calcular a média
        # armazena path de melhor evolução
        if evolucao_apos_busca < best_progress:
            path_best_progress = path
            best_progress = evolucao_apos_busca
        # armazena path de melhor State final
        if h_final < melhor_movimentacao_h:
            melhor_movimentacao_h = h_final
            melhor_movimentacao = path  # movimentos atuais
        #print("DE h={} para h={}".format(path[0].distance_heuristic(),
        #                                                     path[-1].distance_heuristic()))

    print("--- {} buscas realizadas ---".format(n))
    print("- h(State_final) média: {}".format(int(heuristicas.mean())))
    print("- média de movimentos: {}".format(int(n_movements.mean())))
    print("- melhor h(State_final): {}".format(heuristicas.min()))
    print("- média de tempo: {:2.0}s".format(tempo_gasto))
    print("- pior h(State_final): {}".format(heuristicas.max()))
    print("- melhor evolução foi de h={} para h={}".format(path_best_progress[0].distance_heuristic(),
                                                     path_best_progress[-1].distance_heuristic()))


###################################################
### Test and Show Statistics
###################################################
statistics(10)
