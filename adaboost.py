#########################################################################
# Trabalho Prático 2 -  Implementação do Algoritmo de Boosting
# Aluno: Lucas Moura Veloso
# Matrícula: 2021708475
# Email: lmouraveloso@gmail.com 
# Código fonte do trabalho prático 2 da disciplina Aprendizado de Máquina
#########################################################################

import numpy as np
import sys

# Função auxiliar para cálculo do valor alpha
def calc_alpha(error_rate):
    return 0.5 * np.log((1 - error_rate) / error_rate)

# Função auxiliar para recálculo dos pesos dos dados
def calc_weight(w, alpha, misclassified):
    return w * np.exp(alpha * (1 if misclassified else -1))

# Função para treino de preditores AdaBoost, retorna uma lista de preditores treinados
# X: pandas dataframe com as features
# y: list com as classes (0 ou 1).
# n_iter: número inteiro indicando o número de iterações
def train_adaboost(X, y, n_iter):
    # inicialização de variáveis de uso geral do algoritmo
    boost = []
    preditors = []
    total_rows = X.shape[0]
    initial_weight = 1/total_rows
    weights = []
    
    # criação e inicialização dos preditores
    # para cada coluna em X iremos criar um preditor para cada valor possível desta coluna
    # desta forma a implementação apenas trabalha com valores categóricos, o que é razoável para
    # trabalho com a nossa base de dados.
    for col in X:
        col_values = []
        for value in X[col]:
            present = False
            for col_value in col_values:
                if col_value == value:
                    present = True
                    break
            if not present:
                col_values.append(value)
        for col_value in col_values:
            preditors.append({
                "col": col,
                "value": col_value,
                "misc": [],
                "alpha": 0,
                "TRUE": False,
                "FALSE": False
            })
        
        # adicionando preditor que sempre retornará verdadeiro
        preditors.append({
            "col": 'rnd',
            "value": sys.maxsize,
            "misc": [],
            "alpha": 0,
            "TRUE": True,
            "FALSE": False
        })
        
        # adicionando preditor que sempre retornará falso
        preditors.append({
            "col": 'rnd',
            "value": sys.maxsize,
            "misc": [],
            "alpha": 0,
            "TRUE": False,
            "FALSE": True
        })
        
    # inicializando os valores de peso para cada um dos índices de dados
    for i in range(total_rows):
        weights.append(initial_weight)
    
    # calculando quais preditores erram em quais dados
    # para cada valor de X iremos calcular quais preditores acertaram e quais erraram
    # cada preditor terá uma lista 'misc' que contém os índices dos dados que foram erroneamente classificados
    y_index = 0
    for index, x in X.iterrows():
        for p in preditors:
            predicted_value = 0
            if p["TRUE"]:
                predicted_value = 1
            elif p["FALSE"]:
                predicted_value = 0
            else:
                predicted_value = 1 if x[p['col']] == p['value'] else 0
            if predicted_value != y[y_index]:
                p['misc'].append(y_index)
        y_index = y_index+1
        
    # executandos as iterações do AdaBoost
    for i in range(n_iter):
        # encontrando qual é o preditor de menor erro de acordo com a configuração atual de pesos
        min_err = sys.maxsize
        min_err_preditor = {}
        for p in preditors:
            preditor_error_rate = 0
            for misc in p['misc']:
                preditor_error_rate = preditor_error_rate + weights[misc]
            if preditor_error_rate < min_err:
                min_err_preditor = p
                min_err = preditor_error_rate

        # calculo do valor alpha
        alpha = calc_alpha(min_err)
        min_err_preditor["alpha"] = alpha
        
        # adicionando o preditor escolhido na nossa lista de preditores escolhidos em cada iteração
        boost.append(min_err_preditor)
        
        # atualizando os pesos
        weight_sum = 0
        for w_index in range(len(weights)):
            misclassified = False
            for misc in min_err_preditor['misc']:
                if misc == w_index:
                    misclassified = True
                    break
            weights[w_index] = calc_weight(weights[w_index], alpha, misclassified)
            weight_sum = weight_sum + weights[w_index]
        
        # o fator de normalização 'z' foi calculado como a 1/(soma dos pesos) de forma que se aplicado ao
        # vetor 'weights', a sua soma resultará em 1.
        z = 1/weight_sum
        for i in range(len(weights)):
            weights[i] = weights[i] * z

    # retornando os preditores escolhidos em cada iteração
    return boost

# Função para predizer as classes de X dado os preditores AdaBoost, 
# retorna as classes calculadas para cada linha de X em formato list de inteiros (0 ou 1).
# boost: list de preditores AdaBoost
# X: pandas dataframe contendo as features
def predict_adaboost(boost, X):
    y = []
    for index, x in X.iterrows():
        predicted_value = 0
        for b in boost:
            # realizando a predição do preditor 'b'
            prediction = 0
            alpha = b['alpha']
            if b["TRUE"]:
                prediction = 1
            elif b["FALSE"]:
                prediction = 0
            else:
                col = b['col']
                value = b['value']
                x_value = x[col]
                prediction = 1 if x_value == value else -1
            # somando a predição do preditor 'b' ponderada pelo seu valor 'alpha'
            predicted_value = predicted_value + (prediction * alpha)
        # validando se devemos retornar 0 ou 1 baseado no sinal da predição final
        y.append(1 if predicted_value > 0 else 0)
    return y