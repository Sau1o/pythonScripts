"""
Created on Friday March 12/21
PERCEPTRON DE UMA UNICA CAMADA -> LÓGICA AND
"""
import numpy as np

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
output = np.array([0,0,0,1])
weight = np.zeros([2,1])
neta = 0.1

def stepFunction(soma):
	if (soma >= 1):
		return 1
	return 0

def calculaSaida(registro):
	s = registro.dot(weight)
	return stepFunction(s)

def treinar():
	erro = 1
	while (erro != 0):
		erroTotal = 0
		for i in range(len(output)):
			output_calc = calculaSaida(np.array(inputs[i])) 
			erro = abs(output[i]-output_calc)
			erroTotal += erro
			for j in range(len(weight)):
				weight[j] = weight[j] + (neta * inputs[i][j] * erro)
				print('Pesos atualizados: ' + str(weight[j]))
		print('Total de erros: ' + str(erroTotal))

treinar()
