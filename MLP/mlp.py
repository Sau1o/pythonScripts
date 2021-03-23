import numpy as np

def sigmoid(soma):
	return 1 / (1 + np.exp(-soma))

def sigmoidDerivada(sig):
	return sig * (1 - sig)

entradas = np.array([[0,0],
					 [0,1],
					 [1,0],
					 [1,1]])

saidas = np.array([[0],[1],[1],[0]])

pesos1 = np.array([[-0.017, -0.893,-0,148]])

pesos0 = 2 * np.random.random((2,3)) - 1

pesos1 = 2 * np.random.random((3,1)) - 1

epocas = 1000000 #testar com outros valores (10³, 10⁴,ate 10⁶)
taxaAprendizagem = 0.6   #testar outros valores para ver a conversão mais rápida
momento = 1

for j in range(epocas):
	camadaEntrada = entradas
	somaSinapse0 = np.dot(camadaEntrada,pesos0)
	camadaOculta = sigmoid(somaSinapse0)
	
	somaSinapse1 = np.dot(camadaOculta,pesos1)
	camadaSaida = sigmoid(somaSinapse1)

	erroCamadaSaida = saidas - camadaSaida 
	mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
	#print("Erro: " + str(mediaAbsoluta))

	derivadaSaida = sigmoidDerivada(camadaSaida)
	deltaSaida =  erroCamadaSaida * derivadaSaida

	deltaSaidaxPeso = deltaSaida.dot(pesos1.T)
	deltaCamadaOculta = deltaSaidaxPeso * sigmoidDerivada(camadaOculta)

	deltaCamadaOcultaT = camadaOculta.T
	pesosNovo1 = deltaCamadaOcultaT.dot(deltaSaida)
	pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem) 

	camadaEntradaT = camadaEntrada.T
	pesosNovo0 = camadaEntradaT.dot(deltaCamadaOculta)
	pesos0 = (pesos0 * momento) + (pesosNovo0 * taxaAprendizagem)

print("Erro: " + str(mediaAbsoluta))