#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Data input and import libraries
import random
import math
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.metrics.cluster import adjusted_rand_score
import time
import matplotlib
import matplotlib.pyplot
import statistics
from collections import defaultdict 
import datetime
import itertools
from scipy.stats import pearsonr
import pickle
from sklearn.tree import DecisionTreeClassifier as DTC

# datasetteste = pd.read_excel('Pasta de trabalho (5).xlsx')
# df = datasetteste.copy()





datasetteste = pd.read_excel('dataset37.xlsx')
df = datasetteste.copy()
y = df['petrofacie']
df = df.drop(columns = 'petrofacie')
df = df.loc[:,::-1]
name = 'dataset37.xlsx'
p =df['p'][0]
df = df.drop(columns = 'p')


# datasetteste = pd.read_excel('Pasta.xlsx')
# df = datasetteste.copy()

# dataset = pd.read_excel('subtotals_dataset-filtered.xlsx', header = 2)
# df = dataset.copy()
# df = df.drop(columns = "features")
# df = df.drop(0, 0)
# y = df['petrofacie']
# df = df.drop(columns = 'petrofacie')
# df = df.loc[:,::-1]
# name = ' campus basin subtotals_dataset-filtered.xlsx'
# p = ' 100% todas as features'

# dataset = pd.read_excel('Carmopolis_subtotals_dataset-filtered.xlsx', header = 2)
# df = dataset.copy()
# df = df.drop(columns = "features")
# df = df.drop(0, 0)
# y = df['petrofacie']
# df = df.drop(columns = 'petrofacie')
# df = df.loc[:,::-1]
# name = 'Carmopolis subtotals_dataset-filtered.xlsx'
# p = ' 100% todas as features'


# dataset = pd.read_excel('TalaraBasin_subtotals_dataset-filtered.xlsx', header = 2)
# df = dataset.copy()
# df = df.drop(columns = "features")
# df = df.drop(0, 0)
# y = df['petrofacie']
# df = df.drop(columns = 'petrofacie')
# df = df.loc[:,::-1]
# name = 'Talara Basin(western Peru)subtotals_dataset-filtered.xlsx'
# p = ' 100% todas as features'


# dataset = pd.read_excel('Mucuri_subtotals_dataset-filtered.xlsx', header = 2)
# df = dataset.copy()
# df = df.drop(columns = "features")
# df = df.drop(0, 0)
# y = df['petrofacie']
# df = df.drop(columns = 'petrofacie')
# df = df.loc[:,::-1]
# name = 'Mucuri subtotals_dataset-filtered.xlsx'
# p = ' 100% todas as features'


# dataset = pd.read_excel('MargemEquatorial_subtotals_dataset-filtered.xlsx', header = 2)
# df = dataset.copy()
# df = df.drop(columns = "features")
# df = df.drop(0, 0)
# y = df['petrofacie']
# df = df.drop(columns = 'petrofacie')
# df = df.loc[:,::-1]
# name = 'Margem Equatorial subtotals_dataset-filtered.xlsx'
# p = ' 100% todas as features'


# dataset = pd.read_excel('Jequitinhonha_subtotals_dataset-filtered.xlsx', header = 2)
# df = dataset.copy()
# df = df.drop(columns = "features")
# df = df.drop(0, 0)
# y = df['petrofacie']
# df = df.drop(columns = 'petrofacie')
# df = df.loc[:,::-1]
# name = 'Jequitinhonha subtotals_dataset-filtered.xlsx'
# p = ' 100% todas as features'


# In[2]:


#Open log files and create the agent for the genetic algorithm
File = open( datetime.datetime.now().strftime("Esquema4 %I%MDIA%d") + ".txt","w")
Log = open('Esquema4.txt', 'w')
primeira = True
class Agent:

    contador = 0
    def __init__(self,df,Ncluster,generation = 0, vetor = None):
                       
        self.Ncluster = Ncluster
        self.Nfeatures = len(df.columns)
        self.Nsamples = len(df)
        self.indiv = vetor
        if self.indiv == None:
            self.indiv = []
            self.indiv.extend(random.sample(range(self.Nsamples), Ncluster))
        
        self.fitness = -1.0
        self.generation = generation

        self.id = Agent.contador
        Agent.contador += 1
        
        
    def __str__(self):
        return 'Kcentroides: '+ f'{self.indiv}'

    def __len__(self):
        return len(self.indiv)
    


# In[2]:


#Genetic algorithm workflow
def ga(df, population, Ncluster, generations, qntdFilhos, rate, qntdElite, pop = False):
    
    ##### métricas : fitness_indiv      fitMSTindiv        fitSepEDesag           fitSepEDesagComDOM
    calculo = fitness_indiv
    #####  métrica
    
    conter = 0
    listafit = []
    listaari = []
    inicio = time.time()
    resposta= [0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 1, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 5, 5, 3, 3, 6, 0, 0, 0, 6, 0, 3, 3, 3, 3, 3, 3, 3, 0, 7, 0, 0, 8, 8, 8, 8, 9, 9, 9, 8, 8]
    if pop == False:
        agents =  init_agents(df, population, Ncluster)
    if pop == True:
        listavet = recebe_pop()
        agents = init_agents(df, population, Ncluster, listavet)
    File.write('Dataset: '+name+ '  p: '+ str(p)+ '\n' +  'Parâmetros: ' + '\nPopulação: ' + str(population) +'\n'+ 'Gerações: '+ str(generations)+'\n'+'Quantidade de filhos: '+ str(qntdFilhos)+'\n'+ 'Taxa de mutação: '+ str(rate)+'%'+'\n' + 'Quantidade elite: '+ str(qntdElite))
    melhor = []
    for generation in range(generations):
        agents = renumera(agents)
        conter+=1
        print ('Generation: ' + str(generation))
        File.write('\n\nGeneration: ' + str(generation))
        somafit = 0
        listafitGER = []
        listaariGER = []
        maiorfit = agents[0]
        degen = 0
        totalagentes = 0                                     
        for agent in agents:
            totalagentes = totalagentes+1
            featuresPorCluster = []
            for cluster in range(agent.Ncluster):
                features = []
                for bit in agent.indiv[cluster*agent.Nfeatures:(cluster+1)*agent.Nfeatures]:
                    features.append(bit)
                featuresPorCluster.append(features)
            cent = []
#             print('PréKmeans')
            for _ in range(agent.Ncluster):
                cent.append(df.values[agent.indiv[agent.Nfeatures*agent.Ncluster+_]])
            clusterPsample, iters = subKMeans(agent, cent, featuresPorCluster, df, 40)
#             print('pósKmeans')
            set1 = set(range(agent.Ncluster))
            clusterDasSamples = [-1] * agent.Nsamples
            for i, cluster in enumerate(clusterPsample):
                for sample in cluster:
                    clusterDasSamples[sample] = i
            set2 = set(clusterDasSamples)
            if set1.issubset(set2) == False:
                degen += 1
                agent.fitness = -1
            else:
                antigo = agent.indiv.copy()
                agent.indiv[agent.Nfeatures*agent.Ncluster:] = clusterDasSamples
                agent.fitness = calculo(agent, df)
                agent.indiv = antigo
            listafitGER.append(agent.fitness)
            listaariGER.append(adjusted_rand_score(y, clusterDasSamples))
            somafit += agent.fitness
            if agent.fitness > maiorfit.fitness:
                maiorfit = agent
                melhor = clusterDasSamples
#             print('mateimaisum')


        print(f'Total agentes/geração: {totalagentes}')
        
        File.write(f'\nAgentes degenerados: {degen}\n')
        print(f'Maior fitness da geração: {maiorfit.fitness} Id: {maiorfit.id}')
        MatrizConf(resposta, melhor, agents[0].Ncluster)
        File.write(f'\nMaior fitness da geração: {maiorfit.fitness} Id: {maiorfit.id}\n Individuo: {maiorfit.indiv}\n')
        File.write(f'Fitness médio: {somafit/len(agents)}\n')
        File.write(f'Desvio padrao ARI: {statistics.stdev(listaariGER)}   Desvio padrao Fitness: {statistics.stdev(listafitGER)}\n')
        if(len(melhor) == len(y)):
            File.write(f'ARI melhor individuo: {adjusted_rand_score(y, melhor)}')

        
            listaari.append(adjusted_rand_score(y, melhor))
        else:
            File.write(f'deu algum problema no ARI')
            listaari.append(-1)
        listafit.append(maiorfit.fitness)

        #print(f'Len agents préselection: {len(agents)}')
        agents = selection(agents, qntdFilhos, qntdElite)
    
        #print(f'Len agents pósselection: {len(agents)}')
        agents = crossover(agents, population, df, Ncluster, generation)

        #print(f'Len agents póscorssover: {len(agents)}')
        agents = mutation(agents, qntdFilhos, rate)

        #print(f'Len agents pos mutatio: {len(agents)}')

#         if any(agent.fitness >= 0.95 for agent in agents):      
#             break
    fim  = time.time()
    print(f'Samples do melhor individuo: {maiorfit.indiv[maiorfit.Ncluster*maiorfit.Nfeatures:]}')
    print(f'features ativas do melhor individuo: {maiorfit.indiv[:maiorfit.Ncluster*maiorfit.Nfeatures]}')
    print(f'Id: {maiorfit.id} Geração de criação: {maiorfit.generation}')
    print(f'tempo de execução: {fim-inicio} segundos')
    File.write(f'\nCorrelação de Pearson(FitnessxARI): {pearson(listaari, listafit)}\n')
    a,b = pearsonr(listaari, listafit)
    File.write(f'Scipy Pearson: {a}')
    File.write(f'\n\nTempo de execução: {fim-inicio} segundos')
    File.close()
    Log.close()
    matplotlib.pyplot.ylim(-1,maiorfit.fitness+1)
    matplotlib.pyplot.xlim(0, generations)
    matplotlib.pyplot.plot(listaari)
    matplotlib.pyplot.plot(listafit)
    matplotlib.pyplot.savefig(datetime.datetime.now().strftime("Exec %I%MGRAF")+'.png')
    matplotlib.pyplot.show()
    matplotlib.pyplot.scatter(listaari, listafit)
    matplotlib.pyplot.savefig(datetime.datetime.now().strftime("Exec %I%MGRAF2")+'.png')
    matplotlib.pyplot.show()
 


# In[2]:


#Auxiliary funtions
def init_agents(df, population, Ncluster, vetor = None):
    if vetor == None:
        return [Agent(df, Ncluster) for _ in range(population)]
    else:
        return [Agent(df, Ncluster,0, vetor[x]) for x in range(population)]


def novo_indiv(df, Ncluster, generation, vetor):
    return Agent(df, Ncluster,generation, vetor)



def dist(df1, df2, features):
  dist = 0
  for i in range(len(df1)):
    if features[i] == 1:
      dist = dist + (df1[i]-df2[i])**2
  return dist**(1/2)
    
    
def soma_lista(vetor):
  soma = 0
  for i in range(len(vetor)):
    soma = soma + vetor[i]
  return soma

def soma_items(lista1,lista2):
  novalista = []
  for i in range(len(lista1)):
    novalista.append(lista1[i]+lista2[i])
  return novalista

def divide_lista(vetor, valor):
  for i in range(len(vetor)):
    vetor[i] = vetor[i]/valor
  return vetor

def remove_repetidos(lista):
    l = []
    for i in lista:
        if i not in l:
            l.append(i)
    l.sort()
    return l

def remove_zero(lista):
  l = []
  for i in range(len(lista)):
    if lista[i] != 0:
      l.append(lista[i])
    if lista[i] == 0:
      l.append(9999999)
  return l


def pearson(provas, exercicios):

    media_provas = float(sum(provas)) / len(provas)

    media_exercicios = float(sum(exercicios)) / len(exercicios)


    p_cima = 0
    p_baixo_prova = 0
    p_baixo_exercicio = 0


    for nota_prova, nota_exercicio in zip(provas, exercicios):

        p_cima = p_cima + ((nota_prova - media_provas) * (nota_exercicio - media_exercicios))

        p_baixo_prova = p_baixo_prova + ((nota_prova - media_provas) ** 2)
        p_baixo_exercicio = p_baixo_exercicio + ((nota_exercicio - media_exercicios) ** 2)


    p = p_cima / ((p_baixo_prova * p_baixo_exercicio) ** (1/2.0))

    return p


def recebe_pop():
    with open('PopNovaCompleto', 'rb') as filehandle:
        final = pickle.load(filehandle)
    return final

def escreve_pop(agents):
    inicial = [i.indiv for i in agents]
    with open('PopNovaCompleto', 'wb') as filehandle:
        pickle.dump(inicial, filehandle)
        
def MatrizConf(y, agent, Ncluster):
    Log.write('\n')
    samplesPorCluster = []    
    for cluster in range(Ncluster):       ############################################################################################################## 
        samples = []        
        for bit in range(len(agent)):
            if agent[bit] == cluster:
                samples.append(bit)
        samplesPorCluster.append(samples)
    
    resposta = []
    for cluster in range(Ncluster):
        samples = []
        for bit in range(len(agent)):
            if y[bit] == cluster:
                samples.append(bit)
        resposta.append(samples)
    print(f'resposta = {resposta}\nmeuscluter= {samplesPorCluster}')
    
    matriz = [[]for _ in range(Ncluster)]
    
    for cluster in range(Ncluster):
        for i in resposta[cluster]:
            naoachei = 1
            k = 0
            while(naoachei == 1 and k< len(samplesPorCluster)):
                j = 0
                while(naoachei == 1 and j < len(samplesPorCluster[k])):
                    if samplesPorCluster[k][j] == i:
                        samplesPorCluster[k][j] = f'Sample do K {cluster}'
                        naoachei = 0
                    j = j+1
                k = k+1    
    for i, cluster in enumerate(samplesPorCluster):
        Log.write(f'Cluster {i} samples = {cluster}\n')    
    Log.write('\n\n')
    
def arvoreStat(modelo):
    n_nodes = modelo.tree_.node_count
    children_left = modelo.tree_.children_left
    children_right = modelo.tree_.children_right
    feature = modelo.tree_.feature
    threshold = modelo.tree_.threshold


    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # id nó raiz e profundidade do nó pai
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # Caso nó teste
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True    
    
    return n_nodes, feature


# In[2]:


#Fitness functions
def fitness_pop(agents, df, calculo):
    global primeira
    if primeira:
        File.write('\nMétrica usada : Fitness_indiv\n')
        primeira = False
    for i in agents:
        i.fitness = calculo(i, df)
    return agents


def fitness_indiv(agent, df):   
  a = [0] * agent.Nsamples            #a,b e s: Vetor com tamanho == Numero de samples
  b = []                              #b é criado conforme o decorrer de seu cálculo
  s = [-1] * agent.Nsamples 
  cardi = [0] * agent.Ncluster        #VETOR AONDE INDICE É O CLUSTER E VALOR É CARDINALIDADE/ EX: CARDI == [1,1,2] SIGNIFICA: 1 SAMPLE NO CLUSTER 0 || 1 SAMPLE NO CLUSTER 1 || 2 SAMPLES NO CLUSTER 2

  for sample in agent.indiv[agent.Ncluster*agent.Nfeatures:]:
    cardi[sample] = cardi[sample] + 1

  # /\ Crio o meu vetor Cardinalidade com os valores esperados


  #Iteração em todas as minhas samples fazendo o cálculo do parâmetro a:
  #Faço uso de quatro paramêtros p/ iteração:

  #sampleatual: Parâmetro interno ao primeiro laço for (Varia de 0 a (número de samples-1)). É usado para indicar a sample que estou calculando o 'a' em comparação com as outras samples.
  #sampleaux: Parâmetro interno ao segundo laço for (Varia de 0 a (número de samples-1)). É usado para indicar com qual sample a minha sampleatual está sendo comparada.
  #i:Parâmetro do for(varia de (nmr de features*nmr de cluster) a (tamanho total do individuo)). Itera por toda a posição de samples do meu individuo
  #j:Parâmetro do for(varia de (nmr de features*nmr de cluster) a (tamanho total do individuo)). Itera por toda a posição de samples do meu individuo
  
  sampleatual = -1
  for i in range(agent.Ncluster*agent.Nfeatures, len(agent)):
    sampleatual += 1
    sampleaux = -1

    for j in range(agent.Ncluster*agent.Nfeatures, len(agent)):
      sampleaux += 1

      if i == j:           #i == j ambos estão na mesma sample
        continue           #faz nada

      elif agent.indiv[i] == agent.indiv[j]:       #caso (i != j) && (agent.indiv[i] == agent.indiv[j]) significa samples diferentes atribuidas ao mesmo cluster então soma a[posição sampleatual] pois cada sample tem a[i] próprio
        a[sampleatual] += dist(df.values[sampleatual], df.values[sampleaux], agent.indiv[agent.indiv[i]*agent.Nfeatures:(agent.indiv[i]+1)*agent.Nfeatures])

    if (cardi[agent.indiv[i]] -1) > 0:      #ao "sair" do for interno o meu a[sampleatual] é a soma da distância deste para todos as outras samples atribuidas ao mesmo cluster
      a[sampleatual] = a[sampleatual] / (cardi[agent.indiv[i]] -1) 
    else:
        s[sampleatual] = 0
        #Caso cardinalidade > 1(mais de uma única sample no cluster) eu realizo a divisão por (cardinalidade(daquele cluster) - 1)
  #Iteração em todas as minhas samples fazendo o cálculo do parâmetro b
  #Utilizo os mesmos parâmetros do cálculo do 'a' acima

  sampleatual = -1
  for i in range(agent.Ncluster*agent.Nfeatures, len(agent)):
    sampleatual += 1
    menoratual = [0] * len(cardi)
    sampleaux = -1

    for j in range(agent.Ncluster*agent.Nfeatures, len(agent)):
      sampleaux += 1

      if agent.indiv[i]!=agent.indiv[j]: #significa que i e j estão em samples diferentes que estão atribuidos a diferentes clusters
            
            #Cálculo padrão:
        menoratual[agent.indiv[j]] += dist(df.values[sampleatual], df.values[sampleaux], agent.indiv[agent.indiv[i]*agent.Nfeatures:(agent.indiv[i]+1)*agent.Nfeatures])
            #Cálculo com b alterado:
        # menoratual[agent.indiv[j]] += dist(df.values[sampleatual], df.values[sampleaux], agent.indiv[agent.indiv[j]*agent.Nfeatures:(agent.indiv[j]+1)*agent.Nfeatures])
        
        
        
        #Aqui somo a distância no vetor menoratual[agent.indiv[j]]. agent.indiv[j] É o valor do cluster à qual estou comparando, então meu menoratual é um vetor que soma a distância
        #com todas as samples com os clusters diferentes e atribui essa soma na posição respectiva ao cluster no vetor menoratual


    #ao sair do meu for interno, menoratual = [somatório da distância com cluster 0, somatório da distância com cluster 1, ...,  somatório da distância com cluster NmrClusters]
    #Então passo por todo o meu vetor menoratual dividindo o valor no vetor pela cardinalidade respectiva do cluster em questão

    for h in range(len(menoratual)):
        menoratual[h] = menoratual[h]/cardi[h]

    #Esse vetor após a divisão pode ter alguns valores '0' que são errôneos e sem valor para mim, então retiro estes na função remove_zero
    menoratual = remove_zero(menoratual)

    #após ter os zeros removidos eu pego o menor valor dessa lista e adiciono ao 'b', ou seja, a menor distância média com todos os clusters diferentes do meu cluster da sampleatual
    b.append(min(menoratual)) 


  #para todas as samples analiso 'a' e 'b' e cálculo o 's' baseado nisso
  for i in range(len(s)):
    #print(f'a:{a[i]} b: {b[i]}')
    if s[i]!=0:
        if(a[i] > b[i]):
          s[i] = (b[i]/a[i]) - 1
        elif(a[i] < b[i]):
          s[i] = 1-(a[i]/b[i])
        else:
          s[i] = 0

  #Retorna a Soma de todos os meus 's' de cada sample dividido pelo total de samples que é a fitness deste indivíduo
  return soma_lista(s)/agent.Nsamples    

  #soma_lista realiza o somatório de todos os valores de um vetor
    
    
class Grafo: 
  
    def __init__(self,vertices): 
        self.V= vertices #No. of vertices 
        self.grafo = [] # default dictionary  
                                # to store graph 
          
   
    # function to add an edge to graph 
    def add(self,u,v,w): 
        self.grafo.append([u,v,w]) 
  
    # A utility function to find set of an element i 
    # (uses path compression technique) 
    def find(self, parent, i): 
        if parent[i] == i: 
            return i 
        return self.find(parent, parent[i]) 
  
    # A function that does union of two sets of x and y 
    # (uses union by rank) 
    def union(self, parent, rank, x, y): 
        xroot = self.find(parent, x) 
        yroot = self.find(parent, y) 
  
        # Attach smaller rank tree under root of  
        # high rank tree (Union by Rank) 
        if rank[xroot] < rank[yroot]: 
            parent[xroot] = yroot 
        elif rank[xroot] > rank[yroot]: 
            parent[yroot] = xroot 
  
        # If ranks are same, then make one as root  
        # and increment its rank by one 
        else : 
            parent[yroot] = xroot 
            rank[xroot] += 1
 
    def __str__(self):
        b = []
        for i in self.grafo:
            b.append(str(i))
        return str(b)            
        
        
    def KruskalMST(self): 
  
        result =[] #This will store the resultant MST 
  
        i = 0 # An index variable, used for sorted edges 
        e = 0 # An index variable, used for result[] 
  
            # Step 1:  Sort all the edges in non-decreasing  
                # order of their 
                # weight.  If we are not allowed to change the  
                # given graph, we can create a copy of graph 
        self.grafo =  sorted(self.grafo,key=lambda item: item[2]) 
  
        parent = [] ; rank = [] 
  
        # Create V subsets with single elements 
        for node in range(self.V): 
            parent.append(node) 
            rank.append(0) 
      
        # Number of edges to be taken is equal to V-1 
        while e < self.V -1 : 
  
            # Step 2: Pick the smallest edge and increment  
                    # the index for next iteration 
            u,v,w =  self.grafo[i] 
            i = i + 1
            x = self.find(parent, u) 
            y = self.find(parent ,v) 
  
            # If including this edge does't cause cycle,  
                        # include it in result and increment the index 
                        # of result for next edge 
            if x != y: 
                e = e + 1     
                result.append([u,v,w]) 
                self.union(parent, rank, x, y)             
            # Else discard the edge 
  
        # print the contents of result[] to display the built MST 
        return result

    
#DESAGREGAÇÃO QUANDO ARVORE = MST de GRAFO ORIGINAL
def Desag(lista):

    nArestras = len(lista)
    if nArestras == 1:
      #print('\nMAIS MERDA CACETETETETET\n')
      return lista[0][2]

    soma = 0
    for i in lista:
        sample1,sample2,peso = i
        soma+=peso
    return float(soma/nArestras)


def fitMSTindiv(agent, df):
        
    samplesPorCluster = []    
    for cluster in range(agent.Ncluster):        
        samples = []        
        for bit in range(0, agent.Nsamples):
            if agent.indiv[bit+agent.Nfeatures*agent.Ncluster] == cluster:
                samples.append(bit)
        samplesPorCluster.append(samples)
        
    #print(f'SAMPLES POR CLUSTER = {samplesPorCluster}')
        
    featuresPorCluster = []
    for cluster in range(agent.Ncluster):
        features = []
        for bit in agent.indiv[cluster*agent.Nfeatures:(cluster+1)*agent.Nfeatures]:
            features.append(bit)
        featuresPorCluster.append(features)
            
    # print(f'FEATURES POR CLUSTERS = {featuresPorCluster}')
        
    #CRIO OS MODELOS COM N VERTICES
    modelos = []
    for cluster in range(agent.Ncluster):
        modelos.append(Grafo(len(samplesPorCluster[cluster])))


    #CRIO LISTA DE DICIONARIOS
    b = []
    for _ in range(agent.Ncluster):
        b.append({})

    #ATUALIZO OS VALORES DO DICIONARIOS
    j = 0
    for samplesAUX in samplesPorCluster:
        for i in range(len(samplesAUX)):
            b[j].update({i:samplesAUX[i]})
        j+=1  
    

    # print(f'DICIONARIO: {b}')

    for cluster in range(agent.Ncluster):
        x = 0    
        while(x < len(samplesPorCluster[cluster])-1):
            y = x+1
            while(y<=len(samplesPorCluster[cluster])-1):
                modelos[cluster].add(x, y, dist(df.values[b[cluster].get(x)], df.values[b[cluster].get(y)], featuresPorCluster[cluster]))
                y+=1
            x += 1
        if len(samplesPorCluster[cluster]) == 1:
          modelos[cluster].grafo.append([-1,-1,float('inf')])

    if [] in modelos:
      modelos.remove([])
    # for i in range(len(modelos)):
    #   #print(modelos[i])

    desagregacao = []
    for i in range (len(modelos)):
        if modelos[i].V>1:
          desagregacao.append(Desag(modelos[i].KruskalMST()))
          #print(Desag(modelos[i].KruskalMST()))
        else:
          desagregacao.append(0)
    

    distancias = []
    for cluster in range(agent.Ncluster):
        samplesFora = []
        
        for i in range(agent.Nsamples):
            if i not in samplesPorCluster[cluster]:
                samplesFora.append(i)
                
        menor = float('inf')
        for i in range(len(samplesPorCluster[cluster])):
            soma = 0
            for j in range(len(samplesFora)):
                if (dist(df.values[samplesPorCluster[cluster][i]], df.values[samplesFora[j]], featuresPorCluster[cluster]) < menor):
                    menor = dist(df.values[samplesPorCluster[cluster][i]], df.values[samplesFora[j]], featuresPorCluster[cluster])
            soma += menor
        distancias.append(soma / len(samplesPorCluster[cluster]))
    
    qualidade = 0
    for cluster in range(agent.Ncluster):
#         if desagregacao[cluster] == 0:                    ####################
#             qualidade+=0                              ####################
#         else:
        qualidade = qualidade + (distancias[cluster]/(desagregacao[cluster] +1))
    #print(f'qualidadeantes div: {qualidade}')
    qualidade = qualidade/agent.Ncluster
    #print(f'qualidade:{qualidade}')
    return qualidade



def fitSepEDesag(agent,df):
    samplesPorCluster = []    
    for cluster in range(agent.Ncluster):        
        samples = []        
        for bit in range(0, agent.Nsamples):
            if agent.indiv[bit+agent.Nfeatures*agent.Ncluster] == cluster:
                samples.append(bit)
        samplesPorCluster.append(samples)
        
    #print(f'SAMPLES POR CLUSTER = {samplesPorCluster}')
        
    featuresPorCluster = []
    for cluster in range(agent.Ncluster):
        features = []
        for bit in agent.indiv[cluster*agent.Nfeatures:(cluster+1)*agent.Nfeatures]:
            features.append(bit)
        featuresPorCluster.append(features)
        
    #print(f'Samples Por cluster: {samplesPorCluster}\n features por cluster: {featuresPorCluster}')
    desag = []
    for i in range(agent.Ncluster):
        distancia = 0
        for sample in samplesPorCluster[i]:
            for sample2 in samplesPorCluster[i]:
                if sample!=sample2:
                    distancia+= dist(df.values[sample], df.values[sample2], featuresPorCluster[i])
        desag.append(distancia/len(samplesPorCluster[i]))
    
    #print(f'desag: {desag}')
    
    SEP = []
    for cluster in range(agent.Ncluster):
        sep = 0
        maisPertoAtual = float('inf')
        for i in samplesPorCluster[cluster]:
            for j in range(agent.Nsamples):
                if j not in samplesPorCluster[cluster]:
                    if maisPertoAtual>dist(df.values[i], df.values[j], featuresPorCluster[cluster]):
                        maisPertoAtual = dist(df.values[i], df.values[j], featuresPorCluster[cluster])
            sep+=maisPertoAtual
            maisPertoAtual = 0
        SEP.append(sep/len(samplesPorCluster[cluster]))
    
    q = 0
    for cluster in range(agent.Ncluster):
        if len(samplesPorCluster[cluster])>1:
            q+= (SEP[cluster]/(desag[cluster]+1))
            

    return (q/agent.Ncluster)
        
     
def fitSepEDesagComDOM(agent,df):
    samplesPorCluster = []    
    for cluster in range(agent.Ncluster):        
        samples = []        
        for bit in range(0, agent.Nsamples):
            if agent.indiv[bit+agent.Nfeatures*agent.Ncluster] == cluster:
                samples.append(bit)
        samplesPorCluster.append(samples)
        
    #print(f'SAMPLES POR CLUSTER = {samplesPorCluster}')
        
    featuresPorCluster = []
    for cluster in range(agent.Ncluster):
        features = []
        for bit in agent.indiv[cluster*agent.Nfeatures:(cluster+1)*agent.Nfeatures]:
            features.append(bit)
        featuresPorCluster.append(features)
        
        
    desag = []
    for i in range(agent.Ncluster):
        distancia = 0
        for sample in samplesPorCluster[i]:
            for sample2 in samplesPorCluster[i]:
                if sample!=sample2:
                    distancia+= dist(df.values[sample], df.values[sample2], featuresPorCluster[i])
        desag.append(distancia/len(samplesPorCluster[i]))
    
    
    
    SEP = []
    for cluster in range(agent.Ncluster):
        sep = 0
        maisPertoAtual = float('inf')
        for i in samplesPorCluster[cluster]:
            for j in range(agent.Nsamples):
                if j not in samplesPorCluster[cluster]:
                    if maisPertoAtual>dist(df.values[i], df.values[j], featuresPorCluster[cluster]):
                        maisPertoAtual = dist(df.values[i], df.values[j], featuresPorCluster[cluster])
            sep+=maisPertoAtual
            maisPertoAtual = 0
        SEP.append(sep/len(samplesPorCluster[cluster]))
    
    
    q = 0
    for cluster in range(agent.Ncluster):
        if len(samplesPorCluster[cluster])>1:
            q+= (SEP[cluster]/(desag[cluster]+1))
            
            
    rvp = [1]*agent.Ncluster           
    #############COM DOMINIO:::####################
    vgp = df['porosity'].var()
    rvp = []
    for cluster in range(agent.Ncluster):
        rvp.append(vgp/(df['porosity'].iloc[samplesPorCluster[cluster]].var()+1))
    ###############################################  
    q = 0
    for cluster in range(agent.Ncluster):
        if len(samplesPorCluster[cluster])>1:
            q+= (SEP[cluster]/(desag[cluster]+1)*rvp[cluster])
            
              
        

    return (q/agent.Ncluster)
          
          


# In[2]:


#Main genetic algorithm's functions
def selection(agents, ratio, qntdElites = 0):
    
  selected = []

  totalfit = 0.0

  elite = agents[0]

  menorfit = agents[0].fitness

  for agent in agents:
    if agent.fitness < menorfit:
      menorfit = agent.fitness
    
  for agent in agents:
    agent.fitness += abs(menorfit)
 
  for agent in agents:
    totalfit += agent.fitness

  while(len(selected) < qntdElites):
    for agent in agents:
      if agent.fitness > elite.fitness:
        elite = agent
    selected.append(deepcopy(elite))

  while(len(selected) < ratio):
    ponto = random.uniform(0,totalfit)
    i = 0
    atual = 0
    while(atual < ponto):
      atual = atual + agents[i].fitness
      if(atual < ponto):
        i += 1
    selected.append(deepcopy(agents[i]))
      

  for i in selected:
    i.fitness -= abs(menorfit)

  return selected
    
    




def crossover(agents, population, df, Ncluster, generation):

    offspring = []
    while (len(offspring) + len(agents)) < population:
        filho1 = []
        filho2 = []
        mask = []
        for _ in range(len(agents[0].indiv)):
            mask.append(random.choice([0,1]))
        pai, mae = random.sample(agents, 2)
        for bit in range(len(mask)):
            if mask[bit] == 0:
                filho1.append(pai.indiv[bit])
                filho2.append(mae.indiv[bit])
            else:
                filho1.append(mae.indiv[bit])
                filho2.append(pai.indiv[bit])
        offspring.append(novo_indiv(df, Ncluster,generation+1,filho1))
        offspring.append(novo_indiv(df, Ncluster,generation+1,filho2))
    agents.extend(deepcopy(offspring))
    return agents


def ativa_porosidade(agents, df):
  posicao = df.columns.get_loc("porosity")
  for agent in agents:
    for bit in range(posicao, agent.Ncluster*agent.Nfeatures, agent.Nfeatures):
      agent.indiv[bit] = 1
  return agents


def mutation(agents, qntdFilhos, rate):

    for agent in agents[qntdFilhos:]:
        for index in range(len(agent.indiv)):
            if random.choice(range(100)) < rate:
                trocou = False
                while(trocou == False):
                    x = random.choice(range(agent.Nsamples))
                    if x not in agent.indiv:
                        agent.indiv[index] = x
                        trocou = True
    return agents


# In[2]:


#Subspace Kmeans step with feature selection at each iteration.

def centroids(agent, df, samplesPorCluster, featuresPorCluster):
    cent = []

    for cluster, samples in enumerate(samplesPorCluster):
        if samples != [None]:
            mediaAtual = [0] * agent.Nfeatures 
            for sample in samples:
                mediaAtual = soma_items(mediaAtual, df.values[sample])

                mediaAtual = divide_lista(mediaAtual, (len(samplesPorCluster[cluster])))
            cent.append(mediaAtual)
        if samples == [None]:
            cent.append([None])

    return cent
    
        
        
def subKMeans(agent, cent, df, maxIter):
    n_features_total = agent.Nfeatures
    oldCent = [[] for _ in range (agent.Ncluster)]
    K = agent.Ncluster
    iter = 0
    compar = np.array_equiv(cent,oldCent)
    featuresPorCluster = [[1]*agent.Nfeatures]*agent.Ncluster
    while(not(compar) and iter<maxIter):
        #print(f'FeaturesPorCluster: {featuresPorCluster}')
        clusters = [[] for _ in range (K)]
        for samples in range(agent.Nsamples):
            selectedClusterIndex = None
            minDist = float('inf')
            for i in range(K):
                #print(i)
                #print(cent[i])
                #print(featuressPorCluster[i])
                if (len(cent[i])<2 and cent[i] == [None]) or (len(featuresPorCluster[i])<2 and featuresPorCluster[i] == [None]):
                    continue
                else:
                    distancia = dist(df.values[samples], cent[i], featuresPorCluster[i])
                    if distancia < minDist:
                        minDist = distancia
                        selectedClusterIndex = i
            clusters[selectedClusterIndex].append(samples)
        for i, clust in enumerate(clusters):
            if clust == []:
                clusters[i] = [None]
        arvore = DTC(random_state = 1, criterion = 'gini')
        a = [-1]*agent.Nsamples
        
        for i,cluster in enumerate(clusters):
            for sample in cluster:
                if sample!=None:
                    a[sample] = i
                
  
        arvore.fit(df, a)
        n_nodes, features = arvoreStat(arvore)
        #print(n_nodes, features)
        oldCent = cent
        featuresPorCluster = findFeatures(arvore, n_nodes, features, clusters, n_features_total)
        cent = centroids(agent, df, clusters, featuresPorCluster)
        iter += 1
        compar = np.array_equiv(cent,oldCent)
    return clusters, iter



def findFeatures(modelo, n_nodes, features, samplesPorCluster, n_features_total):
    featuresPorCluster = []
    #print(samplesPorCluster)
    for cluster in samplesPorCluster:
        if cluster!= [None]:
            atual = []
            foi = True
            for sample in cluster:
                if foi:
                    caminho = modelo.decision_path([df.values[sample]]).nonzero()[1]
                    for n in caminho:
                        #print(features, n, features[n])
                        if features[n] not in atual and features[n]>-1:
                            atual.append(features[n])
                    feat_indiv = [0]*n_features_total
                    for i in atual:
                        feat_indiv[i] = 1                    
                    foi = False
                    featuresPorCluster.append(feat_indiv)
        else:
            featuresPorCluster.append([None])
    return featuresPorCluster
                        

            


# In[ ]:




