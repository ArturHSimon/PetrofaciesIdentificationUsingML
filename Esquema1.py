#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from sklearn.cluster import SpectralClustering, KMeans
from time import sleep


# datasetteste = pd.read_excel('dataset37.xlsx')
# df = datasetteste.copy()
# y = df['petrofacie']
# df = df.drop(columns = 'petrofacie')
# df = df.loc[:,::-1]
# name = 'dataset37.xlsx'
# p =df['p'][0]
# df = df.drop(columns = 'p')


# datasetteste = pd.read_excel('Pasta.xlsx')
# df = datasetteste.copy()

dataset = pd.read_excel('subtotals_dataset-filtered.xlsx', header = 2)
df = dataset.copy()
df = df.drop(columns = "features")
df = df.drop(0, 0)
y = df['petrofacie']
df = df.drop(columns = 'petrofacie')
df = df.loc[:,::-1]
name = ' campus basin subtotals_dataset-filtered.xlsx'
p = ' 100% todas as features'

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



dataset = pd.read_excel('subtotals_dataset-filtered.xlsx', header = 2)
df = dataset.copy()
df = df.drop(columns = "features")
df = df.drop(0, 0)
y = df['petrofacie']
df = df.drop(columns = 'petrofacie')
df = df.loc[:,::-1]
name = 'subtotals_dataset-filtered.xlsx'
p = ' 100% todas as features'


# In[ ]:


#Open log files and agent for genetic algorithm
File = open( datetime.datetime.now().strftime("Esquema1 %I%MDIA%d") + ".txt","w")

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
          for _ in range(self.Ncluster * self.Nfeatures):
                self.indiv.append(random.choice([0,1]))
          for _ in range(self.Nsamples):
              self.indiv.append(random.choice(range(self.Ncluster)))
          #self.indiv.extend(resposta) 
        self.fitness = -1.0
        self.generation = generation

        self.id = Agent.contador
        Agent.contador += 1
        
        
    def __str__(self):
        return 'Features: ' +  f'{self.indiv[:self.Ncluster*self.Nfeatures]}' +' Cluster/Sample: ' + f'{self.indiv[self.Ncluster*self.Nfeatures:]}'

    def __len__(self):
        return self.Ncluster * self.Nfeatures + self.Nsamples
    


# In[ ]:


#Implementation of the genetic algorithm workflow
def ga(df, population, Ncluster, generations, qntdFilhos, rate, qntdElite):
    listafit = []
    listaari = []
    inicio = time.time()
    agents =  init_agents(df, population, Ncluster)
    File.write('Dataset: '+name+ 'p:'+ str(p)+ '\n' +  'Parâmetros: ' + '\nPopulação: ' + str(population) +'\n'+ 'Gerações: '+ str(generations)+'\n'+'Quantidade de filhos: '+ str(qntdFilhos)+'\n'+ 'Taxa de mutação: '+ str(rate)+'%'+'\n' + 'Quantidade elite: '+ str(qntdElite))
    for generation in range(generations):

        print ('Generation: ' + str(generation))
        File.write('\n\nGeneration: ' + str(generation))
        agents = checagem(agents)

        agents = renumera(agents)
        
        #agents = ativa_porosidade(agents, df)

        agents  = fitness_pop(agents, df)
        somafit = 0
        listafitGER = []
        listaariGER = []
        maiorfit = agents[0]
        for i in agents:
          listafitGER.append(i.fitness)
          listaariGER.append(adjusted_rand_score(y, i.indiv[i.Ncluster*i.Nfeatures:]))
          somafit += i.fitness
          if i.fitness > maiorfit.fitness:
            maiorfit = deepcopy(i)
        print(f'Maior fitness da geração: {maiorfit.fitness} Id: {maiorfit.id}')

        File.write(f'\nMaior fitness da geração: {maiorfit.fitness} Id: {maiorfit.id}\n Individuo: {maiorfit.indiv}\n')
        File.write(f'Fitness médio: {somafit/len(agents)}\n')
        File.write(f'Desvio padrao ARI: {statistics.stdev(listaariGER)}   Desvio padrao Fitness: {statistics.stdev(listafitGER)}\n')
        File.write(f'ARI melhor individuo: {adjusted_rand_score(y, maiorfit.indiv[maiorfit.Ncluster* maiorfit.Nfeatures:])}')


        listaari.append(adjusted_rand_score(y, maiorfit.indiv[maiorfit.Ncluster* maiorfit.Nfeatures:]))
        listafit.append(maiorfit.fitness)


        agents = selection(agents, qntdFilhos, qntdElite)

        agents = crossover(agents, population, df, Ncluster, generation)

        agents = mutation(agents, qntdFilhos, rate)


#         if any(agent.fitness >= 0.95 for agent in agents):      
#             break
    fim  = time.time()
    print(f'Samples do melhor individuo: {maiorfit.indiv[maiorfit.Ncluster*maiorfit.Nfeatures:]}')
    print(f'features ativas do melhor individuo: {maiorfit.indiv[:maiorfit.Ncluster*maiorfit.Nfeatures]}')
    print(f'Id: {maiorfit.id} Geração de criação: {maiorfit.generation}')
    print(f'tempo de execução: {fim-inicio} segundos')
    File.write(f'\n\nTempo de execução: {fim-inicio} segundos')
    File.close()
    matplotlib.pyplot.ylim(-1,maiorfit.fitness+1)
    matplotlib.pyplot.xlim(0, generations)
    matplotlib.pyplot.plot(listaari)
    matplotlib.pyplot.plot(listafit)
    matplotlib.pyplot.savefig(datetime.datetime.now().strftime("Exec %I%MGRAF")+'.png')
    matplotlib.pyplot.show()
    matplotlib.pyplot.scatter(listaari, listafit)
    matplotlib.pyplot.savefig(datetime.datetime.now().strftime("Exec %I%MGRAF2")+'.png')
    matplotlib.pyplot.show()
 
   


# In[ ]:


#Auxiliary functions
def init_agents(df, population, Ncluster):
    return [Agent(df, Ncluster) for _ in range(population)]

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

def remove_zero(lista):
  l = []
  for i in range(len(lista)):
    if lista[i] != 0:
      l.append(lista[i])
    if lista[i] == 0:
      l.append(9999999)
  return l


# In[ ]:


#The implementation of fitness functions for genetic algorithm
def fitness_pop(agents, df):
    global primeira
    if primeira:
        File.write('\nMétrica usada : Coeficiente de silhueta em subespaços((b(i) modificado))\n')
        primeira = False
    for i in agents:
        i.fitness = fitness_indiv(i, df)
        #i.fitness = fitSepEDesag(i, df)
        #i.fitness = fitMSTindiv(i, df)
        #i.fitness = fitSepEDesagComDOM(i, df)
    return agents


def fitness_indiv(agent, df):   
  a = [0] * agent.Nsamples            #a,b e s: Vetor com tamanho == Numero de samples
  b = []                              #b é criado conforme o decorrer de seu cálculo
  s = a
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
      a[sampleatual] = a[sampleatual] / (cardi[agent.indiv[i]] -1)      #Caso cardinalidade > 1(mais de uma única sample no cluster) eu realizo a divisão por (cardinalidade(daquele cluster) - 1)
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
#         menoratual[agent.indiv[j]] += dist(df.values[sampleatual], df.values[sampleaux], agent.indiv[agent.indiv[i]*agent.Nfeatures:(agent.indiv[i]+1)*agent.Nfeatures])
            #Cálculo com b alterado:
        menoratual[agent.indiv[j]] += dist(df.values[sampleatual], df.values[sampleaux], agent.indiv[agent.indiv[j]*agent.Nfeatures:(agent.indiv[j]+1)*agent.Nfeatures])

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
        
          
          


# In[ ]:


#The genetic algorithm's main functions
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
    
    



def crossover(agents, population, df, generation, Ncluster):

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



def renumera(agents):
  for agent in agents:
    novos = []
    for bit in agent.indiv[agent.Ncluster*agent.Nfeatures:]:
      if bit not in novos:
        novos.append(bit)
    dicionario = {}
    for i in range(len(novos)):
      dicionario[novos[i]] = i
    for bit in range(agent.Ncluster*agent.Nfeatures, len(agent)):
      agent.indiv[bit] = dicionario[agent.indiv[bit]]
    b = [0] * (agent.Nfeatures*agent.Ncluster)
    for i in range(0, agent.Nfeatures*agent.Ncluster,  agent.Nfeatures):
      b[dicionario[i//agent.Nfeatures]*agent.Nfeatures:(dicionario[i//agent.Nfeatures]+1)*agent.Nfeatures] = agent.indiv[i:i+agent.Nfeatures]
    agent.indiv[:agent.Nfeatures*agent.Ncluster] = b
    
  return agents



def checagem(agents):
  for agent in agents:

    for i in range(agent.Ncluster):
      somatudo = 0
      for bit in range(i*agent.Nfeatures, (i+1)*agent.Nfeatures):
        if(agent.indiv[bit] != 0):
          somatudo = 1
      if somatudo == 0:
        agent.indiv[i*agent.Nfeatures + random.choice(range(agent.Nfeatures))] = 1

    temalterado = 1
    while(temalterado == 1):
      temalterado = 0
      total = []
      for i in range(agent.Ncluster*agent.Nfeatures, agent.Ncluster*agent.Nfeatures + agent.Nsamples):
        if i not in total:
          total.append(i)


      temalterado = 0
      if len(total) != agent.Ncluster:
        for i in range(agent.Ncluster):
          if i not in agent.indiv[agent.Nfeatures*agent.Ncluster:]:
            agent.indiv[random.choice(range(agent.Nfeatures*agent.Ncluster,agent.Nfeatures*agent.Ncluster+agent.Nsamples))] = i
            temalterado = 1
      
  return agents



def ativa_porosidade(agents, df):
  posicao = df.columns.get_loc("porosity")
  for agent in agents:
    for bit in range(posicao, agent.Ncluster*agent.Nfeatures, agent.Nfeatures):
      agent.indiv[bit] = 1
  return agents



def mutation(agents, qntdFilhos, rate):

    for agent in agents[qntdFilhos:]:
        for index in range(agent.Ncluster * agent.Nfeatures):
            if random.choice(range(100)) < rate:
                if agent.indiv[index] == 1:
                    agent.indiv[index] = 0
                else:
                    agent.indiv[index] = 1
        for index in range(agent.Ncluster * agent.Nfeatures, agent.Ncluster * agent.Nfeatures + agent.Ncluster):
            if random.choice(range(100)) < rate:
                agent.indiv[index] = random.choice(range(agent.Ncluster))
    return agents


# In[ ]:


ga(df= df,population= 200,Ncluster= 10, generations= 200, qntdFilhos= 20, rate = 5, qntdElite = 8)


# In[ ]:





# In[ ]:




