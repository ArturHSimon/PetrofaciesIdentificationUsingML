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


# In[ ]:


#Open Log files and create the agent for genetic algorithm and subspace Kmeans

File = open( datetime.datetime.now().strftime("Kcem %I%MDIA%d") + ".txt","w")
Log = open('Kcem','w')
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
#                   self.indiv.append(1) #subespaço completo
            for _ in range(self.Nsamples):
                self.indiv.append(random.choice(range(self.Ncluster)))
        self.fitness = -1.0
        self.generation = generation

        self.id = Agent.contador
        Agent.contador += 1
        
        
    def __str__(self):
        return 'Features: ' +  f'{self.indiv[:self.Ncluster*self.Nfeatures]}' +' Cluster/Sample: ' + f'{self.indiv[self.Ncluster*self.Nfeatures:]}'

    def __len__(self):
        return self.Ncluster * self.Nfeatures + self.Nsamples
    


# In[ ]:


#Subspace Kmeans in all agents checking the fitness value for all metrics implemented and creating a confusion matrix with the expected vs obtained results
def Kcem(agents, df, qntd, y):
    maiorARI = 0
    medioARI = 0
    ARI = 0
    i = -1
    Log = open('LogMatrizConfusao' + '.txt','w')
    for agent in agents:
        i+= 1
        File.write(f'ARI inicial do agente: {adjusted_rand_score(y, agent.indiv[agent.Ncluster*agent.Nfeatures:])}\n')
        featuresPorCluster = []               
        for cluster in range(agent.Ncluster):
            features = []
            for bit in agent.indiv[cluster*agent.Nfeatures:(cluster+1)*agent.Nfeatures]:
                features.append(bit)
            featuresPorCluster.append(features)

        samplesPorCluster = []    
        for cluster in range(agent.Ncluster):        
            samples = []        
            for bit in range(0, agent.Nsamples):
                if agent.indiv[bit+agent.Nfeatures*agent.Ncluster] == cluster:
                    samples.append(bit)
            samplesPorCluster.append(samples)   

        cent = centroids(agent, df, samplesPorCluster,featuresPorCluster)
    
    #     featuresSamplesProximas = []
    #     for cluster, samples in enumerate(samplesPorCluster):
    #         distMin = float('inf')
    #         for sample in samples:
    #             distancia = dist(df.values[sample], cent[cluster], featuresPorCluster[cluster])
    #             if distMin > distancia:
    #                 selecionada = sample
    #                 distMin = distancia
    #         featuresSamplesProximas.append(df.values[selecionada].tolist())
        #clusters = subKMeans(agent, cent, featuresPorCluster, df, maxIter = 300)

        clusters, itermax = subKMeans(agent, cent, featuresPorCluster, df, maxIter = 100)
        clusterDasSamples = [-1] * agent.Nsamples
        for i, cluster in enumerate(clusters):
            for sample in cluster:
                clusterDasSamples[sample] = i
        agent.indiv[agent.Nfeatures*agent.Ncluster:] = clusterDasSamples
        set1 = set(range(agent.Ncluster))
        set2 = set(agent.indiv[agent.Nfeatures*agent.Ncluster:])
        if set1.issubset(set2) == False:
            File.write(f'Agente que retornou do KMeans é degenerado = Samples: {clusterDasSamples}\n')
            agent = checagem([agent])
            agent = agent[0]
        ARI = adjusted_rand_score(y, agent.indiv[agent.Nfeatures*agent.Ncluster:])
        if maiorARI<ARI:
            maiorARI = ARI
        medioARI+=ARI
        File.write(f'\n\Individuo certo ou corrigido = Samples: {agent.indiv[agent.Nfeatures*agent.Ncluster:]}')
        File.write(f'\nARI Após Kmeans: {ARI} \n')
        File.write(f'fitnessSilhueta = {fitness_indiv(agent,df)}\n')
        File.write(f'fitnessMST = {fitMSTindiv(agent,df)}\n')
        File.write(f'fitSepDesag = {fitSepEDesag(agent, df)}\n')
        File.write(f'fitSepDesagDOM = {fitSepEDesagComDOM(agent,df)}\n\n\n')
        MatrizConf(y, agent)

        
    medioARI = medioARI/qntd
    File.write(f'\n\n\nARI MEDIO: {medioARI}      MAIORARI: {maiorARI}')
    File.close()
    Log.close()
    
    
def MatrizConf(y, agent):
    agent = renumera([agent])
    agent = agent[0]
    
    Log.write('\n')
    samplesPorCluster = []    
    for cluster in range(agent.Ncluster):        
        samples = []        
        for bit in range(0, agent.Nsamples):
            if agent.indiv[bit+agent.Nfeatures*agent.Ncluster] == cluster:
                samples.append(bit)
        samplesPorCluster.append(samples)
    
    resposta = []
    for cluster in range(agent.Ncluster):
        samples = []
        for bit in range(agent.Nsamples):
            if y[bit] == cluster:
                samples.append(bit)
        resposta.append(samples)
    print(f'resposta = {resposta}\nmeuscluter= {samplesPorCluster}')
    
    matriz = [[]for _ in range(agent.Ncluster)]
    
    for cluster in range(agent.Ncluster):
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
    


# In[ ]:


#The subspace Kmeans implementation and auxiliary functions
def centroids(agent, df, samplesPorCluster, featuresPorCluster, oldCent = None):
    cent = []

    for cluster, samples in enumerate(samplesPorCluster):
        mediaAtual = [0] * agent.Nfeatures 
        for sample in samples:
            mediaAtual = soma_items(mediaAtual, df.values[sample])
        #if oldCent != None:
        #    mediaAtual = soma_items(mediaAtual, oldCent[cluster])
        #if oldCent != None:
        #    mediaAtual = divide_lista(mediaAtual, (len(samplesPorCluster[cluster])+1))
        #else:
            mediaAtual = divide_lista(mediaAtual, (len(samplesPorCluster[cluster])))
        cent.append(mediaAtual)
        
    for cluster, samples in enumerate(samplesPorCluster):
        if cent[cluster]==[0] * agent.Nfeatures:
            cent[cluster] = Kseleciona(cent, featuresPorCluster, samplesPorCluster, df, agent, cluster)
    return cent
    
    
def Kseleciona(cent, featuresPorCluster, samplesPorCluster, df, agent, questao):
    maxPeso = float('-inf')
    selecionado = None
    for sample, _ in enumerate(agent.indiv[agent.Nfeatures*agent.Ncluster:]):
        minDist = float('inf')
        for cluster, cents in enumerate(cent):
            distancia = dist(df.values[sample], cent[cluster], featuresPorCluster[questao])
            if distancia<minDist:
                minDist = distancia
        densidade = 0
        for samplej, cc in enumerate(agent.indiv[agent.Nfeatures*agent.Ncluster:]):
            if samplej != sample:
                distancia = dist(df.values[sample], df.values[samplej], featuresPorCluster[questao])
                densidade += distancia
        densidade = densidade/agent.Nsamples*-1
        peso = densidade + minDist
        if peso>maxPeso:
            maxPeso = peso
            selecionado = sample
    return df.values[selecionado].tolist()
        
        
def subKMeans(agent, cent, featuresPorCluster, df, maxIter):
    oldCent = [[] for _ in range (agent.Ncluster)]
    iter = 0
    while(oldCent != cent and iter<maxIter):
        #print(iter)
        clusters = [[] for _ in range (agent.Ncluster)]
        for samples in range(agent.Nsamples):
            selectedClusterIndex = None
            minDist = float('inf')
            for i in range(agent.Ncluster):
                distancia = dist(df.values[samples], cent[i], featuresPorCluster[i])
                if distancia < minDist:
                    minDist = distancia
                    selectedClusterIndex = i
            clusters[selectedClusterIndex].append(samples)
        oldCent = cent
        cent = centroids(agent, df, clusters, featuresPorCluster)
        iter += 1
    return clusters, iter


    


# In[ ]:




