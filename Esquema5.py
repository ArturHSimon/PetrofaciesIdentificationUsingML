#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import libraries
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


# In[ ]:


#Data input

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


dataset = pd.read_excel('Jequitinhonha_subtotals_dataset-filtered.xlsx', header = 2)
df = dataset.copy()
df = df.drop(columns = "features")
df = df.drop(0, 0)
y = df['petrofacie']
df = df.drop(columns = 'petrofacie')
df = df.loc[:,::-1]
name = 'Jequitinhonha subtotals_dataset-filtered.xlsx'
p = ' 100% todas as features'


# In[ ]:


#The ga's agent

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
            for _ in range(self.Nfeatures):
                  self.indiv.append(random.choice([True,False]))
        self.fitness = -1.0
        self.generation = generation
        self.labels = []
        self.id = Agent.contador
        Agent.contador += 1
           
    def __str__(self):
        return 'Features: ' +  f'{self.indiv}'

    def __len__(self):
        return len(self.indiv)


# In[ ]:


#GA's main funtions
def mutation(agents, qntdFilhos, rate):
    for agent in agents[qntdFilhos:]:
        for index in range(len(agent.indiv)):
            if random.choice(range(100)) < rate:
                if agent.indiv[index] == True:
                    agent.indiv[index] = False
                else:
                    agent.indiv[index] = True
    return agents
def ativa_porosidade(agents, df):
    posicao = df.columns.get_loc("porosity")
    for agent in agents:
        agent.indiv[posicao] = True
    return agents
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
def selection(agents, ratio, qntdElites = 0):
    selected = []
    totalfit = 0.0
    menorfit = float('inf')
    
    while(len(selected) < qntdElites):
        elite_idx = 0
        for i,agent in enumerate(agents):
            #print(i)
            if agent.fitness > agents[elite_idx].fitness:
                elite_idx = i
        selected.append(agents.pop(elite_idx))
    
    
    for agent in agents:
        if agent.fitness < menorfit:
            menorfit = agent.fitness
    
    for agent in agents:
        agent.fitness += abs(menorfit)
        
    for agent in agents:
        totalfit = totalfit + agent.fitness


    while(len(selected) < ratio):
        ponto = random.uniform(0,totalfit)
        i = -1
        atual = 0
        while(atual < ponto):
            i+=1
            atual += agents[i].fitness

        totalfit = totalfit-agents[i].fitness
        agents[i].fitness -= abs(menorfit)
        selected.append(agents.pop(i))

    return selected


# In[ ]:


#Auxiliary funtions
def dist(v1,v2):
    soma = 0
    for i in range(len(v1)):
        soma+=(v1[i]-v2[i])**2
    return soma**(1/2)
def init_agents(df, population, Ncluster, vetor = None):
    if vetor == None:
        return [Agent(df, Ncluster) for _ in range(population)]
    else:
        return [Agent(df, Ncluster, 0, vetor[x]) for x in range(population)]


def novo_indiv(df, Ncluster, generation, vetor):
    return Agent(df, Ncluster, generation, vetor)

def InfoGer(agents):
    continue


# In[ ]:


#GA's fitness functions
def fitness_indiv(agent, df):
    clusters = list(agent.labels)
    a = [0]*agent.Nsamples
    b = [0]*agent.Nsamples
    s = [0]*agent.Nsamples
    baux = [0]*agent.Ncluster
    
    
    ############################################# cálculo de a[i]
    for i in range(len(clusters)):
        for j in range(len(clusters)):
            if i == j:
                continue
            elif clusters[i]==clusters[j]:
                a[i]+=dist(df.values[i], df.values[j])
    
    for i in range(len(a)):
        if clusters.count(clusters[i])>1:
            a[i] = a[i]/(clusters.count(clusters[i])-1)
        else:
            a[i] = 0
    #############################################################
    
    ###########################################   cálculo de b[i]
    for i in range(len(clusters)):
        for j in range(len(clusters)):
            if clusters[i]!=clusters[j]:
                baux[clusters[j]]+=dist(df.values[i], df.values[j])
        for j in range(len(baux)):
            baux[j] = baux[j]/clusters.count(j)
        b[i] = min(baux)
    #############################################################
    
    ##############################################  cálculo s[i]
    
    for i in range(len(s)):
        if(a[i] > b[i]):
            s[i] = (b[i]/a[i]) - 1
        elif(a[i] < b[i]):
            s[i] = 1-(a[i]/b[i])
        else:
            s[i] = 0
    ##############################################################
    soma = 0
    for _ in s:
        soma+=_
    return soma/len(s)
    
    
    
    
                
            
    


# In[ ]:


#Genetic algorithm workflow
def GA(df, population, Ncluster, generations,qntdFilhos, rate, qntdElite):
    agents = init_agents(df,population, Ncluster)
    melhor = agents[0]
    for generation in range(generations):
        print(f'Geração: {generation}')
        melhoragente = agents[0]
        #print(f'tamanho agentes: {len(agents)}')
        for agent in agents:
            
            modelo = SpectralClustering(n_clusters = Ncluster, affinity = 'nearest_neighbors', n_neighbors = 22,random_state=1)
            agent.labels = list(modelo.fit(df.iloc[:, agent.indiv]).labels_)
            
            agent.fitness = fitness_indiv(agent, df.iloc[:, agent.indiv])
            
            if agent.fitness>melhoragente.fitness:
                melhoragente = agent
        #InfoGer(agents)
                
        agents = selection(agents, qntdFilhos, qntdElite)
        
        agents = crossover(agents, population, df, generation, Ncluster)
        
        agents = mutation(agents, qntdFilhos, rate)
        
        
        


# In[ ]:


GA(df,population =200,Ncluster = 17,generations = 200, qntdFilhos = 20, rate = 5, qntdElite = 8)


# In[ ]:


#Nearest neighbors optimal value test in Spectral Clustering/Check for errors
for i in range(2,30):
    modelo = SpectralClustering(n_clusters =14, affinity = 'nearest_neighbors', n_neighbors = i,random_state=1)
    print(f'{adjusted_rand_score(list(modelo.fit(df).labels_), y)}')
    sleep(1)


# In[ ]:


#Number of petrofacies of the given dataset based on geological analysis. 
len(set(y))


# In[ ]:





# In[ ]:




