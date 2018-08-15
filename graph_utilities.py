#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip
import urllib.request
import networkx as nx
import numpy as np


def load_grqc(path = 'GA-GrQc.txt'):
    edges = []
    with open('CA-GrQc.txt', 'r') as file:
        for index, line in enumerate(file):
            if index >= 4:
                edges.append([int(i) for i in str(line).replace('\n', '').split('\t')])
    grqc = nx.Graph()
    grqc.add_edges_from(ebunch_to_add=edges)
    return grqc

def load_grqc_from_internet(url = 'https://snap.stanford.edu/data/ca-GrQc.txt.gz', 
                            path = './ca-GrQc.txt.gz'):
    """
    function to download data from internet
    It downloads a .gz file and unzip the data into an array.
    Then it returns a networkx graph """
    
    # downloading .gz file and writing it on the disk
    urllib.request.urlretrieve(url, path)  
    
    # unzipping the file 
    f = gzip.open('./ca-GrQc.txt.gz', 'rb')
    file_content = f.read()
    f.close()
    
    # getting edges
    edges = [tuple([int(val) for val in couple.split('\\t')]) for couple in str(file_content).replace('\\r', '').split('\\n')[4:-1]]
    
    # building a graph
    grqc = nx.Graph()
    grqc.add_edges_from(ebunch_to_add=edges)
    return grqc



def create_fake_edges(graph, seed = 1):
    # setting seed for reproductible results
    np.random.seed(seed= seed)
    # sampling nodes to create nodes
    edges_fake = np.random.choice(graph.nodes, size = (len(graph.edges), 2))
    # setting back the seed to None
    np.random.seed(seed=None)
    # deleting false edges
    edges_fake = edges_fake[edges_fake[:,0]!=edges_fake[:,1],:]
    # eliminating true edges
    edges_fake = set([tuple(val) for val in edges_fake])
    edges_true = set(graph.edges)
    edges_fake = edges_fake.difference(edges_true)
    return list(edges_fake)
    

def building_dataset(graph, embedding_dict, edges_fake, shuffle = True):
    edges_true = graph.edges
    embedding_dimension = len(embedding_dict[list(embedding_dict.keys())[0]])
    
    edges_true_embedding = [[embedding_dict[u],
                             embedding_dict[v]] for u,v in edges_true]
    
    edges_fake_embedding = [[embedding_dict[u],
                             embedding_dict[v]] for u,v in edges_fake]
    
    edges_true_embedding = np.array(edges_true_embedding).reshape(-1, 2*embedding_dimension)
    edges_fake_embedding = np.array(edges_fake_embedding).reshape(-1, 2*embedding_dimension)
    
    y = np.array([1 for i in range(len(edges_true_embedding))]+[0 for i in range(len(edges_fake_embedding))]).reshape(-1, 1)
    
    dataset = np.concatenate([np.concatenate([edges_fake_embedding, edges_true_embedding],axis = 0),
                              y], axis = 1)
    #if shuffle : dataset = dataset[np.random.choice(range(len(dataset)), len(dataset), replace = False),:]
    return dataset