import numpy as np
import networkx as nx
import random
from node2vec import *
from graph_utilities import *

def main():
    print('Starting example')
    
    print('Loading data from internet ...')
    grqc = load_grqc_from_internet()
    print('Data loaded')

    print('Creating Node2Vec model ...')
    node2vec = Node2Vec(dimensions = 100, walk_length = 100, num_walks = 10, window_size = 10, p = 1, q = 1)
    node2vec.load_graph(grqc, directed = True, weighted = False)
    print('Model created')

    print('Starting training with 10 epochs ...')
    node2vec.train(epochs = 10, workers = 8, verbose = True)
    print('Training ended')

    print('Saving model ...')
    node2vec.save_model('./embedding_grqc.emb')
    print('Model saved')

    print('Results : ')
    print(node2vec.get_embedding_dictionnary())

    print('Ending example')
    pass


if __name__ == '__main__':
    main()