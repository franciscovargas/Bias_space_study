#!/usr/bin/python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from sklearn.preprocessing import normalize
from itertools import chain
import random
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import laplacian
import scipy.sparse.linalg as sinalg
from scipy.sparse import identity
from collections import OrderedDict
from scipy import sparse
import numpy as np
import os
from tqdm import tqdm


def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False


random.seed(10)
def parse_digraph(filename, sales_shares_file):
    pwd = os.popen("pwd").read()

    with open(filename) as f:
        # data = [(int(y), int(x)) for (x,y) in csv.reader(f)]
        data = [(int(x), int(y)) for (x,y) in csv.reader(f)]

    with open(sales_shares_file) as f:
        K = [] #[(int(y), int(x)) for (x,y) in csv.reader(f)]
        for i, (x,y)  in enumerate(csv.reader(f)):
            if i == 0 or not isfloat(y): continue
            K.append((int(x), float(y)))
    print("loaded sales")
    nods = set(chain(*data))

    print(">>")
    print("number of nodes: {0}, number of edges:{1}".format(len(nods),
                                                           len(set(data)) ) )
    print(">>")
    G = nx.DiGraph()
    # import ipdb; ipdb.set_trace()
    G.add_edges_from(data)
    print("Graph Created {0}, {1}".format(len(G.degree()), len(nods)))
    degs = list(dict(G.degree()).values())

    print("Average degree: {0}".format( np.mean(degs) ))
    print("deg len: {0}".format(len(degs)))

    for (u, v) in G.edges:
        G[u][v]['weight'] = 1.0

    print("Random weights assinged")
    # import ipdb; ipdb.set_trace()
    return G, np.asarray(degs), G.nodes, K


def return_adj_list(filename="adjacency_matrix2.csv",
                    sales_shares_file='sales_shares_RA.csv'):
    """
    Further processes networkx file

    """
    G, degs, dict_ma, K = parse_digraph(filename, sales_shares_file)

    dict_map = OrderedDict(zip(dict_ma.keys(), range(len(dict_ma))))

    # Cast dag to matrix
    A = nx.to_scipy_sparse_matrix(G, format='csr').astype(np.float32)
    print( "Done casting to matrix" )

    print("Alocating shares matrix")
    Kk = np.zeros(A.shape[0])
    counts = 0
    for x, y in K:
        if x not in dict_map:
            counts += 1
            continue
        i = dict_map[x]
        if Kk[i] != 0:
            # import pdb; pdb.set_trace()
            pass
        Kk[i] = y
    if Kk.sum() != 1.0:
       print( "ALERT KKSUM = {0}".format(Kk.sum()) )
    Kk = Kk / Kk.sum()
    print("dag returned and created")
    # import ipdb; ipdb.set_trace()
    print("Num zeros: ", np.sum(Kk ==  0) ,"/", float(Kk.shape[0]) )
    print("sales not found: ", counts, "/", len(K))
    Kk[Kk ==  0] = 1e-06
    # import pdb; pdb.set_trace()
    return normalize(A, norm='l1', axis=1), dict_map, Kk


def create_flag_vector_old(map_dict, loc="flag.csv"):
    """
    Creating the binary vector to product LAL' and such with
    """
    with open(loc) as f:
        data= np.array([list(x) for x in  csv.reader(f)])

    d = len(map_dict)

    e = np.ones(d)
    count = 0.0
    for x,y,z,w in data[1:]:
        if int(x) in map_dict:
            e[map_dict[int(x)]] = 1 if z != '' and int(z) == 0 else 0
        else:
            count += 1

    print( "Failed: {0}/ {1}".format(count,  len(data[1:])) )
    return sparse.csr_matrix(e.reshape(-1,1))


def create_flag_vector(map_dict, loc="KJDATA_loc2010_hokkaido.csv"):
    """
    Creating the binary vector to product LAL' and such with
    """
    with open(loc) as f:
        data= np.array([list(x) for x in  csv.reader(f)])

    d = len(map_dict)

    e = np.ones(d)
    count = 0.0
    for x,y,z,w,f in data[1:]:
        if int(x) in map_dict:
            e[map_dict[int(x)]] = int(f)
        else:
            count += 1

    print( "Failed: {0}/ {1}".format(count,  len(data[1:])) )
    return sparse.csr_matrix(e.reshape(-1,1))


def compute_covariates(P, e, K=None, power=50, muc=0.5):
    """
    P = 0.5 * A
    e = binary vector
    power = order of the approximation
    """
    d, d = P.shape
    if K is None: K = np.ones(d).reshape(-1,1)

    print("started diag inverse")
    Ki = sparse.eye(d).multiply(1.0 / K)
    K = sparse.eye(d).multiply(K)
    print("Done diag inverse")

    Le = e.copy()
    e1 = e.copy()
    # Le computation
    for i in range( power ):
        e1 = P.dot(e1)

        Le += e1

    # deallocating e1 (need the space)
    del e1

    print("Le Done")
    # diag(A1)*L*e
    one_vec = np.ones(d).reshape(-1,1)
    #  A'K.dot(1)
    AK1 =  ((1.0 / muc )* P.T).dot( ( K.dot( sparse.csr_matrix(one_vec) ) ) )
    LdAKLe = AK1.multiply(Le.copy())
    print("dAKLe  Done")

    # A'KLe computation  Needed for L'ALe
    LAKLe = ((1.0 / muc) * P.T).dot( (K.dot(Le)) ).copy()

    # AK'ALe computation
    LAKALe = ((1.0 / muc) * P.T).dot( K.dot( (1.0 / muc) * P.dot( Le ) ) )

    e1, e2, e3 = LAKALe.copy(), LAKLe.copy(), LdAKLe.copy()
    #  L'A'ALe and L'ALe computations :
    for i in range( power ):

        # L'AKA'Le
        e1 = P.T.dot(e1)
        LAKALe += e1

        # L'ALe
        e2 = P.T.dot(e2)
        LAKLe += e2

        # L'dALe
        e3 = P.T.dot(e3)
        LdAKLe += e3

    del e1, e2, e3

    return Ki.dot(LAKALe - LAKLe), Ki.dot(LdAKLe - LAKALe)



if __name__ == '__main__':

    # Create dag and Cast dag to matrix and row normalize
    A, dict_map, K = return_adj_list(filename="adjacency_matrix2010_2.csv",
                                     sales_shares_file='All_ids_sales_share_2010_RA.csv')
    print("dag returned and created and normalised")
    d, d = A.shape
    muc = 0.6
    P = muc * A
    e = create_flag_vector_old(dict_map, loc="flag.csv")

    covariate1, covariate2  = compute_covariates(P, e, K=K, muc=muc)
    covariate1, covariate2 = covariate1.toarray().flatten(), covariate2.toarray().flatten()

    print("done with covariates")

    kes = list(dict_map.keys())
    psv = pd.DataFrame({"id": kes, "covariate1": covariate1, "covariate2": covariate2})
    print("loaded DataFrame")
    # import pdb; pdb.set_trace()
    psv.to_csv("covariates_50_alireza_fix_epsilon_mu_{0}_stabler_old_2010.csv".format(muc), index=False, header=True)
    print("DONE")
