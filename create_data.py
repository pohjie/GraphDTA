import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *

import pdb

col1 = {'H', 'Li', 'Na', 'K'}
col2 = {'Mg', 'Ca'}
col4 = {'Ti', 'Zr'}
col5 = {'V'}
col6 = {'Cr'}
col7 = {'Mn'}
col8 = {'Fe'}
col9 = {'Co'}
col10 = {'Ni', 'Pd', 'Pt'}
col11 = {'Cu', 'Ag', 'Au'}
col12 = {'Zn', 'Cd', 'Hg'}
col13 = {'Al', 'In', 'Tl'}
col14 = {'C', 'Si', 'Ge', 'Sn', 'Pb'}
col15 = {'N', 'P', 'As', 'Sb'}
col16 = {'O', 'S', 'Se'}
col17 = {'F', 'Cl', 'Br', 'I'}
col18 = {'He'}
periodic_table = [col1, col2, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14,
                  col15, col16, col17, col18]

ionisation_energy = {'C':11.2603, 'N':14.5341, 'O':13.6181, 'S':100.36, 'F':17.4228, 'Si':8.1517,
                     'P':10.4867, 'Cl':12.9676, 'Br':11.8138, 'Mg':7.6462, 'Na':5.1391, 'Ca':6.1132,
                     'Fe':7.9024, 'As':9.8152, 'Al':5.9858, 'I':10.4513, 'B':8.298, 'V':6.7463, 
                     'K':4.3407, 'Tl':6.1083, 'Yb':6.2542, 'Sb':8.64, 'Sn':7.3438, 'Ag':7.5762,
                     'Pd':8.3369, 'Co':7.881, 'Se':9.7524, 'Ti':6.8282, 'Zn':9.3941, 'H':13.5984,
                     'Li':5.3917, 'Ge':7.9, 'Cu':7.7264, 'Au':9.2257, 'Ni':7.6398, 'Cd':8.9937,
                     'In':5.7864, 'Mn':7.434, 'Zn':7.434, 'Cr':6.7666, 'Pt':9, 'Hg':10.4375, 'Pb':7.4167,
                     'Unknown':0}

ionisation_sum = 0
for e in ionisation_energy:
    ionisation_sum += ionisation_energy[e]

electronegativity = {'C':2.55, 'N':3.04, 'O':3.44, 'S':2.58, 'F':3.98, 'Si':1.9,
                     'P':2.19, 'Cl':3.16, 'Br':2.96, 'Mg':1.31, 'Na':0.93, 'Ca':1.0,
                     'Fe':1.83, 'As':2.18, 'Al':1.61, 'I':2.66, 'B':2.04, 'V':1.63, 
                     'K':0.82, 'Tl':1.62, 'Yb':0, 'Sb':2.05, 'Sn':1.96, 'Ag':1.93,
                     'Pd':2.2, 'Co':1.88, 'Se':2.55, 'Ti':1.54, 'Zn':1.65, 'H':2.2,
                     'Li':0.98, 'Ge':2.01, 'Cu':1.9, 'Au':2.58, 'Ni':1.91, 'Cd':1.69,
                     'In':1.78, 'Mn':1.55, 'Zn':1.65, 'Cr':1.66, 'Pt':2.28, 'Hg':2, 'Pb':2.33,
                     'Unknown':0}

electronegativity_sum = 0
for e in electronegativity:
    electronegativity_sum += electronegativity[e]

atomic_radius = {'C':69, 'N':71, 'O':66, 'S':2.58, 'F':64, 'Si':111,
                 'P':107, 'Cl':102, 'Br':120, 'Mg':141, 'Na':166, 'Ca':176,
                 'Fe':132, 'As':119, 'Al':121, 'I':139, 'B':84, 'V':153, 
                 'K':203, 'Tl':145, 'Yb':187, 'Sb':139, 'Sn':139, 'Ag':144,
                 'Pd':139, 'Co':126, 'Se':120, 'Ti':160, 'Zn':122, 'H':31,
                 'Li':121, 'Ge':122, 'Cu':132, 'Au':136, 'Ni':124, 'Cd':144,
                 'In':142, 'Mn':139, 'Zn':122, 'Cr':139, 'Pt':136, 'Hg':132, 'Pb':146,
                 'Unknown':0}

atomic_r_sum = 0
for r in atomic_radius:
    atomic_r_sum += atomic_radius[r]

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()] +
                    one_of_k_encoding_sets(atom.GetSymbol(), periodic_table) +
                    [ionisation_energy[atom.GetSymbol()]/ionisation_sum] +
                    [electronegativity[atom.GetSymbol()]/electronegativity_sum] +
                    [atomic_radius[atom.GetSymbol()]/atomic_r_sum])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_sets(x, allowable_set):
    idx = -1
    for i in range(len(periodic_table)):
        if x in periodic_table[i]:
            idx = i

    encoding = [0] * (len(periodic_table) + 1)
    encoding[idx] = 1
    return encoding

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        pdb.set_trace()
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]
    return x  


# from DeepDTA data
all_prots = []
datasets = ['kiba','davis']
for dataset in datasets:
    print('convert data from DeepDTA for ', dataset)
    fpath = 'data/' + dataset + '/'
    train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
    train_fold = [ee for e in train_fold for ee in e ]
    valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')
    drugs = []
    prots = []
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]),isomericSmiles=True)
        drugs.append(lg)
    for t in proteins.keys():
        prots.append(proteins[t])
    if dataset == 'davis':
        affinity = [-np.log10(y/1e9) for y in affinity]
    affinity = np.asarray(affinity)
    opts = ['train','test']
    for opt in opts:
        rows, cols = np.where(np.isnan(affinity)==False)  
        if opt=='train':
            rows,cols = rows[train_fold], cols[train_fold]
        elif opt=='test':
            rows,cols = rows[valid_fold], cols[valid_fold]
        with open('data/' + dataset + '_' + opt + '.csv', 'w') as f:
            f.write('compound_iso_smiles,target_sequence,affinity\n')
            for pair_ind in range(len(rows)):
                ls = []
                ls += [ drugs[rows[pair_ind]]  ]
                ls += [ prots[cols[pair_ind]]  ]
                ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                f.write(','.join(map(str,ls)) + '\n')       
    print('\ndataset:', dataset)
    print('train_fold:', len(train_fold))
    print('test_fold:', len(valid_fold))
    print('len(set(drugs)),len(set(prots)):', len(set(drugs)),len(set(prots)))
    all_prots += list(set(prots))
    
    
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:i for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

compound_iso_smiles = []
for dt_name in ['kiba','davis']:
    opts = ['train','test']
    for opt in opts:
        df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
        compound_iso_smiles += list( df['compound_iso_smiles'] )
compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g

datasets = ['davis','kiba']
# convert to PyTorch data format
for dataset in datasets:
    processed_data_file_train = 'data/processed/' + dataset + '_train_f.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test_f.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        df = pd.read_csv('data/' + dataset + '_train.csv')
        train_drugs, train_prots,  train_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])
        XT = [seq_cat(t) for t in train_prots]
        train_drugs, train_prots,  train_Y = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y)
        df = pd.read_csv('data/' + dataset + '_test.csv')
        test_drugs, test_prots,  test_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])
        XT = [seq_cat(t) for t in test_prots]
        test_drugs, test_prots,  test_Y = np.asarray(test_drugs), np.asarray(XT), np.asarray(test_Y)

        # make data PyTorch Geometric ready
        print('preparing ', dataset + '_train.pt in pytorch format!')
        train_data = TestbedDataset(root='data', dataset=dataset+'_train_f', xd=train_drugs, xt=train_prots, y=train_Y,smile_graph=smile_graph, oxy=train_oxy)
        print('preparing ', dataset + '_oxy_test.pt in pytorch format!')
        test_data = TestbedDataset(root='data', dataset=dataset+'_test_f', xd=test_drugs, xt=test_prots, y=test_Y,smile_graph=smile_graph, oxy=test_oxy)
        print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')        
    else:
        print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')      