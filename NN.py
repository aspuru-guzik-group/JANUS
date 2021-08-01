#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 19:25:08 2021

@author: akshat
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rdkit
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog('rdApp.*')

import inspect
from collections import OrderedDict

import multiprocessing
manager = multiprocessing.Manager()
lock = multiprocessing.Lock()



def get_rot_bonds_posn(mol):
    '''Return atom indices with Rotatable bonds 
    
    Examples:
    >>> get_rot_bonds_posn('CC1=CC=CC=C1')  # Toluene  (Rotatable Bonds At: CH3 & Benzene)
    ((0, 1),)
    >>> get_rot_bonds_posn('CCC1=CC=CC=C1') # (Rotatable Bonds At: CH3, CH3 & Benzene)
    ((0, 1), (1, 2))
    '''
    RotatableBond = Chem.MolFromSmarts('*-&!@*')
    rot = mol.GetSubstructMatches(RotatableBond)
    return rot

def get_bond_indeces(mol, rot):
    '''Get all the bond indices with Rotatable bonds atoms (generated from 'get_rot_bonds_posn')
    '''
    bonds_idx = []
    for i in range(len(rot)):
        bond = mol.GetBondBetweenAtoms(rot[i][0],rot[i][1])
        bonds_idx.append(bond.GetIdx())
    return bonds_idx


def obtain_rings(smi):
    '''Obtain a list of all rings present in SMILE string smi
    
    Examples:
    >>> obtain_rings('CCC1=CC=CC=C1')
    ['c1ccccc1']
    >>> obtain_rings('C1=CC=C(C=C1)C1=CC=CC=C1')
    ['c1ccccc1', 'c1ccccc1']
    >>> obtain_rings('C1=CC2=C(C=C1)C=CC=C2')
    (None, None)
    
    Parameters:
    smi (string) : SMILE string of a molecule 
    
    Returns
    (list)       : List if all rings in a SMILE string 
    '''
    mol = Chem.MolFromSmiles(smi)
    rot = get_rot_bonds_posn(mol) # Get rotatble bond positions
    
    if len(rot) == 0:
        return None, None
    
    bond_idx = get_bond_indeces(mol, rot)
    new_mol = Chem.FragmentOnBonds(mol, bond_idx, addDummies=False) 
    new_smile = Chem.MolToSmiles(new_mol)
    
    smile_split_list = new_smile.split(".") 
    rings = []
    for item in smile_split_list:
        if '1' in item:
            rings.append(item)
    return rings 


def count_atoms(mol, atomic_num):
    '''Count the number of atoms in mol with atomic number atomic_num
    
    
    Parameters:
    mol (rdkit.Chem.rdchem.Mol) : Molecule in which search is conducted
    atomic_num            (int) : Counting is done in mol for atoms with this atomic number
    Returns:
    (int) :  final count of atom
    '''
    pat = Chem.MolFromSmarts("[#{}]".format(atomic_num))
    return len(mol.GetSubstructMatches(pat))

def get_num_bond_types(mol):
    '''Calculate the ratio of total number of  (single, double, triple, aromatic) bonds to the 
       total number of bonds. 
    
    Parameters:
    mol (rdkit.Chem.rdchem.Mol) : Molecule for which ratios arre retuned 
    
    Returns:
    (list):  [num_single/num_bonds, num_double/num_bonds, num_triple/num_bonds, num_aromatic/num_bonds]
    '''
    bonds = mol.GetBonds()    
    
    num_bonds    = 0
    num_double   = 0
    num_triple   = 0
    num_single   = 0
    num_aromatic = 0
    
    for b in bonds:
        num_bonds += 1
        if b.GetBondType() == rdkit.Chem.rdchem.BondType.SINGLE:
            num_single += 1
        if b.GetBondType() == rdkit.Chem.rdchem.BondType.DOUBLE:
            num_double += 1
        if b.GetBondType() == rdkit.Chem.rdchem.BondType.TRIPLE:
            num_triple += 1
        if b.GetBondType() == rdkit.Chem.rdchem.BondType.AROMATIC:
            num_aromatic += 1
    if num_bonds == 0:
        return [0, 0, 0, 0]
    else:
        return [num_single/num_bonds, num_double/num_bonds, num_triple/num_bonds, num_aromatic/num_bonds]


def count_conseq_double(mol):
    '''Return the number of consequtive double bonds in an entire molecule
       including rings 
    Examples 
    >>> count_conseq_double(Chem.MolFromSmiles('C1=CC=C=C=C1'))
    2
    >>> count_conseq_double(Chem.MolFromSmiles('C1=CC=CC=C1'))
    0
    >>> count_conseq_double(Chem.MolFromSmiles('C1=CC2=C(C=C1)C=C=C=C2'))
    2
    
    Parameters:
    mol (rdkit.Chem.rdchem.Mol) : Molecule for conseq. double bonds are to be counted 
    
    Returns:
    (int):  The integer number of coseq. double bonds 
    '''
    bonds = mol.GetBonds()    
    
    previous_BType    = None
    count_conseq_doub = 0
    for b in bonds:
        curr_BType = b.GetBondType()
        if previous_BType == curr_BType and curr_BType == rdkit.Chem.rdchem.BondType.DOUBLE:
            count_conseq_doub += 1
        previous_BType = curr_BType
    
    return count_conseq_doub


def size_ring_counter(ring_ls):
    '''Get the number of rings of sizes 3 to 20 and the number of consequtive double bonds in a ring
    
    Parameters:
    ring_ls (list)  : list of rings of a molecule 
    
    Returns
    (list)          : Of size 19 (1 for number of conseq. double bonds)
                                 (18 for number of rings between size 3 to 20)
    '''
    ring_counter = []
    
    if ring_ls == (None, None): # Presence of no rings, return 0s for the 19 feature
        return [0 for i in range(19)] 
    
    mol_ring_ls  = [Chem.MolFromSmiles(smi) for smi in ring_ls]
    
    # Cont number consequtive double bonds in ring 
    conseq_dbl_bnd_in_ring = 0
    for item in mol_ring_ls:
        conseq_dbl_bnd_in_ring += count_conseq_double(item)
    ring_counter.append(conseq_dbl_bnd_in_ring) # concatenate onto list ring_counter
    
    # Count the number of consequtive double bonds in rings 
    for i in range(3, 21):
        count = 0
        for mol_ring in mol_ring_ls:
            if mol_ring.GetNumAtoms() == i:
                count += 1
        ring_counter.append(count)
    return ring_counter
            


def get_mol_info(smi):                 
    ''' Calculate a set of 51 RdKit properties, collected from above helper functions. 
    
    Parameters:
    smi (string) : SMILE string of molecule 
    
    Returns:
    (list of float) : list of 51 calculated properties  
    '''
    mol = Chem.MolFromSmiles(smi)
        
    num_atoms   = mol.GetNumAtoms()       
    num_hydro   = Chem.AddHs(mol).GetNumAtoms() - num_atoms 
    num_carbon  = count_atoms(mol, 6)
    num_nitro   = count_atoms(mol, 7)
    num_sulphur = count_atoms(mol, 16)
    num_oxy     = count_atoms(mol, 8)
    num_clorine = count_atoms(mol, 17)
    num_bromine = count_atoms(mol, 35)
    num_florine = count_atoms(mol, 9)

    
    if num_carbon == 0: # Avoid division by zero error, set num_carbon to a very small value 
        num_carbon = 0.0001
    
    basic_props = [num_atoms/num_carbon, num_hydro/num_carbon, num_nitro/num_carbon, 
                     num_sulphur/num_carbon, num_oxy/num_carbon, num_clorine/num_carbon,
                     num_bromine/num_carbon, num_florine/num_carbon]
    
    to_caculate = ["RingCount", "HallKierAlpha", "BalabanJ", "NumAliphaticCarbocycles","NumAliphaticHeterocycles",
                   "NumAliphaticRings","NumAromaticCarbocycles","NumAromaticHeterocycles",
                   "NumAromaticRings","NumHAcceptors","NumHDonors","NumHeteroatoms",
                   "NumRadicalElectrons","NumSaturatedCarbocycles","NumSaturatedHeterocycles",
                   "NumSaturatedRings","NumValenceElectrons"]    

    # Calculate all propoerties listed in 'to_calculate'
    calc_props = OrderedDict(inspect.getmembers(Descriptors, inspect.isfunction))
    for key in list(calc_props.keys()):
        if key.startswith('_'):
            del calc_props[key]
            continue
        if len(to_caculate)!=0 and key not in to_caculate:
            del calc_props[key]
    features = [val(mol) for key,val in calc_props.items()] # List of properties 
    
    
    # Ratio of total number of  (single, double, triple, aromatic) bonds to the total number of bonds. 
    simple_bond_info = get_num_bond_types(mol) 
    
    # Obtain all rings in a molecule and calc. #of triple bonds in rings & #of rings in molecule 
    ring_ls = obtain_rings(smi)
    num_triple = 0      # num triple bonds in ring

    
    if len(ring_ls) > 0 and ring_ls != (None, None):
        for item in ring_ls:
            num_triple += item.count('#')
        simple_bond_info.append(len(ring_ls))     # append number of Rings in molecule 
    else:   simple_bond_info.append(0)            # no rotatable bonds

        
    simple_bond_info.append(num_triple)          # number of triple bonds in rings
                                              
    # Calculate the number of rings of size 3 to 20 & number of conseq. double bonds in rings 
    simple_bond_info = simple_bond_info + size_ring_counter(ring_ls)
    
    # Calculate the number of consequitve double bonds in entire molecule
    simple_bond_info.append(count_conseq_double(mol)) 
    
    return np.array(features + basic_props + simple_bond_info)

def get_mult_mol_info_parr(smiles_list, dataset_x):
    ''' Record calculated rdkit property results for each smile in smiles_list,
    and add record result in dictionary dataset_x.
    '''
    for smi in smiles_list:
        dataset_x['properties_rdkit'][smi] = get_mol_info(smi)


def get_chunks(arr, num_processors, ratio):
    """
    Get chunks based on a list 
    """
    chunks = []  # Collect arrays that will be sent to different processorr 
    counter = int(ratio)
    for i in range(num_processors):
        if i == 0:
            chunks.append(arr[0:counter])
        if i != 0 and i<num_processors-1:
            chunks.append(arr[counter-int(ratio): counter])
        if i == num_processors-1:
            chunks.append(arr[counter-int(ratio): ])
        counter += int(ratio)
    return chunks 

def create_parr_process(chunks):
    '''This function initiates parallel execution (based on the number of cpu cores)
    to calculate all the properties mentioned in 'get_mol_info()'
    
    Parameters:
    chunks (list)   : List of lists, contining smile strings. Each sub list is 
                      sent to a different process
    dataset_x (dict): Locked dictionary for recording results from different processes. 
                      Locking allows communication between different processes. 
                      
    Returns:
    None : All results are recorde in dictionary 'dataset_x'
    '''
    # Assign data to each process 
    process_collector = []
    collect_dictionaries = []
    
    for chunk in chunks:                # process initialization 
        dataset_x         = manager.dict(lock=True)
        smiles_map_props  = manager.dict(lock=True)

        dataset_x['properties_rdkit'] = smiles_map_props
        collect_dictionaries.append(dataset_x)
        
        process_collector.append(multiprocessing.Process(target=get_mult_mol_info_parr, args=(chunk, dataset_x,  )))

    for item in process_collector:      # initite all process 
        item.start()
    
    for item in process_collector:      # wait for all processes to finish
        item.join()   
    
    combined_dict = {}
    for i,item in enumerate(collect_dictionaries):
        combined_dict.update(item['properties_rdkit'])

    return combined_dict


def obtain_discr_encoding(molecules_here, num_processors): 
    dataset_x = []
    for smi in molecules_here: 
        dataset_x.append(get_mol_info(smi))
    return np.array(dataset_x)
    

class Net(torch.nn.Module):
    def __init__(self, n_feature, h_sizes, n_output):
        super(Net, self).__init__()
        # Layers
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes)-1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))
        self.predict = torch.nn.Linear(h_sizes[-1], n_output)


    def forward(self, x):
        for layer in self.hidden:
            x = torch.sigmoid(layer(x))
        output= F.sigmoid(self.predict(x))

        return output

def create_discriminator(init_len, n_hidden, device):
    """
    Define an instance of the discriminator 
    """
    n_hidden.insert(0, init_len)

    net = Net(n_feature=init_len, h_sizes=n_hidden, n_output=1).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    loss_func = torch.nn.BCELoss()
    
    return (net, optimizer, loss_func)



def obtain_initial_discriminator(disc_layers, device):
    ''' Obtain Discriminator initializer
    
    Parameters:
    disc_enc_type        (str)  : (smile/selfie/properties_rdkit)
                                  For calculating num. of features to be shown to discrm.
    disc_layers,         (list) : Intermediate discrm layers (e.g. [100, 10])
    device               (str)  : Device discrm. will be initialized 
    
    Returns:
    discriminator : torch model
    d_optimizer   : Loss function optimized (Adam)
    d_loss_func   : Loss (Cross-Entropy )
    '''
    # Discriminator initialization 
    discriminator, d_optimizer, d_loss_func = create_discriminator(51, disc_layers, device)  
    
    return discriminator, d_optimizer, d_loss_func



def do_x_training_steps(data_x, data_y, net, optimizer, loss_func, steps, graph_x_counter, device):
    
    data_x = torch.tensor(data_x.astype(np.float32), device=device)
    data_y = torch.tensor(data_y, device=device, dtype=torch.float)
    
    net.train()
    for t in range(steps):
        predictions = net(data_x)

        loss = loss_func(predictions, data_y.reshape(len(data_y), 1))
        
        if t % 400 == 0: 
            print('        Epoch:{} Loss:{}'.format(t, loss.item()))

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    return net

def save_model(model, generation_index, dir_name):
    out_dir = './{}/{}'.format(dir_name, generation_index)
    
    if not os.path.isdir(out_dir): 
        os.system('mkdir {}'.format(out_dir))   
        
    torch.save(model, out_dir+'/model')

def load_saved_model(generation_index):
    model = torch.load('./RESULTS/{}/model'.format(generation_index))
    model = model.eval()
    return model 

def do_predictions(discriminator, data_x, device):
    discriminator = discriminator.eval()
    
    data_x = torch.tensor(data_x.astype(np.float32), device=device)
    
    outputs = discriminator(data_x)
    predictions = outputs.detach().cpu().numpy() # Return as a numpy array
    return (predictions)


def train_and_save_model(smiles_ls, pro_val, generation_index): 
    dataset_x = obtain_discr_encoding(smiles_ls, num_processors=1) # multiprocessing.cpu_count()

    avg_val = np.percentile(pro_val, 80) # np.average(pro_val)
    dataset_y  = np.array([1 if x>=avg_val else 0 for x in pro_val ])
    
    disc_layers = [100, 10]
    device      = 'cpu'
    
    discriminator, d_optimizer, d_loss_func = obtain_initial_discriminator(disc_layers, device)
    discriminator = do_x_training_steps(data_x=dataset_x, data_y=dataset_y, net=discriminator, optimizer=d_optimizer, loss_func=d_loss_func, steps=2000, graph_x_counter=1, device=device)
    
    # Save discriminator after training 
    save_model(discriminator, generation_index=generation_index, dir_name='RESULTS')
    

def obtain_new_pred(smiles_ls, generation_index): 


    predictions = []
    model = load_saved_model(generation_index=generation_index) 
    
    for i,smi in enumerate(smiles_ls): 
        if i % 10000 == 0: 
            print('        Predicting: {}/{}'.format(i, len(smiles_ls)))
        data_x  = obtain_discr_encoding([smi], 1)
        data_x  = torch.tensor(data_x.astype(np.float32), device='cpu')
        outputs = model(data_x)
        out_    = outputs.detach().cpu().numpy()
        predictions.append(float(out_[0]))

    return predictions

