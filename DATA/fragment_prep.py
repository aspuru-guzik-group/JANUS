#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 13:50:21 2021

@author: akshat
"""
from selfies import encoder, decoder 
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def sanitize_smiles(smi):
    '''Return a canonical smile representation of smi
    Parameters:
    smi (string) : smile string to be canonicalized 
    Returns:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object                          (None if invalid smile string smi)
    smi_canon (string)          : Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): True/False to indicate if conversion was  successful 
    '''
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        return (mol, smi_canon, True)
    except:
        return (None, None, False)
    
    

def get_frags(smi, radius): 
    mol, smi_canon, _ = sanitize_smiles(smi)
    frags=[]
    for ai in range(mol.GetNumAtoms()):
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, ai)
        amap={}
        submol=Chem.PathToSubmol(mol,env,atomMap=amap)
        frag=mol2smi(submol, isomericSmiles=False, canonical=True)
        frags.append(frag)
    return list(set(frags))

import time 
import os
import numpy as np 
import random
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
# from SAS_calculator.sascorer import calculateScore
from rdkit import RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog('rdApp.*')
import pandas as pd
import dask.dataframe as dd
import multiprocessing


import selfies
from selfies import encoder, decoder


from rdkit.Chem import Mol
from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint, GetBTFingerprint
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import warnings
warnings.filterwarnings("ignore")

# Updated SELFIES constraints: 
default_constraints = selfies.get_semantic_constraints()
new_constraints = default_constraints
new_constraints['S'] = 2
new_constraints['P'] = 3
selfies.set_semantic_constraints(new_constraints)  # update constraints


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# from tensorboardX import SummaryWriter
# writer = SummaryWriter()

def sanitize_smiles(smi):
    '''Return a canonical smile representation of smi
    
    Parameters:
    smi (string) : smile string to be canonicalized 
    
    Returns:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object                          (None if invalid smile string smi)
    smi_canon (string)          : Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): True/False to indicate if conversion was  successful 
    '''
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        return (mol, smi_canon, True)
    except:
        return (None, None, False)

class _FingerprintCalculator:
    """
    Calculate the fingerprint while avoiding a series of if-else.
    See recipe 8.21 of the book "Python Cookbook".

    To support a new type of fingerprint, just add a function "get_fpname(self, mol)".
    """

    def get_fingerprint(self, mol: Mol, fp_type: str):
        method_name = 'get_' + fp_type
        method = getattr(self, method_name)
        if method is None:
            raise Exception(f'{fp_type} is not a supported fingerprint type.')
        return method(mol)

    def get_AP(self, mol: Mol):
        return AllChem.GetAtomPairFingerprint(mol, maxLength=10)

    def get_PHCO(self, mol: Mol):
        return Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)

    def get_BPF(self, mol: Mol):
        return GetBPFingerprint(mol)

    def get_BTF(self, mol: Mol):
        return GetBTFingerprint(mol)

    def get_PATH(self, mol: Mol):
        return AllChem.RDKFingerprint(mol)

    def get_ECFP4(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 2)

    def get_ECFP6(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 3)

    def get_FCFP4(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 2, useFeatures=True)

    def get_FCFP6(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 3, useFeatures=True)


def get_fingerprint(mol: Mol, fp_type: str):
    return _FingerprintCalculator().get_fingerprint(mol=mol, fp_type=fp_type)

    

import re
from scipy.stats.mstats import gmean


class AtomCounter:

    def __init__(self, element: str) -> None:
        """
        Args:
            element: element to count within a molecule
        """
        self.element = element

    def __call__(self, mol: Mol) -> int:
        """
        Count the number of atoms of a given type.

        Args:
            mol: molecule

        Returns:
            The number of atoms of the given type.
        """
        # if the molecule contains H atoms, they may be implicit, so add them
        if self.element == 'H':
            mol = Chem.AddHs(mol)

        return sum(1 for a in mol.GetAtoms() if a.GetSymbol() == self.element)

def parse_molecular_formula(formula: str) :
    """
    Parse a molecular formulat to get the element types and counts.

    Args:
        formula: molecular formula, f.i. "C8H3F3Br"

    Returns:
        A list of tuples containing element types and number of occurrences.
    """
    matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    # Convert matches to the required format
    results = []
    for match in matches:
        # convert count to an integer, and set it to 1 if the count is not visible in the molecular formula
        count = 1 if not match[1] else int(match[1])
        results.append((match[0], count))

    return results



def get_isomer_score(smi, isomer): 
    A = parse_molecular_formula(isomer) # The desired counts! 
    mol = Chem.MolFromSmiles(smi)
    save_counts = [] # The actual Counts! 
    for element, n_atoms in A: 
        save_counts.append((element, AtomCounter(element)(mol)))
    
    total_atoms_desired = sum([x[1] for x in A])
    total_atoms_actual  = sum([x[1] for x in save_counts])
    
    val_ = [np.exp( - ((save_counts[i][1] - A[i][1])**2)/2) for i in range(len(A))]
    val_.append(np.exp( - ((total_atoms_actual - total_atoms_desired)**2)/2))
    
    final_1 = gmean(val_)
    return final_1

start_time = time.time()
with open('./guacamol_v1_train.smiles', 'r') as f: 
    guac_smiles = f.readlines()


isomer = 'C11H24'
guac_smiles = [x.strip() for x in guac_smiles]

fp_scores_guac     = [get_isomer_score(smi, isomer) for smi in guac_smiles]

fp_scores_guac_idx = np.argsort(fp_scores_guac)[::-1]
print('Fp calc time: ', time.time()-start_time)

fp_percentile = np.percentile(fp_scores_guac, 99.8)
all_smiles    = [guac_smiles[i] for i in range(len(fp_scores_guac))  if fp_scores_guac[i]>= fp_percentile ]





for i, smi in enumerate(all_smiles): 
    
    if i%1000 == 0: 
        print(i)
    unique_frags = get_frags(smi, radius=3)
    
    for item in unique_frags: 
        try: 
            sf = encoder(item)
            dec_ = decoder(sf)
            
            m = Chem.MolFromSmiles(dec_)
            Chem.Kekulize(m)
            dearom_smiles = Chem.MolToSmiles(m, canonical=False, isomericSmiles=False, kekuleSmiles=True)
            # print(Chem.MolToSmiles(m, canonical=False, isomericSmiles=False, kekuleSmiles=True))
            dearom_mol = Chem.MolFromSmiles(dearom_smiles)
            if dearom_mol == None: 
                raise Exception('mol dearom failes')
                        
            with open('./fragments_selfies.txt', 'a+') as f: 
                f.writelines(['{}\n'.format(encoder(dearom_smiles))])
            # raise Exception('test')
        except: 
            # print('failure on: ', item)
            # raise Exception('test 1')
            continue
        
