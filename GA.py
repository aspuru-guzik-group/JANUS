#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 04:14:17 2021

@author: akshat
"""
from scipy.stats.mstats import gmean
import re 
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

# from mutate import get_mutated_smiles
from mutate_parr import get_mutated_smiles
# from crossover import crossover_smiles

import selfies
from selfies import encoder, decoder


from rdkit.Chem import Mol
from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint, GetBTFingerprint
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import warnings
warnings.filterwarnings("ignore")

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

def get_fp_scores(smiles_back, target_smi): 
    smiles_back_scores = []
    target    = Chem.MolFromSmiles(target_smi)
    fp_target = AllChem.GetMorganFingerprint(target, 2)
    for item in smiles_back: 
        mol    = Chem.MolFromSmiles(item)
        fp_mol = AllChem.GetMorganFingerprint(mol, 2)
        score  = TanimotoSimilarity(fp_mol, fp_target)
        smiles_back_scores.append(score)
    return smiles_back_scores

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

from scipy.stats.mstats import gmean

def get_isomer_score(smi, isomer='C9H10N2O2PF2Cl'): 
    A = parse_molecular_formula(isomer) # The desired counts! 
    
    # print('Looking at smiles: ', smi)
    mol = Chem.MolFromSmiles(smi)
    
    save_counts = [] # The actual Counts! 
    for element, n_atoms in A: 
        # print('Looking at: ', element)
        # print('Count: ', AtomCounter(element)(mol))
        save_counts.append((element, AtomCounter(element)(mol)))
        # raise Exception('TESTING!')
    
    total_atoms_desired = sum([x[1] for x in A])
    total_atoms_actual  = sum([x[1] for x in save_counts])
    
    val_ = [np.exp( - ((save_counts[i][1] - A[i][1])**2)/2) for i in range(len(A))]
    val_.append(np.exp( - ((total_atoms_actual - total_atoms_desired)**2)/2))
    
    final_1 = gmean(val_)
    return final_1



def get_random_smiles(num_random): 

    alphabet = ['[=N]', '[C]', '[S]','[Branch3_1]','[Expl=Ring3]','[Branch1_1]','[Branch2_2]','[Ring1]', '[#P]','[O]', '[Branch2_1]', '[N]','[=O]','[P]','[Expl=Ring1]','[Branch3_2]','[I]', '[Expl=Ring2]', '[=P]','[Branch1_3]','[#C]','[Cl]', '[=C]','[=S]','[Branch1_2]','[#N]','[Branch2_3]','[Br]','[Branch3_3]','[Ring3]','[Ring2]','[F]'] +  ['[C][=C][C][=C][C][=C][Ring1][Branch1_2]']*2 
    max_smi_len = 81
    collect_random = []
    
    for _ in range(num_random): 
        random_len = random.randint(1, max_smi_len+1)
        random_alphabets = list(np.random.choice(alphabet, random_len)) 
        random_selfies = ''.join(x for x in random_alphabets)
        
        collect_random.append(decoder(random_selfies))
    
    return [x for x in collect_random if x != '']


def get_good_bad_smiles(fitness, population, generation_size): 
    
    fitness             = np.array(fitness)
    idx_sort            = fitness.argsort()[::-1] # Best -> Worst
    keep_ratio          = 0.2
    keep_idx            = int(len(list(idx_sort)) * keep_ratio)
    try: 

        F_50_val  = fitness[idx_sort[keep_idx]]
    
        F_25_val = (np.array(fitness) - F_50_val)
        F_25_val = np.array([x for x in F_25_val if x<0]) + F_50_val
        F_25_sort = F_25_val.argsort()[::-1]
        F_25_val = F_25_val[F_25_sort[0]]
    
        prob_   = 1 / ( 3**((F_50_val-fitness) / (F_50_val-F_25_val)) + 1 )
        
        prob_   = prob_ / sum(prob_)  
        to_keep = np.random.choice(generation_size, keep_idx, p=prob_)    
        to_replace = [i for i in range(generation_size) if i not in to_keep][0: generation_size-len(to_keep)]
        
        keep_smiles = [population[i] for i in to_keep]    
        replace_smiles = [population[i] for i in to_replace]
        
        best_smi = population[idx_sort[0]]
        if best_smi not in keep_smiles: 
            keep_smiles.append(best_smi)
            if best_smi in replace_smiles: replace_smiles.remove(best_smi)
            
        if keep_smiles == [] or replace_smiles == []: 
            raise Exception('Badly sampled population!')
    except: 
        keep_smiles = [population[i] for i in idx_sort[: keep_idx]]
        replace_smiles = [population[i] for i in idx_sort[keep_idx: ]]

    return keep_smiles, replace_smiles


if __name__ == '__main__': 
    
    start_time = time.time()
    with open('./DATA/guacamol_v1_train.smiles', 'r') as f: 
        guac_smiles = f.readlines()
    
    guac_smiles = [x.strip() for x in guac_smiles][0: 100] # TODO! 
    fp_scores_guac     = [get_isomer_score(smi) for smi in guac_smiles]
    fp_scores_guac_idx = np.argsort(fp_scores_guac)[::-1]
    print('Fp calc time: ', time.time()-start_time)
    
    initial_mol = [guac_smiles[i] for i in fp_scores_guac_idx]
    with open('./RESULTS/' + 'init_mols.txt', 'w') as f: 
        f.writelines([ '{} \n'.format(x) for x in initial_mol])
    
    
    smiles_collector = {} # A tracker for all smiles! 
    generations      = 200
    generation_size  = 5000 # 5000
    num_mutation_ls  = [5]
    mutn_prob        = 0.75       # chance of mutation; else a crossover is performed 
    choice_ls        = [1, 2, 3] # Insert; replace; delete 
    
            
    # population = np.random.choice(initial_mol, size=generation_size).tolist()
    population = initial_mol[0: generation_size]

    
    # Calculate fitness for the initial population: 
    unique_pop     = list(set(population))

    
    prop_map = {}
    for item in unique_pop: 
        prop_map[item] = get_isomer_score(item)
    
    fitness        = []
    for item in population: 
        fitness.append(prop_map[item])
        
    # Save fitness onto global collector: 
    for item in prop_map: 
        smiles_collector[item] = [prop_map[item], 1] # [Property_value, Count]

    
    # with open('./DATA/fragments_selfies.txt', 'r') as f: 
    #     alphabet = f.readlines()
    # alphabet = [x.strip() for x in alphabet]
    # print('Mutation alphabets obtained! ')
    
    for gen_ in range(generations): 
        
        pop_best_ = []
        
        # STEP 1: OBTAIN THE NEXT GENERATION OF MOLECULES (FOR EXPLORATION): 
        # Order based on fitness, and decide which part of the population is to be kept/discarded: 
        keep_smiles, replace_smiles = get_good_bad_smiles(fitness, population, generation_size)
        replace_smiles = list(set(replace_smiles))

        # Mutations:     
        mut_smi_dict = get_mutated_smiles(replace_smiles[0: len(replace_smiles)], alphabet=['[C]']) # Half the molecuules are to be mutated     

        all_mut_smiles = []
        del mut_smi_dict["lock"]
        for key in mut_smi_dict: 
            all_mut_smiles.extend(mut_smi_dict[key])
        all_mut_smiles = list(set(all_mut_smiles))
        all_mut_smiles = [x for x in all_mut_smiles if x != '']
        
        all_smiles = list(set(all_mut_smiles))
        all_smiles_unique = [x for x in all_smiles if x not in smiles_collector]
        
        # STEP 2: CONDUCT FITNESS CALCULATION FOR THE EXPLORATION MOLECULES: 
        replaced_pop = random.sample(all_smiles_unique, generation_size-len(keep_smiles) )

        population   = keep_smiles + replaced_pop
        
        unique_pop     = list(set(population))
        prop_map = {}
        for item in unique_pop: 
            prop_map[item] = get_isomer_score(item)
            
        fitness        = []
        for item in population: 
            if item in prop_map: 
                fitness.append(prop_map[item])
            else: 
                fitness.append(smiles_collector[item][0])
            
        # Save fitness onto global collector: 
        for item in population: 
            if item not in smiles_collector: 
                smiles_collector[item] = [prop_map[item], 1] # [Property_value, Count]
            else: 
                smiles_collector[item] = [smiles_collector[item][0], smiles_collector[item][1]+1]
        
        print('On generation {}/{}'.format(gen_, generations) )
        idx_sort = np.argsort(fitness)[::-1]
        top_idx = idx_sort[0]
        print('    (Explr) Top Fitness: {}'.format(fitness[top_idx]))
        print('    (Explr) Top Smile: {}'.format(population[top_idx]))
        

            
        fitness_sort = [fitness[x] for x in idx_sort]
        with open('RESULTS/fitness_explore.txt', 'w') as f: 
            f.writelines(['{} '.format(x) for x in fitness_sort])
            f.writelines(['\n'])
        population_sort = [population[x] for x in idx_sort]
        with open('RESULTS/population_explore.txt', 'w') as f: 
            f.writelines(['{} '.format(x) for x in population_sort])
            f.writelines(['\n'])

        pop_best_.extend(population_sort)

        
        # STEP 3: CONDUCT LOCAL SEARCH: 
        smiles_local_search = [population[top_idx]]
        mut_smi_dict_local  = get_mutated_smiles(smiles_local_search, alphabet=['[C]'], space='Local')
        mut_smi_dict_local  = mut_smi_dict_local[population[top_idx]]
        mut_smi_dict_local  = [x for x in mut_smi_dict_local if x not in smiles_collector]

        fp_scores          = get_fp_scores(mut_smi_dict_local, population[top_idx])
        fp_sort_idx        = np.argsort(fp_scores)[::-1][: generation_size]
        mut_smi_dict_local_calc = [mut_smi_dict_local[i] for i in fp_sort_idx] # TODO: LOCAL SEARCH!!
        
        
        # STEP 4: CALCULATE THE FITNESS FOR THE LOCAL SEARCH: 
        prop_map = {}
        for item in mut_smi_dict_local_calc: 
            prop_map[item] = get_isomer_score(item)
        
        
        fitness_local_search = []
        for item in mut_smi_dict_local_calc: 
            if item in prop_map: 
                fitness_local_search.append(prop_map[item])
            else: 
                fitness.append(smiles_collector[item][0])
        
        idx_sort_lc = np.argsort(fitness_local_search)[::-1]
        print('    (Local) Top Fitness: {}'.format(fitness_local_search[idx_sort_lc[0]]))
        print('    (local) Top Smile: {}'.format(mut_smi_dict_local_calc[idx_sort_lc[0]]))
        
        # Store the results: 
        for item in mut_smi_dict_local_calc: 
            if item not in smiles_collector: 
                smiles_collector[item] = [prop_map[item], 1] # [Property_value, Count]
            else: 
                smiles_collector[item] = [smiles_collector[item][0], smiles_collector[item][1]+1]
        
        # TODO: For the NN! 
        mut_smi_dict_local_remain = [x for x in mut_smi_dict_local if x not in mut_smi_dict_local_calc]
        
        # Logging: 
        fitness_sort = [fitness_local_search[x] for x in idx_sort_lc]
        with open('RESULTS/fitness_local_search.txt', 'w') as f: 
            f.writelines(['{} '.format(x) for x in fitness_sort])
            f.writelines(['\n'])

        population_sort = [mut_smi_dict_local_calc[x] for x in idx_sort_lc]
        pop_best_.extend(population_sort)


        with open('RESULTS/population_local_search.txt', 'w') as f: 
            f.writelines(['{} '.format(x) for x in population_sort])
            f.writelines(['\n'])

        # STEP 5: EXCHANGE THE POPULATIONS:         
        # Introduce changes to 'fitness' & 'population'
        # With replacesments from 'fitness_local_search' & 'mut_smi_dict_local_calc'
        num_exchanges     = 5
        introduce_smiles  = population_sort[0:num_exchanges] # Taking the top 5 molecules
        introduce_fitness = fitness_sort[0:num_exchanges]    # Taking the top 5 molecules
        
        worst_indices = idx_sort[-num_exchanges: ]
        for i,idx in enumerate(worst_indices): 
            try: 
                population[idx] = introduce_smiles[i]
                fitness[idx]    = introduce_fitness[i]
            except: 
                continue 
        
        # Save best of generation!: 
        fit_all_best = np.argmax(fitness)

            
        scores = [get_isomer_score(smi, isomer='C9H10N2O2PF2Cl') for smi in pop_best_] 
        a          = np.argsort(scores)[::-1][:250]
        scores_top = [scores[i] for i in a ]
        smiles_top = [pop_best_[i] for i in a]
        print('GUACAMOL Score: ', gmean(scores_top))

            
        with open('./RESULTS' + '/generation_all_best.txt', 'a+') as f: 
            f.writelines(['Gen:{}, {}, {} {}\n'.format(gen_,  population[fit_all_best], fitness[fit_all_best], gmean(scores_top))])
            





