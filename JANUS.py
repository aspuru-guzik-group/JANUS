#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 04:14:17 2021

@author: akshat
"""
import time 
import numpy as np 
import random
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity


# from mutate import get_mutated_smiles
from mutate_parr import get_mutated_smiles
from crossover_parr import crossover_smiles_parr


from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import warnings
warnings.filterwarnings("ignore")

from params_init import calc_prop, generate_params
from NN import train_and_save_model, obtain_new_pred


def sanitize_smiles(smi):    
    '''
    Return a canonical smile representation of smi 

    Parameters
    ----------
    smi : str
        smile string to be canonicalized 

    Returns
    -------
    mol (rdkit.Chem.rdchem.Mol) : 
        RdKit mol object (None if invalid smile string smi)
    smi_canon (string)          : 
        Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): 
        True/False to indicate if conversion was  successful 
    '''
    
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        return (mol, smi_canon, True)
    except:
        return (None, None, False)


def get_fp_scores(smiles_back, target_smi): 
    '''
    Given a list of SMILES (smiles_back), tanimoto similarities are calculated 
    (using Morgan fingerprints) to SMILES (target_smi). 

    Parameters
    ----------
    smiles_back : (list)
        List of valid SMILE strings. 
    target_smi : (str)
        Valid SMILES string. 

    Returns
    -------
    smiles_back_scores : (list of floats)
        List of figerprint similarity scores of each smiles in input list. 
    '''
    smiles_back_scores = []
    target    = Chem.MolFromSmiles(target_smi)
    fp_target = AllChem.GetMorganFingerprint(target, 2)
    for item in smiles_back: 
        mol    = Chem.MolFromSmiles(item)
        fp_mol = AllChem.GetMorganFingerprint(mol, 2)
        score  = TanimotoSimilarity(fp_mol, fp_target)
        smiles_back_scores.append(score)
    return smiles_back_scores


def get_good_bad_smiles(fitness, population, generation_size): 
    '''
    Given fitness values of all SMILES in population, and the generation size, 
    this function smplits  the population into two lists: keep_smiles & replace_smiles. 
    
    Parameters
    ----------
    fitness : (list of floats)
        List of floats representing properties for molecules in population.
    population : (list of SMILES)
        List of all SMILES in each generation.
    generation_size : (int)
        Number of molecules in each generation.

    Returns
    -------
    keep_smiles : (list of SMILES)
        A list of SMILES that will be untouched for the next generation. .
    replace_smiles : (list of SMILES)
        A list of SMILES that will be mutated/crossed-oved for forming the subsequent generation.

    '''
    
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
    
    params_ = generate_params()
    start_time = time.time()
    with open(params_['start_population'], 'r') as f: 
        init_smiles = f.readlines()
    

    
    init_smiles = [x.strip() for x in init_smiles]
    fp_scores_guac     = [calc_prop(smi) for smi in init_smiles]
    fp_scores_guac_idx = np.argsort(fp_scores_guac)[::-1]
    print('Initial population obtained!')
    
    initial_mol = [init_smiles[i] for i in fp_scores_guac_idx]
    with open('./RESULTS/' + 'init_mols.txt', 'w') as f: 
        f.writelines([ '{} \n'.format(x) for x in initial_mol])
    
    
    smiles_collector = {} # A tracker for all smiles! 
    generations      = params_['generations']
    generation_size  = params_['generation_size'] 

            
    # population = np.random.choice(initial_mol, size=generation_size).tolist()
    population = initial_mol[0: generation_size]

    
    # Calculate fitness for the initial population: 
    unique_pop     = list(set(population))

    
    prop_map = {}
    for item in unique_pop: 
        prop_map[item] = calc_prop(item)
    
    fitness        = []
    for item in population: 
        fitness.append(prop_map[item])
        
    # Save fitness onto global collector: 
    for item in prop_map: 
        smiles_collector[item] = [prop_map[item], 1] # [Property_value, Count]

    
    # CREATUIB OF FRAGMENTS: 
    if params_['use_fragments'] == True: 
        from fragment_prep import form_fragments
        form_fragments(params_)
        with open('./DATA/fragments_selfies.txt', 'r') as f: 
            alphabet = f.readlines()
        alphabet = [x.strip() for x in alphabet]
        alphabet = [x for x in alphabet if len(x) != 0]
    else: 
        alphabet = []
        
    
    for gen_ in range(generations): 
        
        pop_best_ = []
        
        # STEP 1: OBTAIN THE NEXT GENERATION OF MOLECULES (FOR EXPLORATION): 
        # Order based on fitness, and decide which part of the population is to be kept/discarded: 
        keep_smiles, replace_smiles = get_good_bad_smiles(fitness, population, generation_size)
        replace_smiles = list(set(replace_smiles))

        # Mutations:     
        # mut_smi_dict = get_mutated_smiles(replace_smiles[0: len(replace_smiles)], alphabet=['[C]']) # Half the molecuules are to be mutated     
        
        # Mutations:     
        mut_smi_dict = get_mutated_smiles(replace_smiles[0: len(replace_smiles)//2],  alphabet=alphabet) # Half the molecuules are to be mutated     
        # Crossovers: 
        smiles_join = []
        for item in replace_smiles[len(replace_smiles)//2: ]: 
            smiles_join.append(item + 'xxx' + random.choice(keep_smiles))
        cross_smi_dict =  crossover_smiles_parr(smiles_join)              

        all_mut_smiles = []
        del mut_smi_dict["lock"]
        for key in mut_smi_dict: 
            all_mut_smiles.extend(mut_smi_dict[key])
        all_mut_smiles = list(set(all_mut_smiles))
        all_mut_smiles = [x for x in all_mut_smiles if x != '']
        
        
        all_cros_smiles = []
        del cross_smi_dict["lock"]
        for key in cross_smi_dict: 
            all_cros_smiles.extend(cross_smi_dict[key])
        all_cros_smiles = list(set(all_cros_smiles))
        all_cros_smiles = [x for x in all_cros_smiles if x != '']
        
        all_smiles = list(set(all_mut_smiles + all_cros_smiles))
        
        all_smiles_unique = [x for x in all_smiles if x not in smiles_collector]
        
        # STEP 2: CONDUCT FITNESS CALCULATION FOR THE EXPLORATION MOLECULES: 
        # replaced_pop = random.sample(all_smiles_unique, generation_size-len(keep_smiles) )
        if gen_ == 0: 
            replaced_pop = random.sample(all_smiles_unique, generation_size-len(keep_smiles))
        else: 
            if params_['use_NN_classifier'] == True: 
                # The sampling needs to be done by the neural network! 
                print('    Training Neural Net')
                train_smiles, pro_val = [], []
                for item in smiles_collector: 
                    train_smiles.append(item)
                    pro_val.append(smiles_collector[item][0])
                train_and_save_model(train_smiles, pro_val, generation_index=gen_)
                
                # Obtain predictions on unseen molecules: 
                print('    Obtaining Predictions')
                new_predictions  = obtain_new_pred(all_smiles_unique, generation_index=gen_)
                NN_pred_sort     = np.argsort(new_predictions)[::-1]
                replaced_pop     = [all_smiles_unique[NN_pred_sort[i]] for i in range(generation_size-len(keep_smiles))]
            else: 
                replaced_pop = random.sample(all_smiles_unique, generation_size-len(keep_smiles))
        

        population   = keep_smiles + replaced_pop
        
        unique_pop     = list(set(population))
        prop_map = {}
        for item in unique_pop: 
            prop_map[item] = calc_prop(item)
            
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
        mut_smi_dict_local  = get_mutated_smiles(smiles_local_search, alphabet=alphabet, space='Local')
        mut_smi_dict_local  = mut_smi_dict_local[population[top_idx]]
        mut_smi_dict_local  = [x for x in mut_smi_dict_local if x not in smiles_collector]

        fp_scores          = get_fp_scores(mut_smi_dict_local, population[top_idx])
        fp_sort_idx        = np.argsort(fp_scores)[::-1][: generation_size]
        mut_smi_dict_local_calc = [mut_smi_dict_local[i] for i in fp_sort_idx]
        
        
        # STEP 4: CALCULATE THE FITNESS FOR THE LOCAL SEARCH: 
        prop_map = {}
        for item in mut_smi_dict_local_calc: 
            prop_map[item] = calc_prop(item)
        
        
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
        
        # For the NN! 
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
        num_exchanges     = params_['num_exchanges']
        introduce_smiles  = population_sort[0: num_exchanges] # Taking the top 5 molecules
        introduce_fitness = fitness_sort[0: num_exchanges]    # Taking the top 5 molecules
        
        worst_indices = idx_sort[-num_exchanges: ]
        for i,idx in enumerate(worst_indices): 
            try: 
                population[idx] = introduce_smiles[i]
                fitness[idx]    = introduce_fitness[i]
            except: 
                continue 
        
        # Save best of generation!: 
        fit_all_best = np.argmax(fitness)

            
        with open('./RESULTS' + '/generation_all_best.txt', 'a+') as f: 
            f.writelines(['Gen:{}, {}, {} \n'.format(gen_,  population[fit_all_best], fitness[fit_all_best])])
            




