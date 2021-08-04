#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 12:15:57 2021

@author: akshat
"""
from __future__ import print_function
import rdkit
import random
import multiprocessing
from rdkit import Chem
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from selfies import encoder, decoder 

manager = multiprocessing.Manager()
lock = multiprocessing.Lock()

def get_selfie_chars(selfie):
    '''Obtain a list of all selfie characters in string selfie
    
    Parameters: 
    selfie (string) : A selfie string - representing a molecule 
    
    Example: 
    >>> get_selfie_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
    ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']
    
    Returns
    -------
    chars_selfie (list of strings) : 
        list of selfie characters present in molecule selfie
    '''
    chars_selfie = [] # A list of all SELFIE sybols from string selfie
    while selfie != '':
        chars_selfie.append(selfie[selfie.find('['): selfie.find(']')+1])
        selfie = selfie[selfie.find(']')+1:]
    return chars_selfie

def mutate_sf(sf_chars, alphabet): 
    '''
    Given a list of SELFIES alphabets, make random changes to the molecule using 
    alphabet. Opertations to molecules are character replacements, additions and deletions. 

    Parameters
    ----------
    sf_chars : (list of string alphabets)
        List of string alphabets for a SELFIE string.
    alphabet : (list of SELFIE strings)
        Replacements and addition operations are performed using this list of SELFIE strings.

    Returns
    -------
    Muatted SELFIE string.

    '''
    random_char_idx = random.choice(range(len(sf_chars)))
    choices_ls = [1, 2, 3] # 1 = replacement; 2 = addition; 3=delete
    mutn_choice = choices_ls[random.choice(range(len(choices_ls)))] # Which mutation to do: 
        
    if alphabet != []: 
        alphabet = random.sample(alphabet, 200) + ['[=N]', '[C]', '[S]','[Branch3_1]','[Expl=Ring3]','[Branch1_1]','[Branch2_2]','[Ring1]', '[#P]','[O]', '[Branch2_1]', '[N]','[=O]','[P]','[Expl=Ring1]','[Branch3_2]','[I]', '[Expl=Ring2]', '[=P]','[Branch1_3]','[#C]','[Cl]', '[=C]','[=S]','[Branch1_2]','[#N]','[Branch2_3]','[Br]','[Branch3_3]','[Ring3]','[Ring2]','[F]']
    else: 
        alphabet = ['[=N]', '[C]', '[S]','[Branch3_1]','[Expl=Ring3]','[Branch1_1]','[Branch2_2]','[Ring1]', '[#P]','[O]', '[Branch2_1]', '[N]','[=O]','[P]','[Expl=Ring1]','[Branch3_2]','[I]', '[Expl=Ring2]', '[=P]','[Branch1_3]','[#C]','[Cl]', '[=C]','[=S]','[Branch1_2]','[#N]','[Branch2_3]','[Br]','[Branch3_3]','[Ring3]','[Ring2]','[F]']
   
    # Mutate character: 
    if mutn_choice == 1: 
        random_char = alphabet[random.choice(range(len(alphabet)))]
        change_sf  = sf_chars[0:random_char_idx] + [random_char] + sf_chars[random_char_idx+1: ]
        
    # add character: 
    elif mutn_choice == 2: 
        random_char = alphabet[random.choice(range(len(alphabet)))]
        change_sf  = sf_chars[0:random_char_idx] + [random_char] + sf_chars[random_char_idx: ]
        
    # delete character: 
    elif mutn_choice == 3: 
        if len(sf_chars) != 1: 
            change_sf  = sf_chars[0:random_char_idx] + sf_chars[random_char_idx+1: ]
        else: 
            change_sf = sf_chars
    
    return ''.join(x for x in change_sf)


def get_prop_material(smile, alphabet, num_random_samples, num_mutations): 
    '''
    Given an input smile, perform mutations to the strucutre using provided SELFIE
    alphabet list. 'num_random_samples' number of different SMILES orientations are 
    considered & total 'num_mutations' are performed. 

    Parameters
    ----------
    smile : (str)
        Valid SMILES string.
    alphabet : (list of str)
        list of SELFIE strings.
    num_random_samples : (int)
        Number of different SMILES orientations to be formed for the input smile.
    num_mutations : TYPE
        Number of mutations to perform on each of different orientations SMILES.

    Returns
    -------
    mutated_smiles_canon : (list of strings)
        List of unique molecules produced from mutations.
    '''
    mol = Chem.MolFromSmiles(smile)
    Chem.Kekulize(mol)
    
    # Obtain randomized orderings of the SMILES: 
    randomized_smile_orderings = []
    for _ in range(num_random_samples): 
        randomized_smile_orderings.append(rdkit.Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False,  kekuleSmiles=True))
    
    # Convert all the molecules to SELFIES
    selfies_ls = [encoder(x) for x in randomized_smile_orderings]
    selfies_ls_chars = [get_selfie_chars(selfie) for selfie in selfies_ls]
    
    # Obtain the mutated selfies
    mutated_sf    = []
    for sf_chars in selfies_ls_chars: 
        
        for i in range(num_mutations): 
            if i == 0:  mutated_sf.append(mutate_sf(sf_chars, alphabet))
            else:       mutated_sf.append(mutate_sf ( get_selfie_chars(mutated_sf[-1]), alphabet ))
            
    mutated_smiles = [decoder(x) for x in mutated_sf]    
    mutated_smiles_canon = []
    for item in mutated_smiles: 
        try: 
            smi_canon = Chem.MolToSmiles(Chem.MolFromSmiles(item, sanitize=True), isomericSmiles=False, canonical=True)
            if len(smi_canon) <= 81: # Size restriction! 
                mutated_smiles_canon.append(smi_canon)
        except: 
            continue
        
    mutated_smiles_canon = list(set(mutated_smiles_canon))
    return mutated_smiles_canon    

    
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
    
    
def get_chunks(arr, num_processors, ratio):
    '''
    Split list of SMILES int sublists, each of which will be operated on seperate cpus. 

    Parameters
    ----------
    arr : (list of sts)
        A list of SMILES.
    num_processors : (int)
        Number of cpus available for conducting operation.
    ratio : (int)
        number of operations that will be performed on each cpu.

    Returns
    -------
    chunks: (list of lists)
        Each sublist is used by a different cpu to perform operations. 
    '''
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


def calc_parr_prop(unseen_smile_ls, property_name, props_collect, num_random_samples, num_mutations, alphabet):
    '''Calculate logP for each molecule in unseen_smile_ls, and record results
       in locked dictionary props_collect 
    '''
    for smile in unseen_smile_ls: 
        props_collect[property_name][smile] = get_prop_material(smile, alphabet=alphabet, num_random_samples=num_random_samples, num_mutations=num_mutations)  # TODO: TESTING


def create_parr_process(chunks, alphabet, property_name, num_random_samples, num_mutations):
    '''
    Create parallel processes for creating mutating molecules for molecule in sublist chunks. 

    Parameters
    ----------
    chunks : (list of list)
        list of lists containing SMILES.
    property_name : (syr)
        optional name paramtere to enter.

    Returns
    -------
    combined_dict : (dict)
        input smiles -> [List of mutated smiles].

    '''
    process_collector    = []
    collect_dictionaries = []
        
    for item in chunks:
        props_collect  = manager.dict(lock=True)
        smiles_map_    = manager.dict(lock=True)
        props_collect[property_name] = smiles_map_
        collect_dictionaries.append(props_collect)
        
        if property_name == 'logP':
            process_collector.append(multiprocessing.Process(target=calc_parr_prop, args=(item, property_name, props_collect, num_random_samples, num_mutations, alphabet, )))   
    
    for item in process_collector:
        item.start()
    
    for item in process_collector: # wait for all parallel processes to finish
        item.join()   
        
    combined_dict = {}             # collect results from multiple processess
    for i,item in enumerate(collect_dictionaries):
        combined_dict.update(item[property_name])

    return combined_dict


def get_mutated_smiles(smiles, alphabet, space='Explore'): 

    num_processors        = multiprocessing.cpu_count()
    molecules_here_unique = list(set(smiles)) 
    
    ratio            = len(molecules_here_unique) / num_processors 
    chunks           = get_chunks(molecules_here_unique, num_processors, ratio) 
        
    if space == 'Explore': 
        mut_smiles = create_parr_process(chunks, alphabet, 'logP', num_random_samples=5, num_mutations=5)
    else: 
        mut_smiles = create_parr_process(chunks, alphabet, 'logP', num_random_samples=400, num_mutations=400)

    return mut_smiles
    

    
    
if __name__ == '__main__': 
    molecules_here        = ['CCC', 'CCCC', 'CCCCC', 'CCCCCCCC', 'CS', 'CSSS', 'CSSSSS', 'CF', 'CI', 'CBr', 'CSSSSSSSSSSSS', 'CSSSSSSSSSC', 'CSSSSCCSSSC', 'CSSSSSSSSSF', 'SSSSSC']
    A = get_mutated_smiles(molecules_here, alphabet=['[C]']*500, space='Explore')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    