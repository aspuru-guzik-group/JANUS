import random
import rdkit
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import pandas as pd
import dask.dataframe as dd
import selfies
from selfies import encoder, decoder
import multiprocessing

# Updated SELFIES constraints: 
default_constraints = selfies.get_semantic_constraints()
new_constraints = default_constraints
# new_constraints['S'] = 2
new_constraints['P'] = 3
selfies.set_semantic_constraints(new_constraints)  # update constraints


def get_selfie_chars(selfie):
    '''Obtain a list of all selfie characters in string selfie
    
    Parameters: 
    selfie (string) : A selfie string - representing a molecule 
    
    Example: 
    >>> get_selfie_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
    ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']
    
    Returns:
    chars_selfie: list of selfie characters present in molecule selfie
    '''
    chars_selfie = [] # A list of all SELFIE sybols from string selfie
    while selfie != '':
        chars_selfie.append(selfie[selfie.find('['): selfie.find(']')+1])
        selfie = selfie[selfie.find(']')+1:]
    return chars_selfie


def mutate_sf(sf_chars, alphabet): 
    '''
    Provided a list of SELFIE characters, this function will return a modified 
    SELFIES. 
    '''
    random_char_idx = random.choice(range(len(sf_chars)))
    choices_ls = [1, 2, 3] # TODO: 1 = mutate; 2 = addition; 3=delete
    mutn_choice = choices_ls[random.choice(range(len(choices_ls)))] # Which mutation to do: 
        
    # alphabet = random.sample(alphabet, 200) + ['[=N]', '[C]', '[S]','[Branch3_1]','[Expl=Ring3]','[Branch1_1]','[Branch2_2]','[Ring1]', '[#P]','[O]', '[Branch2_1]', '[N]','[=O]','[P]','[Expl=Ring1]','[Branch3_2]','[I]', '[Expl=Ring2]', '[=P]','[Branch1_3]','[#C]','[Cl]', '[=C]','[=S]','[Branch1_2]','[#N]','[Branch2_3]','[Br]','[Branch3_3]','[Ring3]','[Ring2]','[F]']
    alphabet = ['[=N]', '[C]', '[S]','[Branch3_1]','[Expl=Ring3]','[Branch1_1]','[Branch2_2]','[Ring1]', '[#P]','[O]', '[Branch2_1]', '[N]','[=O]','[P]','[Expl=Ring1]','[Branch3_2]','[I]', '[Expl=Ring2]', '[=P]','[Branch1_3]','[#C]','[Cl]', '[=C]','[=S]','[Branch1_2]','[#N]','[Branch2_3]','[Br]','[Branch3_3]','[Ring3]','[Ring2]','[F]']
    # alphabet = ['[=N]', '[C]', '[S]','[Branch3_1]','[Expl=Ring3]','[Branch1_1]','[Branch2_2]','[Ring1]', '[#P]','[O]', '[Branch2_1]', '[N]','[=O]','[P]','[Expl=Ring1]','[Branch3_2]','[I]', '[Expl=Ring2]', '[=P]','[Branch1_3]','[#C]','[Cl]', '[=C]','[=S]','[Branch1_2]','[#N]','[Branch2_3]','[Br]','[Branch3_3]','[Ring3]','[Ring2]','[F]']
    
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
    
    # print('        Mutation successfully performed! ')
    return ''.join(x for x in change_sf)



def get_mutated_smile(smiles, alphabet, num_random_samples, num_mutations): 
    
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol)
    
    # Obtain randomized orderings of the SMILES: 
    randomized_smile_orderings = []
    for _ in range(num_random_samples): 
        randomized_smile_orderings.append(rdkit.Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False,  kekuleSmiles=True))
    # print('    Randomized SMILES: ', randomized_smile_orderings)
    
    # Convert all the molecules to SELFIES
    selfies_ls = [encoder(x) for x in randomized_smile_orderings]
    selfies_ls_chars = [get_selfie_chars(selfie) for selfie in selfies_ls]
    # print('    SELFIE CHARS obtained! ')
    
    # Obtain the mutated selfies
    mutated_sf    = []
    
    for sf_chars in selfies_ls_chars: 
        
        for i in range(num_mutations): 
            if i == 0:  mutated_sf.append(mutate_sf(sf_chars, alphabet))
            else:       mutated_sf.append(mutate_sf ( get_selfie_chars(mutated_sf[-1]), alphabet ))
            # print('    A mutation has been performed!')
            
    mutated_smiles = [decoder(x) for x in mutated_sf]
    # print('    Mutated SMIELS are: ', mutated_smiles)
    
    mutated_smiles_canon = []
    for item in mutated_smiles: 
        try: 
            smi_canon = Chem.MolToSmiles(Chem.MolFromSmiles(item, sanitize=True), isomericSmiles=False, canonical=True)
            if len(smi_canon) <= 81: # Size restriction! 
                mutated_smiles_canon.append(smi_canon)
        except: 
            # print('Failed to canonicalize!!: ', item)
            continue
    # print('Obtained Canonical smiles! ')
        
    mutated_smiles_canon = list(set(mutated_smiles_canon))
    return mutated_smiles_canon



    
def get_mutated_smiles(smiles, alphabet, space='Explore'): 
    
    map_ = {}
    for i,smi in enumerate(smiles):
        if i % 1000 == 0: 
            print('        Mutate: {}/{}'.format(i, len(smiles)))
        if space == 'Explore': 
            mut_smiles = get_mutated_smile(smi, alphabet, num_random_samples=10, num_mutations=1)
        else: 
            mut_smiles = get_mutated_smile(smi, alphabet, num_random_samples=400, num_mutations=400)
            
        # print('Collected {} smiles'.format(len(mut_smiles)))
        mut_smiles = [x for x in mut_smiles if x != '']
        map_[smi] = mut_smiles

    return map_

# with open('./DATA/fragments_selfies.txt', 'r') as f: 
#     alphabet = f.readlines()
# alphabet = [x.strip() for x in alphabet]
# print('Mutation alphabets obtained! ')
    
# A = get_mutated_smiles(['C'], alphabet=alphabet)


