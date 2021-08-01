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


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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



def form_fragments(params_): 
    print('Forming Fragments! ')
    with open(params_['start_population'], 'r') as f: 
        init_smiles = f.readlines()
    init_smiles = [x.strip() for x in init_smiles]
    
    
    for i, smi in enumerate(init_smiles): 
        
        if i%1000 == 0: 
            print('    Fragment creation: {}/{}'.format(i, len(init_smiles)))
        unique_frags = get_frags(smi, radius=3)
        
        for item in unique_frags: 
            try: 
                sf = encoder(item)
                dec_ = decoder(sf)
                
                m = Chem.MolFromSmiles(dec_)
                Chem.Kekulize(m)
                dearom_smiles = Chem.MolToSmiles(m, canonical=False, isomericSmiles=False, kekuleSmiles=True)
                dearom_mol = Chem.MolFromSmiles(dearom_smiles)
                if dearom_mol == None: 
                    raise Exception('mol dearom failes')
                            
                with open('./DATA/fragments_selfies.txt', 'a+') as f: 
                    f.writelines(['{}\n'.format(encoder(dearom_smiles))])
            except: 
                continue
            
