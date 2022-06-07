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

RDLogger.DisableLog("rdApp.*")

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


def sanitize_smiles(smi):
    """Return a canonical smile representation of smi
    Parameters:
    smi (string) : smile string to be canonicalized 
    Returns:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object                          (None if invalid smile string smi)
    smi_canon (string)          : Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): True/False to indicate if conversion was  successful 
    """
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        return (mol, smi_canon, True)
    except:
        return (None, None, False)


def get_frags(smi, radius):
    ''' Create fragments from smi with some radius. Remove duplicates and any
    fragments that are blank molecules.
    '''
    mol = smi2mol(smi, sanitize=True)
    frags = []
    for ai in range(mol.GetNumAtoms()):
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, ai)
        amap = {}
        submol = Chem.PathToSubmol(mol, env, atomMap=amap)
        frag = mol2smi(submol, isomericSmiles=False, canonical=True)
        frags.append(frag)
    return list(filter(None, list(set(frags))))

def form_fragments(smi):
    ''' Create fragments of certain radius. Returns a list of fragments
    using SELFIES characters.
    '''
    selfies_frags = []
    unique_frags = get_frags(smi, radius=3)
    for item in unique_frags:
        sf = encoder(item)
        if sf is None:
            continue
        dec_ = decoder(sf)

        try:
            m = Chem.MolFromSmiles(dec_)
            Chem.Kekulize(m)
            dearom_smiles = Chem.MolToSmiles(
                m, canonical=False, isomericSmiles=False, kekuleSmiles=True
            )
            dearom_mol = Chem.MolFromSmiles(dearom_smiles)
        except:
            continue

        if dearom_mol == None:
            raise Exception("mol dearom failes")

        selfies_frags.append(encoder(dearom_smiles))

    return selfies_frags

    