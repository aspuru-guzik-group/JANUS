import selfies as sf
import yaml
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity

def get_selfies_chars(selfies):
    """Obtain a list of all selfie characters in string selfies
    
    Parameters: 
    selfie (string) : A selfie string - representing a molecule 
    
    Example: 
    >>> get_selfies_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
    ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']
    
    Returns
    -------
    chars_selfies (list of strings) : 
        list of selfie characters present in molecule selfie
    """
    chars_selfies = sf.split_selfies(selfies)
    return list(chars_selfies)

def sanitize_smiles(smi):
    """
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
    """
    if smi == '':
        return None
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        return smi_canon
    except:
        return None

def get_fp_scores(smiles_back, target_smi):
    """
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
        List of fingerprint similarity scores of each smiles in input list. 
    """
    smiles_back_scores = []
    target = smi2mol(target_smi)
    fp_target = AllChem.GetMorganFingerprint(target, 2)
    for item in smiles_back:
        mol = smi2mol(item)
        fp_mol = AllChem.GetMorganFingerprint(mol, 2)
        score = TanimotoSimilarity(fp_mol, fp_target)
        smiles_back_scores.append(score)
    return smiles_back_scores

def from_yaml(work_dir, 
        fitness_function, 
        start_population,
        yaml_file, **kwargs):

    # create dictionary with parameters defined by yaml file 
    with open(yaml_file, 'r') as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    params.update(kwargs)
    params.update({
        'work_dir': work_dir,
        'fitness_function': fitness_function,
        'start_population': start_population
    })

    return params
