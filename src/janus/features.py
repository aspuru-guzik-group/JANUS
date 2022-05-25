import numpy as np
import inspect
import rdkit
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
from collections import OrderedDict
RDLogger.DisableLog('rdApp.*')

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
    ring_smiles = []
    for item in smile_split_list:
        if '1' in item and Chem.MolFromSmiles(item) is not None:
            ring_smiles.append(item)
    
    return ring_smiles 


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
    
    # Count size of rings
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
    
    to_calculate = ["RingCount", "HallKierAlpha", "BalabanJ", "NumAliphaticCarbocycles","NumAliphaticHeterocycles",
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
        if len(to_calculate)!=0 and key not in to_calculate:
            del calc_props[key]
    # features = [val(mol) for key,val in calc_props.items()] # List of properties 

    features = []
    for key, val in calc_props.items():
        try:
            features.append(val(mol))
        except:
            print(f'Failed at: {val}')
            print(f'Failed at: {Chem.MolToSmiles(mol)}')
    
    
    # Ratio of total number of  (single, double, triple, aromatic) bonds to the total number of bonds. 
    simple_bond_info = get_num_bond_types(mol) 
    
    # Obtain all rings in a molecule and calc. #of triple bonds in rings & #of rings in molecule 
    ring_ls = obtain_rings(smi)
    num_triple = 0      # num triple bonds in ring
    
    if len(ring_ls) > 0 and ring_ls != (None, None): 
        for item in ring_ls:
            num_triple += item.count('#')
        simple_bond_info.append(len(ring_ls))     # append number of Rings in molecule 
    else:   
        simple_bond_info.append(0)                # no rotatable bonds

    simple_bond_info.append(num_triple)          # number of triple bonds in rings

                                              
    # Calculate the number of rings of size 3 to 20 & number of conseq. double bonds in rings 
    simple_bond_info = simple_bond_info + size_ring_counter(ring_ls)
    
    # Calculate the number of consequitve double bonds in entire molecule
    simple_bond_info.append(count_conseq_double(mol)) 
    
    return np.array(features + basic_props + simple_bond_info)