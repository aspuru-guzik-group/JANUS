# JANUS: Parallel Tempered Genetic Algorithm Guided by Deep Neural Networks for Inverse Molecular Design
This repository contains code for the paper: [JANUS: Parallel Tempered Genetic Algorithm Guided by Deep Neural Networks for Inverse Molecular Design](https://arxiv.org/abs/2106.04011). 

Originally by: AkshatKumar Nigam, Robert Pollice, Alán Aspuru-Guzik 

Updated by: Gary Tom

<img align="center" src="https://github.com/aspuru-guzik-group/JANUS/blob/main/aux_files/logo.png"/>


## Prerequsites: 

Use [Python 3.7 or up](https://www.python.org/download/releases/3.0/).

You will need to separately install [RDKit](https://www.rdkit.org/docs/Install.html) version >= 2020.03.1. The easiest is to do this on conda.

JANUS uses [SELFIES](https://github.com/aspuru-guzik-group/selfies) version 1.0.3. If you want to use a different version, pip install your desired version; this package will still be compatible. Note that you will have to change your input alphabets to work with other versions of SELFIES.


## Major changes:

- Support the use of any version of SELFIES (please check your installation).
- Improved multiprocessing. Fitness function is not parallelized, in the case that the function already spawns multiple processes.
- GPU acceleration of neural networks.
- Early stopping for classifier. 
- Included SMILES filtering option.
- Additional hyperparameters for controlling JANUS. Defaults used in paper are given in `tests` directory.

## How to run: 

Install JANUS using 

```bash
pip install janus-ga
```

Example script of how to use JANUS is found in [tests/example.py](https://github.com/aspuru-guzik-group/JANUS/blob/main/tests/example.py):

```python
from janus import JANUS, utils
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, RDConfig, Descriptors
RDLogger.DisableLog("rdApp.*")

import selfies

def fitness_function(smi: str) -> float:
    """ User-defined function that takes in individual smiles 
    and outputs a fitness value.
    """
    # logP fitness
    return Descriptors.MolLogP(Chem.MolFromSmiles(smi))

def custom_filter(smi: str):
    """ Function that takes in a smile and returns a boolean.
    True indicates the smiles PASSES the filter.
    """
    # smiles length filter
    if len(smi) > 81 or len(smi) == 0:
        return False
    else:
        return True

# all parameters to be set, below are defaults
params_dict = {
    # Number of iterations that JANUS runs for
    "generations": 200,

    # The number of molecules for which fitness calculations are done, 
    # exploration and exploitation each have their own population
    "generation_size": 5000,
    
    # Number of molecules that are exchanged between the exploration and exploitation
    "num_exchanges": 5,

    # Callable filtering function (None defaults to no filtering)
    "custom_filter": custom_filter,

    # Fragments from starting population used to extend alphabet for mutations
    "use_fragments": True,

    # An option to use a classifier as selection bias
    "use_classifier": True,
}

# Set your SELFIES constraints (below used for manuscript)
default_constraints = selfies.get_semantic_constraints()
new_constraints = default_constraints
new_constraints['S'] = 2
new_constraints['P'] = 3
selfies.set_semantic_constraints(new_constraints)  # update constraints

# Create JANUS object.
agent = JANUS(
    work_dir = 'RESULTS',                                   # where the results are saved
    fitness_function = fitness_function,                    # user-defined fitness for given smiles
    start_population = "./DATA/sample_start_smiles.txt",   # file with starting smiles population
    **params_dict
)

# Alternatively, you can get hyperparameters from a yaml file
# Descriptions for all parameters are found in default_params.yml
params_dict = utils.from_yaml(
    work_dir = 'RESULTS',  
    fitness_function = fitness_function, 
    start_population = "./DATA/sample_start_smiles.txt",
    yaml_file = 'default_params.yml',       # default yaml file with parameters
    **params_dict                           # overwrite yaml parameters with dictionary
)
agent = JANUS(**params_dict)

# Run according to parameters
agent.run()     # RUN IT!
```

Within this file are examples for: 
1. A function for calculting property values (see function `fitness_function`). 
2. Custom filtering of SMILES (see function `custom_filter`).
3. Initializing JANUS from dictionary of parameters.
4. Generating hyperparameters from provided yaml file (see function `janus.utils.from_yaml`).


You can run the file with provided test files

```bash
cd tests
python ./example.py
```


Important parameters the user should provide:
- `work_dir`: directory for outputting results
- `fitness_function`: fitness function defined for an input smiles that will be maximized
- `start_population`: path to text file of starting smiles one each new line
- `generations`: number if evolution iterations to perform
- `generation_size`: number of molecules in the populations per generation
- `custom_filter`: filter function checked after mutation and crossover, returns `True` for accepted molecules
- `use_fragments`: toggle adding fragments from starting population to mutation alphabet
- `use_classifier`: toggle using classifier for selection bias

See [tests/default_params.yml](https://github.com/aspuru-guzik-group/JANUS/blob/main/tests/default_params.yml) for detailed description of adjustable parameters.


## Outputs: 

All results from running JANUS will be stored in specified `work_dir`. 

The following files will be created: 
1. fitness_explore.txt: 
   Fitness values for all molecules from the exploration component of JANUS.    
2. fitness_local_search.txt: 
   Fitness values for all molecules from the exploitation component of JANUS. 
3. generation_all_best.txt: 
   Smiles and fitness value for the best molecule encountered in every generation (iteration). 
4. init_mols.txt: 
   List of molecules used to initialte JANUS. 
5. population_explore.txt: 
   SMILES for all molecules from the exploration component of JANUS. 
6. population_local_search.txt: 
   SMILES for all molecules from the exploitation component of JANUS. 
7. hparams.json:
   Hyperparameters used for initializing JANUS.


## Paper Results/Reproducibility: 
Our code and results for each experiment in the paper can be found here: 
* Experiment 4.1: https://drive.google.com/file/d/1rscIyzpTvtyiEkoP1WsF-XtSHJGQStUU/view?usp=sharing
* Experiment 4.3: https://drive.google.com/file/d/1tlIdfSWwzVeJ5kZ98l8G6osE9zf9wP1f/view?usp=sharing
* GuacaMol: https://drive.google.com/file/d/1FqetwNg6VVc-C3eiPoosGZ4-47WpYBAt/view?usp=sharing


## Questions, problems?
Make a github issue 😄. Please be as clear and descriptive as possible. Please feel free to reach
out in person: (akshat[DOT]nigam[AT]mail[DOT]utoronto[DOT]ca, rob[DOT]pollice[AT]utoronto[DOT]ca)

## License

[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)
