# JANUS: Parallel Tempered Genetic Algorithm Guided by Deep Neural Networks for Inverse Molecular Design
This repository contains code for the paper: [JANUS: Parallel Tempered Genetic Algorithm Guided by Deep Neural Networks for Inverse Molecular Design](https://arxiv.org/abs/2106.04011). 
By: AkshatKumar Nigam, Robert Pollice, Alán Aspuru-Guzik 

<img align="center" src="./aux/logo.png"/>

## Package Requirements: 
- [SELFIES](https://github.com/aspuru-guzik-group/selfies)
- [RDKit](https://www.rdkit.org/docs/Install.html)
- [Pytorch](https://pytorch.org/)
- [Python 3.0 or up](https://www.python.org/download/releases/3.0/)
- [numpy](https://pypi.org/project/numpy/)

## Using The Code: 
The code can be run using: 
```
python ./JANUS.py
```  
Within params_init.py, a user has the option to provide: 
1. A function for calculting property values (see function calc_prop). 
2. Input parameters that are to be used by JANUS (see function generate_params). Initial parameters are provided. These are picked based on prior 
   experience by the authors of the paper.

## Output Generation: 
All results from running JANUS will be stored here. 
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


## Paper Results/Reproducibility: 
Our code and results for each experiment in the paper can be found here: 
* Experiment 4.1: https://drive.google.com/file/d/1rscIyzpTvtyiEkoP1WsF-XtSHJGQStUU/view?usp=sharing
* Experiment 4.3: https://drive.google.com/file/d/1tlIdfSWwzVeJ5kZ98l8G6osE9zf9wP1f/view?usp=sharing


## Questions, problems?
Make a github issue 😄. Please be as clear and descriptive as possible. Please feel free to reach
out in person: (akshat[DOT]nigam[AT]mail[DOT]utoronto[DOT]ca, rob[DOT]pollice[AT]utoronto[DOT]ca)

## License

[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)
