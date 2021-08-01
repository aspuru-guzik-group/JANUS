#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 14:46:02 2021

@author: akshat
"""


def generate_params(): 
    
    params_ = {}
    params_['generations'] = 200
    params_['generation_size'] = 5000
    params_['start_population'] = './DATA/sample_start_smiles.txt'

    
    
    
    return params_






def calc_prop(smi): 
    
    return 1