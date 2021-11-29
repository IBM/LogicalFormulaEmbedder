# python imports
import signal, time, os, sys
import pickle as pkl
import argparse as ap
# numpy imports
import numpy as np
# torch imports
import torch
import torch.autograd as ta
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import data
# code imports
import process_data as pd
import code.dataset as dt

models_path = os.path.join('.', 'models')
results_path = os.path.join('.', 'results')
batch_size = 1
num_workers = 0

dataset_params = { 'batch_size' : batch_size,
                   'shuffle' : True,
                   'num_workers' : num_workers }

if __name__ == '__main__':

    parser = ap.ArgumentParser(description='Test formula classifier module')
    parser.add_argument('--dataset', help='Dataset to test model on')
    parser.add_argument('--model', help='Model name in models directory')
    args = parser.parse_args()

    dataset = args.dataset
    assert dataset in ['holstep', 'mizar'], 'Unknown dataset...'

    model_filename = args.model
    assert model_filename, 'Model specification required...'

    model = pkl.load(open(os.path.join(models_path, model_filename), 'rb'))
    model.eval()

    if dataset == 'holstep':
        te_dobj = pd.hol_val_dobj_loc
        collator = dt.HolstepCollator(model.model_params['depth_cap'],
                                      model.default_pc)
    elif dataset == 'mizar':
        te_dobj = pd.miz_val_dobj_loc
        collator = dt.MizarCollator(model.model_params['depth_cap'],
                                    model.default_pc)

    # get test data
    testing_set = pkl.load(open(te_dobj, 'rb'))
    testing_generator = data.DataLoader(testing_set, collate_fn=collator,
                                        **dataset_params)

    # running analysis on dep-gate
    prem_gate_vals, conj_gate_vals = [], []
    for t_a in sorted(model.type_assignments.keys(), key=lambda x : str(x)):
        t_ind = model.type_assignments[t_a]
        pool_module = model.formula_pair_embedder
        print('Premise gate for ' + str(t_a) + '...')
        t_t = torch.tensor(t_ind, device=model.device)
        prem_g = float(torch.mean(pool_module.prem_gate(t_t)))
        prem_gate_vals.append(prem_g)
        #print(pool_module.prem_gate(e_t))
        print('Average gate value: ' + str(prem_g))
        print()
        print('Conjecture gate for ' + str(t_a) + '...')
        t_t = torch.tensor(t_ind, device=model.device)
        conj_g = float(torch.mean(pool_module.conj_gate(t_t)))
        conj_gate_vals.append(conj_g)
        #print(pool_module.conj_gate(e_t))
        print('Average gate value: ' + str(conj_g))
        print()
        print('Number type uses: ' + str(model.type_uses[t_a]))
        print()
        print('===========\n')

    print('Average premise gate value: ' + str(np.mean(prem_gate_vals)))
    print('Average conjecture gate value: ' + str(np.mean(conj_gate_vals)))
    print()

    input('Gate analysis complete, press \'Enter\' to continue...')

    print('\nAnalyzing attention weights...\n')
    
    # run analysis on attention mechanism here...
    for batch in testing_generator:
        with torch.no_grad():
            outputs, labels, parse_failures = model.run_classifier(batch)
        input('HERE')
