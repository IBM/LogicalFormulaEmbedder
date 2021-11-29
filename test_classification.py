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
import process_sentences as ps
import code.dataset as dt
import train_model as tm

models_path = os.path.join('.', 'models')
results_path = os.path.join('.', 'results')
batch_size = 16
num_workers = 8

dataset_params = { 'batch_size' : batch_size,
                   'shuffle' : True,
                   'num_workers' : num_workers }

if __name__ == '__main__':

    parser = ap.ArgumentParser(description='Test formula classifier module')
    parser.add_argument('--dataset', help='Dataset to test model on')
    parser.add_argument('--model', help='Model name in models directory')
    parser.add_argument('--validation', help='Test validation or not')
    parser.add_argument('--type_smoothing', help='Test with type smoothing')
    parser.add_argument('--edge_type', help='Edge labeling strategy')
    args = parser.parse_args()


    use_edge_type = tm.model_params['edge_type']
    if args.edge_type:
        valid_edge_types = ['typed_ord', 'untyped_ord', 'typed_unord', 'untyped_unord', 'None']
        assert args.edge_type in valid_edge_types, 'Embedding direction must be one of: ' + \
            ', '.join(valid_edge_types)
        use_edge_type = None if args.edge_type == 'None' else args.edge_type

    dataset = args.dataset
    assert dataset in ['holstep', 'mizar', 'mizar_cnf', 'scitail'], 'Unknown dataset...'

    model_filename = args.model
    assert model_filename, 'Model specification required...'

    model = pkl.load(open(os.path.join(models_path, model_filename), 'rb'))

    if args.type_smoothing:
        assert args.type_smoothing in ['True', 'False'], 'Type smoothing must be boolean'
        model.type_smoothing = args.type_smoothing == 'True'
    
    if args.validation:
        assert args.validation in ['True', 'False'], '--validation must be boolean value'
        use_val = args.validation == 'True'
    else:
        use_val = False

    if dataset in ['scitail', 'holstep']:
        if dataset == 'holstep':
            te_dobj = pd.hol_val_dobj_loc if use_val else pd.hol_te_dobj_loc
        elif dataset == 'scitail':
            te_dobj = pd.sci_val_dobj_loc if use_val else pd.sci_te_dobj_loc
        collator = dt.HolstepCollator(model.model_params['depth_cap'],
                                      model.default_pc, 
                                      use_edge_type)
    elif dataset in ['mizar', 'mizar_cnf']:
        if dataset == 'mizar':
            te_dobj = pd.miz_val_dobj_loc if use_val else pd.miz_te_dobj_loc
        else:
            te_dobj = pd.miz_cnf_val_dobj_loc if use_val else pd.miz_cnf_te_dobj_loc
        collator = dt.MizarCollator(model.model_params['depth_cap'],
                                    model.default_pc, 
                                    use_edge_type)
        
    # get test data
    testing_set = pkl.load(open(te_dobj, 'rb'))
    testing_generator = data.DataLoader(testing_set, collate_fn=collator,
                                        **dataset_params)
    
    # test model
    gap_filler = '\n' + '='.join(['' for _ in range(30)]) + '\n'
    
    print(gap_filler)
    print('Starting testing...')
    print(gap_filler)

    te_batch_ct, te_total_ct, corr_ct = 0, 0, 0

    model.eval()
    for batch in testing_generator:
        with torch.no_grad():
            te_batch_ct += len(batch)
            outputs, labels, parse_failures = model.run_classifier(batch)
        for output, label in zip(outputs, labels):
            pred = 1 if output >= 0.5 else 0
            if pred == label: corr_ct += 1
            te_total_ct += 1
        te_total_ct += len(parse_failures)
        print('Testing completion: ' + \
              str(round(te_batch_ct / len(testing_set.data_ids) * 100, 2))+'%')
        print('Current accuracy: ' + str(corr_ct / te_total_ct))
        print()
        
    test_performance = corr_ct / te_total_ct

    print(gap_filler)
    print(test_performance)
    print(gap_filler)

    results_file = model_filename + '_test_results.csv'
    with open(os.path.join(results_path, results_file), 'a') as f:
        f.write(','.join([str(corr_ct), str(te_total_ct), 
                          str(te_total_ct), str(test_performance)]) + '\n')
