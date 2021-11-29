# python imports
import signal, time, os, sys, random
import pickle as pkl
import argparse as ap
from itertools import chain
from datetime import date
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
import code.embedding_modules as em
import code.pooling_modules as pm
import code.classifier_modules as cm

models_path = os.path.join('.', 'models')
results_path = os.path.join('.', 'results')
batch_size = 32
num_workers = 16
max_epochs = None

dataset_params = { 'batch_size' : batch_size,
                   'shuffle' : True,
                   'num_workers' : num_workers }

model_params = { 'device' : torch.device('cpu'),
                 'sparse_grads' : True,
                 'label_hashing' : False,
                 'type_smoothing' : False,
                 'mask_rate' : 0.01,
                 'default_pc' : True,
                 'lr_decay' : False,
                 'att_aggr' : True,
                 'mha_heads' : 2,
                 'edge_type' : None,
                 'dep_gate' : True,
                 # learning rate
                 'lr' : 0.001,
                 'depth_cap' : None,
                 'aggr_type' : 'sum',
                 'dep_match_type' : 'label',
                 'dep_depth' : 0,
                 'dropout' : None,#0.2,
                 # node info
                 'node_ct' : 30000,
                 'node_emb_dim' : None,
                 'node_state_dim' : None,
                 'lstm_state_dim' : None,
                 'pretrained_emb_dim' : 50,
                 # edge info
                 'edge_ct' : 150,
                 'edge_emb_dim' : None,
                 'edge_state_dim' : None,
                 # module used to get initial node embeddings
                 'init_node_embedder' : em.MPNN,
                 # if init node embedder is DagLSTM
                 'init_node_embedder_acc_dir' : em.ACCNN.down_acc,
                 # if init node embedder is GCN or MPNN
                 'num_rounds' : 0,
                 # pooling type used for graph embedding
                 'pooling_module' : pm.SimpleMaxPool,
                 # to use leaf or root pooling
                 'pooling_dir' : em.ACCNN.up_acc }

if __name__ == '__main__':

    parser = ap.ArgumentParser(description='Train formula classifier module')
    parser.add_argument('--dataset', help='Dataset to train model for')
    parser.add_argument('--device', help='Device to use')
    parser.add_argument('--node_embedder', help='Node embedding module')
    parser.add_argument('--pool_type', help='Graph pooling type')
    parser.add_argument('--acc_dir', help='Node embedder update direction (if applicable, e.g., for DAG LSTM).')
    parser.add_argument('--pool_dir', help='Pooling direction (if applicable, e.g., for DAG LSTM)')
    parser.add_argument('--depth_cap', type=int, help='Maximum depth of formulas')
    parser.add_argument('--num_epochs', type=int, help='Epochs over training data')
    parser.add_argument('--num_rounds', type=int,
                        help='Number of rounds of updates (if applicable, e.g., for MPNN or GCN)')
    parser.add_argument('--sparse_grads', help='Use sparse grads')
    parser.add_argument('--type_smoothing', 
                        help='Use type smoothing for unknown symbols')
    parser.add_argument('--default_pc', help='Dataset default training method')
    parser.add_argument('--aggr_type', help='Aggregation method')
    parser.add_argument('--save_model', help='Save model or not')
    parser.add_argument('--edge_type', help='Edge labeling strategy')
    parser.add_argument('--val_is_test', help='Use test set as validation')
    parser.add_argument('--small_network', help='Halve all dimensionalities')
    parser.add_argument('--dep_match_type', help='Dependent embedding match type')
    parser.add_argument('--dep_depth', help='Dependent embedding match depth', type=int)
    parser.add_argument('--batch_size', help='Batch size', type=int)
    parser.add_argument('--mha_heads', help='Number of multi-headed attention heads', type=int)

    args = parser.parse_args()

    val_is_test = False
    if args.val_is_test:
        assert args.val_is_test in ['True', 'False'], 'Validation as test should be True or False'
        val_is_test = args.val_is_test == 'True'

    if args.dep_match_type:
        v_m_types = ['label', 'alpha', 'all', 'type', 'leaf', 'iso',
                     'depth', 'depth_typed', 'leaf_label']
        assert args.dep_match_type in v_m_types, 'Dependent match type must be one of ' + ', '.join(v_m_types)
        model_params['dep_match_type'] = args.dep_match_type

    if args.dep_depth is not None:
        assert args.dep_depth >= 0, 'Non-negative values for dep_depth required'
        model_params['dep_depth'] = args.dep_depth

    if args.mha_heads is not None:
        assert args.mha_heads >= 1, 'Positive values for mha_heads required'
        model_params['mha_heads'] = args.mha_heads

    small_network = False
    if args.small_network:
        assert args.small_network in ['True', 'False'], 'Smallify network should be True or False'
        small_network = args.small_network == 'True'

    save_model = True
    if args.save_model:
        assert args.save_model in ['True', 'False'], 'Save model should be either True or False'
        save_model = args.save_model == 'True'
    
    if args.acc_dir:
        assert args.acc_dir in em.ACCNN.acc_dirs, 'Embedding direction must be one of: ' + \
            ', '.join(em.ACCNN.acc_dirs)
        model_params['init_node_embedder_acc_dir'] = args.acc_dir

    if args.edge_type:
        valid_edge_types = ['typed_ord', 'untyped_ord', 'typed_unord', 'untyped_unord', 'None']
        assert args.edge_type in valid_edge_types, 'Embedding direction must be one of: ' + \
            ', '.join(valid_edge_types)
        model_params['edge_type'] = None if args.edge_type == 'None' else args.edge_type

    if args.pool_dir:
        good_pool_dirs = [em.ACCNN.up_acc, em.ACCNN.down_acc]
        assert args.pool_dir in good_pool_dirs, \
            'Pooling direction must be one of: ' + ', '.join(good_pool_dirs)
        model_params['pooling_dir'] = args.pool_dir
        
    if args.aggr_type:
        good_aggr_types = ['mean', 'sum']
        assert args.aggr_type in good_aggr_types, \
            'Aggregation method must be one of: ' + ', '.join(good_aggr_types)
        model_params['aggr_type'] = args.aggr_type
        
    if args.sparse_grads:
        assert args.sparse_grads in ['True', 'False'], \
            'Sparse grads either True or False...'
        model_params['sparse_grads'] = args.sparse_grads == 'True'

    if args.type_smoothing:
        assert args.type_smoothing in ['True', 'False'], \
            'Type smoothing either True or False...'
        model_params['type_smoothing'] = args.type_smoothing == 'True'

    if args.device:
        model_params['device'] = torch.device(args.device)

    if args.pool_type:
        avail_pts = ['MaxPool', 'DepDagLSTM', 'DagLSTM']
        assert args.pool_type in avail_pts, \
            'Node pooling module not recognized. Option must be one of ' + \
            ', '.join(avail_pts)
        p_d = { 'MaxPool' : pm.SimpleMaxPool, 'DepDagLSTM' : pm.DepDagLSTMPool, 
                'DagLSTM' : pm.DagLSTMPool }
        model_params['pooling_module'] = p_d[args.pool_type]

    if args.node_embedder:
        avail_ets = ['MPNN', 'DagLSTM', 'BidirDagLSTM', 'GCN']
        assert args.node_embedder in avail_ets, \
            'Node embedding module not recognized. Option must be one of ' + \
            ', '.join(avail_ets)
        ne_d = { 'MPNN' : em.MPNN, 'DagLSTM' : em.DagLSTM, 
                 'BidirDagLSTM' : em.BidirDagLSTM, 'GCN' : em.GCN }
        model_params['init_node_embedder'] = ne_d[args.node_embedder]
    
    if args.num_rounds is not None:
        assert args.num_rounds >= 0, 'Round count must be at least 0...'
        model_params['num_rounds'] = args.num_rounds

    model_params['depth_cap'] = None
    if args.depth_cap is not None:
        assert args.depth_cap >= 0, 'Maximum formula depth must be at least 0...'
        model_params['depth_cap'] = args.depth_cap

    if args.default_pc:
        assert args.default_pc in ['True', 'False'], '--default_pc must be True or False'
        model_params['default_pc'] = args.default_pc == 'True'
        
    dataset = args.dataset
    assert dataset in ['holstep', 'mizar', 'mizar_cnf', 'scitail'], 'Unknown dataset...'
    model_params['pretrained_embs'] = dataset == 'scitail' and pd.emb_info_loc
    if dataset == 'holstep':
        div_emb_sz = 4 if small_network else 1
        tr_dobj = pd.hol_tr_dobj_loc
        val_dobj = pd.hol_val_dobj_loc
        te_dobj = pd.hol_te_dobj_loc
        tr_collator = dt.HolstepCollator(model_params['depth_cap'], model_params['default_pc'], 
                                         model_params['edge_type'])
        te_collator = dt.HolstepCollator(model_params['depth_cap'], model_params['default_pc'], 
                                         model_params['edge_type'])
        max_epochs = 5
        model_params['node_emb_dim'] = int(128 / div_emb_sz)
        model_params['node_state_dim'] = int(256 / div_emb_sz)
        model_params['lstm_state_dim'] = int(256 / div_emb_sz)
        # edge info
        model_params['edge_emb_dim'] = int(32 / div_emb_sz)
        model_params['edge_state_dim'] = int(64 / div_emb_sz)
    elif dataset == 'scitail':
        div_emb_sz = 2 if small_network else 1
        tr_dobj = pd.sci_tr_dobj_loc
        val_dobj = pd.sci_val_dobj_loc
        te_dobj = pd.sci_te_dobj_loc
        tr_collator = dt.HolstepCollator(model_params['depth_cap'], model_params['default_pc'], 
                                         model_params['edge_type'])
        te_collator = dt.HolstepCollator(model_params['depth_cap'], model_params['default_pc'], 
                                         model_params['edge_type'])
        max_epochs = 10
        model_params['node_emb_dim'] = int(16 / div_emb_sz)
        #model_params['node_state_dim'] = int(32 / div_emb_sz)
        #model_params['lstm_state_dim'] = int(32 / div_emb_sz)
        model_params['node_state_dim'] = model_params['pretrained_emb_dim']
        model_params['lstm_state_dim'] = model_params['pretrained_emb_dim']
        # edge info
        model_params['edge_emb_dim'] = int(4 / div_emb_sz)
        model_params['edge_state_dim'] = int(8 / div_emb_sz)
    elif dataset in ['mizar', 'mizar_cnf']:
        #if small_network: div_emb_sz = 2 if model_params['default_pc'] else 4
        if small_network: div_emb_sz = 2
        else: div_emb_sz = 1
        if dataset == 'mizar':
            tr_dobj = pd.miz_tr_dobj_loc
            val_dobj = pd.miz_val_dobj_loc
            te_dobj = pd.miz_te_dobj_loc
        elif dataset == 'mizar_cnf':
            tr_dobj = pd.miz_cnf_tr_dobj_loc
            val_dobj = pd.miz_cnf_val_dobj_loc
            te_dobj = pd.miz_cnf_te_dobj_loc
        tr_collator = dt.MizarCollator(model_params['depth_cap'], model_params['default_pc'], 
                                       model_params['edge_type'])
        te_collator = dt.MizarCollator(model_params['depth_cap'], model_params['default_pc'], 
                                       model_params['edge_type'])
        model_params['node_emb_dim'] = int(64 / div_emb_sz)
        model_params['node_state_dim'] = int(128 / div_emb_sz)
        model_params['lstm_state_dim'] = int(128 / div_emb_sz)
        # edge info
        model_params['edge_emb_dim'] = int(16 / div_emb_sz)
        model_params['edge_state_dim'] = int(32 / div_emb_sz)
        # each example in a batch is actually the complete set of positive and 
        # negative premises, so there's actually around 100ish graphs in a batch
        batch_size = 16
        dataset_params['batch_size'] = batch_size
        max_epochs = 30 if model_params['default_pc'] else 5
    dataset_params['num_workers'] = batch_size

    if args.batch_size is not None:
        assert args.batch_size > 0, 'Positive values for batch size required'
        batch_size = args.batch_size
        model_params['batch_size'] = args.batch_size

    if args.num_epochs is not None:
        assert args.num_epochs >= 1, 'Epochs count must be at least 1...'
        max_epochs = args.num_epochs

    def get_tr_val_generators():
        # get training data
        training_set = pkl.load(open(tr_dobj, 'rb'))
        training_generator = data.DataLoader(training_set, collate_fn=tr_collator,
                                             **dataset_params)
        # get validation data
        validation_set = pkl.load(open(val_dobj, 'rb'))
        validation_generator = data.DataLoader(validation_set, collate_fn=te_collator,
                                               **dataset_params)
        if val_is_test:
            training_generator = chain(training_generator, validation_generator)
            validation_set = pkl.load(open(te_dobj, 'rb'))
            validation_generator = data.DataLoader(validation_set, 
                                                   collate_fn=collator,
                                                   **dataset_params)
        return training_set,training_generator,validation_set,validation_generator 

    model = cm.FormulaRelevanceClassifier(**model_params)
    today = date.today()
    model_param_str = '_'.join([model_params[x].__name__
                                for x in ['init_node_embedder',
                                          'pooling_module']]) + '_' + \
                                          '_'.join([str(model_params['num_rounds']),
                                                    model_params['aggr_type'],
                                                    str(model_params['edge_type']),
                                                    model_params['dep_match_type'],
                                                    str(model_params['dep_depth']),
                                                    str(model_params['default_pc']),
                                                    today.strftime('%Y%m%d')])
    
    # actually train model
    best_performance, time_info, loss_info, acc_info, track_ct = -1, [], [], [], 500
    gap_filler = '\n' + '='.join(['' for _ in range(30)]) + '\n'
    for epoch in range(max_epochs):
        if model_params['lr_decay'] and epoch > 0:
            for p_group in model.sparse_optimizer.param_groups:
                p_group['lr'] = p_group['lr'] / 3.0
            for p_group in model.dense_optimizer.param_groups:
                p_group['lr'] = p_group['lr'] / 3.0

        ( training_set, training_generator,
          validation_set, validation_generator ) = get_tr_val_generators()

        print(gap_filler)
        print('Starting training with model params ' + model_param_str + '...')
        print(gap_filler)

        tr_ex_ct, prev_pc, acc_tr_time = 0, None, 0

        tr_acc_nums, tr_loss_nums = [], []
        
        # training
        model.train()
        for batch in training_generator:

            tr_ex_ct += batch_size

            if (not model_params['default_pc']) and dataset == 'mizar':
                random.shuffle(batch)
                pseudo_batches = [batch[i : i + batch_size]
                                  for i in range(0, len(batch), batch_size)]
                if len(pseudo_batches[-1]) < batch_size / 2:
                    up_to = pseudo_batches[:-1]
                    up_to[-1].extend(pseudo_batches[-1])
                    pseudo_batches = up_to
            else:
                pseudo_batches = [batch]

            for pb in pseudo_batches:
                st_time = time.time()
                b_loss, b_acc = model.train_classifier(pb)
                assert not np.isnan(b_loss), 'Loss is nan...'
            
                tr_time = time.time() - st_time
                acc_tr_time += tr_time

                time_info.append(tr_time)
                loss_info.append(b_loss)
                acc_info.append(b_acc)
                if len(time_info) > track_ct: time_info.pop(0)
                if len(loss_info) > track_ct: loss_info.pop(0)
                if len(acc_info) > track_ct: acc_info.pop(0)
                if epoch == 0:
                    tr_acc_nums.append(b_acc)
                    tr_loss_nums.append(b_loss)
                print('At epoch: ' + str(epoch))
                print('Average training loss: ' + str(np.mean(loss_info)))
                print('Average training accuracy: ' + str(np.mean(acc_info)))
                print('Average training time: ' + \
                      str(np.mean(time_info) / len(pb)))
                print('Training completion: ' + \
                      str(round(tr_ex_ct/len(training_set.data_ids)*100, 2)) + '%')
                print()

        # validation
        model.eval()
        val_batch_ct, val_total_ct, corr_ct, acc_val_time = 0, 0, 0, 0
        for batch in validation_generator:
            val_batch_ct += batch_size

            if (not model_params['default_pc']) and dataset == 'mizar':
                random.shuffle(batch)
                pseudo_batches = [batch[i : i + batch_size]
                                  for i in range(0, len(batch), batch_size)]
            else:
                pseudo_batches = [batch]

            for pb in pseudo_batches:
                with torch.no_grad():
                    st_time = time.time()
                    outputs, labels, parse_failures = model.run_classifier(pb)
                    val_time = time.time() - st_time
                acc_val_time += val_time
                for output, label in zip(outputs, labels):
                    pred = 1 if output >= 0.5 else 0
                    if pred == label: corr_ct += 1
                    val_total_ct += 1
                val_total_ct += len(parse_failures)
                print('Validation completion: ' + \
                      str(round(val_batch_ct/len(validation_set.data_ids) * 100, 2))+'%')
                print('Current accuracy: ' + str(corr_ct / val_total_ct))
                print()
                
        epoch_performance = corr_ct / val_total_ct

        print(gap_filler)
        print(epoch_performance)
        print(gap_filler)

        results_file = dataset + '_' + model_param_str + '_val_results.csv'
        with open(os.path.join(results_path, results_file), 'a') as f:
            f.write(','.join([str(epoch), str(corr_ct), str(val_batch_ct),
                              str(val_total_ct), 
                              str(acc_tr_time), str(acc_val_time),
                              str(epoch_performance)]) + '\n')

        if epoch_performance > best_performance and save_model:
            best_spec = dataset + '_best_' + model_param_str
            pkl.dump(model, open(os.path.join(models_path, best_spec), 'wb'))
            best_performance = epoch_performance

        latest_spec = dataset + '_epoch_' + str(epoch) + '_' + model_param_str
        pkl.dump(model, open(os.path.join(models_path, latest_spec), 'wb'))

        if epoch == 0:
            tr_perf_file = dataset + '_' + model_param_str + '_tr_curve.csv'
            with open(os.path.join(results_path, tr_perf_file), 'w') as f:
                f.write(','.join([str(round(num, 5)) for num in tr_acc_nums]))
                f.write('\n')
                f.write(','.join([str(round(num, 5)) for num in tr_loss_nums]))
                f.write('\n')
                
