# python imports
import signal, time, os, sys, random
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
import code.parse_input_forms as pr
import code.node_classes as nc

models_path = os.path.join('.', 'models')
results_path = os.path.join('.', 'results')
ranking_path = os.path.join(pd.miz_path, 'ranking_exp_data')
post_ranking_path = os.path.join(pd.miz_path, 'ranked_files')
nn_data_path = os.path.join(pd.miz_path, 'nndata')
all_files_list_file = os.path.join(pd.miz_path, 'files_in_ranking_exp.txt')
prob_deps_file = os.path.join(pd.miz_path, 'seq.txt')

target_labels = ['C', '+']

test_params = { 'sym_overlap_cutoff' : -1,
                'sym_ranking_cutoff' : 2048,
                'top_k' : [16, 32, 64, 128, 256, 512, 1024, 2048], 
                'check_types' : [nc.PredType, nc.FuncType, nc.ConstType],
                'num_problems' : 4000,
                'use_prev_prob_set' : False,
                'batch_size' : 5112,
                'first_run' : False,
                'keep_prior' : False }

def grab_nth_stmt_from_file(prob_name, n):
    filename = os.path.join(nn_data_path, prob_name)
    with open(filename, 'r') as f:
        at_ct = 0
        all_lines = list(f.readlines())
        for line in all_lines:
            if line[0] in target_labels:
                if at_ct == n:
                    return line[2:].replace('\n', '')
                at_ct += 1
        assert False, 'Should not have gotten here: ' + prob_name

def grab_gr_labels(tup_form, gr_dict, labels_dict):
    if not tup_form in gr_dict:
        gr_dict[tup_form] = dt.convert_expr_to_graph(tup_form)
    expr_gr = gr_dict[tup_form]
    if not tup_form in labels_dict:
        expr_labels = set()
        for node in expr_gr.nodes:
            if expr_gr.nodes[node]['type'] in test_params['check_types']:
                expr_labels.add(expr_gr.nodes[node]['label'])
        labels_dict[tup_form] = expr_labels
    expr_labels = labels_dict[tup_form]
    return expr_gr, expr_labels

def add_to_stmts(stmts, p_name, str_dict, conj=None):
    prems = [stmt for label, stmt in stmts 
             if label in target_labels]
    if conj is not None: prems = [conj] + prems
    new_stmts = []
    for p_i, p in enumerate(prems):
        if not p: continue
        new_stmts.append(p)
        if not p in str_dict:
            use_p_i = p_i + 1 if conj is None else p_i
            str_form = grab_nth_stmt_from_file(p_name, use_p_i)
            assert not 'conjecture' in str_form, str_form
            str_dict[p] = str_form
    return new_stmts

def replace_axiom_in_fof_str(str_form):
    comma_splt = str_form.split(',')
    conj_str = comma_splt[:1] + [' conjecture'] + comma_splt[2:]
    return ','.join(conj_str)

def grab_tup_forms(p_name, load_dict):
    if not p_name in load_dict:
        load_dict[p_name] = torch.load(p_name)
    parsed_conj, parsed_stmts = load_dict[p_name]
    parsed_stmts = [(l, ps) for l, ps in parsed_stmts 
                    if l in target_labels]
    return parsed_conj, parsed_stmts

def write_ranking_to_file(filename, conj_str, scored_stmts):
    with open(filename, 'w') as fw:
        conj_upd = replace_axiom_in_fof_str(conj_str)
        fw.write(conj_upd + '\n')
        for score, _, _, _, stmt in scored_stmts:
            stmt_str = str_dict[stmt]
            fw.write(stmt_str + '\n')

def load_everything(all_files, reconstruct=False):
    # extract files
    data_loc = os.path.join(pd.data_path, 'gen_ranking_data.pkl')
    if reconstruct:
        # build dep structure
        prob_deps = {}
        with open(prob_deps_file, 'r') as f:
            all_lines = list(f.readlines())
            for l_i, line in enumerate(all_lines):
                prob_name = line.replace('\n', '')
                prob_deps[prob_name] = l_i
        # now get statements
        all_files_items, prev_m = list(enumerate(all_files.items())), None
        all_stmts, str_dict, load_dict, prob_info = set(), {}, {}, {}
        canon_t = {}
        for afi, (prob_name, actual_file) in all_files_items:
            if not prob_name in prob_deps: continue
            if round(afi / len(all_files_items), 1) != prev_m:
                prev_m = round(afi / len(all_files_items), 1)
                print('Loaded ' + str(prev_m * 100) + '% of files')
            conj, stmts = grab_tup_forms(actual_file, load_dict)
            if not conj: continue
            conj_str = grab_nth_stmt_from_file(prob_name, 0)
            if not conj in canon_t: canon_t[conj] = conj
            conj = canon_t[conj]
            if not conj in str_dict: str_dict[conj] = conj_str
            if not prob_name in prob_info: prob_info[prob_name] = set()
            prob_info[prob_name].add(conj)
            all_stmts.add(conj)
            add_to_stmts(stmts, prob_name, str_dict)
            for label, stmt in stmts:
                if not stmt: continue
                if not stmt in canon_t: canon_t[stmt] = stmt
                stmt = canon_t[stmt]
                all_stmts.add(stmt)
                prob_info[prob_name].add(stmt)
        pkl.dump((str_dict, all_stmts, load_dict, prob_deps, prob_info), open(data_loc, 'wb'))
        #input('Done constructing...')
    else:
        str_dict, all_stmts, load_dict, prob_deps, prob_info = pkl.load(open(data_loc, 'rb'))
    return str_dict, all_stmts, load_dict, prob_deps, prob_info

if __name__ == '__main__':

    parser = ap.ArgumentParser(description='Test formula ranking module')
    parser.add_argument('--model', help='Model name in models directory')
    parser.add_argument('--sym_overlap_cutoff', type=int, 
                        help='Symbol overlap minimum cutoff')
    parser.add_argument('--sym_ranking_cutoff', type=int, 
                        help='Symbol overlap minimum cutoff')
    parser.add_argument('--top_k', type=int, 
                        help='Only the top k scored will be returned')
    parser.add_argument('--batch_size', type=int, 
                        help='Batch size of ranking')
    parser.add_argument('--num_problems', type=int,
                        help='Randomly select n problems for ranking experiment')
    parser.add_argument('--first_run',
                        help='Set True if first ranking run on particular machine')
    parser.add_argument('--post_ranking_path',
                        help='Path to add ranked files to')
    parser.add_argument('--check_range',
                        help='hyphen separated range for problems')
    parser.add_argument('--use_existing_prob_set', help='Reuse storedd problems')
    args = parser.parse_args()

    if args.first_run is not None:
        assert args.first_run in ['True', 'False'], 'Must be either True or False'
        test_params['first_run'] = args.first_run == 'True'

    if args.post_ranking_path is not None:
        assert os.path.exists(args.post_ranking_path)
        post_ranking_path = args.post_ranking_path

    if args.use_existing_prob_set is not None:
        assert args.use_existing_prob_set in ['True', 'False'], \
            'Must be either True or False'
        test_params['use_prev_prob_set'] = args.use_existing_prob_set == 'True'

    if args.num_problems is not None:
        assert args.num_problems > 0, 'Must give value greater than 0'
        test_params['num_problems'] = args.num_problems

    check_range = None
    if args.check_range is not None:
        cr = args.check_range.split('-')
        assert len(cr) == 2
        check_range = [int(cr[0]), int(cr[1])]
        
    if args.sym_overlap_cutoff is not None:
        assert args.sym_overlap_cutoff >= 0, 'Must give value of at least 0'
        test_params['sym_overlap_cutoff'] = args.sym_overlap_cutoff

    if args.sym_ranking_cutoff is not None:
        assert args.sym_ranking_cutoff >= 0, 'Must give value of at least 0'
        test_params['sym_ranking_cutoff'] = args.sym_ranking_cutoff

    if args.top_k is not None:
        assert args.top_k > 0, 'Must give value greater than 0'
        test_params['top_k'] = args.top_k

    if args.batch_size is not None:
        assert args.batch_size > 0, 'Must give value greater than 0'
        test_params['batch_size'] = args.batch_size

    model_filename = args.model
    assert model_filename, 'Model specification required...'

    model = pkl.load(open(os.path.join(models_path, model_filename), 'rb'))
    model.eval()

    all_files, val_files = {}, set()
    for dobj in [pd.miz_tr_dobj_loc, pd.miz_val_dobj_loc]:#, pd.miz_te_dobj_loc]:
        d_set = pkl.load(open(dobj, 'rb'))
        for f in d_set.data_ids:
            stripped = f.split('_')
            no_end = '_'.join(stripped[:len(stripped) - 2])
            no_data = no_end.replace(pd.data_path + '/', '')
            no_data = no_data.replace(pd.data_path + '\\', '')
            all_files[no_data] = f
            if dobj == pd.miz_val_dobj_loc: val_files.add(no_data)
            
    # getting all axioms
    ( str_dict, all_stmts, load_dict,
      prob_deps, prob_info ) = load_everything(all_files, reconstruct=test_params['first_run'])

    # test model
    gap_filler = '\n' + '='.join(['' for _ in range(30)]) + '\n'
    
    print(gap_filler)
    print('Starting ranking...')
    print(gap_filler)

    prior_ranked_files, dataset_stats, processed_files = [], [], []

    gr_dict, labels_dict = {}, {}
    if test_params['use_prev_prob_set']:
        all_files_to_rank = []
        with open(all_files_list_file, 'r') as f:
            for l in f.readlines():
                all_files_to_rank.append(l.replace('\n', ''))
    else:
        if test_params['keep_prior']:
            with open(all_files_list_file, 'r') as f:
                for l in f.readlines():
                    prior_ranked_files.append(l.replace('\n', ''))
        for f in os.listdir(post_ranking_path):
            if not any(prf in f for prf in prior_ranked_files):
                f_path = os.path.join(post_ranking_path, f)
                #print(f_path)
        all_files_to_rank = list(val_files)
        all_files_to_rank = [f for f in all_files_to_rank
                             if (not f in prior_ranked_files) and \
                             f in prob_info]
        if check_range:
            pr_path = post_ranking_path + '_' + str(check_range[0]) + '_' + str(check_range[1])
            if not os.path.exists(pr_path): os.mkdir(pr_path)
            post_ranking_path = pr_path
            all_files_to_rank = all_files_to_rank[check_range[0] : check_range[1]]
        else:
            random.shuffle(all_files_to_rank)
            all_files_to_rank = all_files_to_rank[:test_params['num_problems']]

    print('Loading statements...\n')
        
    print('Finished loading all ' + str(len(all_stmts)) + ' statements...\n')
    for pr_i, prob_name in enumerate(all_files_to_rank):
        print('Ranking problem ' + str(pr_i + 1) + ' out of ' + \
              str(len(all_files_to_rank)))
        processed_files.append(prob_name)
        # getting premises and conjectures from file to be ranked
        act_conj, act_stmts = grab_tup_forms(all_files[prob_name], load_dict)
        if not act_conj: continue
        conj_str = str_dict[act_conj]
        true_prems = [s for l, s in act_stmts if s]
        
        prob_rank, stmts = prob_deps[prob_name], set()
        for other_prob_name, other_prob_rank in prob_deps.items():
            if (other_prob_rank >= prob_rank) or \
               not other_prob_name in prob_info: continue
            for stmt in prob_info[other_prob_name]:
                stmts.add(stmt)
        stmts = list(stmts)
        
        # ensure true premises available
        stmts.extend(true_prems)
        
        print('Processing symbols...')
        # process into graphs and get symbol overlap numbers
        pc_grs = []
        conj_gr, conj_labels = grab_gr_labels(act_conj, gr_dict, labels_dict)
        for stmt in stmts:
            stmt_gr, stmt_labels = grab_gr_labels(stmt, gr_dict, labels_dict)
            overlap = len(stmt_labels.intersection(conj_labels))
            if overlap > test_params['sym_overlap_cutoff']:
                pc_grs.append((overlap, conj_gr, stmt_gr, act_conj, stmt))
        dataset_stats.append(len(pc_grs))

        print()
        print(prob_name)
        print(prob_rank)
        print(len(all_stmts))
        print(len(stmts))
        print(len(pc_grs))
        print()
        
        #
        ## Ranking by symbol overlap
        #
        sym_sorted = sorted(pc_grs, key=lambda x : x[0], reverse=True)
        sym_pruned = sym_sorted[:test_params['sym_ranking_cutoff']]

        for stmt in true_prems:
            fnd = False
            for pc_gr in pc_grs:
                if fnd: break
                if stmt == pc_gr[4]: fnd = True
            if not fnd:
                print('\n*******************\n')
                print('Statment not found ' + str(stmt))
                print('\n*******************\n')

        #
        ## Ranking with the model
        #
        print('Model ranking...')
        st_time = time.time()
        with torch.no_grad():
            prev_m = None
            model_scored = []
            use_sym_sc = sym_sorted
            for at_i in range(0, len(use_sym_sc), test_params['batch_size']):
                if round(at_i / len(use_sym_sc), 1) != prev_m:
                    prev_m = round(at_i / len(use_sym_sc), 1)
                    print('Model ranking '+str(prev_m * 100)+'% complete')
                s_infos = use_sym_sc[at_i : at_i + test_params['batch_size']]
                batch_rep = []
                for s_info in s_infos:
                    (_, conj_gr, stmt_gr, conj, stmt) = s_info
                    batch_rep.append(((conj_gr, stmt_gr), conj, [(0, stmt)], []))
                scores = model.run_classifier(batch_rep)[0]
                for score, s_info in zip(scores, s_infos):
                    (_, conj_gr, stmt_gr, conj, stmt) = s_info
                    model_scored.append((float(score), conj_gr, stmt_gr,
                                         act_conj, stmt))

        end_time = time.time() - st_time
        print('Model ranked ' + str(len(use_sym_sc)) + ' examples in ' + \
              str(end_time) + ' seconds...')
        print('Average time per example at ' + \
              str(end_time / len(use_sym_sc)) + ' seconds...')

        model_sorted = sorted(model_scored, key=lambda x:x[0],reverse=True)

        for ms in []:#model_sorted[:10]:
            print(ms[3])
            print()
            print(ms[4])
            print()
            print(ms[0])
            print()
            print('-------')
        print()
        
        for p_i, tp in enumerate(true_prems):
            for ms_i, ms in enumerate(model_sorted):
                if ms[4] == tp:
                    print('True premise at ' + str(ms_i) + ' for model ranked...')
                    break
        print()

        for p_i, tp in enumerate(true_prems):
            for ms_i, ms in enumerate([x for x in model_sorted if x[0] > 0.5]):
                if ms[4] == tp:
                    print('True premise at ' + str(ms_i) + ' for model above 0.5...')
                    break
        print()

        for p_i, tp in enumerate(true_prems):
            for sp_i, sp in enumerate(sym_sorted):
                if sp[4] == tp:
                    print('True premise at ' + str(sp_i) + ' for sym ranked...')
                    break
        print()

        print(len([x for x in model_sorted if x[0] > 0.5]))

        #input('=======')
        
        # writing baseline all stmts
        a_s_file = os.path.join(post_ranking_path, prob_name + '_all_stmts')
        write_ranking_to_file(a_s_file, conj_str, pc_grs)
        
        # writing baseline all syms
        #a_r_file = os.path.join(post_ranking_path,prob_name + '_all_sym_ranked')
        #write_ranking_to_file(a_r_file, conj_str, sym_pruned)
        
        s_r_file = os.path.join(post_ranking_path, prob_name + '_sym_ranked_')
        m_r_file = os.path.join(post_ranking_path, prob_name + '_model_ranked_')
        for k in test_params['top_k'] + ['all']:
            for ranked_file, all_scored_stmts in [#[s_r_file, sym_pruned],
                                                  [m_r_file, model_sorted]]:
                if k == 'all': scored_stmts = all_scored_stmts
                else: scored_stmts = all_scored_stmts[:k]
                write_ranking_to_file(ranked_file + str(k), conj_str,
                                      scored_stmts)

    print('\n-----------------\n')
    print('Maximum number of premises: ' + str(max(dataset_stats)))
    print('Minimum number of premises: ' + str(min(dataset_stats)))
    print('Average number of premises: ' + str(np.mean(dataset_stats)))
    print('\n-----------------\n')

    with open(all_files_list_file, 'w') as f:
        for ranked_file in prior_ranked_files + all_files_to_rank:
            f.write(ranked_file + '\n')
