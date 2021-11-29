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
import code.utilities as ut

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

def try_parse(parse_func, item, timeout, is_conj=False):
    try:
        parsed_item = ut.timeout(parse_func, (item,), duration=timeout)
    except (ut.TimeoutError, RuntimeError) as err:
        parsed_item = None
        if is_conj:
            print('PARSE FAILURE FOR CONJECTURE:\n' + item)
        else:
            print('PARSE FAILURE FOR STATEMENT:\n' + item)
    return parsed_item

if __name__ == '__main__':

    parser = ap.ArgumentParser(description='Test formula ranking module')
    parser.add_argument('--model', help='Model name in models directory')
    parser.add_argument('--location', help='Relative path containing files to rank')
    args = parser.parse_args()

    assert args.location, 'location cannot be left unspecified'
    file_loc = os.path.join('.', args.location)
    assert os.path.exists(file_loc), 'file must exist at specified location must exist'
    
    model_filename = args.model
    assert model_filename, 'Model specification required...'

    model = pkl.load(open(os.path.join(models_path, model_filename), 'rb'))
    model.eval()
            
    # test model
    gap_filler = '\n' + '='.join(['' for _ in range(30)]) + '\n'
    
    print(gap_filler)
    print('Starting ranking...')
    print(gap_filler)

    all_files_to_rank = list(os.listdir(file_loc))

    for pr_i, prob_name in enumerate(all_files_to_rank):
        print('Ranking problem ' + str(pr_i + 1) + ' out of ' + \
              str(len(all_files_to_rank)))
        print(prob_name + '\n')
        conjecture, paired_stmts = None, []
        with open(os.path.join(file_loc, prob_name), 'r') as f:
            all_lines = list(f.readlines())
        while all_lines:
            line = all_lines.pop(0)
            if line[:2] == 'C ':
                conjecture = line[2:].replace('\n', '')
                if conjecture[:2] == '|-': conjecture = conjecture[2:]
                tup_conj = try_parse(pr.parse_fof_to_tuple, conjecture,
                                     10, is_conj=True)
            elif line[:2] in ['+ ', '- ']:
                label = line[0]
                new_stmt = line[2:].replace('\n', '')
                if new_stmt[:2] == '|-': new_stmt = new_stmt[2:]
                tup_stmt = try_parse(pr.parse_fof_to_tuple, new_stmt, 10)
                paired_stmts.append((label, tup_stmt, new_stmt))
        assert conjecture, 'No conjecture encountered...'
        with torch.no_grad():
            model_scored = []
            conj_gr = dt.convert_expr_to_graph(tup_conj)
            for at_i in range(0, len(paired_stmts), test_params['batch_size']):
                s_infos = paired_stmts[at_i : at_i + test_params['batch_size']]
                batch_rep = []
                for _, tup_stmt, stmt_str in s_infos:
                    stmt_gr = dt.convert_expr_to_graph(tup_stmt)
                    batch_rep.append(((conj_gr, stmt_gr), tup_conj, [(0, tup_stmt)], []))
                scores = model.run_classifier(batch_rep)[0]
                for score, s_info in zip(scores, s_infos):
                    stmt_str = s_info[-1]
                    model_scored.append((float(score), stmt_str))
        model_scored = sorted(model_scored, key=lambda x : x[0], reverse=True)
        out_file = os.path.join(file_loc, prob_name + '_model_scored')
        with open(out_file, 'w') as f:
            f.write('C ' + conjecture + '\n')
            for score, stmt_str in model_scored:
                f.write(str(round(score, 3)) + ' ' + stmt_str + '\n')
