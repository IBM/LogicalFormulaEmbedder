# python imports
import signal, time, os, sys, subprocess
import pickle as pkl
import argparse as ap
import generate_ranking as gr
import train_model as tm

def run_for(test_on, post_ranking_path):
    proof_found_str = 'Proof found!'
    no_proof_found_str = 'No proof found!'
    skip_nums = []

    k_res = dict([(str(k), { 'model_ranked' : 0, 'sym_ranked' : 0 })
                  for k in gr.test_params['top_k'] + ['all']])
    k_res['-1'] = {}
    k_res['-1']['all_sym_ranked'] = 0
    k_res['-1']['all_stmts'] = 0
    k_tot = dict([(str(k), { 'model_ranked' : 0, 'sym_ranked' : 0 })
                  for k in gr.test_params['top_k'] + ['all']])
    k_tot['-1'] = {}
    k_tot['-1']['all_sym_ranked'] = 0
    k_tot['-1']['all_stmts'] = 0

    kv = dict([(k, i) for i, k in enumerate(['-1','all'] + \
                                            [str(k) for k in gr.test_params['top_k']])])
    as_success = {}
    all_files, ord_files = sorted(list(os.listdir(post_ranking_path))), []
    if test_on == 'model':
        num_lst = ['all', 16, 32, 64, 128, 256, 512, 1024, 2048]
    elif test_on == 'baseline':
        num_lst = ['all_stmts']
    for tgt in [str(x) for x in num_lst]:
        for file_name in all_files:
            if 'all_stmts' in file_name:
                file_num = 'all_stmts'
            else:
                file_num = file_name.split('_')[-1]
            if tgt == file_num:
                ord_files.append(file_name)
    for prob_name in ord_files:
        base_name = prob_name.split('_all_sym_ranked')[0]
        base_name = base_name.split('_all_stmts')[0]
        base_name = base_name.split('_sym_ranked')[0]
        base_name = base_name.split('_model_ranked')[0]
        #if test_on == 'model' and 'all_stmts' in prob_name: continue
        if test_on == 'baseline' and '_model_ranked' in prob_name: continue
        if base_name in as_success and as_success[base_name]: continue
        if '_sym_ranked' in prob_name: continue
        if '_all_sym_ranked' in prob_name: continue
        if any(str(skip_num) in prob_name for skip_num in skip_nums): continue
        
        if test_on == 'model': cpu_limit = '--cpu-limit=10'
        else: cpu_limit = '--cpu-limit=90'

        print('Running ' + prob_name + '...')
        f_loc = os.path.join(post_ranking_path, prob_name)
        command = ['eprover', '--auto-schedule', '--proof-object', '--free-numbers',
                   '--silent', cpu_limit, f_loc]
        ret_str = ''
        try:
            ret_str = str(subprocess.check_output(command))
        except subprocess.CalledProcessError as e:
            ret_str = str(e.output)
            
        if not ('all_sym_ranked' in f_loc or 'all_stmts' in f_loc): 
            is_k = prob_name.split('_')[-1]
        if 'model_ranked' in f_loc:
            k_tot[is_k]['model_ranked'] += 1
        elif 'all_sym_ranked' in f_loc:
            k_tot['-1']['all_sym_ranked'] += 1
        elif 'all_stmts' in f_loc:
            k_tot['-1']['all_stmts'] += 1
        elif 'sym_ranked' in f_loc:
            k_tot[is_k]['sym_ranked'] += 1

        proof_found = proof_found_str in ret_str
        if not base_name in as_success: as_success[base_name] = proof_found
        as_success[base_name] = as_success[base_name] or proof_found

        if proof_found:
            print('Proof found!')
            if 'model_ranked' in f_loc:
                k_res[is_k]['model_ranked'] += 1
            elif 'all_sym_ranked' in f_loc:
                k_res['-1']['all_sym_ranked'] += 1
            elif 'all_stmts' in f_loc:
                k_res['-1']['all_stmts'] += 1
            elif 'sym_ranked' in f_loc:
                k_res[is_k]['sym_ranked'] += 1
        else:
            print('No proof found')
        print('\n---- Current results ----')
        for k in sorted(list(k_res.keys()), key=lambda x : kv[x]):
            if k != '-1': print('-')
            for v in sorted(list(k_res[k].keys())):
                if (k in ['1024', '2048'] and 'model' in v): j_s = '\t'
                elif k == '-1' and 'sym_ranked' in v: j_s = '\t'
                elif k != '-1': j_s = '\t\t'
                elif 'all_stmts' in v: j_s = '\t\t'
                else: j_t = '\t'
                print(' '.join(v.split('_')).capitalize() + ' for ' + \
                      str(k) + ' at' + j_s + str(k_res[k][v]) + \
                      ' out of ' + str(k_tot[k][v]))
        print('=========================\n')

    path_parts = os.path.split(post_ranking_path)
    rank_res_file = os.path.join(tm.results_path, test_on + '_' + path_parts[-1] + \
                                 '_ranking_results.csv')
    with open(rank_res_file, 'a') as f:
        srtd_keys = sorted(list(k_res.keys()), key=lambda x : kv[x])
        for k in ['Ranking method'] + srtd_keys:
            f.write(str(k) + ',')
        f.write('\n')
        for v in ['all_stmts', 'all_sym_ranked', 'model_ranked', 'sym_ranked']:
            f.write(v + ',')
            for k in srtd_keys:
                if v in k_res[k]:
                    f.write(str(k_res[k][v]) + ',')
                else:
                    f.write('N/A,')
            f.write('\n')

if __name__ == '__main__':

    parser = ap.ArgumentParser(description='Test formula premise selection')
    parser.add_argument('--test_on', help='Which method to test for premise selection')
    parser.add_argument('--ranked_path',
                        help='Path ranked files are found in')
    args = parser.parse_args()

    post_ranking_path = gr.post_ranking_path
    if args.ranked_path is not None:
        assert os.path.exists(args.ranked_path)
        post_ranking_path = args.ranked_path

    t_os = ['model']
    if args.test_on:
        test_types = ['model', 'baseline', 'all']
        assert args.test_on in test_types, 'Method type should be one of ' + \
            ', '.join(test_types)
        if args.test_on == 'all': t_os = ['model', 'baseline']
        else: t_os = [args.test_on]

    for t_o in t_os:
        run_for(t_o, post_ranking_path)
