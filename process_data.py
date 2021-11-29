# python imports
import os, random, subprocess
import pickle as pkl
import argparse as ap
# torch imports
import torch
# natural language imports
import spacy
import penman
# code imports
import code.dataset as dt
import code.parse_input_forms as pr
import code.utilities as ut
import process_sentences as ps

# dataset / data paths
dataset_path = os.path.join('.', 'datasets')
data_path = os.path.join('.', 'data')

# holstep
hol_path = os.path.join(dataset_path, 'holstep')
hol_tr_path = os.path.join(hol_path, 'train')
hol_te_path = os.path.join(hol_path, 'test')

# holstep dataset objects
hol_tr_dobj_loc = os.path.join(data_path, 'tr_holstep_data.pkl')
hol_val_dobj_loc = os.path.join(data_path, 'val_holstep_data.pkl')
hol_te_dobj_loc = os.path.join(data_path, 'te_holstep_data.pkl')

# mizar
miz_path = os.path.join(dataset_path, 'mizar')
miz_data_path = os.path.join(miz_path, 'nndata')
miz_tr_splt_file = os.path.join(miz_path, 'tr_file.txt')
miz_val_splt_file = os.path.join(miz_path, 'dev_file.txt')
miz_te_splt_file = os.path.join(miz_path, 'te_file.txt')

# mizar dataset objects
miz_tr_dobj_loc = os.path.join(data_path, 'tr_mizar_data.pkl')
miz_val_dobj_loc = os.path.join(data_path, 'val_mizar_data.pkl')
miz_te_dobj_loc = os.path.join(data_path, 'te_mizar_data.pkl')

# mizar cnf
miz_cnf_path = os.path.join(dataset_path, 'mizar_cnf')
miz_cnf_data_path = os.path.join(miz_path, 'nndata')
miz_cnf_tr_splt_file = os.path.join(miz_path, 'tr_file.txt')
miz_cnf_val_splt_file = os.path.join(miz_path, 'dev_file.txt')
miz_cnf_te_splt_file = os.path.join(miz_path, 'te_file.txt')

# mizar cnf dataset objects
miz_cnf_tr_dobj_loc = os.path.join(data_path, 'tr_mizar_cnf_data.pkl')
miz_cnf_val_dobj_loc = os.path.join(data_path, 'val_mizar_cnf_data.pkl')
miz_cnf_te_dobj_loc = os.path.join(data_path, 'te_mizar_cnf_data.pkl')

# scitail
sci_path = os.path.join(dataset_path, 'SciTailV1.1')
sci_tr_file = os.path.join(sci_path, 'tsv_format', 'scitail_1.0_train.tsv')
sci_val_file = os.path.join(sci_path, 'tsv_format', 'scitail_1.0_dev.tsv')
sci_te_file = os.path.join(sci_path, 'tsv_format', 'scitail_1.0_test.tsv')

# scitail dataset objects
sci_tr_dobj_loc = os.path.join(data_path, 'tr_scitail_data.pkl')
sci_val_dobj_loc = os.path.join(data_path, 'val_scitail_data.pkl')
sci_te_dobj_loc = os.path.join(data_path, 'te_scitail_data.pkl')
lang_path = os.path.join('.', 'language_info')
emb_info_loc = os.path.join(lang_path, 'pretrained_embs')
sci_sents_obj_loc = os.path.join(lang_path, 'scitail_sentences.pkl')
sci_amr_obj_loc = os.path.join(lang_path, 'scitail_sentences_amr.pkl')
sci_spacy_obj_loc = os.path.join(lang_path, 'scitail_sentences_spacy.pkl')
sci_emb_obj_loc = os.path.join(lang_path, 'scitail_pretrained_word_embeddings.pkl')

def process_holstep(resume_from_exists=False):
    # file assumes location of holstep data is in ./datasets/holstep/
    tr_files = list(os.listdir(hol_tr_path))
    # shuffling with random seed such that we always get the same split
    random.Random(1).shuffle(tr_files)
    val_files = tr_files[:int(len(tr_files) / 10)]
    tr_files = tr_files[int(len(tr_files) / 10):]
    te_files = list(os.listdir(hol_te_path))
    to_process = [(hol_tr_dobj_loc, tr_files, hol_tr_path), 
                  (hol_val_dobj_loc, val_files, hol_tr_path),
                  (hol_te_dobj_loc, te_files, hol_te_path)]
    for split, (dobj_loc, file_src, src_path) in enumerate(to_process):
        is_test = split == 2
        added_files, prev_at = [], 0
        for at_f, filename in enumerate(file_src):
            if round(100 * at_f / len(file_src), 2) != prev_at:
                prev_at = round(100 * at_f / len(file_src), 2)
                print('Holstep processing for split ' + str(split) + ' at ' + \
                      str(prev_at) + '% complete...')
            with open(os.path.join(src_path, filename), 'r') as f:
                conjecture, paired_stmts = None, []
                all_lines = list(f.readlines())
                while all_lines:
                    line = all_lines.pop(0)
                    if line[:2] == 'C ':
                        conjecture = line[2:].replace('\n', '')
                        if conjecture[:2] == '|-': conjecture = conjecture[2:]
                    elif line[:2] in ['+ ', '- ']:
                        label = line[0]
                        new_stmt = line[2:].replace('\n', '')
                        if new_stmt[:2] == '|-': new_stmt = new_stmt[2:]
                        paired_stmts.append((label, new_stmt))
                assert conjecture, 'No conjecture encountered...'
                parsed_conjecture, conjecture_failed = None, False
                for st_num, (label, stmt) in enumerate(paired_stmts):
                    new_filename = '_'.join([filename,'holstep',str(st_num)])+'.pt'
                    new_file = os.path.join(data_path, new_filename)
                    if resume_from_exists and os.path.exists(new_file): continue
                    if parsed_conjecture is None and not conjecture_failed:
                        parsed_conjecture = try_parse(pr.parse_s_expr_to_tuple,
                                                      conjecture, timeout, 
                                                      is_conj=True)
                        conjecture_failed = parsed_conjecture == None
                    if parsed_conjecture:
                        parsed_stmt = try_parse(pr.parse_s_expr_to_tuple,
                                                stmt, timeout, is_conj=False)
                    else:
                        parsed_stmt = None
                    if parsed_stmt:
                        stmt_ex = (label, parsed_conjecture, parsed_stmt)
                    else:
                        stmt_ex = (None, None, None)
                    if parsed_stmt or is_test:
                        # if we can't parse the test example, we count it
                        # as a missed question, if we can't parse a training
                        # example, we just skip it
                        torch.save(stmt_ex, new_file)
                        added_files.append(new_file)
        pkl.dump(dt.Dataset(added_files), open(dobj_loc, 'wb'))

def process_mizar(resume_from_exists=False):
    # file assumes location of mizar data is in ./datasets/mizar/
    tr_files, val_files, te_files = [], [], []
    for splt_file, file_st in [(miz_tr_splt_file, tr_files), 
                               (miz_val_splt_file, val_files),
                               (miz_te_splt_file, te_files)]:
        with open(splt_file, 'r') as f:
            file_st.extend([l.replace('\n', '') for l in f.readlines()])
    for split, (dobj_loc, file_src) in enumerate([(miz_tr_dobj_loc, tr_files), 
                                                  (miz_val_dobj_loc, val_files), 
                                                  (miz_te_dobj_loc, te_files)]):
        is_test = split == 2
        added_files, prev_at = [], 0
        for at_f, filename in enumerate(file_src):
            if round(100 * at_f / len(file_src), 2) != prev_at:
                prev_at = round(100 * at_f / len(file_src), 2)
                print('Mizar processing for split ' + str(split) + ' at ' + \
                      str(prev_at) + '% complete...')
            with open(os.path.join(miz_data_path, filename), 'r') as f:
                conjecture, paired_stmts = None, []
                all_lines = list(f.readlines())
                while all_lines:
                    line = all_lines.pop(0)
                    if line[:2] == 'C ':
                        conjecture = line[2:].replace('\n', '')
                    elif line[:2] in ['+ ', '- ']:
                        label = line[0]
                        new_stmt = line[2:].replace('\n', '')
                        paired_stmts.append((label, new_stmt))
                assert conjecture, 'No conjecture encountered...'
                parsed_conjecture, conjecture_failed = None, False
                parsed_stmts = []
                new_filename = filename + '_mizar_proc.pt'
                new_file = os.path.join(data_path, new_filename)
                if resume_from_exists and os.path.exists(new_file): continue
                for st_num, (label, stmt) in enumerate(paired_stmts):
                    if parsed_conjecture is None and not conjecture_failed:
                        parsed_conjecture = try_parse(pr.parse_fof_to_tuple,
                                                      conjecture, timeout, 
                                                      is_conj=True)
                        conjecture_failed = parsed_conjecture == None
                    if parsed_conjecture:
                        parsed_stmt = try_parse(pr.parse_fof_to_tuple, 
                                                stmt, timeout, is_conj=False)
                    else:
                        parsed_stmt = None
                    if parsed_stmt:
                        parsed_stmts.append((label, parsed_stmt))
                    else:
                        parsed_stmts.append((label, None))
                assert len(parsed_stmts) == len(paired_stmts)
                torch.save((parsed_conjecture, parsed_stmts), new_file)
                added_files.append(new_file)
        pkl.dump(dt.Dataset(added_files), open(dobj_loc, 'wb'))

def process_mizar_cnf(resume_from_exists=False):
    # file assumes location of mizar data is in ./datasets/mizar/
    tr_files, val_files, te_files = [], [], []
    for splt_file, file_st in [(miz_cnf_tr_splt_file, tr_files), 
                               (miz_cnf_val_splt_file, val_files),
                               (miz_cnf_te_splt_file, te_files)]:
        with open(splt_file, 'r') as f:
            file_st.extend([l.replace('\n', '') for l in f.readlines()])
    for split, (dobj_loc, file_src) in enumerate([(miz_cnf_tr_dobj_loc, tr_files), 
                                                  (miz_cnf_val_dobj_loc, val_files),
                                                  (miz_cnf_te_dobj_loc, te_files)]):
        is_test = split == 2
        added_files, prev_at = [], 0
        for at_f, filename in enumerate(file_src):
            if round(100 * at_f / len(file_src), 2) != prev_at:
                prev_at = round(100 * at_f / len(file_src), 2)
                print('Mizar cnf processing for split ' + str(split) + ' at ' + \
                      str(prev_at) + '% complete...')
            with open(os.path.join(miz_cnf_data_path, filename), 'r') as f:
                conjecture, paired_stmts = None, []
                all_lines = list(f.readlines())
                while all_lines:
                    line = all_lines.pop(0)
                    if line[:2] == 'C ':
                        conjecture = line[2:].replace('\n', '')
                        conjecture_lst = convert_fof_to_cnf(conjecture, negate=False)
                    elif line[:2] in ['+ ', '- ']:
                        label = line[0]
                        new_stmt = line[2:].replace('\n', '')
                        stmt_lst = convert_fof_to_cnf(new_stmt)
                        paired_stmts.append((label, stmt_lst))
                assert conjecture, 'No conjecture encountered...'
                parsed_conjecture, conjecture_failed = None, False
                parsed_stmts = []
                new_filename = filename + '_mizar_cnf_proc.pt'
                new_file = os.path.join(data_path, new_filename)
                if resume_from_exists and os.path.exists(new_file): continue
                for st_num, (label, stmts_lst) in enumerate(paired_stmts):
                    if parsed_conjecture is None and not conjecture_failed:
                        parsed_conjecture = try_parse(pr.parse_cnf_lst_to_tuple,
                                                      conjecture_lst, timeout, 
                                                      is_conj=True)
                        conjecture_failed = parsed_conjecture == None
                    if parsed_conjecture:
                        parsed_stmt = try_parse(pr.parse_cnf_lst_to_tuple, 
                                                stmts_lst, timeout, is_conj=False)
                    else:
                        parsed_stmt = None
                    if parsed_stmt:
                        parsed_stmts.append((label, parsed_stmt))
                    else:
                        parsed_stmts.append((label, None))
                assert len(parsed_stmts) == len(paired_stmts)
                torch.save((parsed_conjecture, parsed_stmts), new_file)
                added_files.append(new_file)
        pkl.dump(dt.Dataset(added_files), open(dobj_loc, 'wb'))
    
def convert_fof_to_cnf(formula, negate=False):
    if negate: formula = pr.negate_fof_formula(formula)
    tmp_f_name = 'cnf_tmp'
    with open(tmp_f_name, 'w') as tmp_f:
        tmp_f.write(formula)
    command = ['eprover', '--free-numbers', '--cnf', tmp_f_name]
    ret_str = ''
    try: ret_str = str(subprocess.check_output(command))
    except subprocess.CalledProcessError as e:
        ret_str = str(e.output)
        raise e
    cnf_stmts = []
    for l in ret_str.split('\\n'):
        if l[:3] != 'cnf': continue
        cnf_stmts.append(l)
    return cnf_stmts

def process_scitail(resume_from_exists=False, data_range=None):
    #nlp = spacy.load("en_core_web_sm")
    # shuffling with random seed such that we always get the same split
    to_process = [(sci_tr_dobj_loc, sci_tr_file, 'train'), 
                  (sci_val_dobj_loc, sci_val_file, 'val'),
                  (sci_te_dobj_loc, sci_te_file, 'test')]
    sents = []
    amr_map = dict(pkl.load(open(sci_amr_obj_loc, 'rb')))
    if os.path.exists(sci_spacy_obj_loc): spacy_map = pkl.load(open(sci_spacy_obj_loc, 'rb'))
    else: spacy_map = {}
    if data_range == None: start, end = 0, float('inf')
    else: start, end = data_range
    for split, (dobj_loc, filename, f_type) in enumerate(to_process):
        is_test = split == 2
        added_files, prev_at = [], 0
        with open(filename, 'r') as f:
            conjecture, paired_stmts = None, []
            all_lines = list(f.readlines())
            for st_num, line in enumerate(all_lines):
                if round(100 * st_num / len(all_lines), 2) != prev_at:
                    prev_at = round(100 * st_num / len(all_lines), 2)
                    print('Scitail processing for split ' + f_type + ' at ' + \
                          str(prev_at) + '% complete...')
                if not (st_num >= start and st_num <= end): continue
                new_filename_base = '_'.join([f_type, 'scitail', str(st_num)])
                new_filename = new_filename_base + '.pt'
                new_file = os.path.join(data_path, new_filename)
                if resume_from_exists and os.path.exists(new_file): continue
                premise, hypothesis, label = line.replace('\n', '').split('\t')
                label = '-' if label == 'neutral' else '+'
                if not premise in spacy_map:
                    p_doc = nlp(premise)
                    spacy_map[premise] = p_doc
                if not hypothesis in spacy_map:
                    h_doc = nlp(hypothesis)
                    spacy_map[hypothesis] = h_doc
                parsed_conjecture, conj_text_lst = ut.parse_nl_stmt(hypothesis, spacy_map, amr_map)
                parsed_stmt, prem_text_lst = ut.parse_nl_stmt(premise, spacy_map, amr_map)
                sents.append((new_filename_base, [conj_text_lst, prem_text_lst]))
                stmt_ex = (label, parsed_conjecture, parsed_stmt, new_filename_base)
                torch.save(stmt_ex, new_file)
                added_files.append(new_file)
        pkl.dump(dt.Dataset(added_files), open(dobj_loc, 'wb'))
    if not os.path.exists(sci_spacy_obj_loc): pkl.dump(spacy_map, open(sci_spacy_obj_loc, 'wb'))
    sents = dict(sents)
    pkl.dump(sents, open(sci_sents_obj_loc, 'wb'))
    ps.process_sentences(sents)

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
    #sys.setrecursionlimit(10000)
    parser = ap.ArgumentParser(description='Process dataset for experiments')
    parser.add_argument('--dataset', help='Dataset to process')
    parser.add_argument('--resume', help='Skip existing files')
    parser.add_argument('--timeout', help='Timeout for parsing an example', type=int)
    parser.add_argument('--data_range', help='Range for data processing')
    args = parser.parse_args()

    timeout = int(args.timeout) if args.timeout else 750
    resume_from_exists = args.resume == 'True'
    assert (not args.resume) or args.resume in ['True', 'False'], \
        'Invalid resume command'
    allowed_datasets = ['holstep', 'mizar', 'mizar_cnf', 'scitail', 'all']
    dataset = args.dataset
    assert dataset in allowed_datasets, 'Unknown dataset, options are ' + \
        ', '.join(allowed_datasets)
    data_range = None
    if args.data_range: data_range = [int(x) for x in args.data_range.split('-')]
    if dataset in ['holstep', 'all']:
        process_holstep(resume_from_exists=resume_from_exists)
    if dataset in ['scitail', 'all']:
        process_scitail(resume_from_exists=resume_from_exists, data_range=data_range)
    if dataset in ['mizar', 'all']:
        process_mizar(resume_from_exists=resume_from_exists)
    if dataset in ['mizar_cnf']:#, 'all']:
        process_mizar_cnf(resume_from_exists=resume_from_exists)
