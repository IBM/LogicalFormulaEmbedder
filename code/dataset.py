# python imports
import signal, time, random, itertools, copy, math, os, string
import pickle as pkl
# numpy imports
import numpy as np
# torch imports
import torch
from torch.utils import data
# code imports
from code.parse_input_forms import *
from code.utilities import *

#######################
# dataset class
#######################

class Dataset(data.Dataset):
    def __init__(self, data_ids):
        self.data_ids = data_ids

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, index):
        f_loc = self.data_ids[index]
        X = torch.load(f_loc)
        return X

#######################
# collate class
#######################

class Collator: pass

class HolstepCollator(Collator):

    def __init__(self, max_depth, default_pc, edge_spec):
        self.max_depth = max_depth
        self.default_pc = default_pc
        self.edge_spec = edge_spec

    def __call__(self, data):
        graph_examples = []
        for data_ex in data:
            label, conjecture, stmt = data_ex[0], data_ex[1], data_ex[2]
            binary_label = (1 if label == '+' else 0)
            stmts, parse_failures, all_grs = [], [], []
            if conjecture is not None:
                stmts.append(('conj', conjecture))
            if stmt == None or conjecture == None:
                parse_failures.append(binary_label)
            else:
                stmts.append((binary_label, stmt))
            # 
            if conjecture == None:
                ret_grs, conj = None, None
            elif self.default_pc:
                all_grs = [gr for _, gr in stmts]
                conj = [x for x in stmts if x[0] == 'conj'][0][1]
                stmts = [x for x in stmts if x[0] != 'conj']
                conj_gr, st_grs = all_grs[0], all_grs[1:]
                conj_gr = convert_expr_to_graph(all_grs[0],
                                                depth_limit=self.max_depth,
                                                edge_spec=self.edge_spec,
                                                is_hol=True)
                st_gr = convert_expr_list_to_graph(all_grs[1:], 
                                                   depth_limit=self.max_depth,
                                                   edge_spec=self.edge_spec,
                                                   is_hol=True)
                ret_grs = (conj_gr, st_gr)
            else:
                raise ValueError('Cannot have default_pc == False on Holstep')
            graph_examples.append((ret_grs, conj, stmts, parse_failures))
        return graph_examples

class MizarCollator(Collator):

    def __init__(self, max_depth, default_pc, edge_spec):
        self.max_depth = max_depth
        self.default_pc = default_pc
        self.edge_spec = edge_spec
        self.fixed_neg_ct = 16

    def __call__(self, data):
        graph_examples = []
        if not self.default_pc:
            compl_stmts = set()
            for conjecture, all_stmts in data:
                for label, stmt in all_stmts:
                    if stmt != None and conjecture != None:
                        compl_stmts.add(conjecture)
                        compl_stmts.add(stmt)

        for conjecture, all_stmts in data:
            stmts, parse_failures, all_grs = [], [], []
            if conjecture is not None:
                stmts.append(('conj', conjecture))
            for label, stmt in all_stmts:
                binary_label = (1 if label == '+' else 0)
                if stmt == None or conjecture == None:
                    parse_failures.append(binary_label)
                else:
                    stmts.append((binary_label, stmt))
            #
            if conjecture == None:
                ret_grs = None
            elif self.default_pc:
                all_grs = [gr for _, gr in stmts]
                conj = [x for x in stmts if x[0] == 'conj'][0][1]
                stmts = [x for x in stmts if x[0] != 'conj']
                conj_gr, st_grs = all_grs[0], all_grs[1:]
                conj_gr = convert_expr_to_graph(all_grs[0],
                                                depth_limit=self.max_depth,
                                                edge_spec=self.edge_spec)
                st_gr = convert_expr_list_to_graph(all_grs[1:],
                                                   depth_limit=self.max_depth,
                                                   edge_spec=self.edge_spec)
                ret_grs = (conj_gr, st_gr)
                graph_examples.append((ret_grs, conj, stmts, parse_failures))
            else:
                all_grs = [gr for _, gr in stmts]
                conj = [x for x in stmts if x[0] == 'conj'][0][1]
                stmts = [x for x in stmts if x[0] != 'conj']
                conj_gr, st_grs = all_grs[0], all_grs[1:]
                conj_gr = convert_expr_to_graph(all_grs[0],
                                                depth_limit=self.max_depth,
                                                edge_spec=self.edge_spec)
                new_stmts = [(0, stmt) for stmt in compl_stmts if not stmt in all_grs]
                random.shuffle(new_stmts)
                new_stmts = stmts + new_stmts[:self.fixed_neg_ct]
                random.shuffle(new_stmts)
                for st_i, (label, stmt) in enumerate(new_stmts):
                    use_fs = parse_failures if st_i == 0 else []
                    st_gr = convert_expr_to_graph(stmt, depth_limit=self.max_depth,
                                                  edge_spec=self.edge_spec)
                    ret_grs = (conj_gr, st_gr)
                    graph_examples.append((ret_grs, conjecture, [(label, stmt)], use_fs))
                    
        return graph_examples
