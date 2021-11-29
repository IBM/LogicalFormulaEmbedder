import signal, time, random, itertools, copy, math, sys
import pickle as pkl
# numpy imports
import numpy as np
# networkx imports
import networkx as nx
# torch imports
import torch
import torch.autograd as ta
import torch.nn.functional as F
import torch.nn as nn
# code imports
import code.parse_input_forms as pr
from code.simple_modules import *
from code.utilities import *
from code.node_classes import *
from code.pooling_modules import *

class FormulaRelevanceClassifier(nn.Module):

    def __init__(self, **kw_args):
        super().__init__()
        self.model_params = kw_args
        self.device = kw_args['device']
        self.lr = kw_args['lr']
        if kw_args['pretrained_embs']:
            self.pre_embs, self.pre_emb_map = torch.load(kw_args['pretrained_embs'])
            self.pre_embs = self.pre_embs.to(self.device)
        else: self.pre_embs, self.pre_emb_map = None, None
        self.label_hashing = kw_args['label_hashing']
        self.node_ct = kw_args['node_ct']
        self.edge_ct = kw_args['edge_ct']
        self.type_ct = kw_args['edge_ct']
        self.depth_ct = kw_args['edge_ct']
        self.type_smoothing = kw_args['type_smoothing']
        self.dropout = kw_args['dropout']
        self.mask_rate = kw_args['mask_rate']
        assert self.mask_rate >= 0 and self.mask_rate <= 1, \
            'Mask rate must be in [0, 1]'
        self.default_pc = kw_args['default_pc']
        self.node_assignments, self.available_nodes = {}, list(range(self.node_ct))
        self.edge_assignments, self.available_edges = {}, list(range(self.edge_ct))
        self.type_assignments, self.available_types = {}, list(range(self.type_ct))
        self.type_information = {}
        self.type_uses = {}
        self.exchangeable_types = set([PredType, FuncType, ConstType, 
                                       VarType, VarFuncType])
        self.dep_match_type = kw_args['dep_match_type']
        self.dep_depth = kw_args['dep_depth']
        
        self.formula_pair_embedder = kw_args['pooling_module'](**kw_args)

        if kw_args['pooling_module'] in [DagLSTMPool, DepDagLSTMPool]:
            self.classifier_inp_dim = kw_args['lstm_state_dim'] * 2
        else:
            self.classifier_inp_dim = kw_args['node_state_dim'] * 2
        self.classifier = MLP(self.classifier_inp_dim, 1,
                              hid_dim=int(self.classifier_inp_dim / 2),
                              dropout=0.2,#self.dropout,
                              mlp_act='sigmoid', device=self.device)
        
        self.sparse_grads = kw_args['sparse_grads']
        sparse_param_ids = []
        for m in self.modules():
            if type(m) == torch.nn.modules.sparse.Embedding and self.sparse_grads:
                sparse_param_ids.extend([id(p) for p in m.parameters()])
        dense_params = [param for param in self.parameters()
                        if not id(param) in sparse_param_ids]
        sparse_params = [param for param in self.parameters()
                         if id(param) in sparse_param_ids]
        self.dense_optimizer = torch.optim.Adam(dense_params, lr=self.lr)
        if self.sparse_grads:
            self.sparse_optimizer = torch.optim.SparseAdam(sparse_params,lr=self.lr)

    def assign_node_emb(self, node_label, node_info, is_masked):
        node_type, node_arity = node_info
        node_key = (node_label, node_type)
        if self.pre_emb_map and node_type == ConstType and node_label in self.pre_emb_map:
            return self.node_ct
        if self.training and self.type_smoothing and \
           node_info in self.type_information and is_masked:
            return self.type_information[node_info]
        if node_key in self.node_assignments:
            return self.node_assignments[node_key]
        if self.training:
            assert self.available_nodes or self.label_hashing, \
                'Insufficient node label embeddings...'
            if not node_info in self.type_information:
                t_assignment = self.available_nodes.pop()
                self.type_information[node_info] = t_assignment
            if self.available_nodes:
                n_assignment = self.available_nodes.pop()
                self.node_assignments[node_key] = n_assignment
            elif node_info in self.type_information and self.type_smoothing:
                self.node_assignments[node_key] = self.type_information[node_info]
            else:
                self.node_assignments[node_key] = self.node_ct
            return self.node_assignments[node_key]
        else:
            if self.type_smoothing and node_info in self.type_information:
                self.node_assignments[node_key] = self.type_information[node_info]
                return self.node_assignments[node_key]
            # padding index
            return self.node_ct

    def assign_edge_emb(self, edge_label):
        if edge_label in self.edge_assignments:
            return self.edge_assignments[edge_label]
        if self.training:
            assert self.available_edges, 'Insufficient edge label embeddings...'
            self.edge_assignments[edge_label] = self.available_edges.pop()
            return self.edge_assignments[edge_label]
        else:
            # padding index
            return self.edge_ct

    def assign_type_emb(self, type_label):
        if type_label in self.type_assignments:
            self.type_uses[type_label] += 1
            return self.type_assignments[type_label]
        if self.training:
            assert self.available_types, 'Insufficient type embeddings...'
            self.type_uses[type_label] = 1
            self.type_assignments[type_label] = self.available_types.pop()
            return self.type_assignments[type_label]
        else:
            # padding index
            return self.type_ct

    def assign_depth_emb(self, depth_info, stdv=5):

        #
        # quick fix
        return 0
        #
        
        noise1, noise2, noise3 = 0, 0, 0
        if self.training and stdv is not None:
            noise1, noise2, noise3 = [int(np.random.normal(0, stdv))
                                      for _ in range(3)]
        min_b = max(0, min(self.depth_ct, np.min(depth_info)) + noise1)
        avg_b = max(0, min(self.depth_ct, int(np.mean(depth_info))) + noise2)
        max_b = max(0, min(self.depth_ct, np.min(depth_info)) + noise3)
        depth_spec = (min_b, avg_b, max_b)
        return depth_spec
    
    def assign_graph_inds(self, graph):
        node_emb_inds, edge_emb_inds, type_emb_inds, depth_emb_inds = [], [], [], []
        topological_grouping = topologically_group(graph)
        node_ind_assigns = {}
        upd_layers, roots, leaves = [], [], []

        # masking should improve generalizability, we don't
        # want to mask operators or variables though
        mask_types = [PredType, FuncType, ConstType]
        all_syms, mask_syms = set(), set()
        if self.training and self.type_smoothing:
            for node in graph.nodes:
                if graph.nodes[node]['type'] in mask_types:
                    all_syms.add(graph.nodes[node]['label'])
            for sym in all_syms:
                if random.uniform(0, 1) <= self.mask_rate:
                    mask_syms.add(sym)

        # building up matrices
        for layer in topological_grouping:
            upd_layer = []
            for node in layer:
                node_ind = len(node_emb_inds)
                if graph.in_degree(node) == 0: roots.append(node_ind)
                if graph.out_degree(node) == 0: leaves.append(node_ind)
                node_ind_assigns[node] = node_ind
                node_label = graph.nodes[node]['label']
                node_type = graph.nodes[node]['type']
                node_depths = graph.nodes[node]['depth']
                # similar to edge labels, we bucket connectives and quantifiers
                # by their actual label
                if node_type in [QuantType, OpType]: node_type = node_label
                node_arity = graph.out_degree(node)
                node_info = (node_type, node_arity)
                # anonymizing variables, not sure what to do with generated variables
                # so we'll just keep them separate
                if node_type == VarType: node_label = '*__VAR__*'
                elif node_type == VarFuncType: node_label = '*__VARFUNC__*'
                elif node_type == GenVarType: node_label = '*__GENVAR__*'
                elif node_type == GenVarFuncType: node_label = '*__GENVARFUNC__*'
                elif node_type == UniqVarType: node_label = '*__UNIQVAR__*'
                elif node_type == UniqVarFuncType: node_label = '*__UNIQVARFUNC__*'
                elif node_type == ApplyType: node_label = '*__APPLY__*'
                elif node_type == NASType: node_label = '*__NAS__*'

                # get node embedding ind
                is_masked = node_label in mask_syms
                node_emb_inds.append(self.assign_node_emb(node_label, node_info,
                                                          is_masked))
                
                # getting edge embedding inds
                for arg in graph.successors(node):
                    arg_ind = node_ind_assigns[arg]
                    edge_ind = len(edge_emb_inds)
                    edge_label = graph.edges[node, arg]['label']
                    edge_emb_inds.append(self.assign_edge_emb(edge_label))
                    upd_layer.append((node_ind, arg_ind, edge_ind))

                # getting depth embedding ind
                depth_emb_inds.append(self.assign_depth_emb(node_depths))
                    
                # getting type embedding ind
                type_emb_inds.append(self.assign_type_emb(node_type))
                
                # adding leaves
                if not graph.out_degree(node):
                    upd_layer.append((node_ind, None, None))
            upd_layers.append(upd_layer)

        emb_ind_info = (node_emb_inds, edge_emb_inds, type_emb_inds, depth_emb_inds)
        return emb_ind_info, upd_layers, node_ind_assigns

    def vectorize_graphs(self, graph_examples):
        # input is [ <pc_graph, stmt>, ...]
        is_training = self.training
        self.eval()
        node_emb_inds, edge_emb_inds, type_emb_inds, depth_emb_inds = [], [], [], []
        upd_layers, graph_info = [], []
        for bg_ex in graph_examples:
            pc_graph, stmt = bg_ex[0], bg_ex[1]
            n_offset, e_offset = len(node_emb_inds), len(edge_emb_inds)
            em_info, updates,node_assigns = self.assign_graph_inds(pc_graph)
            ns, es, ts, ds = em_info
            node_emb_inds.extend(ns)
            edge_emb_inds.extend(es)
            type_emb_inds.extend(ts)
            depth_emb_inds.extend(ds)
            upd_layers = merge_updates(updates,upd_layers,n_offset,e_offset)
            prem_assigns = node_assigns
            stmt_info = get_item_info(stmt, pc_graph, node_assigns, n_offset)
            pc_info = (stmt_info, [stmt_info], [])
            graph_info.append(pc_info)
        emb_ind_info = (node_emb_inds, edge_emb_inds, type_emb_inds, depth_emb_inds, [], [])
        embs = self.formula_pair_embedder.compute_graph_reprs(emb_ind_info, upd_layers, graph_info,
                                                              just_use_conj=True)
        if is_training: self.train()
        return embs
    
    def vectorize_batch_examples(self, batch_graph_examples):
        # vectorizing is tricky, we actually just turn everything into one
        # giant disconnected graph for processing
        node_emb_inds, edge_emb_inds, type_emb_inds, depth_emb_inds = [], [], [], []
        upd_layers, graph_info = [], []
        targets, parse_fails = [], []
        pre_embs, emb_pairs = [], []
        for bg_ex in batch_graph_examples:
            pc_graphs, conjecture, premises, parse_failures = bg_ex[0], bg_ex[1], bg_ex[2], bg_ex[3]
            conj_pre_embs, prem_pre_embs = {}, {}
            parse_fails.extend(parse_failures)
            if pc_graphs is None: continue
            for pc_i, pc_graph in enumerate(pc_graphs):
                n_offset, e_offset = len(node_emb_inds), len(edge_emb_inds)
                if pc_i == 0: conj_offset = n_offset
                else: prem_offset = n_offset
                em_info, updates,node_assigns = self.assign_graph_inds(pc_graph)
                ns, es, ts, ds = em_info
                node_emb_inds.extend(ns)
                edge_emb_inds.extend(es)
                type_emb_inds.extend(ts)
                depth_emb_inds.extend(ds)
                upd_layers = merge_updates(updates,upd_layers,n_offset,e_offset)
                if pc_i == 0: conj_assigns = node_assigns
                else: prem_assigns = node_assigns
            # ident pairs is a dictionary of symbols found in both prem and conj
            ident_pairs = {}
            conj_info = get_item_info(conjecture, pc_graphs[0], conj_assigns, 
                                      conj_offset)
            # conj_info is tuple of (conj_inds, conj_roots, conj_leaves)
            all_prem_info = []
            for label, premise in premises:
                targets.append(label)
                prem_info = get_item_info(premise, pc_graphs[1], prem_assigns,
                                          prem_offset)
                all_prem_info.append(prem_info)
            ident_pairs = get_ident_pairs(conjecture, pc_graphs[0],
                                          [pr for _, pr in premises], pc_graphs[1],
                                          conj_offset, conj_assigns, 
                                          prem_offset, prem_assigns,
                                          self.dep_match_type, self.dep_depth)
            if self.pre_emb_map:
                for assigns, offset_is in [[conj_assigns, conj_offset],
                                           [prem_assigns, prem_offset]]:
                    for const, const_pos in assigns.items():
                        if type(const) != tuple and pr.sep_tok_id(const) in self.pre_emb_map:
                            offset_pos = offset_is + const_pos
                            emb_pairs.append((offset_pos, len(pre_embs)))
                            pre_embs.append(self.pre_emb_map[pr.sep_tok_id(const)])
            
            pc_info = (conj_info, all_prem_info, ident_pairs)
            graph_info.append(pc_info)

        pre_embs = self.pre_embs.index_select(0, torch.tensor(pre_embs).to(self.device)) if pre_embs else None
        emb_ind_info = (node_emb_inds, edge_emb_inds, type_emb_inds, depth_emb_inds,
                        emb_pairs, pre_embs)
        
        return emb_ind_info, upd_layers, graph_info, targets, parse_fails
        
    def train_classifier(self, batch_graph_examples):
        self.dense_optimizer.zero_grad()
        if self.sparse_grads: self.sparse_optimizer.zero_grad()
        vectorized = self.vectorize_batch_examples(batch_graph_examples)
        emb_ind_info, upd_layers, graph_info, targets, parse_failures = vectorized

        outputs = self.classify_batch(emb_ind_info, upd_layers, graph_info).squeeze(-1)
        target_tensor = torch.tensor(targets, device=self.device, dtype=torch.float)
        loss = nn.BCELoss()(outputs, target_tensor)
        
        binary_outputs = [(1 if output >= 0.5 else 0) for output in outputs]
        c = sum([(1 if x[0] == x[1] else 0) for x in zip(binary_outputs, targets)])
        acc = c / (len(targets) + len(parse_failures))

        loss.backward()
        self.dense_optimizer.step()
        if self.sparse_grads: self.sparse_optimizer.step()
        return float(loss), acc

    def run_classifier(self, batch_graph_examples):
        vectorized = self.vectorize_batch_examples(batch_graph_examples)
        emb_ind_info, upd_layers, graph_info, targets, parse_fails = vectorized
        outputs = self.classify_batch(emb_ind_info, upd_layers,
                                      graph_info).squeeze(-1)
        return outputs, targets, parse_fails

    def classify_batch(self, emb_ind_info, upd_layers, graph_info):
        graph_reprs = self.formula_pair_embedder.compute_graph_reprs(emb_ind_info,
                                                                     upd_layers,
                                                                     graph_info)
        return self.classifier(graph_reprs)


###
# Misc helper functions
###

def merge_updates(updates, upd_layers, n_offset, e_offset):
    # helper function for merging update lists, this could be made more complex
    # to include better load-balancing
    for i, upd_lyr in enumerate(updates):
        offset_layer = []
        for (n_i, n_j, e_ij) in upd_lyr:
            if n_j is not None:
                edge = (n_i + n_offset, n_j + n_offset, e_ij + e_offset)
            else:
                edge = (n_i + n_offset, None, None)
            offset_layer.append(edge)
        if len(upd_layers) > i:
            upd_layers[i].extend(offset_layer)
        else:
            upd_layers.append(offset_layer)
    return upd_layers

def get_item_info(item, src_graph, node_assigns, n_offset):
    # helper function to build up identical symbol dictionary as well as 
    # get the item-graph roots, leaves, and all indices
    item_inds, item_leaves = [], []
    item_roots = [node_assigns[item] + n_offset]
    expr_set, par_dict = deconstruct_expr(item)
    for subexpr in expr_set:
        if not subexpr in src_graph.nodes: continue
        assignment = node_assigns[subexpr] + n_offset
        if type(subexpr) != tuple:
            item_leaves.append(assignment)
        item_inds.append(assignment)
    return (item_inds, item_roots, item_leaves)

def get_ident_pairs(conjecture, conj_graph, premises, prem_graph,
                    conj_offset, conj_assigns, prem_offset, prem_assigns,
                    key_type, max_depth=1000):
    conj_d = {}
    conj_expr_set, cpd = deconstruct_expr(conjecture)
    for subexpr in conj_expr_set:
        if not subexpr in conj_graph.nodes: continue
        assignment = conj_assigns[subexpr] + conj_offset
        c_key = get_ident_key(subexpr, conj_graph, cpd, key_type,
                              max_depth=max_depth)
        if not c_key in conj_d: conj_d[c_key] = []
        conj_d[c_key].append((subexpr, assignment))
    ident_pairs = set()
    for premise in premises:
        prem_expr_set, ppd = deconstruct_expr(premise)
        for subexpr in prem_expr_set:
            if not subexpr in prem_graph.nodes: continue
            assignment = prem_assigns[subexpr] + prem_offset
            p_key = get_ident_key(subexpr, prem_graph, ppd, key_type,
                                  max_depth=max_depth)
            if not p_key in conj_d: continue
            for conj_subexpr, conj_ind in conj_d[p_key]:
                if matches_by_key_type(conj_subexpr, conj_graph, cpd,
                                       subexpr, prem_graph, ppd, key_type,
                                       max_depth=max_depth):
                    ident_pairs.add((conj_ind, assignment))
    return ident_pairs

def get_ident_key(expr, src_graph, pd, key_type, max_depth=1000):
    if key_type == 'label':
        if src_graph.nodes[expr]['type'] == VarType: ident_key = get_var_name(expr)
        else: ident_key = expr[0] if type(expr) == tuple else expr
        ident_key = (ident_key, src_graph.nodes[expr]['type'])
    elif key_type == 'leaf_label':
        if src_graph.nodes[expr]['type'] == VarType: ident_key = get_var_name(expr)
        else: ident_key = expr[0] if type(expr) == tuple else 'leaf'
        ident_key = (ident_key, src_graph.nodes[expr]['type'])
    elif key_type == 'leaf':
        ident_key = 'leaf' if type(expr) != tuple else 'non_leaf'
    elif key_type == 'type':
        ident_key = src_graph.nodes[expr]['type']
    elif key_type == 'depth':
        lc = get_ancestor_label_chains(expr, pd, src_graph=src_graph)
        orig_type = src_graph.nodes[expr]['type']
        is_QO = orig_type in [QuantType, OpType]
        ident_key = min([len([t for _, t in l]) for l in lc])
    elif key_type == 'depth_typed':
        lc = get_ancestor_label_chains(expr, pd,  src_graph=src_graph)
        orig_type = src_graph.nodes[expr]['type']
        is_QO = orig_type in [QuantType, OpType]
        ident_key = min([len([t for _, t in l if t == orig_type or \
                              (t in [QuantType, OpType] and is_QO)])
                         for l in lc])
        if is_QO: orig_type = 'QuantOp'
        ident_key = (ident_key, orig_type)
    elif key_type == 'iso':
        ident_key = len(list(formula_elements_lst(expr, src_graph, depth=max_depth)))
    elif key_type == 'alpha':
        ident_key = tuple(sorted(list(formula_elements_lst(expr, src_graph, depth=max_depth))))
    elif key_type == 'all':
        ident_key = None
    else:
        raise ValueError('Dependent embedding key method ' + key_type + ' unknown')
    return ident_key

def matches_by_key_type(conj_expr, conj_graph, cpd, prem_expr, prem_graph, ppd,
                        key_type, max_depth):
    if key_type in ['label', 'depth', 'leaf_label']:
        c_k = get_ident_key(conj_expr, conj_graph, cpd, key_type)
        p_k = get_ident_key(prem_expr, prem_graph, ppd, key_type)
        return c_k == p_k
    elif key_type == 'type':
        conj_type = conj_graph.nodes[conj_expr]['type']
        prem_type = prem_graph.nodes[prem_expr]['type']
        return conj_type == prem_type
    elif key_type == 'leaf':
        return type(conj_expr) != tuple and type(prem_expr) != tuple
    elif key_type == 'alpha':
        return is_alpha_equiv(conj_expr, conj_graph, prem_expr, prem_graph, depth=max_depth)
    elif key_type == 'iso':
        return is_prob_iso(conj_expr, conj_graph, prem_expr, prem_graph, depth=max_depth)
    elif key_type == 'all':
        return True
    
