import sys, os
# numpy imports
import numpy as np
# torch imports
import torch
import torch.autograd as ta
import torch.nn.functional as F
import torch.nn as nn
# code imports
from code.simple_modules import *
from code.embedding_modules import *

###
# Pair pooling
###
        
class DepDagLSTMPool(nn.Module):

    def __init__(self, **kw_args):
        super().__init__()
        self.device = kw_args['device']
        self.is_cuda = (self.device != torch.device('cpu'))
        self.pool_dir = kw_args['pooling_dir']
        ne_kw_args = dict(kw_args)
        if 'init_node_embedder_acc_dir' in kw_args:
            ne_kw_args['acc_dir'] = kw_args['init_node_embedder_acc_dir']
        # this makes sure the output dimensionality of the node embedder
        # remains the node_state_dim
        self.node_state_dim = kw_args['node_state_dim']
        if kw_args['init_node_embedder'] in [DagLSTM, BidirDagLSTM]:
            ne_kw_args['lstm_state_dim'] = self.node_state_dim
        self.node_embedder = kw_args['init_node_embedder'](**ne_kw_args)
        pool_kw_args = dict(kw_args)
        pool_kw_args['acc_dir'] = ACCNN.up_acc
        pool_kw_args['node_embedder'] = self.node_embedder

        self.att_dim = self.node_state_dim * 2
        pool_kw_args['node_state_dim'] = self.node_state_dim + self.att_dim
        self.dag_lstm = DagLSTM(**pool_kw_args)
        
        self.ne_norm = nn.LayerNorm(self.node_state_dim).to(self.device)
        #self.ne_norm = nn.BatchNorm1d(self.node_state_dim).to(self.device)

        self.type_ct = kw_args['edge_ct']
        self.type_state_dim = kw_args['edge_state_dim']
        sparse_grads = kw_args['sparse_grads']
        
        # gating mechanism
        prem_type_embs = Emb(self.type_ct, self.type_state_dim,
                             sparse_grads=sparse_grads, device=self.device)
        prem_lin = nn.Linear(self.type_state_dim, self.att_dim).to(self.device)
        self.prem_gate = nn.Sequential(*[prem_type_embs, prem_lin,
                                         nn.Sigmoid()]).to(self.device)
        conj_type_embs = Emb(self.type_ct, self.type_state_dim,
                             sparse_grads=sparse_grads, device=self.device)
        conj_lin = nn.Linear(self.type_state_dim, self.att_dim).to(self.device)
        self.conj_gate = nn.Sequential(*[conj_type_embs, conj_lin,
                                         nn.Sigmoid()]).to(self.device)
        
        # attention matrix
        num_heads = kw_args['mha_heads']
        self.head_dim = int(self.att_dim / num_heads)
        self.b_q = np.sqrt(self.head_dim)
        self.W_op = nn.Linear(self.head_dim * num_heads, self.att_dim,
                              bias=False).to(self.device)
        self.W_oc = nn.Linear(self.head_dim * num_heads, self.att_dim,
                              bias=False).to(self.device)
        MHA_p, MHA_c = [], []
        for _ in range(num_heads):
            trip_p = [nn.Linear(self.node_state_dim, self.head_dim, bias=False)
                      for _ in range(3)]
            MHA_p.append(nn.ModuleList(trip_p))
            trip_c = [nn.Linear(self.node_state_dim, self.head_dim, bias=False)
                      for _ in range(3)]
            MHA_c.append(nn.ModuleList(trip_c))
        self.MHA_p = nn.ModuleList(MHA_p).to(self.device)
        self.MHA_c = nn.ModuleList(MHA_c).to(self.device)
            
    def compute_graph_reprs(self, emb_ind_info, upd_layers, graph_info):
        node_emb_inds, edge_emb_inds, type_emb_inds, depth_emb_inds, _, _ = emb_ind_info
        node_reprs = self.node_embedder.compute_node_reprs(emb_ind_info,
                                                           upd_layers)
        
        node_reprs = self.ne_norm(node_reprs)

        type_tensor = torch.tensor(type_emb_inds, device=self.device)
        type_prem_gate = self.prem_gate(type_tensor)
        type_conj_gate = self.conj_gate(type_tensor)
        
        # getting the pairing information for both graphs
        just_conj, just_prem = [], []
        for _, _, ident_pairs in graph_info:
            for conj_ind, prem_ind in ident_pairs:
                just_conj.append((conj_ind, prem_ind))
                just_prem.append((prem_ind, conj_ind))
        just_conj, just_prem = sorted(just_conj), sorted(just_prem)
        comb_reprs = torch.zeros(len(node_reprs), self.att_dim,
                                 device=self.device)

        for W_o, MHA, pairs, gate in [[self.W_op, self.MHA_p, just_prem, type_prem_gate],
                                      [self.W_oc, self.MHA_c, just_conj, type_conj_gate]]:
            mha_reprs = []
            for W_q, W_k, W_v in MHA:
                gr_reprs = torch.zeros(len(node_reprs), self.head_dim, device=self.device)
                for grouped_pairs in group_similar_tup_sizes(pairs,
                                                             no_split=self.is_cuda):
                    just_pairs = sorted(grouped_pairs)
                    prob_matr = compute_att_aggr(node_reprs, just_pairs,
                                                 W_q, W_k, self.b_q, self.device)
                    if prob_matr is None: continue
                    prob_matr = prob_matr.unsqueeze(1)
                    prob_to_node_sz = torch.Size([len(node_reprs), len(prob_matr)])
                    tgt_tensor = torch.tensor([x[1] for x in just_pairs], device=self.device)
                    tgt_matr = node_reprs.index_select(0, tgt_tensor)
                    tgt_matr = W_v(tgt_matr)
                    comb_matr = prob_matr * tgt_matr
                    adj_pairs = [(x[0], acc_pos) for acc_pos, x in enumerate(just_pairs)]
                    adj_matr = get_adj_matr(adj_pairs, prob_to_node_sz,
                                            is_cuda=self.is_cuda).to(self.device)
                    grouped_reprs = torch.mm(adj_matr, comb_matr)
                    gr_reprs = torch.add(grouped_reprs, gr_reprs)
                mha_reprs.append(gr_reprs)
            mha_reprs = gate * W_o(torch.cat(mha_reprs, 1))
            comb_reprs = torch.add(comb_reprs, mha_reprs)
            
        node_reprs = torch.cat((node_reprs, comb_reprs), 1)
        
        # now we pool the combined representations
        pooled_reprs = self.dag_lstm.compute_node_reprs(emb_ind_info, upd_layers,
                                                        node_states=node_reprs)

        prem_lst, conj_lst = [], []
        for c_info, prem_sets, _ in graph_info:
            c_inds = c_info[1] if self.pool_dir == ACCNN.up_acc else c_info[2]
            conj_tensor = torch.tensor(c_inds, device=self.device)
            conj_set = pooled_reprs.index_select(0, conj_tensor)
            conj_repr = torch.max(conj_set, dim=0)[0]
            for p_info in prem_sets:
                p_inds = p_info[1] if self.pool_dir == ACCNN.up_acc else p_info[2]
                prem_tensor = torch.tensor(p_inds, device=self.device)
                prem_set = pooled_reprs.index_select(0, prem_tensor)
                prem_repr = torch.max(prem_set, dim=0)[0]
                # for every premise, we concatenate it with the conjecture
                prem_lst.append(prem_repr)
                conj_lst.append(conj_repr)
        pc_r = torch.cat((torch.stack(prem_lst), torch.stack(conj_lst)), 1)
        return pc_r
 
###
# Special graph pooling
###


###
# Individual pooling
###

class DagLSTMPool(nn.Module):
    
    def __init__(self, **kw_args):
        super().__init__()
        self.device = kw_args['device']
        self.pool_dir = kw_args['pooling_dir']
        self.node_state_dim = kw_args['node_state_dim']
        ne_kw_args = dict(kw_args)
        if 'init_node_embedder_acc_dir' in kw_args:
            ne_kw_args['acc_dir'] = kw_args['init_node_embedder_acc_dir']
        # this makes sure the output dimensionality of the node embedder
        # remains the node_state_dim
        if kw_args['init_node_embedder'] in [DagLSTM, BidirDagLSTM]:
            ne_kw_args['lstm_state_dim'] = self.node_state_dim
        self.node_embedder = kw_args['init_node_embedder'](**ne_kw_args)
        pool_kw_args = dict(kw_args)
        pool_kw_args['acc_dir'] = ACCNN.up_acc
        pool_kw_args['node_embedder'] = self.node_embedder
        self.dag_lstm = DagLSTM(**pool_kw_args)
        
        self.ne_norm = nn.LayerNorm(self.node_state_dim).to(self.device)
        #self.ne_norm = nn.BatchNorm1d(self.node_state_dim).to(self.device)

    def compute_graph_reprs(self, emb_ind_info, upd_layers, graph_info, just_use_conj=False):
        node_emb_inds, edge_emb_inds, type_emb_inds, depth_emb_inds, _, _ = emb_ind_info
        node_reprs = self.node_embedder.compute_node_reprs(emb_ind_info, upd_layers)
        node_reprs = self.ne_norm(node_reprs)
        pooled_reprs = self.dag_lstm.compute_node_reprs(emb_ind_info, upd_layers,
                                                        node_states=node_reprs)
        prem_lst, conj_lst = [], []
        for c_info, prem_sets, _ in graph_info:
            c_inds = c_info[1] if self.pool_dir == ACCNN.up_acc else c_info[2]
            conj_tensor = torch.tensor(c_inds, device=self.device)
            conj_set = pooled_reprs.index_select(0, conj_tensor)
            conj_repr = torch.max(conj_set, dim=0)[0]
            if just_use_conj: conj_lst.append(conj_repr)
            else:
                for p_info in prem_sets:
                    p_inds = p_info[1] if self.pool_dir == ACCNN.up_acc else p_info[2]
                    prem_tensor = torch.tensor(p_inds, device=self.device)
                    prem_set = pooled_reprs.index_select(0, prem_tensor)
                    prem_repr = torch.max(prem_set, dim=0)[0]
                    # for every premise, we concatenate it with the conjecture
                    prem_lst.append(prem_repr)
                    conj_lst.append(conj_repr)
        if just_use_conj: return torch.stack(conj_lst)
        return torch.cat((torch.stack(prem_lst), torch.stack(conj_lst)), 1)

class SimpleMaxPool(nn.Module):
   
    def __init__(self, **kw_args):
        super().__init__()
        self.device = kw_args['device']
        ne_kw_args = dict(kw_args)
        if 'init_node_embedder_acc_dir' in kw_args:
            ne_kw_args['acc_dir'] = kw_args['init_node_embedder_acc_dir']
        self.node_embedder = kw_args['init_node_embedder'](**ne_kw_args)

    def compute_graph_reprs(self, emb_ind_info, upd_layers, graph_info, just_use_conj=False):
        node_emb_inds, edge_emb_inds, type_emb_inds, depth_emb_inds, _, _ = emb_ind_info
        node_reprs = self.node_embedder.compute_node_reprs(emb_ind_info, upd_layers)
        prem_lst, conj_lst = [], []
        for c_info, prem_sets, _ in graph_info:
            c_inds = c_info[0]
            conj_tensor = torch.tensor(c_inds, device=self.device)
            conj_set = node_reprs.index_select(0, conj_tensor)
            conj_repr = torch.max(conj_set, dim=0)[0]
            if just_use_conj: conj_lst.append(conj_repr)
            else:
                for p_info in prem_sets:
                    p_inds = p_info[0]
                    prem_tensor = torch.tensor(p_inds, device=self.device)
                    prem_set = node_reprs.index_select(0, prem_tensor)
                    prem_repr = torch.max(prem_set, dim=0)[0]
                    # for every premise, we concatenate it with the conjecture
                    prem_lst.append(prem_repr)
                    conj_lst.append(conj_repr)
        if just_use_conj: return torch.stack(conj_lst)
        return torch.cat((torch.stack(prem_lst), torch.stack(conj_lst)), 1)

