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
from code.utilities import *

class ACCNN(nn.Module):
    up_acc = 'up'
    down_acc = 'down'
    bidir_acc = 'bidir'
    acc_dirs = [up_acc, down_acc, bidir_acc]

class BidirDagLSTM(ACCNN):
    
    def __init__(self, **kw_args):
        super().__init__()
        self.device = kw_args['device']
        self.node_state_dim = kw_args['node_state_dim']
        up_kw_args = dict(kw_args)
        up_kw_args['acc_dir'] = ACCNN.up_acc
        self.up_lstm = DagLSTM(**up_kw_args)
        down_kw_args = dict(kw_args)
        down_kw_args['acc_dir'] = ACCNN.down_acc
        self.down_lstm = DagLSTM(**down_kw_args)
        self.combiner = MLP(self.node_state_dim * 2, self.node_state_dim,
                            norm_type='batch', device=self.device)
        
    def compute_node_reprs(self, emb_ind_info, upd_layers):
        up = self.up_lstm.compute_node_reprs(emb_ind_info, upd_layers)
        down = self.down_lstm.compute_node_reprs(emb_ind_info, upd_layers)
        comb_repr = self.combiner(torch.cat((up, down), 1))
        return comb_repr
        
class DagLSTM(ACCNN):
    # attention-based dag lstm accumulator neural network
    def __init__(self, **kw_args):
        super().__init__()
        self.device = kw_args['device']
        self.is_cuda = (self.device != torch.device('cpu'))
        if 'acc_dir' in kw_args: self.acc_dir = kw_args['acc_dir']
        else: self.acc_dir = ACCNN.up_acc
        assert self.acc_dir in ACCNN.acc_dirs, 'Unknown accumulation direction...'
        self.node_emb_dim = kw_args['node_emb_dim']
        self.edge_emb_dim = kw_args['edge_emb_dim']
        self.node_state_dim = kw_args['node_state_dim']
        self.lstm_state_dim = kw_args['lstm_state_dim']
        self.edge_state_dim = kw_args['edge_state_dim']
        self.aggr_type = kw_args['aggr_type']
        self.node_ct = kw_args['node_ct']
        self.edge_ct = kw_args['edge_ct']
        sparse_grads = kw_args['sparse_grads']
        
        self.dropout = kw_args['dropout']

        self.pretrained_embs = kw_args['pretrained_embs']
        if self.pretrained_embs:
            self.emb_map = nn.Sequential(*[nn.BatchNorm1d(kw_args['pretrained_emb_dim']),
                                           MLP(kw_args['pretrained_emb_dim'], self.node_state_dim,
                                               mlp_layers=2,dropout=self.dropout)]).to(self.device)
            self.emb_map = nn.BatchNorm1d(kw_args['pretrained_emb_dim'])
            #self.emb_map = nn.Identity()
            
        self.gen_node_embeddings = not 'node_embedder' in kw_args
        if self.gen_node_embeddings:
            self.node_embedder = EmbProj(self.node_ct, self.node_emb_dim,
                                         self.node_state_dim, device=self.device,
                                         sparse_grads=sparse_grads)
        self.edge_embedder = EmbProj(self.edge_ct, self.edge_emb_dim,
                                     self.edge_state_dim, device=self.device,
                                     sparse_grads=sparse_grads)

        stdv = 1. / math.sqrt(self.lstm_state_dim)
        edge_matr_size = (self.edge_ct + 1, self.lstm_state_dim, self.lstm_state_dim)
        
        self.W_i = nn.Linear(self.node_state_dim, self.lstm_state_dim, 
                             bias=False).to(self.device)
        U_i = torch.zeros(edge_matr_size, dtype=torch.float, device=self.device)
        U_i.uniform_(-stdv, stdv)
        self.U_i = nn.Parameter(U_i)
        self.b_i = nn.Parameter(torch.tensor(0., device=self.device))

        self.W_o = nn.Linear(self.node_state_dim, self.lstm_state_dim, 
                             bias=False).to(self.device)
        U_o = torch.zeros(edge_matr_size, dtype=torch.float, device=self.device)
        U_o.uniform_(-stdv, stdv)
        self.U_o = nn.Parameter(U_o)
        self.b_o = nn.Parameter(torch.tensor(0., device=self.device))

        self.W_c = nn.Linear(self.node_state_dim, self.lstm_state_dim, 
                             bias=False).to(self.device)
        U_c = torch.zeros(edge_matr_size, dtype=torch.float, device=self.device)
        U_c.uniform_(-stdv, stdv)
        self.U_c = nn.Parameter(U_c)
        self.b_c = nn.Parameter(torch.tensor(0., device=self.device))

        self.W_f = nn.Linear(self.node_state_dim, self.lstm_state_dim, 
                             bias=False).to(self.device)
        U_f = torch.zeros(edge_matr_size, dtype=torch.float, device=self.device)
        U_f.uniform_(-stdv, stdv)
        self.U_f = nn.Parameter(U_f)
        self.b_f = nn.Parameter(torch.tensor(0., device=self.device))

        self.nr_w_i = nn.LayerNorm(self.lstm_state_dim).to(self.device)
        self.nr_u_i = nn.LayerNorm(self.lstm_state_dim).to(self.device)
        self.nr_w_o = nn.LayerNorm(self.lstm_state_dim).to(self.device)
        self.nr_u_o = nn.LayerNorm(self.lstm_state_dim).to(self.device)
        self.nr_w_c = nn.LayerNorm(self.lstm_state_dim).to(self.device)
        self.nr_u_c = nn.LayerNorm(self.lstm_state_dim).to(self.device)
        self.nr_w_f = nn.LayerNorm(self.lstm_state_dim).to(self.device)
        self.nr_u_f = nn.LayerNorm(self.lstm_state_dim).to(self.device)
        self.nr_cell = nn.LayerNorm(self.lstm_state_dim).to(self.device)
        
    def compute_node_reprs(self, emb_ind_info, upd_layers, node_states=None):
        if self.acc_dir == ACCNN.down_acc: upd_layers = flip_upd_layers(upd_layers)
        node_inputs, edge_emb_inds, type_emb_inds, depth_emb_inds, pre_assigns, pre_embs = emb_ind_info

        if self.gen_node_embeddings:
            node_tensor = torch.tensor(node_inputs, device=self.device)
            node_embs = self.node_embedder(node_tensor)
            if pre_assigns:
                emb_red = self.emb_map(pre_embs)
                emb_to_node_size = torch.Size([len(node_embs), len(emb_red)])
                adj_matr = get_adj_matr(pre_assigns, emb_to_node_size, 
                                        mean=False, is_cuda=self.is_cuda).to(self.device)
                eraser = 1. - torch.mm(adj_matr, torch.ones(emb_red.size()).to(self.device))
                node_embs = eraser * node_embs
                node_embs = torch.mm(adj_matr, emb_red) + node_embs
        else:
            node_embs = node_states

        orig_node_states = node_embs
        
        node_zero_vec = torch.zeros((1, len(orig_node_states[0])), device=self.device)
        node_states = torch.cat((orig_node_states, node_zero_vec), 0)
        
        edges_w_zv = edge_emb_inds + [self.edge_ct]
        edge_tensor = torch.tensor(edge_emb_inds + [self.edge_ct],
                                   device=self.device)
        
        node_reprs = torch.zeros(len(node_states), self.lstm_state_dim,
                                 device=self.device)
        node_mem = torch.zeros(len(node_states), self.lstm_state_dim,
                               device=self.device)

        node_to_node_sz = torch.Size([len(node_states), len(node_states)])
        
        W_i = self.nr_w_i(self.W_i(node_states))
        W_o = self.nr_w_o(self.W_o(node_states))
        W_c = self.nr_w_c(self.W_c(node_states))
        W_f = self.nr_w_f(self.W_f(node_states))
        
        for dir_upd_layer in upd_layers:

            diag = [(x, x) for x in set([y[0] for y in dir_upd_layer])]
            upd_diag = get_adj_matr(diag, node_to_node_sz,
                                    is_cuda=self.is_cuda).to(self.device)

            w_i = torch.mm(upd_diag, W_i)
            w_o = torch.mm(upd_diag, W_o)
            w_c = torch.mm(upd_diag, W_c)
            
            upd_layer = add_zv_to_no_deps(dir_upd_layer, len(node_states) - 1,
                                          len(edge_tensor) - 1)

            cat_upd_layer, i_matrs, o_matrs, c_matrs, f_matrs = [], [], [], [], []
            # we group together every update using the same edge matrix
            ext_upd_layer = [(ni, nj, eij, edges_w_zv[eij]) 
                             for ni, nj, eij in upd_layer]
            for grp_layer in group_similar_tup_sizes(ext_upd_layer, 3,
                                                     no_split=True):
                cont_layers, ind_map = [], {}
                for upd in grp_layer:
                    if not upd[3] in ind_map: 
                        ind_map[upd[3]] = len(cont_layers)
                        cont_layers.append([])
                    cont_layers[ind_map[upd[3]]].append(upd)
                cl_lens = [len(cl) for cl in cont_layers]
                max_sz = max(cl_lens)
                zv_tup = (len(node_states) - 1, len(node_states) - 1, 
                          len(edge_tensor) - 1, self.edge_ct)

                # we pad different for length matrix operations
                pad_layers = [cl + [zv_tup for _ in range(max_sz - len(cl))] 
                              for cl in cont_layers]
                cs_add_inds = torch.tensor([[add for _, add, _, _ in cl]
                                            for cl in pad_layers],device=self.device)

                # edge_lst is the minimal list of exactly which edges are to be used
                edge_lst = [cl[0][3] for cl in pad_layers]
                edge_inds = torch.tensor(edge_lst, device=self.device)

                # we get all the hidden states in the order they need to be 
                # multiplied and similarly with the edge matrices
                add_matrs = node_reprs[cs_add_inds]
                i_edge_matrs = self.U_i[edge_inds]
                o_edge_matrs = self.U_o[edge_inds]
                c_edge_matrs = self.U_c[edge_inds]
                f_edge_matrs = self.U_f[edge_inds]
                
                # we multiply the padded hidden states with the edge matrices
                i_res_matr = torch.bmm(add_matrs, i_edge_matrs)
                o_res_matr = torch.bmm(add_matrs, o_edge_matrs)
                c_res_matr = torch.bmm(add_matrs, c_edge_matrs)
                f_res_matr = torch.bmm(add_matrs, f_edge_matrs)
                
                # now we flatten our results into one matrix and discard the padding 
                for c_i, cl in enumerate(cont_layers):
                    cat_upd_layer.extend([(ni, nj, eij) for ni, nj, eij, _ in cl])
                    i_matrs.append(i_res_matr[c_i, :len(cl)])
                    o_matrs.append(o_res_matr[c_i, :len(cl)])
                    c_matrs.append(c_res_matr[c_i, :len(cl)])
                    f_matrs.append(f_res_matr[c_i, :len(cl)])

            i_edge_matr = self.nr_u_i(torch.cat(i_matrs, 0))
            o_edge_matr = self.nr_u_o(torch.cat(o_matrs, 0))
            c_edge_matr = self.nr_u_c(torch.cat(c_matrs, 0))

            # in the order specified by the sub-batching operation we just did
            # we now grab our input and argument results
            src_inds = torch.tensor([src for src, _, _ in cat_upd_layer],
                                    device=self.device)
            add_inds = torch.tensor([add for _, add, _ in cat_upd_layer],
                                    device=self.device)

            edge_to_node_sz = torch.Size([len(node_reprs), len(src_inds)])
            adj_pairs = [(x[0], acc_pos) for acc_pos, x in enumerate(cat_upd_layer)]
            adj_matr = get_adj_matr(adj_pairs, edge_to_node_sz,
                                    is_cuda=self.is_cuda).to(self.device)

            u_i = torch.mm(adj_matr, i_edge_matr)
            i_gate = nn.Sigmoid()(w_i + u_i + self.b_i)

            u_o = torch.mm(adj_matr, o_edge_matr)
            o_gate = nn.Sigmoid()(w_o + u_o + self.b_o)

            u_c = torch.mm(adj_matr, c_edge_matr)
            ch_gate = nn.Tanh()(w_c + u_c + self.b_c)

            u_f = self.nr_u_f(torch.cat(f_matrs, 0))
            w_f = W_f.index_select(0, src_inds)
            f_gate = nn.Sigmoid()(w_f + u_f + self.b_f)

            arg_mem = node_mem.index_select(0, add_inds)
            par_mem = i_gate * ch_gate + torch.mm(adj_matr, f_gate * arg_mem)

            # ensure only memory cells in current layer are updated
            restr_par_mem = torch.mm(upd_diag, par_mem)
            node_mem = torch.add(node_mem, restr_par_mem)
            
            # ensure only node hidden states in current layer are updated
            out_reprs = o_gate * nn.Tanh()(self.nr_cell(node_mem))
            new_node_reprs = torch.mm(upd_diag, out_reprs)

            node_reprs = torch.add(node_reprs, new_node_reprs)

        # removing 0 vector
        node_reprs = node_reprs[:len(node_reprs)-1]

        return node_reprs

###
# Basic GNN classes
###

class MPNN(nn.Module):
    # message passing neural network
    def __init__(self, **kw_args):
        super().__init__()
        self.device = kw_args['device']
        self.is_cuda = (self.device != torch.device('cpu'))
        self.node_emb_dim = kw_args['node_emb_dim']
        self.edge_emb_dim = kw_args['edge_emb_dim']
        self.node_state_dim = kw_args['node_state_dim']
        self.edge_state_dim = kw_args['edge_state_dim']
        self.node_ct = kw_args['node_ct']
        self.edge_ct = kw_args['edge_ct']
        sparse_grads = kw_args['sparse_grads']

        self.dropout = kw_args['dropout']

        self.pretrained_embs = kw_args['pretrained_embs']
        if self.pretrained_embs:
            #self.emb_map = nn.Identity()
            self.emb_map = nn.Sequential(*[nn.BatchNorm1d(kw_args['pretrained_emb_dim']),
                                           MLP(kw_args['pretrained_emb_dim'], self.node_state_dim,
                                               mlp_layers=2,dropout=self.dropout)]).to(self.device)
            self.emb_map = nn.BatchNorm1d(kw_args['pretrained_emb_dim'])

        
        
        self.num_rounds = kw_args['num_rounds']
        self.aggr_type = kw_args['aggr_type']
        
        self.gen_node_embeddings = not 'node_embedder' in kw_args
        if self.gen_node_embeddings:
            self.node_embedder = EmbProj(self.node_ct, self.node_emb_dim,
                                         self.node_state_dim, device=self.device,
                                         sparse_grads=sparse_grads)
        self.par_edge_embedder = EmbProj(self.edge_ct, self.edge_emb_dim, 
                                         self.edge_state_dim, device=self.device,
                                         sparse_grads=sparse_grads)
        self.arg_edge_embedder = EmbProj(self.edge_ct, self.edge_emb_dim, 
                                         self.edge_state_dim, device=self.device,
                                         sparse_grads=sparse_grads)

        layers = []
        for _ in range(self.num_rounds):
            par_messenger = MLP(self.node_state_dim * 2 + self.edge_state_dim,
                                self.node_state_dim, device=self.device,
                                dropout=self.dropout)
            arg_messenger = MLP(self.node_state_dim * 2 + self.edge_state_dim,
                                self.node_state_dim, device=self.device,
                                dropout=self.dropout)
            accumulator = MLP(self.node_state_dim * 3, self.node_state_dim,
                              device=self.device, dropout=self.dropout)
            layer = nn.ModuleList([par_messenger, arg_messenger, accumulator])
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def compute_node_reprs(self, emb_ind_info, upd_layers):
        node_inputs, edge_emb_inds, type_emb_inds, depth_emb_inds, pre_assigns, pre_embs = emb_ind_info
        # assumes upd_layers contains topologically sorted updates
        par_updates = [upd for upd_l in flip_upd_layers(upd_layers) for upd in upd_l]
        arg_updates = [upd for upd_l in upd_layers for upd in upd_l]
        if self.gen_node_embeddings:
            node_tensor = torch.tensor(node_inputs, device=self.device)
            node_embs = self.node_embedder(node_tensor)
            if pre_assigns:
                emb_red = self.emb_map(pre_embs)
                emb_to_node_size = torch.Size([len(node_embs), len(emb_red)])
                adj_matr = get_adj_matr(pre_assigns, emb_to_node_size, 
                                        mean=False, is_cuda=self.is_cuda).to(self.device)
                eraser = 1. - torch.mm(adj_matr, torch.ones(emb_red.size()).to(self.device))
                node_embs = eraser * node_embs
                node_embs = torch.mm(adj_matr, emb_red) + node_embs
        else:
            node_embs = node_inputs
            
        node_states = node_embs

        node_zero_vec = torch.zeros((1, len(node_states[0])), device=self.device)
        node_states = torch.cat((node_states, node_zero_vec), 0)
        
        node_to_node_size = torch.Size([len(node_states),len(node_states)])
        
        # edges will be zero vector for leaf / root
        edge_zero_vec = torch.zeros((1, self.edge_state_dim), device=self.device)
        edge_tensor = torch.tensor(edge_emb_inds, device=self.device)

        # handling parent / argument edges
        upd_sets = []
        for edge_embedder, par_arg_updates in [[self.par_edge_embedder,par_updates],
                                               [self.arg_edge_embedder,arg_updates]]:
            edge_states = edge_embedder(edge_tensor)
            edge_states = torch.cat((edge_states, edge_zero_vec), 0)
            updates = add_zv_to_no_deps(par_arg_updates, len(node_states) - 1,
                                        len(edge_states) - 1)
            src_inds = torch.tensor([ni for ni, _, _ in updates], device=self.device)
            add_inds = torch.tensor([nj for _, nj, _ in updates], device=self.device)
            edge_inds = torch.tensor([eij for _, _, eij in updates], device=self.device)
            upd_sets.append([edge_states, updates, src_inds, add_inds, edge_inds])
            
        # now doing message passing
        for par_messenger, arg_messenger, accumulator in self.layers:
            msg_lst, par_arg_msgs = [par_messenger, arg_messenger], []
            for messenger, upd_set in zip(msg_lst, upd_sets):
                (edge_states, updates, src_inds, add_inds, edge_inds) = upd_set
                src_matr = node_states.index_select(0, src_inds)
                add_matr = node_states.index_select(0, add_inds)
                edge_matr = edge_states.index_select(0, edge_inds)
                msg_matr = torch.cat((src_matr, add_matr, edge_matr), 1)
                msgs = messenger(msg_matr)
                edge_to_node_size = torch.Size([len(node_states), len(msgs)])
                adj_pairs = [(x[0], acc_pos) for acc_pos, x in enumerate(updates)]
                adj_matr = get_adj_matr(adj_pairs, edge_to_node_size, 
                                        mean=self.aggr_type=='mean',  
                                        is_cuda=self.is_cuda).to(self.device)
                par_arg_msgs.append(torch.mm(adj_matr, msgs))
                
            acc_matr = torch.cat([node_states] + par_arg_msgs, 1)
            acc = accumulator(acc_matr)
            node_states = torch.add(node_states, acc)

        # removing 0 vector
        node_states = node_states[:len(node_states) - 1]
        
        return node_states

class GCN(nn.Module):
    # message passing neural network
    def __init__(self, **kw_args):
        super().__init__()
        self.device = kw_args['device']
        self.is_cuda = (self.device != torch.device('cpu'))
        self.node_emb_dim = kw_args['node_emb_dim']
        self.node_state_dim = kw_args['node_state_dim']
        self.node_ct = kw_args['node_ct']
        sparse_grads = kw_args['sparse_grads']
        self.pretrained_embs = kw_args['pretrained_embs']

        self.dropout = kw_args['dropout']
        
        if self.pretrained_embs:
            #self.emb_map = nn.Identity()
            self.emb_map = nn.Sequential(*[nn.BatchNorm1d(kw_args['pretrained_emb_dim']),
                                           MLP(kw_args['pretrained_emb_dim'], self.node_state_dim,
                                               mlp_layers=2,dropout=self.dropout)]).to(self.device)
            self.emb_map = nn.BatchNorm1d(kw_args['pretrained_emb_dim'])

            
        self.num_rounds = kw_args['num_rounds']
        self.aggr_type = kw_args['aggr_type']
        
        self.gen_node_embeddings = not 'node_embedder' in kw_args
        if self.gen_node_embeddings:
            self.node_embedder = EmbProj(self.node_ct, self.node_emb_dim,
                                         self.node_state_dim, device=self.device,
                                         sparse_grads=sparse_grads)
        
        layers = []
        for _ in range(self.num_rounds):
            accumulator = nn.Linear(self.node_state_dim, self.node_state_dim,
                                    bias=False).to(self.device)
            layer = nn.Sequential(*[accumulator, nn.ReLU()])
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def compute_node_reprs(self, emb_ind_info, upd_layers):
        node_inputs, edge_emb_inds, type_emb_inds, depth_emb_inds, pre_assigns, pre_embs = emb_ind_info
        # assumes upd_layers contains topologically sorted updates
        par_updates = [upd for upd_l in flip_upd_layers(upd_layers) for upd in upd_l]
        arg_updates = [upd for upd_l in upd_layers for upd in upd_l]
        updates = [upd for upd in list(set(par_updates + arg_updates))
                   if upd[1] != None]
        if self.gen_node_embeddings:
            node_tensor = torch.tensor(node_inputs, device=self.device)
            node_embs = self.node_embedder(node_tensor)
            if pre_assigns:
                emb_red = self.emb_map(pre_embs)
                emb_to_node_size = torch.Size([len(node_embs), len(emb_red)])
                adj_matr = get_adj_matr(pre_assigns, emb_to_node_size, 
                                        mean=False, is_cuda=self.is_cuda).to(self.device)
                eraser = 1. - torch.mm(adj_matr, torch.ones(emb_red.size()).to(self.device))
                node_embs = eraser * node_embs
                node_embs = torch.mm(adj_matr, emb_red) + node_embs
        else:
            node_embs = node_inputs
        node_states = node_embs

        node_zero_vec = torch.zeros((1, len(node_states[0])), device=self.device)
        node_states = torch.cat((node_states, node_zero_vec), 0)

        nodes = set([ni for ni, nj, _ in updates if ni != nj])
        # add self connections 
        for ni in nodes: updates.append((ni, ni, None))
        add_inds = torch.tensor([nj for _, nj, _ in updates], device=self.device)
        
        out_deg = {}
        for ni, _, _ in updates:
            if not ni in out_deg: out_deg[ni] = 0
            out_deg[ni] += 1

        gcn_agg = {}
        for ni, od in out_deg.items(): gcn_agg[(0, ni)] = np.sqrt(out_deg[ni])
        for acc_pos, (_, nj, _) in enumerate(updates):
            gcn_agg[(1, acc_pos)] = np.sqrt(out_deg[nj])

        # now doing message passing
        for neigh_agg_layer in self.layers:
            add_matr = node_states.index_select(0, add_inds)
            adj_pairs = [(x[0], acc_pos) for acc_pos, x in enumerate(updates)]
            edge_to_node_size = torch.Size([len(node_states), len(add_matr)])
            adj_matr = get_adj_matr(adj_pairs, edge_to_node_size,
                                    gcn_agg=gcn_agg, 
                                    is_cuda=self.is_cuda).to(self.device)
            agg = torch.mm(adj_matr, add_matr)
            node_states = neigh_agg_layer(agg)

        # removing 0 vector
        node_states = node_states[:len(node_states) - 1]
        
        return node_states


