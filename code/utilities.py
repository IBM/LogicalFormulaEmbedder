# python imports
import sys, signal, math, copy, random, os, re
# torch imports
import torch
import torch.nn as nn
# numpy imports
import numpy as np
# sklearn imports
import sklearn.cluster as skcl
# graphviz imports
from graphviz import Digraph
import networkx as nx
# natural language imports
import spacy
import penman
# code imports
import code.node_classes as nc

#################
# General utilities
#################

def uniquify(lst):
    seen = set()
    seen_add = seen.add
    return [x for x in lst if not (x in seen or seen_add(x))]

def position(el, lst, key=None):
    if key:
        return next(i for i, x in enumerate(lst) if key(x) == key(el))
    return next(i for i, x in enumerate(lst) if x == el)

def calc_metrics(true_pos, false_pos, false_neg):
    prec_denom = (true_pos + false_pos) if (true_pos + false_pos) > 0 else 1
    precision = true_pos / prec_denom
    rec_denom = (true_pos + false_neg) if (true_pos + false_neg) > 0 else 1
    recall = true_pos / rec_denom
    pr_denom = (precision + recall) if (precision + recall) > 0 else 1
    f1 = 2 * precision * recall / pr_denom
    return precision, recall, f1

class TimeoutError(Exception): pass

def timeout(func, args=(), kwargs={}, duration=1):
    def handler(signum, frame):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, handler) 
    signal.alarm(duration)
    try:
        result = func(*args, **kwargs)
    finally:
        signal.alarm(0)
    return result

def deconstruct_expr(expr, par_d=None):
    if par_d == None: par_d = {}
    # assumes s-expr where expr[0] is NOT a nested expression
    ret_set = set([expr])
    if type(expr) == tuple:
        assert type(expr[0]) != tuple
        for i, el in enumerate(expr):
            if i == 0: continue
            if not el in par_d: par_d[el] = set()
            par_d[el].add(expr)
            n_els, _ = deconstruct_expr(el, par_d)
            ret_set = ret_set.union(n_els)
    return ret_set, par_d

def get_ancestor_label_chains(orig_expr, par_dict, src_graph=None, depth=None, ret_all=True):
    def key_form(expr):
        label_of = expr[0] if type(expr) == tuple else expr
        if src_graph != None: return (label_of, src_graph.nodes[expr]['type'])
        else: return label_of
    expl = [[orig_expr]]
    ret_lsts = set()
    while expl:
        curr = expl.pop()
        k_f = curr[:-1] + [key_form(c) for c in curr[-1:]]
        last = curr[len(curr) - 1]
        if last in par_dict and (depth == None or len(curr) < depth):
            if ret_all:
                ret_lsts.add(tuple(k_f[1:]))
            for p in par_dict[last]:
                expl.append(k_f + [p])
        elif len(k_f) > 1:
            ret_lsts.add(tuple(k_f[1:]))
    return ret_lsts

def make_anon_formula(expr, src_graph):
    new_tup = []
    if type(expr) == tuple:
        assert type(expr[0]) != tuple
        for i, el in enumerate(expr):
            new_tup.append(make_anon_formula(el, src_graph))
        return tuple(new_tup)
    elif expr in src_graph.nodes and src_graph.nodes[expr]['type'] == nc.VarType:
        return 'VAR'
    return expr

def formula_elements_lst(expr, src_graph, anon_var=True, anon_leaf=False, depth=1000):
    els = []
    if depth == 0:
        return expr if type(expr) != tuple else expr[0]
    elif type(expr) == tuple:
        if expr in src_graph.nodes[expr]['type'] == nc.SkolemFuncType:
            return ['VAR']
        els.append(expr[0])
        for el in expr:
            els.extend(formula_elements_lst(el, src_graph, depth=depth-1))
        return els
    elif anon_leaf:
        return ['LEAF']
    elif expr in src_graph.nodes and anon_var and \
         src_graph.nodes[expr]['type'] in [nc.VarType, nc.SkolemConstType]:
        return ['VAR']
    return [expr]
 
def make_debrujin_formula(expr, src_graph, assignment=None):
    if assignment == None: assignment = [0, {}]
    new_tup = []
    if type(expr) == tuple:
        assert type(expr[0]) != tuple
        for i, el in enumerate(expr):
            new_tup.append(make_debrujin_formula(el, src_graph, assignment))
        return tuple(new_tup)
    elif expr in src_graph.nodes and src_graph.nodes[expr]['type'] == nc.VarType:
        if not expr in assignment[1]: 
            assignment[0] += 1
            assignment[1][expr] = 'VAR_' + str(assignment[0])
        return assignment[1][expr]
    return expr

def group_similar_tup_sizes(tuples, key_in=0, no_split=False, grp_sp=10, min_bk=10):
    if no_split: return [tuples]
    indiv_buckets = {}
    for tup in tuples:
        src = tup[key_in]
        if not src in indiv_buckets: indiv_buckets[src] = []
        indiv_buckets[src].append(tup)
    buckets = {}
    for src, tups in indiv_buckets.items():
        if len(tups) <= min_bk: bucket_id = -min_bk
        else: bucket_id = round(len(tups) / grp_sp)
        if not bucket_id in buckets: buckets[bucket_id] = []
        buckets[bucket_id].extend(tups)
    return list(buckets.values())

#################
# Expression variant checking
#################

def is_alpha_equiv(conj_expr, conj_graph, prem_expr, prem_graph, depth=1000):
    if conj_expr == prem_expr: return True
    if var_check(conj_expr, conj_graph) == var_check(prem_expr, prem_graph):
        return True
    if type(conj_expr) == type(prem_expr) and \
       type(conj_expr) != tuple:
        # if one is variable and one isn't, return False
        if var_check(conj_expr, conj_graph) != var_check(prem_expr, prem_graph):
            return False
        # otherwise, return True if variable and False if not
        return var_check(conj_expr, conj_graph)
    # this happens more than you would think...
    hme = hash_matching_exprs(conj_expr, conj_graph, prem_expr, prem_graph, depth=depth)
    return hme

def is_prob_iso(conj_expr, conj_graph, prem_expr, prem_graph, depth=1000):
    if type(conj_expr) == type(prem_expr) and \
       type(conj_expr) != tuple:
        return var_check(conj_expr, conj_graph) == var_check(prem_expr, prem_graph)
    if conj_expr == prem_expr: return True
    hme = hash_matching_exprs(conj_expr, conj_graph, prem_expr, prem_graph,
                              use_labels=False, const_matching=True, ignore_ord=True,
                              depth=depth)
    return hme

def hash_matching_exprs(conj_expr, conj_graph, prem_expr, prem_graph,
                        use_labels=True, const_matching=False, ignore_ord=False,
                        depth=1000):
    # getting ent hashes here
    conj_hashes, prem_hashes = {}, {}
    extract_var_hashes(conj_expr, conj_graph, conj_hashes,
                       ignore_ord=ignore_ord, use_labels=use_labels, depth=depth)
    extract_var_hashes(prem_expr, prem_graph, prem_hashes,
                       ignore_ord=ignore_ord, use_labels=use_labels, depth=depth)
    # we require a perfect bipartite matching to be considered alpha-equivalent
    if len(conj_hashes.keys()) != len(prem_hashes.keys()): return False
    assignments = set()
    for c_ent, c_hv in conj_hashes.items():
        c_ent_f = c_ent if type(c_ent) != tuple else c_ent[0]
        found = False
        for p_ent, p_hv in prem_hashes.items():
            if c_hv == p_hv:
                # just a sanity check, probably isn't necessary
                if (not const_matching) and \
                   var_check(c_ent, conj_graph) != var_check(p_ent, prem_graph):
                    return False
                p_ent_f = p_ent if type(p_ent) != tuple else p_ent[0]
                if const_matching or var_check(c_ent, conj_graph) or c_ent_f == p_ent_f:
                    found = p_ent
                    break
                else: return False
        if found == False: return False
        assignments.add((c_ent, found))
        del prem_hashes[p_ent]
    # if we get here, prem_entity_hashes should be empty
    return not prem_hashes

def extract_var_hashes(expr, graph, hashes, src_hash=None, ignore_ord=False,
                       use_labels=True, depth=10000):
    if src_hash == None: src_hash = 0
    new_tup = []
    gn = graph.nodes[expr]
    if (not var_check(expr, graph)) and type(expr) == tuple and depth > 0:
        if use_labels: lead = (gn['label'], gn['type'], len(expr))
        else: lead = (gn['type'], len(expr))
        for el in expr[1:]:
            # partial ordering edge labels will account for orderedness of lead
            if ignore_ord: edge_hash = hash(lead)
            else: edge_hash = hash((lead, graph.edges[expr, el]['label']))
            new_src_hash = hash(src_hash + edge_hash)
            extract_var_hashes(el, graph, hashes, new_src_hash, ignore_ord=ignore_ord,
                               use_labels=use_labels, depth=depth-1)
    else:
        if var_check(expr, graph): label = 'VAR'
        else: label = gn['label']
        if use_labels: lead = (label, gn['type'], 0)
        else: lead = ('const', 0)
        if not expr in hashes: hashes[expr] = hash(lead)
        hashes[expr] += hash(src_hash + hashes[expr])

def var_check(expr, graph):
    if type(expr) == tuple:
        return graph.nodes[expr]['type'] in [nc.SkolemFuncType]
    return graph.nodes[expr]['type'] in [nc.VarType, nc.SkolemConstType]

#################
# Variable compression
#################

def is_rn_var(expr):
    return type(expr) != tuple and 'SYM_EXT' in expr

def get_var_name(v):
    return v.split('_SYM_EXT_')[0]

def get_av_els(expr):
    if type(expr) == tuple:
        els = [expr[0]]
        for el in expr: els.extend(get_av_els(el))
        return els
    elif is_rn_var(expr): return ['VAR']
    else: return [expr]

def get_av_key(expr):
    return tuple(sorted(list(get_av_els(expr))))

def var_compress_stmts(exprs):
    def add_to_e_info(expr, expr_info):
        expr_info[expr] = {}
        subexprs, par_info = deconstruct_expr(expr)
        for se in subexprs:
            av_k = hash(get_av_key(se))
            if not av_k in expr_info[expr]: expr_info[expr][av_k] = []
            expr_info[expr][av_k].append(se)
        return subexprs, par_info
    expr_info, all_subexpr_info = {}, []
    for expr in exprs: all_subexpr_info.append(add_to_e_info(expr, expr_info))
    var_info = {}
    for expr, (subexprs, par_info) in zip(exprs, all_subexpr_info):
        var_info[expr] = {}
        for se in subexprs:
            if is_rn_var(se):
                l_chains = get_ancestor_label_chains(se, par_info, depth=2, ret_all=False)
                var_info[expr][se] = l_chains
                #var_info[expr][se] = np.max([len(x) for x in l_chains])
                #var_info[expr][se] = np.mean([len(x) for x in l_chains])
                #var_info[expr][se] = np.min([len(x) for x in l_chains])
                #var_info[expr][se] = len(l_chains)
    wts, var_hashes = {}, {}
    lg_e_ind = max([(i, max([len(get_av_key(se)) for se in se_lst]))
                    for i, (se_lst, _) in enumerate(all_subexpr_info)],
                   key=lambda x : x[1])[0]
    lg_expr, lg_subexpr_info = exprs[lg_e_ind], all_subexpr_info[lg_e_ind]
    expr_size = {}
    for p_i, (expr, subexpr_info) in enumerate(zip(exprs, all_subexpr_info)):
        all_expr_subexprs, par_info = subexpr_info
        expr_subexprs = []
        for se in all_expr_subexprs:
            av_k = get_av_key(se)
            expr_subexprs.append((av_k, hash(av_k), se))
        expr_subexprs = sorted(expr_subexprs, key=lambda x : len(x[0]),
                               reverse=True)
        expr_size[p_i] = len(expr_subexprs[0][0])
        substs = []
        for a_i, alt_expr in enumerate(exprs):
            if a_i <= p_i: continue
            else:
                supp_sc, subst = find_good_subst(expr_subexprs, expr_info[alt_expr],
                                                 var_info[expr], var_info[alt_expr], var_hashes)
                substs.append((supp_sc, subst, a_i))
        if not p_i in wts: wts[p_i] = {}
        wts[p_i][p_i] = (0., {})
        for score, subst, a_i in substs:
            if not a_i in wts: wts[a_i] = {}
            wts[a_i][p_i] = (score, subst)
            wts[p_i][a_i] = (score, dict([(v, k) for k, v in subst.items()]))
    sym_matr = [[None for _ in range(len(exprs))] for _ in range(len(exprs))]
    norm_c = max([wts[i][j][0] for i in range(len(exprs)) for j in range(len(exprs))])
    if norm_c == 0: norm_c = 1
    for i in range(len(exprs)):
        sym_matr[i][i] = 1.
        max_i = max([wts[i][k][0] for k in range(len(exprs))])
        for j in range(len(exprs)):
            if i == j: continue
            max_j = max([wts[k][j][0] for k in range(len(exprs))])
            sym_matr[i][j] = wts[i][j][0] / norm_c
    sym_matr = 1. - np.matrix(sym_matr)
    db = skcl.DBSCAN(eps=0.1, min_samples=2, metric='precomputed')
    cluster_inds = db.fit_predict(sym_matr)
    #cluster_inds = [1 for _ in range(len(exprs))]
    clusters = {}
    for i, cl in enumerate(cluster_inds):
        if not cl in clusters: clusters[cl] = []
        clusters[cl].append(i)
    new_stmts = [None for _ in range(len(exprs))]
    for cl, inds in clusters.items():
        medioid = [-math.inf, -1]
        for i in range(len(exprs)):
            s_sum = sum([wts[i][j][0] for j in range(len(exprs))])
            if s_sum > medioid[0]: medioid = [s_sum, i]
        md_ind = medioid[1]
        #md_ind = 0
        for j in inds:
            sc, subst = wts[md_ind][j]
            new_expr = apply_subst(exprs[j], subst)
            new_stmts[j] = new_expr
    return new_stmts

def find_good_subst(expr_subexprs, alt_info, expr_par_info, alt_par_info,
                    var_hashes):
    # build up substitution for each expression
    subst, supp_by, change, nogoods = {}, [], True, set()
    all_vars = set([s for a, h, s in expr_subexprs if a == ('VAR',)])
    while all_vars:
        best_subst = [0, subst]
        for av_k, hash_av_k, subexpr in expr_subexprs:
            if type(subexpr) != tuple: continue
            if not hash_av_k in alt_info: continue
            if not subexpr in var_hashes:
                var_hashes[subexpr] = {}
                get_var_hashes(subexpr, var_hashes[subexpr])
            s1_hashes = var_hashes[subexpr]
            for a_i, alt_subexpr in enumerate(alt_info[hash_av_k]):
                if (hash_av_k, a_i) in nogoods: continue
                if is_rn_var(alt_subexpr):
                    if get_var_name(alt_subexpr) == get_var_name(subexpr): a_sc = 2
                    else: a_sc = 1
                    a_sc = match_par_info(expr_par_info[subexpr],
                                           alt_par_info[alt_subexpr])
                else: a_sc = len(av_k)
                if a_sc <= best_subst[0]: continue
                if not alt_subexpr in var_hashes:
                    var_hashes[alt_subexpr] = {}
                    get_var_hashes(alt_subexpr, var_hashes[alt_subexpr])
                s2_hashes = var_hashes[alt_subexpr]
                fnd_subst = find_valid_subst(s1_hashes, s2_hashes, dict(subst))
                if fnd_subst != False and any(not k in subst for k in fnd_subst.keys()):
                    best_subst = [a_sc, fnd_subst]
                else: nogoods.add((hash_av_k, a_i))
        # exit if nothing found
        if best_subst[0] == 0: break
        supp_by.append(best_subst[0])
        subst = best_subst[1]
        for k in subst.keys():
            if k in all_vars: all_vars.remove(k)
    return np.sum(supp_by), subst

def match_par_info(se1_info, se2_info):
    #return 1 / (1 + abs(se1_info - se2_info))
    ct = 0
    for p in se1_info:
        if p in se2_info:
            ct += 1
    return ct

def find_good_subst_2(expr_subexprs, alt_info, expr_par_info, alt_par_info,
                    var_hashes):
    # build up substitution for each expression
    subst, supp_by = {}, []
    for av_k, hash_av_k, subexpr in expr_subexprs:
        if type(subexpr) == tuple: continue
        if not hash_av_k in alt_info: continue
        if not subexpr in var_hashes:
            var_hashes[subexpr] = {}
            get_var_hashes(subexpr, var_hashes[subexpr])
        s1_hashes = var_hashes[subexpr]
        for alt_subexpr in alt_info[hash_av_k]:
            if not alt_subexpr in var_hashes:
                var_hashes[alt_subexpr] = {}
                get_var_hashes(alt_subexpr, var_hashes[alt_subexpr])
            s2_hashes = var_hashes[alt_subexpr]
            fnd_subst = find_valid_subst(s1_hashes, s2_hashes, dict(subst))
            if fnd_subst != False and any(not k in subst for k in fnd_subst.keys()):
                subst = fnd_subst
                supp_by.append(len(av_k))
    return np.sum(supp_by), subst

def apply_subst(expr, subst):
    if type(expr) == tuple:
        new_expr = [expr[0]]
        for i, el in enumerate(expr):
            if i == 0: continue
            new_expr.append(apply_subst(el, subst))
        return tuple(new_expr)
    elif expr in subst: return subst[expr] + ''
    else: return expr + ''

def find_valid_subst(e1_hashes, e2_hashes, subst=None):
    if subst == None: subst = {}
    rev_subst = dict([(v, k) for k, v in subst.items()])
    e1_hashes, e2_hashes = dict(e1_hashes), dict(e2_hashes)
    # we require a perfect bipartite matching to be considered alpha-equivalent
    if len(e1_hashes.keys()) != len(e2_hashes.keys()): return False
    assignments = set()
    for ent1, c_hv in e1_hashes.items():
        found = False
        for ent2, p_hv in e2_hashes.items():
            if c_hv != p_hv: continue
            if is_rn_var(ent1) != is_rn_var(ent2): continue
            if ent1 in subst and subst[ent1] != ent2: continue
            if ent2 in rev_subst and rev_subst[ent2] != ent1: continue
            if is_rn_var(ent1) or ent1 == ent2:
                found = ent2
                break
        if found == False: return False
        assignments.add((ent1, found))
        del e2_hashes[found]
    # if we get here, prem_entity_hashes should be empty
    if e2_hashes != {}: return False
    for a, b in assignments: subst[a] = b
    return subst

def get_var_hashes(expr, hashes, src_hash=None):
    if src_hash == None: src_hash = 0
    new_tup = []
    if (not is_rn_var(expr)) and type(expr) == tuple:
        lead = (expr[0], len(expr))
        for i, el in enumerate(expr):
            if i == 0: continue
            # partial ordering edge labels will account for orderedness of lead
            edge_hash = hash((lead, i))
            new_src_hash = hash(src_hash + edge_hash)
            get_var_hashes(el, hashes, new_src_hash)
    else:
        if is_rn_var(expr): lead = ('VAR', 0)
        else: lead = (expr, 0)
        if not expr in hashes: hashes[expr] = hash(lead)
        hashes[expr] += hash(src_hash + hashes[expr])
        
#################
# Matching utilities
#################

def maximal_var_subst(paths1, paths2):
    all_wts = []
    for p_k, p_paths in paths1.items():
        if not is_rn_var(p_k): continue
        for c_k, c_paths in paths2.items():
            pc_wt = get_alignment_score(p_paths, c_paths)
            if pc_wt > 0: all_wts.append((p_k, c_k, pc_wt))
    score_of, var_subst = 0, {}
    while all_wts:
        best_l, best_r, best_wt = max(all_wts, key=lambda x : x[2])
        var_subst[best_l] = best_r
        all_wts = [(l, r, w) for l, r, w in all_wts
                   if l != best_l and r != best_r]
        score_of += best_wt
    return score_of, var_subst

def get_alignment_score(p_paths, c_paths, cos=True):
    dot_prod = sparse_dot_prod(p_paths, c_paths)
    if cos:
        n1 = np.sqrt(sum([pow(x[1], 2) for x in p_paths]))
        n2 = np.sqrt(sum([pow(x[1], 2) for x in c_paths]))
        if n1 * n2 == 0: score = 0
        else: score = dot_prod / (n1 * n2)
    else:
        score = dot_prod
    return score

def sparse_dot_prod(lst1, lst2):
    dot_prod, i, j = 0, 0, 0
    while i < len(lst1) and j < len(lst2):
        if lst1[i][0] == lst2[j][0]:
            dot_prod += lst1[i][1] * lst2[j][1]
            i += 1
            j += 1
        elif lst1[i][0] < lst2[j][0]: i += 1
        else: j += 1
    return dot_prod
        
def get_paths_upto(set_lst, prov, path_len=3, just_syms=True, dp_form=True, all_len=True):
    paths = [[s_l] for s_l in set_lst]
    fin_paths = []
    for i in range(path_len - 1):
        new_paths = []
        for p in paths:
            last_el = p[-1]
            if last_el in prov:
                for new_el in prov[last_el]:
                    new_paths.append(p + [new_el])
            if all_len:
                fin_paths.append(p)
        paths = new_paths
    ret_paths = fin_paths + paths
    if just_syms: ret_paths = [[(r[0] if type(r) == tuple else r) for r in p]
                               for p in ret_paths]
    if dp_form:
        d = {}
        for el in ret_paths:
            k = '___'.join(el)
            if not k in d: d[k] = 0
            d[k] += 1
        return sorted(d.items(), key=lambda x : x[0])
    return ret_paths
    
#################
# Graph utilities
#################

def topologically_group(graph):
    par_dict = {}
    for node in graph.nodes:
        if not node in par_dict: par_dict[node] = set()
        for par in graph.predecessors(node):
            par_dict[node].add(par)
        # should be redundant, but just in case...
        for arg in graph.successors(node):
            if not arg in par_dict: par_dict[arg] = set()
            par_dict[arg].add(node)

    # actual layers
    update_layers = []
    rem_nodes = list(graph.nodes) + []
    while rem_nodes:
        layer_nodes = [node for node in rem_nodes if not par_dict[node]]
        for node in layer_nodes:
            for arg in graph.successors(node):
                if node in par_dict[arg]: par_dict[arg].remove(node)
        rem_nodes = [node for node in rem_nodes if not node in layer_nodes]
        update_layers.append(layer_nodes)

    # ensures leaf nodes are in the very first layer
    # and root nodes in the very last
    leaf_nodes, non_leaf_nodes = [], []
    for layer in reversed(update_layers):
        new_layer = []
        for node in layer:
            if graph.out_degree(node):
                new_layer.append(node)
            else:
                leaf_nodes.append(node)
        if new_layer:
            non_leaf_nodes.append(new_layer)
    assert len(set([el for lst in ([leaf_nodes] + non_leaf_nodes) for el in lst])) == len(graph.nodes)
    return [leaf_nodes] + non_leaf_nodes

#################
# Encoder utilities
#################

def flip_upd_layers(upd_layers):
    new_upd_layers = []
    restr_upd_layers = [[(a, d, e) for a, d, e in upd_layer if d != None]
                        for upd_layer in upd_layers]
    restr_upd_layers = [upd_layer for upd_layer in restr_upd_layers if upd_layer]
    desc = set([y for upd_layer in restr_upd_layers for _, y, _ in upd_layer])
    asc = set([x for upd_layer in restr_upd_layers for x, _, _ in upd_layer])
    roots = [(x, None, None) for x in asc.difference(desc)]
    for upd_layer in reversed(restr_upd_layers):
        new_upd_layers.append([(d, a, e) for a, d, e in upd_layer])
    return [x for x in ([roots] + new_upd_layers) if x]

def add_zv_to_no_deps(dir_upd_layer, node_zv, edge_zv):
    upd_layer = []
    for src, add, edge in dir_upd_layer:
        add_triple = (src, add, edge)
        if add == None: add_triple = (src, node_zv, edge_zv)
        upd_layer.append(add_triple)
    return upd_layer

#################
# PyTorch utilities
#################

def get_adj_matr(pairs, size, is_cuda=False, mean=False, gcn_agg=None):
    if is_cuda:
        i = torch.cuda.LongTensor(pairs)
    else:
        i = torch.LongTensor(pairs)
    if gcn_agg != None:
        n_lst = [1 / (gcn_agg[(0, src)] * gcn_agg[(1, add)])
                 for src, add in pairs]
        if is_cuda: v = torch.cuda.FloatTensor(n_lst)
        else: v = torch.FloatTensor(n_lst)
    elif mean:
        src_ct = {}
        for src, _ in pairs:
            if not src in src_ct: src_ct[src] = 0
            src_ct[src] += 1
        if is_cuda:
            v = torch.cuda.FloatTensor([1 / src_ct[src] for src, _ in pairs])
        else:
            v = torch.FloatTensor([1 / src_ct[src] for src, _ in pairs])
    else:
        if is_cuda:
            v = torch.cuda.FloatTensor([1 for _ in range(len(pairs))])
        else:
            v = torch.FloatTensor([1 for _ in range(len(pairs))])
    if is_cuda:
        return torch.cuda.sparse.FloatTensor(i.t(), v, size)
    return torch.sparse.FloatTensor(i.t(), v, size)

def compute_att_aggr(node_matr, pairs, W_q, W_k, b_q, device, softmax=True):
    all_ms, at_src = [], None
    for src, tgt in pairs:
        if src != at_src:
            if at_src != None: all_ms.append(bmm_lst)
            bmm_lst, at_src = [], src
        bmm_lst.append(tgt)
    if at_src != None: all_ms.append(bmm_lst)
    ch_lens = [len(lst) for lst in all_ms]
    if not ch_lens: return None
    max_len = max(ch_lens)
    src_bmm_tensor = torch.tensor(uniquify([src for src, _ in pairs]),
                                  device=device)
    mask, zv_added = [], []
    for lst in all_ms:
        if len(lst) == max_len:
            mask.append(torch.zeros(len(lst), device=device))
            zv_added.append(torch.tensor(lst, device=device))
        else:
            zv = torch.zeros(len(lst), device=device)
            ov = torch.ones(max_len - len(lst), device=device)
            zo_tensor = torch.cat((zv, ov), 0)
            mask.append(zo_tensor)
            # this doesn't matter because we mask it anyway
            padding = [0 for _ in range(max_len - len(lst))]
            zv_added.append(torch.tensor(lst + padding, device=device))
    mask = torch.stack(mask).unsqueeze(1)
    bmm_tgt = W_k(torch.stack([node_matr.index_select(0, x) for x in zv_added]))
    bmm_src = W_q(node_matr.index_select(0, src_bmm_tensor).unsqueeze(1))
    att_matr = bmm_src.matmul(bmm_tgt.transpose(1, 2)) / b_q

    mask_matr = att_matr.masked_fill(mask==True, float('-inf'))
    if softmax: probs = nn.Softmax(dim=2)(mask_matr).squeeze(1)
    else: probs = mask_matr.squeeze(1)
    exp_rngs = []
    for ch_len in ch_lens:
        a_rng = torch.arange(ch_len, device=device)
        if device == torch.device('cpu'): exp_rngs.append(torch.LongTensor(a_rng))
        else: exp_rngs.append(torch.cuda.LongTensor(a_rng))
    prob_matr = torch.cat([prob_m.index_select(0, exp_rng)
                           for prob_m, exp_rng in zip(probs, exp_rngs)], 0)
    return prob_matr

#################
# Visualization
#################

def visualize_alignment(nodes1, nodes2, alignments, file_app='', col='green'):
    dag = Digraph(filename=sf.vis_data_loc + file_app)

    for i, nodes in enumerate([nodes1, nodes2]):
        gr_name = 'base' if i == 0 else 'target'
        # graph name must begin with 'cluster' for graphviz
        with dag.subgraph(name='cluster_' + gr_name) as g:
            g.attr(color='black')
            g.attr(label=gr_name)
            tsrt = reversed(topological_sort(nodes))
            for layer in tsrt:
                for node in layer:
                    n_shape = 'ellipse' if node.ordered else 'rectangle'
                    g.node(str(id(node)), label=node.label, shape=n_shape)
            for node in nodes:
                for arg in node.args:
                    g.edge(str(id(node)), str(id(arg)))

    if col == 'green':
        col_val = '0.33 '
    elif col == 'blue':
        col_val = '0.5 '

    for prob, (n1_ind, n2_ind) in alignments:
        prob = max(prob, 0)
        dag.edge(str(id(nodes1[n1_ind])), str(id(nodes2[n2_ind])), 
                 constraint='false', dir='none', color=col_val + str(prob) + ' 1')
    dag.view()

def visualize_alignment_ps(nodes1, nodes2, alignments, file_app='', col='green'):
    dag = Digraph(filename=sf.vis_data_loc + file_app)
    dag.attr(nodesep='0.4')
    dag.attr(ranksep='0.35')

    for i, nodes in enumerate([nodes1, nodes2]):
        tsrt = reversed(topological_sort(nodes))
        for layer in tsrt:
            for node in layer:
                n_shape = 'ellipse'# if node.ordered else 'rectangle'
                dag.node(str(id(node)), label=node.label, shape=n_shape)
        for node in nodes:
            for arg in node.args:
                dag.edge(str(id(node)), str(id(arg)))

    col_val = '0.33 '
    col_val = '0.6 '

    for prob, (n1_ind, n2_ind) in alignments:
        prob = max(prob, 0)
        dag.edge(str(id(nodes1[n1_ind])), str(id(nodes2[n2_ind])), 
                 constraint='false', dir='none', color=col_val + str(prob) + ' 1')
    dag.view()

def visualize_graph(graph, filename='graph_img'):
    g = Digraph(filename=filename)
    g.attr(color='black')
    #g.attr(label=gr_name)
    
    tsrt = reversed(topologically_group(graph))
    good_nodes = set()
    for node in graph.nodes:
        for arg in graph.successors(node):
            edge_label = graph.edges[node, arg]['label']
            if ':' in edge_label[0] or 'word_node' in node:
                good_nodes.add(node)
                good_nodes.add(arg)

    for layer in tsrt:
        for node in layer:
            if not node in good_nodes: continue
            n_shape = 'ellipse'
            #label_is = str(graph.nodes[node]['label']).replace('-','=').replace('.','dt').replace('/\\', '&')
            label_is = str(graph.nodes[node]['label']).replace('/\\', '&')
            if len(label_is) == 1 and (not label_is == '&') and list(graph.successors(node)):
                label_is = 'amr-' + label_is
            g.node(str(hash(node)), label=label_is, shape=n_shape)
    for node in graph.nodes:
        if not node in good_nodes: continue
        for arg in graph.successors(node):
            edge_label = graph.edges[node, arg]['label']
            edge_label = edge_label if edge_label[0] == ':' else ''
            if edge_label == ':pred-is-named': edge_label = ':word'
            #edge_label = ''
            g.edge(str(hash(node)), str(hash(arg)), label=edge_label)
    
    try: g.view()
    except: pass

#################
# Language Utilities
#################

def parse_nl_stmt(stmt, spacy_map, amr_map):
    #doc = nlp(stmt)
    doc = spacy_map[stmt]
    graph, tok_map = convert_to_graph(doc)
    amr_s_exprs = get_amr_graph(stmt, amr_map)
    s_exprs = convert_graph_to_s_exprs(graph)
    s_exprs = [expr for expr in s_exprs if not ('pos_' in expr[0] or 'word_node' in expr[0])]
    s_exprs = [(expr[1] if expr[0] == 'end_sent' else expr) for expr in s_exprs ]
    s_expr = tuple(['/\\'] + amr_s_exprs + s_exprs)
    sent = [incl_pos(tok_map[ind_func(tok)]) for sent in doc.sents for tok in sent]
    return s_expr, sent

TOK_SP = '_ID_POS_PT_'
def incl_pos(arg):
    return arg.label.lower()
    return arg.label.lower() + TOK_SP + '_'.join([str(x) for x in arg.position])

def convert_graph_to_s_exprs(graph):
    s_exprs = []
    for gr_expr in graph:
        s_exprs.append(convert_graph_to_s_expr(gr_expr))
    return s_exprs

def convert_graph_to_s_expr(gr_expr):
    new_label = gr_expr.label.lower()
    new_args = [convert_graph_to_s_expr(a) for a in gr_expr.arguments]
    if new_args: return tuple([new_label] + new_args)
    return new_label
    
def convert_to_graph(doc):
    graphs = []
    tok_map = {}
    for s_num, sentence in enumerate(doc.sents):
        for t_num, tok in enumerate(sentence):
            if not ind_func(tok) in tok_map:
                tok_map[ind_func(tok)] = ParseNode(canon_str(tok))
            tok_map[ind_func(tok)].position = (s_num, t_num)
    prev_node = None
    for tok, tok_node in tok_map.items():
        s_num, t_num = tok_node.position
        pos_node = ParseNode('pos_' + str(t_num))
        #new_node = ParseNode('word_node', [tok_node, pos_node])
        #new_node = ParseNode('pos_' + str(t_num), [tok_node] + ([prev_node] if prev_node else []))
        new_node = ParseNode('word_node', [tok_node] + ([prev_node] if prev_node else []))
        prev_node = new_node
        graphs.append(new_node)
    graphs.append(ParseNode('end_sent', [prev_node]))

    # unary token features
    #graphs.extend(get_sp_graph(list(doc), tok_map))
    graphs.extend(get_fine_tag_graph(list(doc), tok_map))

    # dependency features
    graphs.extend(get_dep_graph(doc, constr_rels=set(['punct']), tok_nodes=tok_map))

    return graphs, tok_map

def ind_func(tok):
    #return tok.text
    return tok

def canon_str(tok):
    r_label = tok.text.lower()
    #r_label = tok.lemma_.lower()
    return r_label

def lemma_str(tok):
    r_label = tok.lemma_.lower()
    return r_label

def skip_tok(tok):
    return tok.is_punct or tok.is_stop

def is_ann_tok(tok):
    if tok.pos_ in 'ADV' and tok.text[-2:] == 'er':
        return True
    return tok.pos_ in ['NOUN', 'VERB']

def is_comp_tok(tok):
    return tok.pos_ in 'ADV' and tok.text[-2:] == 'er'

def get_sp_graph(doc, tok_map=None):
    if tok_map == None: tok_map = {}
    nodes = []
    for i in range(len(doc)):
        tok = doc[i]
        t_node = tok_map[ind_func(tok)]
        # entity type
        if tok.ent_type_:
            et_node = ParseNode(tok.ent_type_ + '_Ent_Type', [t_node])
            t_node.parents.append(et_node)
            nodes.append(et_node)
    return nodes

def get_fine_tag_graph(doc, tok_map=None):
    if tok_map == None: tok_map = {}
    nodes = []
    for i in range(len(doc)):
        tok = doc[i]
        t_node = tok_map[ind_func(tok)]
        if is_ann_tok(tok):
            tag_node = ParseNode(tok.tag_ + '_Fine_Pos', [t_node])
            t_node.parents.append(tag_node)
            nodes.append(tag_node)
    return nodes

def get_amr_graph(sent, amr_map):
    graph = amr_map[sent]
    amr_tuple = parse_amr_str(graph)
    return amr_tuple

def parse_amr_str(graph_str):
    lines = graph_str.split('\n')
    node_map = {}
    for l in lines:
        if '::tok' in l: toks = l.split()[2:]
        elif '::node' in l:
            comp = l.split()
            try:
                node_info = comp[3]
                node_span = [int(x) for x in comp[4].split('-')]
                node_tok = '-'.join(toks[node_span[0] : node_span[1]]).lower()
                node_map[node_info.lower()] = node_tok
            except: pass
    use_str = ' '.join([l for l in lines if l and l[0] != '#'])
    use_str = [el for el_str in use_str.split() for el in re.split('(\(|\))', el_str)]
    use_str_lst = [el for el in use_str if el]
    amr_tup = parse_amr_lst(use_str_lst, node_map)
    #print(graph_str)
    #print(amr_tup)
    #input()
    
    return amr_tup

def parse_amr_lst(toks, node_map):
    stack, add_lst, seen_dict = [], [], {}
    for tok in toks:
        if tok == '(':
            stack.append(add_lst)
            add_lst = []
        elif tok == ')':
            assert len(stack) > 0, 'Imbalanced parentheses:\n' + sexpr_str
            assert add_lst, 'Empty list found:\n' + sexpr_str
            old_expr = reformat_amr_expr(add_lst, node_map)
            if not old_expr in seen_dict: seen_dict[old_expr] = old_expr
            old_expr = seen_dict[old_expr]
            add_lst = stack.pop()
            add_lst.append(old_expr)
        else:
            add_lst.append(tok.lower())
    assert len(add_lst) == 1
    return add_lst

def reformat_amr_expr(lst, node_map):
    assert lst[1] == '/'
    pred = lst[2]
    new_lst = [(':amr-name', pred)]
    if pred in node_map: new_lst.append((':orig-word', node_map[pred]))
    i = 3
    while i < len(lst):
        j = i + 1
        while j < len(lst) and lst[j][0] != ':': j += 1
        arg_pt = lst[i]
        splt = lst[i + 1 : j]
        if len(splt) == 1: arg_n = splt[0]
        else: arg_n = '_'.join(splt)
        new_lst.append((arg_pt, arg_n))
        i = j
    #return tuple([lst[0]] + new_lst)
    return tuple(['amr_rel'] + new_lst)
    
def get_dep_graph(doc, tok_nodes=None, constr_rels=None):
    if constr_rels == None: constr_rels = set()
    if tok_nodes == None: tok_nodes = {}
    sentences = list(doc.sents)
    unprocessed = [sentence.root for sentence in sentences]
    seen = set()
    fin_graph = []
    while unprocessed:
        tok = unprocessed.pop()
        tok_node = tok_nodes[ind_func(tok)]
        seen.add(tok)
        # dependency information
        for child in tok.children:
            dep_label = child.dep_ + '_dep_info'
            ch_node = tok_nodes[ind_func(child)]
            dep_node = ParseNode(dep_label, [tok_node, ch_node])
            l_dep_node = ParseNode(dep_label + '_1', [tok_node])
            r_dep_node = ParseNode(dep_label + '_2', [ch_node])
            if not (child.dep_ in constr_rels or \
                    any(dep in child.dep_ for dep in constr_rels)):
                tok_node.parents.append(dep_node)
                ch_node.parents.append(dep_node)
                fin_graph.append(dep_node)
            if not child in seen:
                unprocessed.append(child)
    vals = list(set(fin_graph))
    return vals

class ParseNode:

    def __init__(self, label, arguments=None, parents=None, ordered=True, position=None):
        if arguments == None: arguments = []
        if parents == None: parents = []
        self.position = position
        self.label = label
        self.arguments = arguments
        self.parents = parents
        self.ordered = ordered
        
    def keyForm(self):
        return (self.label, len(self.arguments), len(self.parents) > 0, self.ordered)
        
    def __str__(self):
        args_str = ''
        if self.arguments:
            args_str = '(' + ', '.join([str(x) for x in self.arguments]) + ')'
        return self.label + args_str

    def __repr__(self):
        return str(self)
