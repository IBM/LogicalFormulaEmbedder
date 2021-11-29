# python imports
import re, random
from io import StringIO
# networkx imports
import networkx as nx
# code imports
from code.utilities import *
from code.node_classes import *

apply_node_label = '_EVAL_'
quantifiers = { '\\', '?!', '?', '!', '@' }
ord_operators = { 'IN', 'SUBSET', '>', '<', '..', '<=', '==>', '>=', 'MOD', '|-',
                  ',', 'INSERT', 'INTER', 'DIFF', 'PCROSS', 'DELETE', 'HAS_SIZE',
                  'EXP', 'PSUBSET', 'DIV', 'CROSS', 'o', '<=_c', '<_c', 'treal_le', 
                  '>=_c', '$', 'EVAL', '=>', '~' }
unord_operators = { '=', '-', '+', '*', '==', 'UNION', '\\/', '=_c', 'treal_eq',
                    'treal_mul', 'treal_add', '/\\', '!=', '&', '|', '<=>', '<==>' }

operators = unord_operators | ord_operators

###
# S-expr handling
###

def parse_s_expr_to_tuple(sexpr_str):
    toks = re.split('([()])', sexpr_str)
    toks = [x for x in toks if x]
    stack, add_lst, seen_dict = [], [], {}
    for tok in toks:
        if tok == '(':
            stack.append(add_lst)
            add_lst = []
        elif tok == ')':
            assert len(stack) > 0, 'Imbalanced parentheses:\n' + sexpr_str
            assert add_lst, 'Empty list found:\n' + sexpr_str
            old_expr = tuple(reformat_expr_list(add_lst))
            if not old_expr in seen_dict: seen_dict[old_expr] = old_expr
            old_expr = seen_dict[old_expr]
            add_lst = stack.pop()
            add_lst.append(old_expr)
        else:
            add_lst.extend(get_syms(tok))
    assert len(stack) == 0, 'Imbalanced parentheses:\n' + sexpr_str
    if len(add_lst) > 1:
        ops = [el for el in add_lst if el in operators and el != ',']
        assert ops, 'Multi-expression list with no known operators:\n' + sexpr_str
        assert len(ops) == 1, 'Multiple operators at top-level:\n' + sexpr_str
        op_pos = position(ops[0], add_lst)
        # sequents A1, ..., An |- B1, ..., Bn are interpreted as 
        # (A1 AND ... AND An) |- (B1 OR ... OR Bn)
        lhs = [tuple(['/\\'] + [el for el in add_lst[:op_pos] if el != ','])]
        rhs = [tuple(['\\/'] + [el for el in add_lst[op_pos + 1 : len(add_lst)]
                                if el != ','])]
        ret_tup = tuple([ops[0]] + lhs + rhs)
    else:
        ret_tup = add_lst[0]
    return ret_tup

def get_syms(string):
    parsed_syms = []
    for sym in string.split():
        quant = None
        if sym[:2] in quantifiers: quant = sym[:2]
        if not quant and sym[0] in quantifiers: quant = sym[0]
        if quant and sym[len(sym) - 1] == '.':
            sym = sym[1 : -1]
            parsed_syms.extend([quant, sym])
        else:
            parsed_syms.append(sym)
    return parsed_syms

def reformat_expr_list(lst, uncurry=True):
    if len(lst) == 3:
        if lst[1] in operators:
            return [lst[1], lst[0], lst[2]]
    if type(lst[0]) == tuple:
        return [apply_node_label] + [lst[0]] + lst[1:]
    if lst[0] in quantifiers and uncurry:
        if type(lst[len(lst) - 1]) == tuple:
            if lst[len(lst) - 1][0] == lst[0]:
                expr_to_ext = list(lst[len(lst) - 1])
                expr_vars = expr_to_ext[1 : len(expr_to_ext) - 1]
                expr_content = expr_to_ext[len(expr_to_ext) - 1]
                new_vars = lst[1 : len(lst) - 1] + expr_vars
                return [lst[0]] + new_vars + [expr_content]
    if lst[0] in unord_operators and uncurry:
        new_args = []
        for arg in lst[1:]:
            if type(arg) == tuple:
                if arg[0] == lst[0]:
                    new_args.extend(arg[1:])
                else:
                    new_args.append(arg)
            else:
                new_args.append(arg)
        return [lst[0]] + new_args
    return lst

###
# TPTP form handling
###

def negate_fof_formula(string):
    stream = StringIO(string)
    pos = stream.tell()
    while peek(stream):
        skip_whitespace(stream)
        sym = has_symbol(stream)
        if sym == 'fof' and stream.read(1) == '(':
            fof_name = has_symbol(stream)
            skip_whitespace(stream)
            if fof_name and stream.read(1) == ',':
                fof_type = has_symbol(stream)
                skip_whitespace(stream)
                if fof_type and stream.read(1) == ',':
                    rem_f = stream.read(len(string))
                    rem_f = '~( ' + rem_f.replace('.', ' ).')
                    new_formula = ', '.join(['fof(' + fof_name, fof_type, rem_f])
                    return new_formula
        raise ValueError('Parsing error on input formula:\n' + string)

def parse_cnf_lst_to_tuple(string_lst):
    clause_lst = []
    for string in string_lst:
        # lazy cnf
        assert string[:3] == 'cnf'
        fof_str = 'fof' + string[3:]
        fof_str = fof_str.replace('|', ' | (')
        l_p_ct = len([x for x in fof_str if x == '('])
        r_p_ct = len([x for x in fof_str if x == ')'])
        rem_ct = l_p_ct - r_p_ct
        new_rp = ''.join([')' for _ in range(rem_ct)])
        fof_str = fof_str.replace('.', new_rp + '.')
        fof_tup = parse_fof_to_tuple(fof_str)
        clause_lst.append(fof_tup)
    conjunct = tuple(['&'] + clause_lst)
    return conjunct

def parse_fof_to_tuple(string):
    stream = StringIO(string)
    pos = stream.tell()
    while peek(stream):
        skip_whitespace(stream)
        sym = has_symbol(stream)
        if sym == 'fof' and stream.read(1) == '(':
            fof_name = has_symbol(stream)
            skip_whitespace(stream)
            if fof_name and stream.read(1) == ',':
                fof_type = has_symbol(stream)
                skip_whitespace(stream)
                if fof_type and stream.read(1) == ',':
                    formula = is_formula(stream)
                    skip_whitespace(stream)
                    if formula and stream.read(1) == ')':
                        skip_whitespace(stream)
                        if stream.read(1) == '.':
                            # consistent with s-expr parser
                            return reformat_tptp_list_form(formula)
        raise ValueError('Parsing error on input formula:\n' + string)

def reformat_tptp_list_form(lst, seen_dict=None):
    if seen_dict == None: seen_dict = {}
    if type(lst) == list:
        tup = tuple(reformat_expr_list([reformat_tptp_list_form(x) for x in lst]))
        if not tup in seen_dict: seen_dict[tup] = tup
        return seen_dict[tup]
    else:
        if not lst in seen_dict: seen_dict[lst] = lst
        return seen_dict[lst]

def is_formula(stream):
    pos = stream.tell()
    skip_whitespace(stream)
    negated_formula = is_negated_formula(stream)
    if negated_formula: return negated_formula
    nested_connective_formula = is_nested_connective_formula(stream)
    if nested_connective_formula: return nested_connective_formula
    nested_formula = is_nested_formula(stream)
    if nested_formula: return nested_formula
    quantified_formula = is_quantified_formula(stream)
    if quantified_formula: return quantified_formula
    atomic_formula = is_atomic_formula(stream)
    if atomic_formula: return atomic_formula
    connective_formula = is_connective_formula(stream)
    if connective_formula: return connective_formula
    stream.seek(pos)

def is_nested_connective_formula(stream):
    pos = stream.tell()
    skip_whitespace(stream)
    if stream.read(1) == '(':
        formula1 = is_formula(stream)
        if formula1:
            skip_whitespace(stream)
            connective = has_connective(stream)
            if connective:
                formula2 = is_formula(stream)
                if formula2:
                    skip_whitespace(stream)
                    if stream.read(1) == ')':
                        return [connective, formula1, formula2]
    stream.seek(pos)

def is_nested_formula(stream):
    pos = stream.tell()
    skip_whitespace(stream)
    if stream.read(1) == '(':
        formula = is_formula(stream)
        if formula:
            skip_whitespace(stream)
            if stream.read(1) == ')':
                return formula
    stream.seek(pos)
    
def is_connective_formula(stream):
    pos = stream.tell()
    skip_whitespace(stream)
    formula1 = is_formula(stream)
    if formula1:
        skip_whitespace(stream)
        connective = has_connective(stream)
        if connective:
            formula2 = is_formula(stream)
            if formula2:
                return [connective, formula1, formula2]
    stream.seek(pos)

def is_quantified_formula(stream):
    pos = stream.tell()
    skip_whitespace(stream)
    sym = stream.read(1)
    if sym in quantifiers:
        var_lst = is_var_list(stream)
        if var_lst:
            skip_whitespace(stream)
            if stream.read(1) == ':':
                formula = is_formula(stream)
                return [sym] + var_lst + [formula]
    stream.seek(pos)

def is_var_list(stream):
    pos = stream.tell()
    skip_whitespace(stream)
    if stream.read(1) == '[':
        var1 = has_symbol(stream)
        if var1:
            var_lst = [var1]
            skip_whitespace(stream)
            while peek(stream) == ',':
                stream.read(1)
                var = has_symbol(stream)
                if var:
                    var_lst.append(var)
                    skip_whitespace(stream)
                else:
                    break
            skip_whitespace(stream)
            if stream.read(1) == ']':
                return var_lst
    stream.seek(pos)

def is_negated_formula(stream):
    pos = stream.tell()
    neg_sym = has_neg_sym(stream)
    if neg_sym:
        formula = is_formula(stream)
        if formula:
            return [neg_sym, formula]
    stream.seek(pos)

def is_atomic_formula(stream):
    pos = stream.tell()
    eq_formula = is_equality_formula(stream)
    if eq_formula: return eq_formula
    basic_formula = is_internal_arg(stream)
    if basic_formula: return basic_formula
    stream.seek(pos)

def is_equality_formula(stream):
    pos = stream.tell()
    skip_whitespace(stream)
    formula1 = is_internal_arg(stream)
    if formula1:
        eq_sym = has_eq_sym(stream)
        if eq_sym:
            formula2 = is_internal_arg(stream)
            if formula2:
                return [eq_sym, formula1, formula2]
    stream.seek(pos)

def is_internal_arg(stream):
    pos = stream.tell()
    skip_whitespace(stream)
    functor = has_symbol(stream)
    if functor:
        if peek(stream) == '(':
            # term
            stream.read(1)
            arg_lst = is_arg_list(stream)
            if arg_lst:
                skip_whitespace(stream)
                if stream.read(1) == ')':
                    return [functor] + arg_lst
        else:
            # constant
            return functor
    stream.seek(pos)

def is_arg_list(stream):
    pos = stream.tell()
    arg_lst = []
    arg1 = is_internal_arg(stream)
    if arg1:
        arg_lst.append(arg1)
        skip_whitespace(stream)
        while peek(stream) == ',':
            stream.read(1)
            arg = is_internal_arg(stream)
            if arg:
                arg_lst.append(arg)
                skip_whitespace(stream)
            else:
                break
    if arg_lst:
        return arg_lst
    stream.seek(pos)

def has_symbol(stream):
    pos = stream.tell()
    skip_whitespace(stream)
    label = ''
    while is_acceptable_char(peek(stream)):
        label += stream.read(1)
    if label:
        return label
    stream.seek(pos)

def has_neg_sym(stream):
    pos = stream.tell()
    skip_whitespace(stream)
    if stream.read(1) == '~': return '~'
    stream.seek(pos)

def has_eq_sym(stream):
    pos = stream.tell()
    skip_whitespace(stream)
    if peek(stream, 2) == '!=': 
        stream.read(2)
        return '!='
    elif peek(stream, 1) == '=':
        if not any(peek(stream, i) in operators for i in range(2, 4)):
            stream.read(1)
            return '='
    stream.seek(pos)

whitespace_chars = [' ', '\n', '\t']

def has_connective(stream):
    pos = stream.tell()
    skip_whitespace(stream)
    # explicitly avoid equality when parsing connectives
    eq_sym = has_eq_sym(stream)
    if not eq_sym:
        label = ''
        while peek(stream) not in whitespace_chars + ['(', ')']:
            label += stream.read(1)
        if label in operators:
            return label
    stream.seek(pos)

char_pattern = re.compile('\w')
    
def is_acceptable_char(character):
    return char_pattern.match(character)

def skip_whitespace(stream):
    while peek(stream) in whitespace_chars: 
        stream.read(1)
    
def peek(stream, size=1):
    pos = stream.tell()
    next = stream.read(size)
    stream.seek(pos)
    return next

###
# Creating graphs from tuple forms
###

def convert_expr_list_to_graph(expr_lst, label_conj=True, depth_limit=None, edge_spec=None,
                               is_hol=False):
    graph = nx.DiGraph()
    if depth_limit is None: depth_limit = 100000000
    for expr_i, expr in enumerate(expr_lst):
        fill_graph_from_expr(expr, graph, depth_limit, edge_spec, is_hol)
    return graph

def convert_expr_to_graph(expr, depth_limit=None, edge_spec=None, is_hol=False):
    graph = nx.DiGraph()
    if depth_limit is None: depth_limit = 100000000
    fill_graph_from_expr(expr, graph, depth_limit, edge_spec, is_hol)
    return graph

def fill_graph_from_expr(expr, graph, depth_limit, edge_spec, is_hol,
                         orig_depth=None, par_type=None):
    if orig_depth == None: orig_depth = depth_limit
    at_depth = orig_depth - depth_limit
    def get_const_type(expr, par_type):
        if likely_var(expr):
           return VarType
        elif likely_skolem(expr):
            return SkolemConstType
        elif par_type in [PredType, FuncType]:
            return ConstType
        else:
            #return PredType
            # otherwise default to const type, will get corrected as needed
            return ConstType

    # if depth limit is reached, don't explore further
    if depth_limit < 0: 
        # generate new token
        if type(expr) != tuple and expr in graph.nodes: return expr
        new_label = expr[0] if type(expr) == tuple else expr
        graph.add_node(expr)
        graph.nodes[expr]['label'] = new_label
        if type(expr) != tuple: graph.nodes[expr]['type'] = get_const_type(expr, par_type)
        else: graph.nodes[expr]['type'] = NASType
        graph.nodes[expr]['depth'] = [at_depth]
        return expr

    # if it's already been defined, just return it
    if expr in graph.nodes:
        if not 'src' in graph.nodes[expr]: graph.nodes[expr]['src'] = set()
        graph.nodes[expr]['depth'].append(at_depth)
        return expr
    
    if type(expr) == tuple:
        graph.add_node(expr)
        graph.nodes[expr]['label'] = expr[0]
        graph.nodes[expr]['depth'] = [at_depth]
        if expr[0] in quantifiers: graph.nodes[expr]['type'] = QuantType
        elif expr[0] == apply_node_label: graph.nodes[expr]['type'] = ApplyType
        elif expr[0] in operators: graph.nodes[expr]['type'] = OpType
        elif likely_gen_var(expr[0]): graph.nodes[expr]['type'] = VarFuncType
        elif likely_uniq_var(expr[0]): graph.nodes[expr]['type'] = VarFuncType
        elif likely_skolem(expr[0]): graph.nodes[expr]['type'] = SkolemFuncType
        elif expr[0] in graph.nodes and \
             graph.nodes[expr[0]]['type'] == VarType:
            # reassign type of lead element as well as the expression
            graph.nodes[expr[0]]['type'] = VarFuncType
            graph.nodes[expr]['type'] = VarFuncType
        elif par_type == PredType and not is_hol: graph.nodes[expr]['type'] = FuncType
        else: graph.nodes[expr]['type'] = PredType
        is_type = graph.nodes[expr]['type']
        for a_i, orig_arg in enumerate(expr):
            if a_i == 0: continue
            if type(orig_arg) == tuple and len(orig_arg) == 2 and orig_arg[0][0] == ':':
                arg_label, arg = orig_arg[0], orig_arg[1]
            else: arg_label, arg = None, orig_arg
            arg_node = fill_graph_from_expr(arg, graph, depth_limit - 1,
                                            edge_spec, is_hol,
                                            orig_depth=orig_depth, par_type=is_type)
            if expr[0] in quantifiers and a_i < len(expr) - 1 and \
               arg == graph.nodes[arg_node]['label']:
                graph.nodes[arg_node]['type'] = VarType
            # adding edge
            graph.add_edge(expr, arg_node)
            if edge_spec and 'unord' in edge_spec: arg_rank = 1
            elif edge_spec and 'ord' in edge_spec: arg_rank = a_i
            elif expr[0] in quantifiers: arg_rank = 1 if a_i < len(expr) - 1 else 2
            elif expr[0] in unord_operators: arg_rank = 1
            else: arg_rank = a_i
            par_type = graph.nodes[expr]['type']
            par_part = expr[0] if par_type in [QuantType, OpType] else par_type.__name__
            if edge_spec and 'untyped' in edge_spec: par_part = 'untyped'
            elif edge_spec and 'typed' in edge_spec: par_part = par_part
            edge_label = par_part + '_' + str(arg_rank)
            if arg_label: edge_label = arg_label
            graph.edges[expr, arg_node]['label'] = edge_label
    else:
        graph.add_node(expr)
        graph.nodes[expr]['depth'] = [at_depth]
        graph.nodes[expr]['label'] = sep_tok_id(expr)
        # constants with _ in front, e.g., _9234, are implicitly
        # universally quantified variables
        graph.nodes[expr]['type'] = get_const_type(expr, par_type)
    return expr

def sep_tok_id(el):
    return el.split(TOK_SP)[0]

def likely_var(expr):
    return likely_gen_var(expr) or likely_uniq_var(expr) or likely_implicit_univ_var(expr)

def likely_skolem(expr):
    return expr[:3] == 'esk'

def likely_implicit_univ_var(expr):
    return expr[0] == 'X' and expr[1:].isdigit()

def likely_gen_var(expr):
    return ('GEN' in expr and 'PVAR' in expr)

def likely_uniq_var(expr):
    return (expr[0] == '_' and expr[1:].isnumeric())

def get_el_info(expr):
    if type(expr) == tuple:
        els = [expr[0]]
        if expr[0] in quantifiers: type_of = QuantType
        elif expr[0] in operators: type_of = OpType

        for el in expr: els.extend(get_av_els(el))
        return els
    elif is_rn_var(expr): return ['VAR']
    else: return [expr]

def make_univ_sink(expr, known_vars=None):
    def make_sink_node(arg, is_v):
        new_a = 'GENPVAR_' + arg if is_v else arg
        return (new_a, 'UNIV_SINK')
    if known_vars == None: known_vars = set()
    if type(expr) == tuple:
        new_expr = [expr[0]]
        if expr[0] in quantifiers:
            for a_i, arg in enumerate(expr):
                if a_i == 0: continue
                if a_i < len(expr) - 1:
                    known_vars.add(arg)
                    new_expr.append(make_sink_node(arg, True))
                else:
                    new_expr.append(make_univ_sink(arg, known_vars))
        else:
            for a_i, arg in enumerate(expr):
                if a_i == 0: continue
                new_expr.append(make_univ_sink(arg, known_vars))
        return tuple(new_expr)
    if expr in known_vars: return make_sink_node(expr, True)
    if likely_var(expr) or likely_skolem(expr): return make_sink_node(expr, True)
    return make_sink_node(expr, False)

def reorder_expr(expr):
    def key_form(x):
        ls = []
        for y in x:
            if type(y) == tuple: ls.extend([z[0] for z in y])
            else: ls.append(y)
        return ls
    
    if type(expr) == tuple:
        new_expr = []
        if expr[0] in quantifiers:
            for a_i, arg in enumerate(expr):
                if a_i == 0: continue
                if a_i < len(expr) - 1: new_expr.append(arg)
            new_expr = sorted(new_expr)
            new_expr.append(reorder_expr(expr[len(expr) - 1]))
            return tuple([expr[0]] + new_expr)
        for a_i, arg in enumerate(expr):
            if a_i == 0: continue
            new_expr.append(reorder_expr(arg))
        if expr[0] in unord_operators:
            non_tups = [el for el in new_expr if type(el) != tuple]
            tups = [el for el in new_expr if type(el) == tuple]
            new_expr = sorted(non_tups) + sorted(tups, key=lambda x : key_form(x))
        return tuple([expr[0]] + new_expr)
    else:
        return expr
    
def rename_all_vars(expr, var_assigns=None, ext=None):
    if ext == None: ext = random.randint(0, 100000)
    if var_assigns == None: var_assigns = {}
    if type(expr) == tuple:
        new_expr = [expr[0]]
        if expr[0] in quantifiers:
            for a_i, arg in enumerate(expr):
                if a_i == 0: continue
                if a_i < len(expr) - 1:
                    new_var_ext = ext + random.randint(1, 1000)
                    var_assigns[arg] = arg + '_SYM_EXT_' + str(new_var_ext)
                    new_expr.append(var_assigns[arg])
                else:
                    new_expr.append(rename_all_vars(arg, var_assigns=var_assigns,
                                                    ext=ext))
        else:
            for a_i, arg in enumerate(expr):
                if a_i == 0: continue
                new_expr.append(rename_all_vars(arg, var_assigns=var_assigns,
                                                ext=ext))
        return tuple(new_expr)
    if expr in var_assigns: return var_assigns[expr]
    if likely_var(expr) or likely_skolem(expr): return expr + '_SYM_EXT_' + str(ext)
    return expr

def unrename_all_vars(expr):
    if type(expr) == tuple:
        new_expr = [expr[0]]
        for i, el in enumerate(expr):
            if i == 0: continue
            new_expr.append(unrename_all_vars(el))
        return tuple(new_expr)
    elif '_SYM_EXT_' in expr: return expr.split('_SYM_EXT_')[0]
    else: return expr
