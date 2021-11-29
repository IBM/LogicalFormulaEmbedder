from code.utilities import *
from code.parse_input_forms import *

def main():
    fof = 'fof(t20_latsubgr,conjecture,( ! [A] : ( ((((~ v3_struct_0(A) & v3_group_1(A)) & v4_group_1(A)) & l1_group_1(A)) ) => ! [B] : ( ( v1_group_1(B) & m1_group_2(B,A) ) => k1_funct_1(k1_latsubgr(A),B) != k1_xboole_0 ) ) )).'

    tup_form = parse_fof_to_tuple(fof)
    gr_form = convert_expr_to_graph(tup_form)
    print(gr_form.nodes)
    
    visualize_graph(gr_form)

if __name__ == '__main__':
    main()
