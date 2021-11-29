# python imports
import sys, os, random, time, signal, subprocess
import pickle as pkl
import argparse as ap
# torch imports
import torch
# natural language imports
import spacy
# code imports
import code.dataset as dt
import code.parse_input_forms as pr
import code.utilities as ut
from process_data import *

def main(reset):
    sents = pkl.load(open(sci_sents_obj_loc, 'rb'))
    process_sentences(sents, reset)

def process_sentences(sents, reset=True):
    wrds = set()
    for (prob, conj_prem_sents) in sents.items():
        for sent in conj_prem_sents:
            for wrd in sent: wrds.add(pr.sep_tok_id(wrd))
    emb_map = {}
    embs = []
    emb_file = 'glove.6B.50d.txt' #'glove.6B.100d.txt'
    #emb_file = 'glove.6B.300d.txt'
    with open(os.path.join(lang_path, emb_file), 'r') as f:
        for n, l in enumerate(f.readlines()):
            wrd = ''
            i = 0
            while l[i] != ' ':
                wrd += l[i]
                i += 1
            if not wrd in wrds: continue
            if wrd in emb_map: continue
            print(100 * float(n / len(wrds)))
            vec = torch.tensor([float(x) for x in l.replace('\n', '').split(' ')[1:]])
            emb_map[wrd] = len(embs)
            embs.append(vec)
    torch.save((torch.stack(embs), emb_map), emb_info_loc)
    
def process_elmo_sentences(sents, reset=True):
    from allennlp.commands.elmo import ElmoEmbedder
    elmo = ElmoEmbedder()
    prev_at, num_sents = 0, len(sents)
    num_sents = len(sents)
    for st_num, (prob, conj_prem_sents) in enumerate(sents.items()):
        if round(100 * st_num / num_sents, 0) != prev_at:
            prev_at = round(100 * st_num / num_sents, 0)
            print('Scitail embeddings at ' + str(prev_at) + '% complete...')
        if (not reset) and os.path.exists(os.path.join(emb_path, prob)): continue
        embedding_matrs = []
        for sent in conj_prem_sents:
            to_parse = [pr.sep_tok_id(el) for el in sent]
            vectors = elmo.embed_sentence(to_parse)
            embedding_matrs.append(dict(zip(sent, vectors[-1])))
        torch.save(embedding_matrs, os.path.join(emb_path, prob))
    
if __name__ == '__main__':
    parser = ap.ArgumentParser(description='Get sentence embeddings')
    parser.add_argument('--reset', help='Reset embedding computation')
    args = parser.parse_args()
    
    main(args.reset=='True')
