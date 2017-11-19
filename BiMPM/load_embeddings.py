import numpy as np
import os
import spacy
import argparse
import json
from collections import Counter

def main(args):

    vocab = np.load(args.source + args.vocab_file).item()
    file = open(args.source+args.emb_file, 'r')

    emb = {k:0 for k,v in vocab.items()} 

    for line in file:
        temp = line.strip().split(' ')
        if temp[0] in emb.keys():
            emb[temp[0]] = temp[1:]

    np.save(args.saveto+args.save_label+".npy", emb)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-source', type=str, default="../intermediate/")
    parser.add_argument('-saveto', type=str, default="../intermediate/")
    parser.add_argument('-save_label', type=str, default='snli_emb')
    parser.add_argument('-vocab_file', type=str, default='vocabsnli.npy')
    parser.add_argument('-emb_file', type=str, default='glove.840B.300d.txt')
    args = parser.parse_args()
    main(args)




