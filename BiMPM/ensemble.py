import numpy as np
import logging
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
from glob import glob
import os
import math
from collections import OrderedDict
import time
import matplotlib.pyplot as plt
import argparse
from sklearn.manifold import TSNE
import random

from datastream import Batch
from datastream import loadData,loadData_sample, all_vocab_emb, test_model
from MultiPerspective_Matching import MatchingLayer, AggregationLayer, PredictionLayer
from BiMPM_prediction import BiMPM


def load_checkpoint(model, optimizer, best_model_file):

    if os.path.isfile(best_model_file):

        print("=> loading checkpoint '{}'".format(best_model_file))
        checkpoint = torch.load(best_model_file)
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
            .format(best_model_file, checkpoint['epoch']))
        print("best validation accuracy is %s" % best_val_acc)
    else:
        print("=> no checkpoint found at '{}'".format(best_model_file))

    return model, optimizer

def model_prob_dist(model, batch, args):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0

    model.eval()

    prob_dist_all = []
    all_labels = []

    for labels, s1, s2, ch1, ch2 in batch:

        ## uncomment this if you want to test
        #if batch.start > 200:
        #    break

        if args.cuda:
            labels, s1, s2, ch1, ch2 = labels.cuda(), s1.cuda(), s2.cuda(), ch1.cuda(), ch2.cuda()


        prob_dist = model(labels, s1, s2, ch1, ch2)

        prob_dist_all.append(prob_dist)
        all_labels.append(labels.data)

    model.train()

    return torch.cat(prob_dist_all, 0), torch.cat(all_labels)


def ensemble_prediction(prob_all_models, labels):
    
    correct = 0
    prob_all_ave = torch.cat(prob_all_models, 2).mean(dim=2)
    ave_predicted = (prob_all_ave.data.max(1)[1]).long().view(-1)
    correct += (ave_predicted == labels).sum()
    
    return 100 * correct / labels.size(0)

def main(args):

    vocab, emb, vocab_chars = all_vocab_emb(args)

    batch_size = args.batch

    model = BiMPM(vocab, vocab_chars, emb, args)

    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    test_data = loadData(vocab, args, mode='test')

    best_model_list = args.best_model_list

    probs_all_models = []
    for best_model_file in best_model_list:
        test_batch = Batch(test_data, batch_size, vocab, vocab_chars)

        model, _ = load_checkpoint(model, optimizer, args.saveto + best_model_file)

        probs, labels = model_prob_dist(model, test_batch, args)

        probs_all_models.append(probs.unsqueeze(2))

    test_acc = ensemble_prediction(probs_all_models, labels)

    print("Ensemble Test Acc is: %s %%" % round(test_acc, 3))
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-emb', type=int, default=300)
    parser.add_argument('-hid', type=int, default=100)
    parser.add_argument('-num_context_layers', type=int, default=1)
    parser.add_argument('-num_aggr_layers', type=int, default=1)
    parser.add_argument('-batch', type=int, default=60)
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-seed', type=int, default=123)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-lambda_l2', type=float, default=0)
    parser.add_argument('-clipper', type=float, default=5.0)
    parser.add_argument('-num_classes', type=int, default=3)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-perspective', type=int, default=20)
    parser.add_argument('-vocab_trn', type=str)
    parser.add_argument('-vocab_dev', type=str)
    parser.add_argument('-vocab_tst', type=str)
    parser.add_argument('-vocab_trn_char', type=str)
    parser.add_argument('-vocab_dev_char', type=str)
    parser.add_argument('-vocab_tst_char', type=str)
    parser.add_argument('-file_trn', type=str)
    parser.add_argument('-file_dev', type=str)
    parser.add_argument('-file_tst', type=str)
    parser.add_argument('-file_emb', type=str)
    
    parser.add_argument('-best_model_list', type=list, default=[])
    parser.add_argument('-source', type=str)
    parser.add_argument('-saveto', type=str)

    parser.add_argument('-cuda', type=bool, default=False)

    parser.add_argument('-char_emb_size', type=int, default=20)
    parser.add_argument('-char_size', type=int, default=50)

    parser.add_argument('--log_dir', type=str, default='../log')

    parser.add_argument('--log_fname', type=str, default='gradClamp_combinelayer.log')

    args = parser.parse_args()

    args.vocab_trn = "vocabsnli_trn.npy"
    args.vocab_dev = "vocabsnli_dev.npy"
    args.vocab_tst = "vocabsnli_tst.npy"

    args.vocab_trn_char = "vocab_charssnli_trn.npy"
    args.vocab_dev_char = "vocab_charssnli_dev.npy"
    args.vocab_tst_char = "vocab_charssnli_tst.npy"

    args.file_trn = 'snli_trn.txt'
    args.file_dev = 'snli_dev.txt'
    args.file_tst = 'snli_tst.txt'

    args.file_emb = 'snli_emb.npy'

    args.source = "../intermediate/"
    args.saveto = "../results/"
    args.save_stamp = 'snli_1028'

    args.emb_file = "snli.npy"

    args.best_model_list = ['best_model.path.tar', 'best_model.path.tar']

    args.cuda = False

    main(args)