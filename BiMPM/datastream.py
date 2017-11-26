import numpy as np

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
import random

def loadData_sample(vocab, num_samples, args, mode = 'train'):

    keys = list(vocab.keys())
    data = []
    count = 0
    missing = 0

    start = time.time()
    print('loading data... \n')

    if mode == 'train':
        filename = args.file_trn
    elif mode == 'eval':
        filename = args.file_dev
    elif mode == 'test':
        filename = args.file_tst

    with open(args.source+filename, 'r') as file_list:

        for file in file_list:
            if count < num_samples:
                line = file.strip().split('\t')

                if len(line) < 3:
                    continue

                label = line[0]
                s1 = line[1].strip(' ').split(' ')
                s2 = line[2].strip(' ').split(' ')
            
                if label == '-':
                    missing += 1
                else:
                    data.append((label, s1, s2))
                    count += 1

                if count%10000 == 0:
                    print("processed %s docs"%count)
                    print("%s seconds elapsed"%(time.time()-start))

    file_list.close()

    print('... %s data loaded, total %s samples, missing %s \n' % (mode, len(data), missing))
    return data


def loadData(vocab, args, mode = 'train'):

    keys = list(vocab.keys())
    data = []
    count = 0
    missing = 0
    start = time.time()
    print('loading data... \n')

    if mode == 'train':
        filename = args.file_trn
    elif mode == 'eval':
        filename = args.file_dev
    elif mode == 'test':
        filename = args.file_tst

    with open(args.source+filename, 'r') as file_list:

        for file in file_list:
            line = file.strip().split('\t')

            if len(line) < 3:
                continue

            label = line[0]
            s1 = line[1].strip(' ').split(' ')
            s2 = line[2].strip(' ').split(' ')
            if label == '-':
                missing += 1
            else:
                data.append((label, s1, s2))
            
            count += 1
            if count%10000 == 0:
                print("processed %s docs"%count)
                print("%s seconds elapsed"%(time.time()-start))

    file_list.close()

    print('... %s data loaded, total %s samples, %s missing \n' % (mode, len(data), missing))
    return data


class Batch():
    
    def __init__(self, data, batch_size, vocab, vocab_char):

        assert len(data) > 0
        assert batch_size > 0
        
        self.data = data
        self.vocab = vocab
        self.vocab_char = vocab_char
        self.label_map = {'entailment':0, 'neutral':1, 'contradiction':2}
        self.batch_size = batch_size
        self.start = 0
        
    def __iter__(self):
        return(self)
    
    def __next__(self):

        if self.start >= len(self.data):
            self.start = 0
            raise StopIteration


        labels, s1, s2, ch1, ch2 = self.create_batch(self.data[self.start:self.start+self.batch_size], self.vocab, self.vocab_char)
        self.start += self.batch_size
        
        return labels, s1, s2, ch1, ch2

    def max_len(self, alist):
        return np.max([len(x) for x in alist])

    def create_batch_chars(self, text, s, vocab_char, max_len):

        max_length_char = np.max([self.max_len(x) for x in text])

        s_char_mat = torch.zeros(s.size()[1], int(max_length_char), int(max_len))

        counter = 0
        for s_c in text:
            idxs = list(map(lambda w: [vocab_char[c] for c in w], s_c))
            s_char_temp = []
            s_char = []
            for idx in idxs:
                temp = np.pad(np.array(idx), 
                              pad_width=((0,max_length_char-len(idx))), 
                              mode="constant", constant_values=0)
                s_char_temp.append(temp)

            s_char_temp = np.array(s_char_temp).transpose()
            
            for tp in s_char_temp: 
                tp = np.pad(tp, 
                            pad_width=((0,max_len-len(tp))), 
                            mode="constant", constant_values=0)
                s_char.append(tp)
            
            s_char_mat[counter,:,:] = torch.LongTensor(np.array(s_char))
            counter += 1

        return s_char_mat

    def create_batch(self, raw_batch, vocab, vocab_char):
        
        #batch inputs
        all_txt = list(zip(*raw_batch))
        #print(all_txt)
        idxs = list(map(lambda w: self.label_map[w], all_txt[0]))
        labels = Variable(torch.LongTensor(idxs).view(len(raw_batch),1))
        #print(labels)

        #sentence 1
        idxs = list(map(lambda output: [vocab[w] for w in output], all_txt[1]))
        max_length_s1 = np.max(list(map(len, idxs)))
        #print(max_length)
        
        s1 = []
        for idx in idxs:
            temp = np.pad(np.array(idx), 
                          pad_width=((0,max_length_s1-len(idx))), 
                          mode="constant", constant_values=0)
            s1.append(temp)


        s1 = Variable(torch.LongTensor(np.array(s1).transpose()))
        #sentence 2
        idxs = list(map(lambda output: [vocab[w] for w in output], all_txt[2]))
        max_length_s2 = np.max(list(map(len, idxs)))
        s2 = []
        for idx in idxs:
            temp = np.pad(np.array(idx), 
                          pad_width=((0,max_length_s2-len(idx))), 
                          mode="constant", constant_values=0)
            s2.append(temp)

        s2 = Variable(torch.LongTensor(np.array(s2).transpose()))


        ### create character tensor
        s1_char_mat = Variable(self.create_batch_chars(all_txt[1], s1, vocab_char, max_length_s1).type(torch.LongTensor))
        s2_char_mat = Variable(self.create_batch_chars(all_txt[2], s2, vocab_char, max_length_s2).type(torch.LongTensor))
        labels = labels.squeeze(1)
        return labels, s1, s2, s1_char_mat, s2_char_mat

def all_vocab_emb(args):
    emb = np.load(args.source+args.file_emb).item()
    
    vcb_trn = np.load(args.source+args.vocab_trn).item()
    vcb_dev = np.load(args.source+args.vocab_dev).item()
    vcb_tst = np.load(args.source+args.vocab_tst).item()

    vcb_trn_char = np.load(args.source+args.vocab_trn_char).item()
    vcb_dev_char = np.load(args.source+args.vocab_dev_char).item()
    vcb_tst_char = np.load(args.source+args.vocab_tst_char).item()
    
    vcb_all = vcb_trn.copy()

    # zero is reserved for padding
    count = len(vcb_trn)+1

    # stack vocab_trn, vocab_dev and vocab_tst
    for i in vcb_dev.keys():
        if i not in vcb_all.keys():
            vcb_all[i] = count
            count += 1

    for i in vcb_tst.keys():
        if i not in vcb_all.keys():
            vcb_all[i] = count
            count += 1
    

    vcb_size = len(vcb_all)

    # initialize random embedding
    all_emb = np.random.normal(0, 1, (len(vcb_all)+1, args.emb))
    
    trn_keys = list(vcb_trn.keys())
    count = 0
    
    # replace with pre_trained embedding if exists
    for r in range(len(trn_keys)):
        k = trn_keys[r]
        if type(emb[k]) != int:
            all_emb[r+1, :] = list(map(float, emb[k]))
            count += 1

    # stack character vocabulary
    vcb_all_char = vcb_trn_char.copy()
    count = len(vcb_trn_char) + 1

    for i in vcb_dev_char.keys():
        if i not in vcb_all_char.keys():
            vcb_all_char[i] = count
            count += 1

    for i in vcb_tst_char.keys():
        if i not in vcb_all_char.keys():
            vcb_all_char[i] = count
            count += 1

    return vcb_all, all_emb, vcb_all_char
        
class PredictionLayer(nn.Module):

    """ Feed a fixed-length matching vector to probability distribution
    We employ a two layer feed-forward neural network to consume the fixed-length matching vector,
    and apply the softmax function in the output layer.

    Attributes:
        input_size: 4*hidden_size(4*100)
        hidden_size: default 100
        output_size: num of classes, default 3 in our SNLI task
        dropout: default 0.1

    Dimensions:
        Input: batch_size * sequence_length * (4* hidden_size)(4*100)
        output: batch_size * sequence_length * num_classes
    """

    def __init__(self, input_size, hidden_size=100, output_size=3, dropout=0.1):
        super(PredictionLayer, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.linear1(x)
        out = self.linear2(out)
        out = self.dropout(out)
        out = self.softmax(out)
        _, predicted = torch.max(out.data, 1)
        return out

def test_model(model, batch, args):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0

    model.eval()

    for labels, s1, s2, ch1, ch2 in batch:
            
        if args.cuda:
            labels, s1, s2, ch1, ch2 = labels.cuda(), s1.cuda(), s2.cuda(), ch1.cuda(), ch2.cuda()

        outputs = model(labels, s1, s2, ch1, ch2)

        predicted = (outputs.data.max(1)[1]).long().view(-1)

        total += labels.size(0)

        correct += (predicted == labels.data).sum()
    model.train()

    return (100 * correct / total)

