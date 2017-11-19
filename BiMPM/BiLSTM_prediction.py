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
from sklearn.manifold import TSNE
import random

from bilstm import Batch
from bilstm import loadData,loadData_sample, all_vocab_emb
from MultiPerspective_Matching import MatchingLayer, ContextLayer, PredictionLayer


class BiLSTM(nn.Module):
    def __init__(self, vocab, vocab_char, emb, args):

        super(BiLSTM, self).__init__()

        self.num_words = len(vocab)
        self.num_chars = len(vocab_char)
        self.embed_size = emb.shape[1]
        self.char_embed_size = args.char_emb_size
        self.char_size = args.char_size

        self.hid_size = args.hid
        self.batch_size = args.batch
        self.num_layers = args.num_layers
        self.num_classes = args.num_classes
        self.dropout = args.dropout
        self.sent_length = 0
        self.word_length = 0

        ### embedding layer
        self.emb = nn.Embedding(self.num_words + 1, self.embed_size, sparse=True)
        self.emb.weight = Parameter(torch.FloatTensor(emb))

        ### character representation
        self.emb_char = nn.Embedding(self.num_chars + 1, self.char_embed_size)
        self.emb_char.weight = Parameter(torch.FloatTensor(self.num_chars + 1, self.char_embed_size).uniform_(-1, 1))

        self.char_representation = nn.LSTM(self.char_embed_size, self.char_size, self.num_layers, bias=False,
                                           bidirectional=False)

        ### lstm layer
        self.context = ContextLayer(input_size=self.embed_size+self.char_size, hidden_size=self.hid_size, num_layers = self.num_layers)
        self.pre = PredictionLayer(input_size = 4 * self.hid_size, hidden_size = self.hid_size, output_size= self.num_classes, dropout = self.dropout)

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hid_size)),
                autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hid_size)))

    def forward(self, labels, s1, s2, ch1, ch2):

        use_cuda = self.emb.weight.is_cuda

        batch_size = s1.size()[0]

        ### look up the embedding for both sencetences
        s1_emb = self.emb(s1)
        s2_emb = self.emb(s2)

        ### create character representations through LSTM

        # s1
        self.word_length = ch1.size()[1]
        self.sent_length = ch1.size()[2]

        all_char1 = []
        for ch in range(ch1.size()[0]):
            ch_emb = self.emb_char(ch1[ch, :, :].squeeze(0))
            _, s1_chr = self.char_representation(ch_emb)
            # the first element of s1_chr is the last hidden state
            # we use this as character embedding
            all_char1.append(s1_chr[0].squeeze(0).unsqueeze(1))
        all_char1 = torch.cat(all_char1, 1)

        # s2
        self.word_length = ch2.size()[1]
        self.sent_length = ch2.size()[2]

        all_char2 = []
        for ch in range(ch2.size()[0]):
            ch_emb = self.emb_char(ch2[ch, :, :].squeeze(0))
            _, s2_chr = self.char_representation(ch_emb)
            all_char2.append(s2_chr[0].squeeze(0).unsqueeze(1))

        all_char2 = torch.cat(all_char2, 1)

        ### Context Layer
        s1 = torch.cat([s1_emb, all_char1], 2)
        s2 = torch.cat([s2_emb, all_char2], 2)

        s1_context_out, _ = self.context(s1)
        s2_context_out, _ = self.context(s2)

        ### Prediction Layer
        pre_list = []
        pre_list.append(s1_context_out[-1, :, :self.hid_size])  # the last vector in forward
        pre_list.append(s1_context_out[0, :, self.hid_size:])  # the last vector in backward
        pre_list.append(s2_context_out[-1, :, :self.hid_size])
        pre_list.append(s2_context_out[0, :, self.hid_size:])
        out = torch.cat(pre_list, -1)
        out = self.pre(out)

        return out


def evaluation(model, batch):
    model.eval()
    correct = 0
    total = 0
    for labels, s1, s2, ch1, ch2 in batch:
        output = model(labels, s1, s2, ch1, ch2)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
    return correct / float(total)


def main(args):
    vocab, emb, vocab_chars = all_vocab_emb(args)

    batch_size = args.batch

    data = loadData_sample(vocab, 10000, args)

    random.shuffle(data)

    n_batch = int(np.ceil(len(data) / batch_size))

    model = BiLSTM(vocab, vocab_chars, emb, args)

    batch = Batch(data, batch_size, vocab, vocab_chars)

    if args.cuda:
        model.cuda()

    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    losses = []

    print("start training model...\n")
    start_time = time.time()
    count = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for labels, s1, s2, ch1, ch2 in batch:

            if args.cuda:
                labels, s1, s2, ch1, ch2 = labels.cuda(), s1.cuda(), s2.cuda(), ch1.cuda(), ch2.cuda()

            if batch.start % 1000 == 0:
                print("training epoch %s: completed %s %%" % (
                str(epoch), str(round(100 * batch.start / len(data), 2))))

            model.zero_grad()
            out = model(labels, s1, s2, ch1, ch2)
            loss = criterion(out, labels)
            total_loss += loss

            loss.backward()
            optimizer.step()

            # total_loss+=loss.data.cpu().numpy()[0]

        ave_loss = total_loss/n_batch
        print("average loss is: %s" % str(ave_loss))
        losses.append(ave_loss)
        end_time = time.time()
        print("%s seconds elapsed" % str(end_time - start_time))
        accuracy = evaluation(model, batch)
        print("Current accuracy: " , accuracy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-emb', type=int, default=300)
    parser.add_argument('-hid', type=int, default=100)
    parser.add_argument('-num_layers', type=int, default=1)
    parser.add_argument('-batch', type=int, default=5)
    parser.add_argument('-epochs', type=int, default=1)
    parser.add_argument('-seed', type=int, default=123)
    parser.add_argument('-lr', type=float, default=0.05)
    parser.add_argument('-num_classes', type=int, default=3)
    parser.add_argument('-dropout', type=float, default=0.1)

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

    parser.add_argument('-source', type=str)
    parser.add_argument('-saveto', type=str)

    parser.add_argument('-cuda', type=bool, default=False)

    parser.add_argument('-char_emb_size', type=int, default=20)
    parser.add_argument('-char_size', type=int, default=50)

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

    args.cuda = False

    main(args)
