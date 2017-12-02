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


class BiMPM(nn.Module):
    def __init__(self, vocab, vocab_char, emb, args):

        super(BiMPM, self).__init__()

        self.num_words = len(vocab)
        self.num_chars = len(vocab_char)
        self.embed_size = emb.shape[1]
        self.char_embed_size = args.char_emb_size
        self.char_size = args.char_size

        self.hid_size = args.hid
        self.batch_size = args.batch
        self.num_classes = args.num_classes
        self.dropout = args.dropout
        self.num_context_layers = args.num_context_layers
        self.sent_length = 0
        self.word_length = 0
        self.epsilon = 1e-6
        self.perspective = args.perspective

        self.use_cuda = args.cuda
        ### embedding layer
        self.emb = nn.Embedding(self.num_words+1, self.embed_size, sparse=True)
        self.emb.weight = Parameter(torch.FloatTensor(emb))
        self.emb.weight.requires_grad = False


        ### character representation
        ### character representation
        self.emb_char = nn.Embedding(self.num_chars + 1, self.char_embed_size)
        self.emb_char.weight = Parameter(torch.FloatTensor(self.num_chars + 1, self.char_embed_size).uniform_(-1, 1))

        self.char_representation = nn.LSTM(self.char_embed_size,
                                           self.char_size,
                                           args.num_context_layers,
                                           dropout = self.dropout,
                                           bias=False,
                                           bidirectional=False)
        #self.char_hid = self.init_hidden_chars()

        ### lstm layer
        self.lstm_s1 = nn.LSTM(self.embed_size+self.char_size,
                               self.hid_size, args.num_context_layers,
                               bias = False,
                               dropout=self.dropout,
                               bidirectional=True)
        #self.lstm_s2 = nn.LSTM(self.embed_size+self.char_size, self.hid_size, args.num_context_layers, bias = False, dropout=self.dropout, bidirectional=True)

        #self.s1_hid = self.init_hidden(self.batch_size)
        #self.s2_hid = self.init_hidden(self.batch_size)


        ### dropout layer
        self.dropout_layer = nn.Dropout(p=self.dropout)

        ### other layers
        self.matching = MatchingLayer(embed_dim=self.hid_size,
                                      epsilon=self.epsilon,
                                      perspective=self.perspective,
                                      type='all')
        self.aggregation = AggregationLayer(input_size=8 * self.perspective,
                                            hidden_size=self.hid_size,
                                            num_layers=args.num_aggr_layers,
                                            dropout=self.dropout)
        self.pre = PredictionLayer(input_size =4 * self.hid_size,
                                   hidden_size = self.hid_size,
                                   output_size=self.num_classes,
                                   dropout = self.dropout)

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if self.use_cuda:
            return (Variable(torch.zeros(self.num_context_layers * 2, batch_size, self.hid_size)).cuda(),
                    Variable(torch.zeros(self.num_context_layers * 2, batch_size, self.hid_size)).cuda())
        else:
            return (Variable(torch.zeros(self.num_context_layers * 2, batch_size, self.hid_size)),
                    Variable(torch.zeros(self.num_context_layers * 2, batch_size, self.hid_size)))

    def init_hidden_chars(self):
        if self.use_cuda:
            return (Variable(torch.zeros(self.num_context_layers, self.sent_length, self.char_size)).cuda(),
                    Variable(torch.zeros(self.num_context_layers, self.sent_length, self.char_size)).cuda())
        else:
            return (Variable(torch.zeros(self.num_context_layers, self.sent_length, self.char_size)),
                    Variable(torch.zeros(self.num_context_layers, self.sent_length, self.char_size)))

    def forward(self, labels, s1, s2, ch1, ch2):

        use_cuda = self.emb.weight.is_cuda

        batch_size = s1.size()[1]

        ### look up the embedding for both sencetences
        s1_emb = self.emb(s1)
        s2_emb = self.emb(s2)

        ### create character representations through LSTM

        # s1
        self.word_length = ch1.size()[1]
        self.sent_length = ch1.size()[2]

        all_char1 = []
        for ch in range(ch1.size()[2]):
            ch_emb = self.emb_char(ch1[:,:,ch].squeeze(-1))
            s1_chr, _ = self.char_representation(ch_emb)
            # the last element of s1_chr is the last hidden state
            # we use this as character embedding
            all_char1.append(s1_chr[:,-1:,].squeeze(1).unsqueeze(0))
        all_char1 = torch.cat(all_char1, 0)

        # s2
        self.word_length = ch2.size()[1]
        self.sent_length = ch2.size()[2]

        all_char2 = []
        for ch in range(ch2.size()[2]):
            ch_emb = self.emb_char(ch2[:, :, ch].squeeze(-1))
            #self.char_hid = self.init_hidden_chars()
            s2_chr, _ = self.char_representation(ch_emb)
            all_char2.append(s2_chr[:,-1:,].squeeze(1).unsqueeze(0))

        all_char2 = torch.cat(all_char2, 0)

        """
        all_char1 = []
        for ch in range(ch1.size()[0]):
            ch_emb = self.emb_char(ch1[ch, :, :].squeeze(0))
            self.char_hid = self.init_hidden_chars()
            _, s1_chr = self.char_representation(ch_emb, self.char_hid)
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
            self.char_hid = self.init_hidden_chars()
            _, s2_chr = self.char_representation(ch_emb, self.char_hid)
            all_char2.append(s2_chr[0].squeeze(0).unsqueeze(1))

        all_char2 = torch.cat(all_char2, 1)

        """

        ## Dropout after representation layer
        s1_input = self.dropout_layer(torch.cat([s1_emb, all_char1], 2))
        s2_input = self.dropout_layer(torch.cat([s2_emb, all_char2], 2))
        #s1_input = torch.cat([s1_emb, all_char1], 2)
        #s2_input = torch.cat([s2_emb, all_char2], 2)

        ### Context Layer
        #self.s1_hid = self.init_hidden(batch_size)
        #self.s2_hid = self.init_hidden(batch_size)
        out1, _ = self.lstm_s1(s1_input)
        out2, _ = self.lstm_s1(s2_input)

        
        ### Matching Layer
        out3 = self.matching(out1, out2)
        out4 = self.matching(out2, out1)

        ## Dropout after matching                                                                 
        out3 = self.dropout_layer(out3)
        out4 = self.dropout_layer(out4)

        ### Aggregation Layer
        out5, _ = self.aggregation(out3)
        out6, _ = self.aggregation(out4)

        ### Prediction Layer
        pre_list = []
        pre_list.append(out5[-1, :, :self.hid_size])  # the last vector in forward
        pre_list.append(out5[0, :, self.hid_size:])  # the last vector in backward
        pre_list.append(out6[-1, :, :self.hid_size])
        pre_list.append(out6[0, :, self.hid_size:])
        out = torch.cat(pre_list, -1)
        out = self.pre(out)

        return out

def main(args):
    

    # initialize the logger
    # create logger
    logger_name = "mylog"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # file handler
    fh = logging.FileHandler(args.log_dir + args.log_fname)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # stream handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)


    for arg in vars(args):
        logger.info(str(arg) + ' ' + str(getattr(args, arg)))

    logger.info('keeping dropout layers with updated matching layers')
    # load train/dev/test data
    # train data
    logger.info('loading data...')


    vocab, emb, vocab_chars = all_vocab_emb(args)

    batch_size = args.batch

    data = loadData(vocab, args)
    #data = loadData_sample(vocab,10000, args)
    random.shuffle(data)

    n_batch = int(np.ceil(len(data) / batch_size))

    model = BiMPM(vocab, vocab_chars, emb, args)

    batch = Batch(data, batch_size, vocab, vocab_chars)

    if args.cuda:
        model.cuda()

    para = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    losses = []
    eval_acc_hist = []
    train_acc_hist = []
    best_acc = 0.000

    logger.info("start training model...\n")
    start_time = time.time()
    count = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        for labels, s1, s2, ch1, ch2 in batch:

            if args.cuda:
                labels, s1, s2, ch1, ch2 = labels.cuda(), s1.cuda(), s2.cuda(), ch1.cuda(), ch2.cuda()

            model.zero_grad()
            out = model(labels, s1, s2, ch1, ch2)

            loss = criterion(out, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clipper)
            optimizer.step()

            total_loss += loss.data.cpu().numpy()[0]
            predicted = (out.data.max(1)[1]).long().view(-1)
            correct += (predicted == labels.data).sum()

            if batch.start % 50000 == 0:
                logger.info("training epoch %s: completed %s %%" % (str(epoch), str(round(100 * batch.start / len(data), 2))))
                logger.info("current training loss %.3f: " % loss.data.cpu().numpy()[0])

        ave_loss = total_loss / n_batch
        logger.info("average trainning loss is: %s" % str(ave_loss))
        losses.append(ave_loss)
        
        train_acc = 100 * correct / len(data)
        train_acc_hist.append(train_acc)
        
        ### evaluate the model after each epoch
        eval_data = loadData(vocab, args, mode='eval')
        eval_batch = Batch(eval_data, batch_size, vocab, vocab_chars)
        eval_acc = test_model(model, eval_batch, args)
        if eval_acc > best_acc:
            model_fname = '%s_epoch-%d_val-acc-%.3f' % (args.log_fname.split('.')[0], epoch, eval_acc)
            torch.save(model.state_dict(), model_fname + 'bimpm.pt')
            best_acc = eval_acc
        eval_acc_hist.append(eval_acc)
        logger.info("completed epoch %s, training accuracy is: %s %%, evaluation accuracy is: %s %%" % (epoch, round(train_acc, 2), round(eval_acc, 2)))
        end_time = time.time()
        logger.info("%s seconds elapsed" % str(end_time - start_time))
    
    logger.info("training loss history: ")
    logger.info(losses)
    logger.info("training accuracy history: ")
    logger.info(train_acc_hist)
    logger.info("evaluation accuracy history: ")
    logger.info(eval_acc_hist)
    test_data = loadData(vocab, args, mode='test')
    test_batch = Batch(test_data, batch_size, vocab, vocab_chars)
    test_acc = test_model(model, test_batch, args)
    logger.info("testing accuracy is: %s %%" % test_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-emb', type=int, default=300)
    parser.add_argument('-hid', type=int, default=100)
    parser.add_argument('-num_context_layers', type=int, default=1)
    parser.add_argument('-num_aggr_layers', type=int, default=1)
    parser.add_argument('-batch', type=int, default=60)
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-seed', type=int, default=123)
    parser.add_argument('-lr', type=float, default=0.0005)
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

    parser.add_argument('-source', type=str)
    parser.add_argument('-saveto', type=str)

    parser.add_argument('-cuda', type=bool, default=False)

    parser.add_argument('-char_emb_size', type=int, default=20)
    parser.add_argument('-char_size', type=int, default=50)

    parser.add_argument('--log_dir', type=str, default='../log')

    parser.add_argument('--log_fname', type=str, default='test4.log')

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

    args.cuda = True

    print('no lamba_l2, train with pers = 20, num_layer_aggr = 1, keep dropout layers for all \n')
    main(args)
