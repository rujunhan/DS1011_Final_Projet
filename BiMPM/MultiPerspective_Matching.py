"""
Layer construction for Context Layer/Multi-perspective matching Layer/Prediction Layer
"""


import torch
import torch.nn as nn
from torch.autograd import Variable


class MatchingLayer(nn.Module):

    def __init__(self, embed_dim = 100, epsilon=1e-6, perspective=10, type = 'all'):
        super(MatchingLayer , self).__init__()
        self.epsilon = epsilon
        self.embed_dim = embed_dim
        self.perspective = perspective
        #self.batch_size = batch_size

        w1 = torch.Tensor(perspective, embed_dim)
        w2 = torch.Tensor(perspective, embed_dim)
        w3 = torch.Tensor(perspective, embed_dim)
        w4 = torch.Tensor(perspective, embed_dim)
        w5 = torch.Tensor(perspective, embed_dim)
        w6 = torch.Tensor(perspective, embed_dim)
        w7 = torch.Tensor(perspective, embed_dim)
        w8 = torch.Tensor(perspective, embed_dim)

        nn.init.uniform(w1, -0.01, 0.01)
        nn.init.uniform(w2, -0.01, 0.01)
        nn.init.uniform(w3, -0.01, 0.01)
        nn.init.uniform(w4, -0.01, 0.01)
        nn.init.uniform(w5, -0.01, 0.01)
        nn.init.uniform(w6, -0.01, 0.01)
        nn.init.uniform(w7, -0.01, 0.01)
        nn.init.uniform(w8, -0.01, 0.01)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.w3 = nn.Parameter(w3)
        self.w4 = nn.Parameter(w4)
        self.w5 = nn.Parameter(w5)
        self.w6 = nn.Parameter(w6)
        self.w7 = nn.Parameter(w7)
        self.w8 = nn.Parameter(w8)

    def forward(self, s1, s2):
        # separate forward states and backward states
        s1_for = s1[:,:,:self.embed_dim]
        s1_back = s1[:,:,self.embed_dim:]
        s2_for = s2[:,:, :self.embed_dim]
        s2_back = s2[:,:, self.embed_dim:]

        cos_matrix_for = self.cosine_matrix(s1_for, s2_for)
        cos_matrix_back = self.cosine_matrix(s1_back, s2_back)

        # full matching
        temp = []
        full_matching_for = self.full_matching(s1_for, s2_for, self.w1)
        full_matching_back = self.full_matching(s1_back, s2_back, self.w2)
        temp.append(full_matching_for)
        temp.append(full_matching_back)
        full_matching = torch.cat(temp,2)
        # print(full_matching.size())

        # max pooling matching
        temp = []
        maxpooling_matching_for = self.max_pooling_matching(s1_for, s2_for, self.w3)
        maxpooling_matching_back = self.max_pooling_matching(s1_back, s2_back, self.w4)
        temp.append(maxpooling_matching_for)
        temp.append(maxpooling_matching_back)
        maxpooling_matching = torch.cat(temp, 2)
        # print(maxpooling_matching.size())

        # mean attentive matching
        temp = []
        mean_attentive_matching_for = self.mean_attentive_matching(s1_for, s2_for, self.w5, cos_matrix_for)
        mean_attentive_matching_back = self.mean_attentive_matching(s1_back, s2_back, self.w6, cos_matrix_back)
        temp.append(mean_attentive_matching_for)
        temp.append(mean_attentive_matching_back)
        mean_attentive_matching = torch.cat(temp, 2)
        #print(mean_attentive_matching.size())

        # max attentive matching
        temp = []
        max_attentive_matching_for = self.max_attentive_matching(s1_for, s2_for, self.w7, cos_matrix_for)
        max_attentive_matching_back = self.mean_attentive_matching(s1_back, s2_back, self.w8, cos_matrix_back)
        temp.append(max_attentive_matching_for)
        temp.append(max_attentive_matching_back)
        max_attentive_matching = torch.cat(temp, 2)
        # print(max_attentive_matching.size())

        if type == 'full':
            out = full_matching
        elif type == 'maxpooling':
            out = maxpooling_matching
        elif type == 'mean_attentive':
            out = mean_attentive_matching
        elif type == 'max_attentive':
            out = max_attentive_matching
        else:
            out = torch.cat([full_matching, maxpooling_matching, mean_attentive_matching, max_attentive_matching], dim=-1)
        return out

    def cos_calc(self, v1, v2):
        """
        calculate cosine similarity between two tensors
        :param v1:[..., 1, ..., embed_dim ]
        :param v2: [1, ..., ..., embed_dim]
        :return: [..., ..., ...]
        """
        cos = (v1 * v2).sum(-1)
        v1_norm = torch.sqrt(torch.sum(v1 ** 2, -1).clamp(min=self.epsilon))
        v2_norm = torch.sqrt(torch.sum(v2 ** 2, -1).clamp(min=self.epsilon))
        return cos / (v1_norm * v2_norm)

    def cosine_matrix(self, s1, s2):
        """
        calculate cosine similarity between each forward(or backward) contextual embedding and
        every forward(or backward) contextual embedding of the other sentences
        :param x1: [n_states1, batch_size, embed_dim]
        :param x2: [n_state2, batch_size, embed_dim]
        :return: [n_state1, n_state2, batch_size]
        """
        # expand x1 shape to (n_states1, 1, batch_size, embed_size)
        s1 = s1.unsqueeze(1)
        # expand x2 shape to (1, n_states2, batch_size, embed_size)
        s2 = s2.unsqueeze(0)
        # return cosine matrix (n_states1, n_states2, batch_size)
        return self.cos_calc(s1, s2)

    def element_multiply_vector(self, v, w):
        """
        calculate element_wise multiplication between a state vector(tensor) and weights matrix W
        :param v: [batch_size, embed_dim]
        :param W: [perspective, embed_dim]
        :return: [batch_size, perspective, embed_size]
        """
        # expand state vector shape to: [batch_size, 1, embed_dim]
        v = v.unsqueeze(1)
        # expand W shape to: [1, perspective, embed_dim]
        w = w.unsqueeze(0)
        # return element-wise multiply
        out = v*w
        return out

    def element_multiply_states(self, s, w):
        """
        calculate element_wise multiplication between every state vectors and weights matrix W without for loop
        :param s: [n_state, batch_size, embed_dim]
        :param W: [perspective, embed_dim]
        :return: [n_state, batch_size, perspective, embed_dim]
        """
        # concatenate state dimension and batch size dimention: [n_state*batch_size, embed_dim]
        batch_size = s.size()[1]
        s = s.contiguous().view(-1, self.embed_dim)
        # reshape to (n_state*batch_size, 1, embedding_size)
        s = torch.unsqueeze(s, 1)
        # reshape weights to (1, perspective, embed_size)
        w = torch.unsqueeze(w, 0)
        # element-wise multiply
        out = s * w
        # reshape to original shape
        out = out.view(-1, batch_size, self.perspective, self.embed_dim)
        return out

    def full_matching(self, s1, s2, w):
        """
        return full matching strategy result
        :param s1: [n_state1, batch_size, embed_dim]
        :param s2: [n_state2, batch_size, embed_dim]
        :param W: [perspective, embed_dim]
        :return: [n_state1, batch_size, perspective]
        """

        # get s2 forward last step hidden vector: [batch_size, embed_dim]
        s2_last_state = s2[-1, :, :]
        # get weighted s1: [n_state1, batch_size, perspective, embed_size]
        weighted_s1 = self.element_multiply_states(s1, w)
        # get weighted s2 last state: [batch_size, perspective, embed_size]
        s2 = self.element_multiply_vector(s2_last_state, w)
        #reshape weighted s2 last state to: [1, batch_size, perspective, embed_size]
        weighted_s2 = s2.unsqueeze(0)
        result = self.cos_calc(weighted_s1, weighted_s2)
        return result

    def max_pooling_matching(self, s1, s2, w):
        """
        return max pooling matching strategy result
        :param s1: [n_state1, batch_size, embed_dim]
        :param s2: [n_state2, batch_size, embed_dim]
        :param W: [perspective, embed_dim]
        :return: [n_state1, batch_size, perspective]
        """
        # get weighted s1: [n_state1, batch_size, perspective, embed_size]
        weighted_s1 = self.element_multiply_states(s1, w)
        # get weighted s2: [n_state2, batch_size, perspective, embed_size]
        weighted_s2 = self.element_multiply_states(s2, w)

        # reshape weighted s1 to [n_state1, 1, batch_size, perspective, embed_size]
        weighted_s1 = weighted_s1.unsqueeze(1)
        # reshape weighted s2 to  [1, n_state2, batch_size, perspective, embed_size]
        weighted_s2 = weighted_s2.unsqueeze(0)
        # cosine similarity, [n_state1, n_state2, batch_size, perspective]
        cos = self.cos_calc(weighted_s1, weighted_s2)
        result = cos.max(1)[0]
        return result

    def mean_attentive_vectors(self, s, cosine_matrix):
        """
        return attentive vectors for the entire sentence s2 by weighted summing all the contextual embeddings of s2
        :param s: [n_state2, batch_size, embed_dim]
        :param cosine_matrix: [n_state1, n_state2, batch_size]
        :return: [n_state1, batch_size, embed_size]
        """
        # reshape cosine matrix to : [n_state1, n_state2, batch_size, 1]
        expanded_cosine_matrix = cosine_matrix.unsqueeze(-1)
        # reshape state vectors to : [1, n_state2, batch_size, embed_size]
        s = s.unsqueeze(0)
        # weighted summing up to : [n_state1, batch_size, embed_dim]
        weighted_sum_cos = (expanded_cosine_matrix * s).sum(1)
        # summing up: [n_state1, batch_size, 1]
        sum_cos = (cosine_matrix.sum(1) + self.epsilon).unsqueeze(-1)
        return weighted_sum_cos / sum_cos

    def mean_attentive_matching(self, s1, s2, w, cos_matrix):
        """
        return mean attentive matching stretegy result
        :param s1: [n_state1, batch_size, embed_dim]
        :param s2: [n_state2, batch_size, embed_dim]
        :param W: [perspective, embed_dim]
        :return: [n_state1, batch_size, perspective]
        """
        # get weighted_s1: [n_state1, batch_size, perspective, embed_dim]
        weighted_s1 = self.element_multiply_states(s1, w)
        # get attentive vector for s2: [n_state1, batch_size, embed_dim]
        attentive_vector = self.mean_attentive_vectors(s2, cos_matrix)
        # get weighted s2: [n_state1, batch_size, perspective, embed_dim]
        weighted_s2 = self.element_multiply_states(attentive_vector, w)
        result = self.cos_calc(weighted_s1, weighted_s2)
        return result

    def max_attentive_vectors(self, s, cosine_matrix):
        """
        return attentive vectors for the entire sentence s2 by picking the contextual embedding with the highest cosine similarity
        :param s: [n_state2, batch_size, embed_dim]
        :param cosine_matrix: [n_state1, n_state2, batch_size]
        :return: [n_state1, batch_size, embed_size]
        """
        batch_size = s.size()[1]
        _, max_s = cosine_matrix.max(1)
        # concatenate max_x to [n_state1*batch_size]
        max_s = max_s.contiguous().view(-1)
        s = s.contiguous().view(-1, self.embed_dim)
        max_s_vectors = s[max_s]
        result = max_s_vectors.view(-1, batch_size, self.embed_dim)
        return result

    def max_attentive_matching(self, s1, s2, w, cos_matrix):
        """
        return max attentive matching strategy result
        :param s1: [n_state1, batch_size, embed_dim]
        :param s2: [n_state2, batch_size, embed_dim]
        :param W: [perspective, embed_dim]
        :return: [n_state1, batch_size, perspective]
        """
        weighted_s1 = self.element_multiply_states(s1, w)
        # get attentive vector for s2: [n_state1, batch_size, embed_dim]
        attentive_vector = self.max_attentive_vectors(s2, cos_matrix)
        # get weighted s2: [n_state1, batch_size, perspective, embed_dim]
        weighted_s2 = self.element_multiply_states(attentive_vector, w)
        result = self.cos_calc(weighted_s1, weighted_s2)
        return result


class ContextLayer(nn.Module):

    """ Feed a concatenated [WordEmbedding, CharEmbedding] into a BiLSTM layer

    Attributes:
        input_size: embed_dim
        hidden_size: default 100
        dropout: default 0.1
    Dimensions:
        Input: batch_size * sequence_length * (4*embed_size)
        Output: batch_size * sequence_length * (2 * hidden_size)

    """
    def __init__(self, input_size, hidden_size=100,  num_layers = 1, dropout=0.1):
        super(ContextLayer, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, bias=False, dropout=dropout, bidirectional = True)

    def forward(self, x):
        out = self.rnn(x)
        return out


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
        return out


if __name__ == '__main__':
    s1 = torch.randn(14, 5, 350)
    s2 = torch.randn(17, 5, 350)
    batch_size = 5
    embed_size = 100
    perspective = 50
    hidden_size = 100
    n_layers = 1

    s1 = Variable(s1)
    s2 = Variable(s2)

    context = ContextLayer(input_size = 350, hidden_size=100)
    matching = MatchingLayer(embed_dim=100, epsilon=1e-6, perspective=50, type = 'all')
    aggregation = ContextLayer(input_size = 8*perspective, hidden_size=100,  dropout=0.1)
    pre = PredictionLayer(input_size=4*hidden_size, hidden_size=100, output_size=3, dropout=0.1)

    out1, _ = context(s1)
    out2, _ = context(s2)
    # [n_state, batch_size, 2*hidden_size]
    print(out1.size())
    print(out2.size())
    out3 = matching(out1, out2)
    out4 = matching(out2, out1)
    # [n_state, batch_size, 8*perspective]
    print(out3.size())
    print(out4.size())
    out5, _ = aggregation(out3)
    out6, _ = aggregation(out4)
    #[n_state, batch_size, 2*hidden_size]
    print(out5.size())
    print(out6.size())
    # get fixed-length vector from the last time-step of the BiLSTM model
    pre_list = []
    pre_list.append(out5[-1, :, :hidden_size]) # last timestamp from forward
    pre_list.append(out5[0, :, hidden_size:]) # last timestamp from backward
    pre_list.append(out6[-1, :, :hidden_size])
    pre_list.append(out6[0, :, :hidden_size])
    prediction = torch.cat(pre_list, -1)
    # [batch_size, 4*hidden_size]
    print(prediction.size())
    out = pre(prediction)
    print(out)
