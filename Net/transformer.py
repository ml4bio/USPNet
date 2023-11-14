import numpy as np
import torch
import torch.nn as nn
from Net.log import logger
import math


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, dim_model, attn_dropout=0.1):
        # root square of dimension size
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(dim_model, 0.5)
        self.softmax = nn.Softmax()
        self.attention_dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        ''' Returns the softmax scores and attention tensor '''
        attention = torch.matmul(q, k.transpose(-2, -1)) / self.temper
        attention = self.softmax(attention)
        attention = self.attention_dropout(attention)
        output = torch.bmm(attention, v)
        return output, attention


class MultiHeadAttention(nn.Module):
    ''' Multihead attention '''

    def __init__(self, qty_head, dim_model, dim_k, dim_v, dropout=0.1, attn_dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.qty_head = qty_head
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_model = dim_model

        self.weight_q = nn.Parameter(torch.FloatTensor(qty_head, dim_model, dim_k))
        self.weight_k = nn.Parameter(torch.FloatTensor(qty_head, dim_model, dim_k))
        self.weight_v = nn.Parameter(torch.FloatTensor(qty_head, dim_model, dim_v))

        self.attention_model = ScaledDotProductAttention(dim_model, attn_dropout=attn_dropout)
        self.layer_norm = nn.LayerNorm(dim_model)
        # V vectors of each head are concatenated
        self.projection = nn.Linear(qty_head * dim_v, dim_model)

        self.dropout = nn.Dropout(dropout)

        torch.nn.init.xavier_normal_(self.weight_q)
        torch.nn.init.xavier_normal_(self.weight_k)
        torch.nn.init.xavier_normal_(self.weight_v)

    def forward(self, q, k, v):
        residual = q

        batch_size, q_len, dim_model = q.size()

        _, k_len, _ = k.size()
        _, v_len, _ = v.size()

        # Reshaping considering number of heads
        q_vector = q.repeat(self.qty_head, 1, 1).view(self.qty_head, -1, self.dim_model)
        k_vector = k.repeat(self.qty_head, 1, 1).view(self.qty_head, -1, self.dim_model)
        v_vector = v.repeat(self.qty_head, 1, 1).view(self.qty_head, -1, self.dim_model)

        q_vector = torch.bmm(q_vector, self.weight_q).view(-1, q_len, self.dim_k)
        k_vector = torch.bmm(k_vector, self.weight_k).view(-1, k_len, self.dim_k)
        v_vector = torch.bmm(v_vector, self.weight_v).view(-1, v_len, self.dim_v)

        outputs, attentions = self.attention_model(q_vector, k_vector, v_vector)

        outputs = torch.cat(torch.split(outputs, batch_size, dim=0), dim=-1)
        outputs = self.projection(outputs)
        outputs = self.dropout(outputs)

        return self.layer_norm(outputs + residual), attentions


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, dim_hidden, dim_inner_hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.layer_1 = nn.Conv1d(dim_hidden, dim_inner_hidden, 1)  # position-wise
        self.layer_2 = nn.Conv1d(dim_inner_hidden, dim_hidden, 1)  # position-wise
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_hidden)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        # print("Input of fnn {}".format(x.size()))
        # print("transposed Input of fnn {}".format(x.transpose(1, 2).size()))
        output = self.relu(self.layer_1(x.transpose(1, 2)))
        # print("First convolution of fnn {}".format(output.size()))
        output = self.layer_2(output).transpose(2, 1)
        # print("Second convolution of fnn {}".format(output.size()))
        output = self.dropout(output)
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    ''' Transformer encoder layer '''

    def __init__(self, dim_model, dim_inner_hidden, qty_head, dim_k, dim_v, dropout=0.1, attn_dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(qty_head, dim_model, dim_k,
                                                 dim_v, dropout=dropout, attn_dropout=attn_dropout)
        self.feedforward = PositionwiseFeedForward(dim_model, dim_inner_hidden, dropout)

    def forward(self, input_tensor):
        output, attention = self.self_attention(input_tensor, input_tensor, input_tensor)
        output = self.feedforward(output)
        return output, attention


def position_encoding_init(positions, dim_word_vector):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_encoder = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim_word_vector) for j in range(dim_word_vector)]
        if pos != 0 else np.zeros(dim_word_vector) for pos in range(positions)])

    position_encoder[1:, 0::2] = np.sin(position_encoder[1:, 0::2])  # dim 2i
    position_encoder[1:, 1::2] = np.cos(position_encoder[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_encoder).type(torch.FloatTensor)


class TransformerEncoder(nn.Module):
    ''' A neural network Transformer Encoder '''

    def __init__(self, vocab_size, max_sequence_length, qty_encoder_layer=1, qty_attention_head=8,
                 dim_k=32, dim_v=32, dim_word_vector=256, dim_model=256, dim_inner_hidden=128, output_size=3,
                 dropout=0.2, attn_dropout=0.1, embedding=False):
        super(TransformerEncoder, self).__init__()
        positions = max_sequence_length  # counting UNK

        self.max_sequence_length = max_sequence_length
        self.dim_model = dim_model

        # Embedding containing sentence order information
        self.position_encoder = nn.Embedding(positions, dim_word_vector, padding_idx=0)
        self.position_encoder.weight.data = position_encoding_init(positions, dim_word_vector)

        # Embedding vector of words. TODO: test with word2vec
        self.word_embedding_layer = nn.Embedding(vocab_size, dim_word_vector, padding_idx=0)

        # Create a set of encoder layers, given the quantity informed in
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(dim_model, dim_inner_hidden, qty_attention_head, dim_k, dim_v, dropout=dropout, attn_dropout=attn_dropout)
            for _ in range(qty_encoder_layer)
        ])

        # whether do embedding before attention module
        self.embedding = embedding
        logger.info('''Transformer Model:
                    - max sequence length = {}
                    - encoder layers = {}
                    - attention heads = {}
                    '''.format(max_sequence_length, qty_encoder_layer, qty_attention_head))

    def get_trainable_parameters(self):
        """ Avoid updating the position encoding """
        position_parameters = set(map(id, self.position_encoder.parameters()))
        return (p for p in self.parameters() if id(p) not in position_parameters)

    def forward(self, sequence):
        if(self.embedding):
            # lookup word embedding layer
            word_embedding = self.word_embedding_layer(sequence)
        else:
            word_embedding = sequence
        encoder_output = word_embedding

        for encoder_layer in self.encoder_layers:
            encoder_output, attentions = encoder_layer(encoder_output)

        return encoder_output

    def get_positions(self, sequence):
        """
            Get position
        :param sequence: input tensor
        :return: array with the order of each element. Example: [23, 45, 67, 54, PAD, PAD] ---> [1, 2, 3, 4, 0, 0]
        """

        PADDING = 0
        positions = [[pos + 1 if word != PADDING else 0 for pos, word in enumerate(instance)] for instance in sequence]
        return torch.autograd.Variable(torch.LongTensor(positions), volatile=False).cuda()