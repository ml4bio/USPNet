import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from Net.LSTM import *
from Net.CNN import *
from Net.SelfAttentionTorch import MultiHeadAttention
from Net.transformer import TransformerEncoder
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np
from Net.CRF import CRF
from Net.LSTM_Attention import LSTM_attention

embedding_feature_dim_msa = 768
embedding_feature_dim_pro = 1024
class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out
class Attention_CRF(nn.Module):

    def __init__(self, config, config1, cnn_configs, lstm_lan_config, lstm_config,
                 use_CRF=False, use_attention=True, reweight_ratio=None):
        super(Attention_CRF, self).__init__()

        self.num_classes = 20
        self.max_len = config1['max_text_len']

        self.embedding = nn.Embedding(num_embeddings=config['vocab_size'], embedding_dim=config['embedding_size'])

        self.ef1 = 512
        self.ef2 = 144
        self.ef3 = 32
        self.csef = 11
        self.ef4 = 256
        self.ef5 = 256
        self.ef6 = 64

        if (use_CRF):
            self.crf = CRF(num_tags=11, reweight_ratio=reweight_ratio)#original: num_tags=9
            self.use_CRF = True

        else:
            self.use_CRF = False

        self.linear = nn.Sequential(nn.Linear(config['embedding_size'], config1['input_dim'] - 4), nn.ReLU())
        self.linear2 = nn.Linear(self.max_len, self.max_len)
        self.lstm = BLSTM(config1)
        self.cnn = TextCNN(cnn_configs[0])
        self.cnn1 = TextCNN(cnn_configs[1])

        self.lstm2 = BLSTM(lstm_config)

        self.lstm_lan = LSTM_attention(lstm_lan_config)

        # att weight layer
        self.fcn1 = nn.Sequential(nn.Linear(self.ef1, self.ef2), nn.Tanh())
        self.fcn2 = nn.Linear(self.ef2, self.ef3)
        self.fcn2_ = nn.Linear(self.ef3, self.csef)

        self.use_attention = use_attention
        if (self.use_attention):
            self.fcn3 = nn.Linear(config1['max_text_len'], 1)
            self.fcn4 = nn.Sequential(nn.Linear(self.ef1 * self.ef3 + self.ef6, self.ef4), nn.ReLU())
        else:
            self.fcn3 = nn.Linear(config1['max_text_len'], 1)
            self.fcn4 = nn.Sequential(nn.Linear(self.ef1 + self.ef6, self.ef4), nn.ReLU())

        self.fcn5 = NormedLinear(self.ef4, 6)

        self.fcn_embedding_msa = \
            nn.Sequential(nn.Linear(embedding_feature_dim_msa, self.ef5), NormedLinear(self.ef5, self.ef6))

    def forward(self, input):
        input = input.float()
        aux = input[:, 70:74]
        embedding = input[:, 74:]
        aux = aux.unsqueeze(dim=1)
        aux = torch.repeat_interleave(aux, repeats=self.max_len, dim=1)

        batch = input.shape[0]
        input = input[:, :70]
        input = self.embedding(input.long())

        input = self.linear(input)

        input = torch.cat([input, aux], dim=2)
        input_1 = self.lstm_lan(input, input)
        input_2 = self.cnn(input)
        input_2 = self.cnn1(input_2)

        input = self.lstm2(input_1 + input_2)
        # att weight layer
        input2 = self.fcn1(input)
        input2 = self.fcn2(input2)

        cs_model = self.fcn2_(input2)

        if (self.use_attention):
            input = input.unsqueeze(dim=3)

            input = torch.repeat_interleave(input, repeats=self.ef3, dim=3)

            input = F.softmax(input, dim=1)

            input2 = input2.unsqueeze(dim=2)

            input2 = torch.repeat_interleave(input2, repeats=self.ef1, dim=2)

            outputs = input * input2

            outputs = outputs.permute(0, 2, 3, 1)

            outputs = outputs.sum(dim=3)

            outputs = outputs.squeeze(dim=-1)

            outputs = torch.cat([outputs.reshape(outputs.shape[0], -1),
                                 self.fcn_embedding_msa((embedding))], dim=1)

            outputs = self.fcn4(outputs)

        else:
            outputs = input.permute(0, 2, 1)

            outputs = self.fcn3(outputs)

            outputs = outputs.squeeze(dim=-1)

            outputs = torch.cat([outputs.reshape(outputs.shape[0], -1),
                                 self.fcn_embedding_msa((embedding))], dim=1)

            outputs = self.fcn4(outputs)


        model = self.fcn5(outputs)

        return model, cs_model

class baseline_model(nn.Module):

    def __init__(self, config, config1, cnn_configs, lstm_lan_config, lstm_config,
                 use_CRF=False, use_attention=True, reweight_ratio=None):
        super(baseline_model, self).__init__()

        self.num_classes = 20
        self.max_len = config1['max_text_len']

        self.embedding = nn.Embedding(num_embeddings=config['vocab_size'], embedding_dim=config['embedding_size'])

        self.ef1 = 512
        self.ef2 = 144
        self.ef3 = 32
        self.csef = 11
        self.ef4 = 256
        self.ef5 = 256
        self.ef6 = 64

        if (use_CRF):
            self.crf = CRF(num_tags=11, reweight_ratio=reweight_ratio)#original: num_tags=9
            self.use_CRF = True

        else:
            self.use_CRF = False

        self.linear = nn.Sequential(nn.Linear(config['embedding_size'], config1['input_dim'] - 4), nn.ReLU())
        self.linear2 = nn.Linear(self.max_len, self.max_len)
        self.lstm = BLSTM(config1)
        self.cnn = TextCNN(cnn_configs[0])
        self.cnn1 = TextCNN(cnn_configs[1])

        self.lstm2 = BLSTM(lstm_config)

        self.lstm_lan = LSTM_attention(lstm_lan_config)

        # att weight layer
        self.fcn1 = nn.Sequential(nn.Linear(self.ef1, self.ef2), nn.Tanh())
        self.fcn2 = nn.Linear(self.ef2, self.ef3)
        self.fcn2_ = nn.Linear(self.ef3, self.csef)

        self.use_attention = use_attention
        if (self.use_attention):
            self.fcn3 = nn.Linear(config1['max_text_len'], 1)
            self.fcn4 = nn.Sequential(nn.Linear(self.ef1 * self.ef3, self.ef4), nn.ReLU())
        else:
            self.fcn3 = nn.Linear(config1['max_text_len'], 1)
            self.fcn4 = nn.Sequential(nn.Linear(self.ef1, self.ef4), nn.ReLU())

        self.fcn5 = NormedLinear(self.ef4, 6)

    def forward(self, input):
        input = input.float()
        aux = input[:, 70:74]
        aux = aux.unsqueeze(dim=1)
        aux = torch.repeat_interleave(aux, repeats=self.max_len, dim=1)

        batch = input.shape[0]
        input = input[:, :70]
        input = self.embedding(input.long())

        input = self.linear(input)

        input = torch.cat([input, aux], dim=2)
        input_1 = self.lstm_lan(input, input)
        input_2 = self.cnn(input)
        input_2 = self.cnn1(input_2)

        input = self.lstm2(input_1 + input_2)
        # att weight layer
        input2 = self.fcn1(input)
        input2 = self.fcn2(input2)


        cs_model = self.fcn2_(input2)

        if (self.use_attention):
            input = input.unsqueeze(dim=3)
            input = torch.repeat_interleave(input, repeats=self.ef3, dim=3)
            input = F.softmax(input, dim=1)
            input2 = input2.unsqueeze(dim=2)
            input2 = torch.repeat_interleave(input2, repeats=self.ef1, dim=2)
            outputs = input * input2
            outputs = outputs.permute(0, 2, 3, 1)
            outputs = outputs.sum(dim=3)
            outputs = outputs.squeeze(dim=-1)
            outputs = outputs.reshape(outputs.shape[0], -1)

            outputs = self.fcn4(outputs)
        else:
            outputs = input.permute(0, 2, 1)
            outputs = self.fcn3(outputs)
            outputs = outputs.squeeze(dim=-1)
            outputs = outputs.reshape(outputs.shape[0], -1)
            outputs = self.fcn4(outputs)

        model = self.fcn5(outputs)

        return model, cs_model

if __name__ == '__main__':
    # The code below is used for test!

    # Training
    batch_size = 128
    nb_epoch = 100

    # Embedding
    vocab_size = 27
    embedding_size = 128

    # Convolution
    # First CNN Module：filter_length = 3
    filter_length1 = 3
    pool_length = 2
    feature_size = 32

    # LSTM
    lstm_output_size = 64

    x = np.random.randn(64, 70, 64)
    x = torch.tensor(x, dtype=torch.float32).cuda()

    lstm_config = {'dropout_rate': 0.2, 'input_dim': 64,
                   'hidden_dim':64, 'output_dim': 2, 'num_layers': 2, 'max_text_len': 70, 'classifier': True,
                   'use_norm': True, 'use_blstm': True}

    model = BLSTM(lstm_config).cuda()
    print(x.dtype)
    y = model(x)
    sum = torch.sum(y)

    grads = torch.autograd.grad(sum, model.parameters())

    # print(grads)


