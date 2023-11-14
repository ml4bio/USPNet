import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from Net.ONLSTM import OnLSTM
import numpy as np
import torch.nn.init as init

class BLSTM(nn.Module):
    """
        Implementation of BLSTM Concatenation for sentiment classification task
    """
    # config
    # properties: input_dim, hidden_dim, num_layers, output_dim, max_text_len=70, dropout_rate=0.5
    # According to the paper, hidden_dim=64, output_dim=64
    def __init__(self, config):
        super(BLSTM, self).__init__()

        #self.emb = nn.Embedding(num_embeddings=embeddings.size(0),
                                #embedding_dim=embeddings.size(1),
                                #padding_idx=0)
        #self.emb.weight = nn.Parameter(embeddings)
        print(config['input_dim'])
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']

        self.num_layers = config['num_layers']
        self.num_directions = 2 if config['use_blstm'] else 1
        self.dropout_rate = config['dropout_rate']
        self.max_text_len = config['max_text_len']
        self.attention = config['attention']
        if(self.attention):
            self.attention_layer = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=True)
            )

        self.lstm = nn.LSTM(input_size=config['input_dim'],
                               hidden_size=config['hidden_dim'],
                               num_layers=config['num_layers'],
                               dropout=config['dropout_rate'],
                               batch_first=True,
                               bidirectional=config['use_blstm'])
        self.batch_norm = nn.BatchNorm1d(self.num_directions * self.hidden_dim)

        self.start_dim = int(self.num_directions * self.hidden_dim * self.max_text_len )



    def attention_net_with_w(self, lstm_out):
        '''
        :param lstm_out: [batch_size, n_step, n_hidden * num_directions(=2)]
        :return:
        '''
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        # [batch_size, time_step, hidden_dims]
        atten_w = self.attention_layer(h)
        m = nn.Tanh()(h)
        # [batch_size, time_step, time_step]
        atten_context = torch.bmm(m, atten_w.transpose(1, 2))

        softmax_w = F.softmax(atten_context, dim=-1)

        context = torch.bmm(h.transpose(1,2), softmax_w)
        result = torch.sum(context, dim=-1)
        result = nn.Dropout(self.dropout_rate)(result)
        return result

    def forward(self, input):
        """
        :param X: (batch, sen_length, feature_size), tensor for sentence sequence
        :return:
        """

        ''' Embedding Layer | Padding | Sequence_length 70'''
        # X = self.emb(X)
        # X shape=(batch_size, 70, 32)
        # (batch_size, max_len, feature_size)

        batch_size = input.shape[0]
        # input : [batch_size, seq_len, feature_size]

        hidden_state = torch.randn(self.num_layers*self.num_directions, batch_size, self.hidden_dim).cuda()  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.randn(self.num_layers*self.num_directions, batch_size, self.hidden_dim).cuda()   # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        # input=torch.tensor(input, dtype=torch.float32).cuda()

        self.lstm.flatten_parameters()
        outputs, (_, _) = self.lstm(input.float(), (hidden_state, cell_state))
        # outputs shape = (batch_size, max_len=70, num_direction*feature_size)
        if(self.attention):
            atten_out = self.attention_net_with_w(outputs)

            outputs=atten_out
        else:

            outputs = outputs

        # outputs = outputs.reshape(batch_size, -1)
        # outputs = self.batch_norm(outputs)
        # outputs = F.dropout(input=outputs, p=self.dropout_rate)
        if(self.attention):
            outputs = self.batch_norm(outputs)
        else:
            outputs = outputs.permute(0, 2, 1)
            outputs = self.batch_norm(outputs)
            outputs = outputs.permute(0, 2, 1)
        # outputs = outputs.reshape(batch_size, -1)
        # outputs = F.dropout(input=outputs, p=self.dropout_rate)
        model = outputs

        return model

if __name__ == '__main__':
    # The code below is used for test!

    # Training
    batch_size = 128
    nb_epoch = 100

    # Embedding
    # 暂时考虑amino acid用26个字母表示+1个为空
    vocab_size = 27
    embedding_size = 128

    # Convolution
    # 第一个CNN Module：filter_length = 3
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
    #original: hidden_dim = 64

    model = BLSTM(lstm_config).cuda()
    print(x.dtype)
    y = model(x)
    sum = torch.sum(y)

    grads = torch.autograd.grad(sum, model.parameters())

    # print(grads)


