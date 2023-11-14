import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn import Parameter, init

class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.is_training = True
        self.dropout_rate = config['dropout_rate']
        self.config = config
        if (config["activation_function_type"]=="Sigmoid"):
            self.activation = nn.Sigmoid()
        elif (config["activation_function_type"]=="Tanh"):
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        self.conv = nn.Sequential(nn.Conv1d(in_channels=config['embedding_size'],
                                    out_channels=config['feature_size'],
                                    kernel_size=config["kernel_size"], padding=math.floor(config["kernel_size"]/2)),

            # 原文声明activation统一使用ReLU
            self.activation)


    def forward(self, x):
        # x shape=(batch_size, max_text_len, 148)

        embed_x = x
        # embed_x shape=(batch_size, max_text_len,  embedding_dim=148)
        embed_x = embed_x.permute(0, 2, 1)

        # embed_x shape=(batch_size, embedding_dim=148, max_text_len)
        # conv in_channels=148, out_channel=40/20

        # out shape=(batch_size, out_channel=40/20, max_text_len)
        out = self.conv(embed_x.float())

        # out shape=(batch_size, max_text_len, out_channel=40/20)
        out=out.permute(0 ,2, 1)
        return out

if __name__ == '__main__':
    # The code below is used for test!


    # Convolution
    # 第一个CNN Module：kernel_size = 5
    kernel_size=5
    feature_size=40


    x=np.random.random((128, 3000, 148))
    x=torch.tensor(x)
    config={'dropout_rate':0.5, 'kernel_size':kernel_size, 'embedding_size':148,
            'feature_size':feature_size, 'activation_function_type':"Sigmoid"}

    model=TextCNN(config)
    y=model(x)

