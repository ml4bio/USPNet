import torch
import torch.nn as nn


class OnLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, level_hidden_size=None, bias=False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.level_hidden_size = level_hidden_size
        self.n_repeat = hidden_size // level_hidden_size

        self.lstm_weight = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=bias)
        self.level_weight = nn.Linear(input_size + hidden_size, 2 * level_hidden_size, bias=bias)

    def forward(self, input, prev_state):
        """
        :param input: shape of (bsz, 1, input_size)
        :param prev_state:  h,c from prev step witch shape of (1, hidden_dim)
        :return:
        """

        h_prev, c_prev = prev_state

        combined = torch.cat([input, h_prev], dim=-1)

        cc_i, cc_f, cc_o, cc_g = torch.split(self.lstm_weight(combined), self.hidden_size, dim=1)

        cc_i_h, cc_f_h = torch.split(self.level_weight(combined), self.level_hidden_size, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)

        c = torch.tanh(cc_g)

        p_f = torch.softmax(cc_f_h, dim=-1)
        p_i = torch.softmax(cc_i_h, dim=-1)

        # level mask
        f_h = torch.cumsum(p_f, dim=-1)  # (1, level_hidden_size)
        i_h = torch.cumsum(p_i.flip(dims=[-1]), dim=-1).flip(dims=[-1])  # (1, level_hidden_size)

        # d_i = i_h.sum(dim=-1)
        # d_f = 1. - f_h.sum(dim=-1)

        # (1, level_hidden_size, 1) -> (1, level_hidden_size, n_repeat) -> (1, hidden_size)
        i_h = i_h.unsqueeze(dim=-1).expand((*i_h.shape, self.n_repeat)).flatten(1)
        f_h = f_h.unsqueeze(dim=-1).expand((*f_h.shape, self.n_repeat)).flatten(1)

        w = i_h * f_h

        # combine information from lower and higher layer
        c = w * (f * c_prev + i * c) + (f_h - w) * c_prev + (i_h - w) * c
        h = o * torch.tanh(c)

        return h, c


class OnLSTM(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 level_hidden_size=None,
                 num_layers=1,
                 bias=True,
                 batch_first=True,
                 bidirectional=False):
        super().__init__()

        assert num_layers >= 1, 'Need at least one layer'

        if level_hidden_size is None:
            level_hidden_size = hidden_size
        else:
            assert hidden_size % level_hidden_size == 0, \
                'level_hidden_size should be divisible by hidden_size'

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.bias = bias
        self.bidirectional = bidirectional

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_size if i == 0 else self.hidden_size
            cell_list.append(OnLSTMCell(input_size=cur_input_dim,
                                        hidden_size=hidden_size,
                                        level_hidden_size=level_hidden_size,
                                        bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward_impl(self, input, hx):
        """
        :param input: shape of (bsz, seq_len, input_size)
        :param hx: The number of features in the hidden state `h`
        :return:
        """
        seq_len = input.size(1)
        cur_layer_input = input

        h_hold, c_hold = [], []

        h, c = hx
        for layer_idx in range(self.num_layers):
            h_t, c_t = h[layer_idx], c[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h_t, c_t = self.cell_list[layer_idx](cur_layer_input[:, t], (h_t, c_t))
                output_inner.append(h_t)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            h_hold.append(h_t)
            c_hold.append(c_t)

        h = torch.stack(h_hold, dim=0)
        c = torch.stack(c_hold, dim=0)

        return layer_output, (h, c)

    def forward(self, input, hx=None):
        """
        :param input: 3-D Tensor either of shape (b, t, d) or (t, b, d)
        :param hidden_state: come from existed hidden state or inited by zeros if None.
        :return:
        """

        if not self.batch_first:
            input = input.permute(1, 0, 2)

        num_directions = 2 if self.bidirectional else 1

        if hx is None:
            hidden_state = torch.zeros(self.num_layers * num_directions,
                                       input.shape[0],
                                       self.hidden_size,
                                       dtype=input.dtype,
                                       device=input.device)
            hx = (hidden_state, hidden_state)

        h, c = hx
        if num_directions == 2:
            h_f, h_b = torch.split(h, self.num_layers, 0)
            c_f, c_b = torch.split(c, self.num_layers, 0)
            layer_output_f, hx_f = self.forward_impl(input, (h_f, c_f))
            layer_output_b, hx_b = self.forward_impl(input.flip(dims=[1]), (h_b, c_b))

            layer_output_b = layer_output_b.flip(dims=[1])

            layer_output = torch.cat([layer_output_f, layer_output_b], dim=-1)
            h = torch.stack([h_f, h_b], dim=1).reshape(-1, input.shape[0], self.hidden_size)
            c = torch.stack([c_f, c_b], dim=1).reshape(-1, input.shape[0], self.hidden_size)
            hx = (h, c)
        else:
            layer_output, hx = self.forward_impl(input, hx)

        if not self.batch_first:
            layer_output = layer_output.permute(1, 0, 2)

        return layer_output, hx