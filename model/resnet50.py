'''Deep Hierarchical Classifier using resnet50 with cbam as the base.
'''

import torch
import torch.nn as nn


class TransformerNN(nn.Module):
    '''classification architecture with transformer blocks.
    '''

    def __init__(self, use_rnn=False, rnn_type='lstm', rnn_init_states='zeros' ,num_classes=[4, 9, 16]):
        '''Params init and build arch.
        '''
        super(TransformerNN, self).__init__()

        self.use_rnn = use_rnn
        self.rnn_type = rnn_type
        self.rnn_init_states = rnn_init_states

        # define transformer blocks here
        self.transformer_embed_size = 256

        # use RNN layer if transformer word embeddings are being used
        self.recurrent = None
        if use_rnn:
            if rnn_type == 'lstm':
                self.recurrent = nn.LSTM(input_size=self.transformer_embed_size,
                                         hidden_size=self.transformer_embed_size,
                                         batch_first=True)
            elif rnn_type == 'gru':
                self.recurrent = nn.GRU(input_size=self.transformer_embed_size,
                                         hidden_size=self.transformer_embed_size,
                                         batch_first=True)
            else:
                raise NotImplementedError('rnn type not found!')

        self.linear_lvl1 = nn.Linear(self.transformer_embed_size, num_classes[0])
        self.linear_lvl2 = nn.Linear(self.transformer_embed_size, num_classes[1])
        self.linear_lvl3 = nn.Linear(self.transformer_embed_size, num_classes[2])

        self.softmax_reg1 = nn.Linear(num_classes[0], num_classes[0])
        self.softmax_reg2 = nn.Linear(num_classes[0]+num_classes[1], num_classes[1])
        self.softmax_reg3 = nn.Linear(num_classes[1]+num_classes[2], num_classes[2])


    def make_layer(self, out_channels, num_blocks, stride, use_cbam):
        '''To construct the bottleneck layers.
        '''
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BottleNeck(in_channels=self.in_channels, out_channels=out_channels, stride=stride, expansion=self.expansion, use_cbam=use_cbam))
            self.in_channels = out_channels * self.expansion
        return nn.Sequential(*layers)

    def _forward_rnn(self, dummy_x):
        inp = dummy_x[:, 1:, :]
        if self.rnn_init_states == 'zeros':
            if self.rnn_type == 'lstm':
                output, hn, cn = self.recurrent(inp)
            elif self.rnn_type == 'gru':
                output, hn = self.recurrent(inp)
            else:
                raise NotImplementedError()
        elif self.rnn_init_states == 'transformer_hxs':
            h0 = c0 = dummy_x[:, 1, :].unsqueeze(0)
            if self.rnn_type == 'lstm':
                output, hn, cn = self.recurrent(inp, (h0, c0))
            elif self.rnn_type == 'gru':
                output, hn = self.recurrent(inp, h0)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError('Error!')
        return output, hn

    def forward(self, x):
        '''Forward propagation of ResNet-50.
        '''

        # Forward pass for transformer block
        with torch.no_grad():
            dummy_x = torch.randn((8, 101, 256))

        # use RNN layer if transformer word embeddings are being used
        if self.use_rnn:
            output, hn = self._forward_rnn(dummy_x)
            x = hn.squeeze()
        else:
            x = dummy_x[:, 0, :]

        import pdb
        pdb.set_trace()
        #x = self.avgpool(x_conv)
        #x = nn.Flatten()(x) #flatten the feature maps.
        #import pdb
        #pdb.set_trace()

        level_1 = self.softmax_reg1(self.linear_lvl1(x))
        level_2 = self.softmax_reg2(torch.cat((level_1, self.linear_lvl2(x)), dim=1))
        level_3 = self.softmax_reg3(torch.cat((level_2, self.linear_lvl3(x)), dim=1))

        return level_1, level_2, level_3
