import torch
import torch.nn as nn
from .transformer import Residual, PreNorm, Attention, FeedForward
import torch.nn.functional as F

def conv_block(in_channel, out_channel,drop):
    layer = nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(5,5), padding=(2,2)),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
        nn.Dropout(drop),
    )
    return layer

class dense_block(nn.Module):
    def __init__(self, in_channel, growth_rate, num_layers, drop):
        super(dense_block, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(conv_block(channel, growth_rate, drop))
            channel += growth_rate
        self.net = nn.Sequential(*block)
    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x


class densenet(nn.Module):
    def __init__(self, growth, dense_layers, drop, Final):
        super(densenet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(drop))
        self.dense=dense_block(in_channel=64,growth_rate=growth,num_layers=dense_layers,drop=drop)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64+growth*dense_layers, out_channels=64, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv2d(in_channels=64, out_channels=Final, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(Final),
            nn.ReLU(),
            nn.Dropout(drop))
    def forward(self, x):
        x = self.block1(x)
        x = self.dense(x)
        x = self.block2(x)
        return x

class PrompterModel(nn.Module):
    def __init__(self,
                 input_dim,
                 embedding_dim,
                 depth_transformer,
                 heads_transformer,
                 dim_head_transformer,
                 attn_dropout_transformer,
                 ff_dropout_transformer,
                 dropout_CNN,
                 mat_size):
        super().__init__()

        self.embeds_seq = nn.Linear(input_dim, embedding_dim)
        a=dropout_CNN
        Final_channel=16 #8

        self.net=densenet(growth=16, dense_layers=3, drop=a, Final=Final_channel)

        # #transformer layers for dna seq
        self.layers_dna = nn.ModuleList([])
        for _ in range(depth_transformer):
            self.layers_dna.append(nn.ModuleList([
                Residual(
                    PreNorm(embedding_dim, Attention(embedding_dim, heads=heads_transformer, dim_head=dim_head_transformer,
                                                     dropout=attn_dropout_transformer))),
                Residual(PreNorm(embedding_dim, FeedForward(embedding_dim, dropout=ff_dropout_transformer))),
            ]))

        self.output_layer = nn.Sequential(nn.Linear(Final_channel * mat_size * mat_size+48 * embedding_dim, 1))


    def forward(self, seq_data, CGR_graph):
        flat_x0=torch.flatten(self.net(CGR_graph),-3,-1)
        x1 = self.embeds_seq(seq_data)
        for attn, ff in self.layers_dna:
            x1 = attn(x1)
            x1 = ff(x1)
        flat_x1 = torch.flatten(x1, -2, -1)
        return self.output_layer(torch.cat([flat_x0,flat_x1],dim=1))