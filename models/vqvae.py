import torch.nn as nn
from models.encdec import Encoder, Decoder
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset,LFQ,QuantizeEMAResetSim
import math
import torch
import numpy as np
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class TSVQVAE(nn.Module):
    def __init__(self,
                 args,
                 nb_code=1024,
                 code_dim=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = args.quantizer
        output_emb_width = self.code_dim
        if args.quantizer == "lfq":
            output_emb_width = math.ceil(math.log2(self.num_code))
            self.code_dim = output_emb_width
        elif args.dataname == 'stock':
            input_emb_width = 6
        elif args.dataname == 'energy':
            input_emb_width = 28
        elif args.dataname == 'sine':
            input_emb_width = 5
        elif args.dataname == 'etth':
            input_emb_width = 7
        elif args.dataname == 'fmri':
            input_emb_width = 50
        elif args.dataname == 'mujoco':
            input_emb_width = 14
        self.encoder = Encoder(input_emb_width, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm, quantizer=args.quantizer)
        self.decoder = Decoder(input_emb_width, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        if args.quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(nb_code, code_dim, args)
        elif args.quantizer == "orig":
            self.quantizer = Quantizer(nb_code, code_dim, 1.0)
        elif args.quantizer == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim, args)
        elif args.quantizer == "reset":
            self.quantizer = QuantizeReset(nb_code, code_dim, args)
        elif args.quantizer == "lfq":
            self.quantizer = LFQ(self.num_code, self.code_dim, args)
        elif args.quantizer == "ema_reset_sim":
            self.quantizer = QuantizeEMAResetSim(nb_code, code_dim, args)


    def preprocess(self, x):
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        x = x.permute(0,2,1)
        return x

    def encode2(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        N, _, _ = x_encoder.shape
        x_encoder = self.postprocess(x_encoder)
        return x_encoder



    def encode(self, x):

        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        N, _, _ = x_encoder.shape
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx


    def forward(self, x):

        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)

        ## quantization
        x_quantized, loss, perplexity  = self.quantizer(x_encoder)

        ## decoder
        x_decoder = self.decoder(x_quantized)###shared2 x.shape[-1]
        x_out = self.postprocess(x_decoder)

        return x_out, loss, perplexity


    def forward_decoder(self, x):
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(x.shape[0], -1, self.code_dim).permute(0, 2, 1).contiguous()

        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out


class VQVAE(nn.Module):
    def __init__(self,
                 args,
                 nb_code=512,
                 code_dim=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        
        self.vqvae = TSVQVAE(args, nb_code, code_dim, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

    def encode(self, x):

        quants = self.vqvae.encode(x) # (N, T)
        return quants

    def encode2(self, x):

        quants = self.vqvae.encode2(x) # (N, T)
        return quants

    def forward(self, x):

        x_out, loss, perplexity = self.vqvae(x)
        return x_out, loss, perplexity

    def forward_decoder(self, x):
        x_out = self.vqvae.forward_decoder(x)
        return x_out
        