import math

import torch
import torch.nn.functional as F

from local.CDVQVAE.layers import ( Conditions, Conv1d_Layernorm_LRelu, DeConv1d_Layernorm_GLU,
                      log_loss)
from local.CDVQVAE.layers_vq import CDVectorQuantizer, Jitter

class Model(torch.nn.Module):
    def __init__(self, arch):
        super(Model, self).__init__()

        self.encoder = torch.nn.ModuleDict()
        self.decoder = torch.nn.ModuleDict()

        self.encoder['img'] = Encoder(arch['encoder']['img'], arch['z_dim'])
        self.decoder['img'] = Decoder(arch['decoder']['img'], arch['y_dim'])

        self.encoder['mel'] = Encoder(arch['encoder']['mel'], arch['z_dim'])
        self.decoder['mel'] = Decoder(arch['decoder']['mel'], arch['y_dim'])

        self.quantizer = CDVectorQuantizer( arch['z_num'], arch['z_dim'], normalize=arch['embed_norm'], reduction='sum')
        self.embeds = Conditions( arch['y_num'], arch['y_dim'], normalize=True)

        if arch['jitter_p'] > 0.0:
            self.jitter = Jitter(probability=arch['jitter_p'])
        else:
            self.jitter = None

        self.beta = arch['beta']
        self.arch = arch        

    def forward(self, input, encoder_kind='mcc', decoder_kind='mcc'):
        # Preprocess
        img, mel, spk_id = input
        y = self.embeds(spk_id).transpose(1,2).contiguous()   # Size( N, y_dim, nframes)

        if self.training:

            # Encode
            z_img = self.encoder['img'](img)
            z_mel = self.encoder['mel'](mel)

            # Vector Quantize
            z_vq, z_qut_loss, z_enc_loss, entropy = self.quantizer([z_img,z_mel], ref='mean')
            if self.jitter is None:
                z_vq_img, z_vq_mel = z_vq
            else:
                z_vq_img, z_vq_mel = z_vq
                z_vq_img = self.jitter(z_vq_img)
                z_vq_mel = self.jitter(z_vq_mel)

            # Decode
            xh_img_img = self.decoder['img']((z_vq_img,y))
            xh_img_mel = self.decoder['img']((z_vq_mel,y))
            xh_mel_img = self.decoder['mel']((z_vq_img,y))
            xh_mel_mel = self.decoder['mel']((z_vq_mel,y))

            # Loss
            batch_size = img.size(0)

            z_qut_loss = z_qut_loss / batch_size
            z_enc_loss = z_enc_loss / batch_size

            img_img_loss = log_loss(xh_img_img, img)
            img_mel_loss = log_loss(xh_img_mel, img)
            mel_img_loss = log_loss(xh_mel_img, mel)
            mel_mel_loss = log_loss(xh_mel_mel, mel)

            x_recon = (img_img_loss + mel_mel_loss) / batch_size
            x_cross = (img_mel_loss + mel_img_loss) / batch_size

            loss = x_recon + x_cross + z_qut_loss + self.beta * z_enc_loss
            
            losses = {'Total': loss.item(),
                      'VQ loss': z_qut_loss.item(),
                      'Entropy': entropy.item(),
                      'X recon': x_recon.item(),
                      'X cross': x_cross.item()}

            return loss, losses

        else:
            # Encode
            z = self.encoder[encoder_kind](x)
            # Vector Quantize
            z = self.quantizer(z)
            # Decode
            xhat = self.decoder[encoder_kind]((z,y))

            return xhat.transpose(1,2).contiguous()

    def quantize(self, x):
        z_img = self.encoder['img'](x)
        z_idx = self.quantizer.quantize_the_input(z_img)
        return z_idx



class Encoder(torch.nn.Module):
    def __init__(self, arch, z_dim):
        super(Encoder, self).__init__()

        self.layers = torch.nn.ModuleList()
        for ( i, o, k, s) in zip( arch['input'], 
                                  arch['output'], 
                                  arch['kernel'], 
                                  arch['stride']):
            self.layers.append(
                Conv1d_Layernorm_LRelu( i, o, k, stride=s)
            )

        self.mlp = torch.nn.Conv1d( in_channels=arch['output'][-1],
                                    out_channels=z_dim,
                                    kernel_size=1)


    def forward(self, input):
        x = input   # Size( N, x_dim, nframes)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        z = self.mlp(x)

        return z   # Size( N, z_dim, nframes)

class Decoder(torch.nn.Module):
    def __init__(self, arch, y_dim):
        super(Decoder, self).__init__()

        self.layers = torch.nn.ModuleList()
        for ( i, o, k, s) in zip( arch['input'], 
                                  arch['output'], 
                                  arch['kernel'], 
                                  arch['stride']):
            if len(self.layers) == len(arch['output']) - 1:
                self.layers.append(
                    torch.nn.ConvTranspose1d( in_channels=i+y_dim,
                                              out_channels=o,
                                              kernel_size=k,
                                              stride=s,
                                              padding=int((k-1)/2))
                )                
            else:
                self.layers.append(
                    DeConv1d_Layernorm_GLU( i+y_dim, o, k, stride=s)
                )

    def forward(self, input):
        x, y = input   # ( Size( N, z_dim, nframes), Size( N, y_dim, nframes))

        for i in range(len(self.layers)):
            x = torch.cat((x,y),dim=1)
            x = self.layers[i](x)

        return x   # Size( N, x_dim, nframes)
