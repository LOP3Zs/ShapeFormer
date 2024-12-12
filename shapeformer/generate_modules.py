import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

from .functions import vq, vq_st


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, dim, K=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

        self.codebook = VQEmbedding(K, dim)

        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x



class ConditionalVQEmbedding(nn.Module):
    def __init__(self, K, D, n_conditions):
        super().__init__()
        self.n_conditions = n_conditions
        self.K = K
        self.embedding = nn.Embedding(n_conditions * K, D)
        self.embedding.weight.data.uniform_(-1./(K*n_conditions), 1./(K*n_conditions))
        self.embedding.weight.data = self.embedding.weight.data.view(self.n_conditions, self.K, -1)
        '''
        for i in range(n_conditions):
            self.embeddings.append(nn.Embedding(K, D))
            self.embeddings[i].weight.data.uniform_(-1./K, 1./K)
        '''

    def straight_through(self, z_e_x, C):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_ = torch.zeros_like(z_e_x_)
        z_q_x_bar_ = torch.zeros_like(z_e_x_)

        ''' this option is faster where I try to group z_e_x_ by C
        however, it is likely to have unexpected login bugs
        since I am not sure if the order of z_e_x_ is preserved

        for c in C:
            original_indice = torch.where(C == c)[0]
            c_z_e_x_ = z_e_x_[C == c]
            c_z_q_x_, c_indices = vq_st(c_z_e_x_, self.embedding.weight[c].detach())
            c_z_q_x_bar_flatten = torch.index_select(self.embedding.weight[c], dim=0, index=c_indices)
            c_z_q_x_bar_ = c_z_q_x_bar_flatten.view_as(c_z_e_x_)

            z_q_x_[original_indice] = c_z_q_x_
            z_q_x_bar_[original_indice] = c_z_q_x_bar_
        '''

        for vi in range(len(z_e_x_)):
            c_z_e_x_ = z_e_x_[vi]
            c = C[vi]
            c_z_q_x_, c_indices = vq_st(c_z_e_x_, self.embedding.weight[c].detach())
            c_z_q_x_bar_flatten = torch.index_select(self.embedding.weight[c], dim=0, index=c_indices)
            c_z_q_x_bar_ = c_z_q_x_bar_flatten.view_as(c_z_e_x_)

            z_q_x_[vi] = c_z_q_x_
            z_q_x_bar_[vi] = c_z_q_x_bar_

        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar


class ConditionalVectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, dim, n_conditions, K=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

        self.codebook = ConditionalVQEmbedding(K, dim, n_conditions)

        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)


    def forward(self, x, C):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x, C)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x
