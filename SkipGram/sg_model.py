import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tools import _getNegSamples

class word2vec(nn.module):

    def __init__(self, n_words, dims, noise_dist):
        super.__init__(self)

        self.n_words = n_words
        self.dims = dims
        self.noise_dist = noise_dist

        self.in_embed = nn.Embedding(n_words, dims)
        self.out_embed = nn.Embedding(n_words, dims)


    def forward_input(self, input_words):
        input_vector = self.in_embed(input_words)
        return input_vector


    def forward_output(self, output_words):
        output_vector = self.out_embed(output_words)
        return output_vector

    def forward_noise(self, size, N_SAMPLE):
        noise_dist = self.noise_dist

        noise_words = torch.multinomial(noise_dist,
                                        size * N_SAMPLE,
                                        replacement=True)

        noise_vectors = self.out_embed(noise_words)
