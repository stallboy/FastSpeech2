# coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Models import Encoder, Decoder
from transformer.Layers import PostNet
from modules import VarianceAdaptor
from utils import get_mask_from_lengths
import hparams as hp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, use_postnet=True):
        super(FastSpeech2, self).__init__()

        self.encoder = Encoder()
        self.variance_adaptor = VarianceAdaptor()

        self.decoder = Decoder()
        self.mel_linear = nn.Linear(hp.decoder_hidden, hp.n_mel_channels)

        self.use_postnet = use_postnet
        if self.use_postnet:
            self.postnet = PostNet()

    # src_seq <16, 110> 是音素id的列表， d_target 是音素id对应的帧数, max_src_len=110， max_mel_len=965
    # p_target<16, 965> 是f0， e_target是energy
    def forward(self, src_seq, src_len, mel_len=None, d_target=None, p_target=None, e_target=None, max_src_len=None,
                max_mel_len=None):
        # mask[false, false, false, true, true], 前面是src_len个false，后面max_src_len个true，无用的信息是true
        src_mask = get_mask_from_lengths(src_len, max_src_len)
        mel_mask = get_mask_from_lengths(mel_len, max_mel_len) if mel_len is not None else None

        # encoder_output <16, 110, 256>
        encoder_output = self.encoder(src_seq, src_mask)
        if d_target is not None:
            variance_adaptor_output, d_prediction, p_prediction, e_prediction, _, _ = self.variance_adaptor(
                encoder_output, src_mask, mel_mask, d_target, p_target, e_target, max_mel_len)
        else:
            variance_adaptor_output, d_prediction, p_prediction, e_prediction, mel_len, mel_mask = self.variance_adaptor(
                encoder_output, src_mask, mel_mask, d_target, p_target, e_target, max_mel_len)
        # variance_adaptor_output, decoder_output <16, 965, 256>
        decoder_output = self.decoder(variance_adaptor_output, mel_mask)
        # mel_output <16, 965, 80>
        mel_output = self.mel_linear(decoder_output)

        if self.use_postnet:
            mel_output_postnet = self.postnet(mel_output) + mel_output
        else:
            mel_output_postnet = mel_output

        return mel_output, mel_output_postnet, d_prediction, p_prediction, e_prediction, src_mask, mel_mask, mel_len


if __name__ == "__main__":
    # Test
    model = FastSpeech2(use_postnet=False)
    print(model)
    print(sum(param.numel() for param in model.parameters()))
