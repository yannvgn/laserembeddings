# The code contained in this file was copied/pasted from LASER's source code (source/embed.py)
# and nearly kept untouched besides:
# - code formatting
# - buffered_arange: fix to avoid unnecessary warning on PyTorch >= 1.4.0

# pylint: disable=redefined-builtin, consider-using-enumerate, arguments-differ, fixme, abstract-method, consider-using-from-import

from collections import namedtuple

import re
import numpy as np

import torch
import torch.nn as nn

__all__ = ['SentenceEncoder', 'Encoder']

SPACE_NORMALIZER = re.compile(r'\s+')
Batch = namedtuple('Batch', 'srcs tokens lengths')


def buffered_arange(max):
    if not hasattr(buffered_arange,
                   'buf') or max > buffered_arange.buf.numel():
        buffered_arange.buf = torch.LongTensor()
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


# TODO Do proper padding from the beginning
def convert_padding_direction(src_tokens,
                              padding_idx,
                              right_to_left=False,
                              left_to_right=False):
    assert right_to_left ^ left_to_right
    pad_mask = src_tokens.eq(padding_idx)
    if not pad_mask.any():
        # no padding, return early
        return src_tokens
    if left_to_right and not pad_mask[:, 0].any():
        # already right padded
        return src_tokens
    if right_to_left and not pad_mask[:, -1].any():
        # already left padded
        return src_tokens
    max_len = src_tokens.size(1)
    range = buffered_arange(max_len).type_as(src_tokens).expand_as(src_tokens)
    num_pads = pad_mask.long().sum(dim=1, keepdim=True)
    if right_to_left:
        index = torch.remainder(range - num_pads, max_len)
    else:
        index = torch.remainder(range + num_pads, max_len)
    return src_tokens.gather(1, index)


class SentenceEncoder:
    def __init__(self,
                 model_path,
                 max_sentences=None,
                 max_tokens=None,
                 cpu=False,
                 fp16=False,
                 sort_kind='quicksort'):
        self.use_cuda = torch.cuda.is_available() and not cpu
        self.max_sentences = max_sentences
        self.max_tokens = max_tokens
        if self.max_tokens is None and self.max_sentences is None:
            self.max_sentences = 1

        state_dict = torch.load(model_path)
        self.encoder = Encoder(**state_dict['params'])
        self.encoder.load_state_dict(state_dict['model'])
        self.dictionary = state_dict['dictionary']
        self.pad_index = self.dictionary['<pad>']
        self.eos_index = self.dictionary['</s>']
        self.unk_index = self.dictionary['<unk>']
        if fp16:
            self.encoder.half()
        if self.use_cuda:
            self.encoder.cuda()
        self.sort_kind = sort_kind

    def _process_batch(self, batch):
        tokens = batch.tokens
        lengths = batch.lengths
        if self.use_cuda:
            tokens = tokens.cuda()
            lengths = lengths.cuda()
        self.encoder.eval()
        embeddings = self.encoder(tokens, lengths)['sentemb']
        return embeddings.detach().cpu().numpy()

    def _tokenize(self, line):
        tokens = SPACE_NORMALIZER.sub(" ", line).strip().split()
        ntokens = len(tokens)
        ids = torch.LongTensor(ntokens + 1)
        for i, token in enumerate(tokens):
            ids[i] = self.dictionary.get(token, self.unk_index)
        ids[ntokens] = self.eos_index
        return ids

    def _make_batches(self, lines):
        tokens = [self._tokenize(line) for line in lines]
        lengths = np.array([t.numel() for t in tokens])
        indices = np.argsort(-lengths, kind=self.sort_kind)  # pylint: disable=invalid-unary-operand-type

        def batch(tokens, lengths, indices):
            toks = tokens[0].new_full((len(tokens), tokens[0].shape[0]),
                                      self.pad_index)
            for i in range(len(tokens)):
                toks[i, -tokens[i].shape[0]:] = tokens[i]
            return Batch(srcs=None,
                         tokens=toks,
                         lengths=torch.LongTensor(lengths)), indices

        batch_tokens, batch_lengths, batch_indices = [], [], []
        ntokens = nsentences = 0
        for i in indices:
            if nsentences > 0 and ((self.max_tokens is not None
                                    and ntokens + lengths[i] > self.max_tokens)
                                   or (self.max_sentences is not None
                                       and nsentences == self.max_sentences)):
                yield batch(batch_tokens, batch_lengths, batch_indices)
                ntokens = nsentences = 0
                batch_tokens, batch_lengths, batch_indices = [], [], []
            batch_tokens.append(tokens[i])
            batch_lengths.append(lengths[i])
            batch_indices.append(i)
            ntokens += tokens[i].shape[0]
            nsentences += 1
        if nsentences > 0:
            yield batch(batch_tokens, batch_lengths, batch_indices)

    def encode_sentences(self, sentences):
        indices = []
        results = []
        for batch, batch_indices in self._make_batches(sentences):
            indices.extend(batch_indices)
            results.append(self._process_batch(batch))
        return np.vstack(results)[np.argsort(indices, kind=self.sort_kind)]


class Encoder(nn.Module):
    def __init__(self,
                 num_embeddings,
                 padding_idx,
                 embed_dim=320,
                 hidden_size=512,
                 num_layers=1,
                 bidirectional=False,
                 left_pad=True,
                 padding_value=0.):
        super().__init__()

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.padding_idx = padding_idx
        self.embed_tokens = nn.Embedding(num_embeddings,
                                         embed_dim,
                                         padding_idx=self.padding_idx)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths):
        if self.left_pad:
            # convert left-padding to right-padding
            src_tokens = convert_padding_direction(
                src_tokens,
                self.padding_idx,
                left_to_right=True,
            )

        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.cpu())

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.data.new(*state_size).zero_()
        c0 = x.data.new(*state_size).zero_()
        packed_outs, (final_hiddens,
                      final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outs, padding_value=self.padding_value)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:

            def combine_bidir(outs):
                return torch.cat([
                    torch.cat([outs[2 * i], outs[2 * i + 1]], dim=0).view(
                        1, bsz, self.output_units)
                    for i in range(self.num_layers)
                ],
                                 dim=0)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        # Set padded outputs to -inf so they are not selected by max-pooling
        padding_mask = src_tokens.eq(self.padding_idx).t().unsqueeze(-1)
        if padding_mask.any():
            x = x.float().masked_fill_(padding_mask, float('-inf')).type_as(x)

        # Build the sentence embedding by max-pooling over the encoder outputs
        sentemb = x.max(dim=0)[0]

        return {
            'sentemb':
            sentemb,
            'encoder_out': (x, final_hiddens, final_cells),
            'encoder_padding_mask':
            encoder_padding_mask if encoder_padding_mask.any() else None
        }
