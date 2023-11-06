# -*- coding: utf-8 -*-
# Python version: 3.8+
#
# SFU CMPT413/825 Fall 2023, HW4
# default solution
# Simon Fraser University
# Jetic Gū
#
#
import os
import re
import sys
import optparse
from tqdm import tqdm
from heapq import heappop, heappush, nlargest
import spacy

import torch
from torch import nn
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


class hp:
    pad_idx = 0
    sos_idx = 1
    eos_idx = 2
    unk_idx = 3
    lex_min_freq = 1

    # architecture
    hidden_dim = 256
    embed_dim = 256
    n_layers = 2
    dropout = 0.2
    batch_size = 32
    num_epochs = 10
    lexicon_cap = 25000

    # training
    max_lr = 1e-4
    cycle_length = 3000

    # generation
    max_len = 50

    # system
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---YOUR ASSIGNMENT---
# -- Step 1: Baseline ---
# The attention module is completely broken now. Fix it using the definition
# given in the HW description.
class AttentionModule(nn.Module):
    def __init__(self, attention_dim):
        """
        You shouldn't deleted/change any of the following defs, they are
        essential for successfully loading the saved model.
        """
        super(AttentionModule, self).__init__()
        self.W_enc = nn.Linear(attention_dim, attention_dim, bias=False)
        self.W_dec = nn.Linear(attention_dim, attention_dim, bias=False)
        self.V_att = nn.Linear(attention_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim = 0)

    # Start working from here, both 'calcAlpha' and 'forward' need to be fixed
    def calcAlpha(self, decoder_hidden, encoder_out):
        """
        param encoder_out: (seq, batch, dim),
        param decoder_hidden: (seq, batch, dim)
        """
        enc = self.W_enc( encoder_out )
        dec = self.W_dec( decoder_hidden )
        scores = enc + dec
        beta = self.V_att( torch.nn.functional.tanh( scores ) )
        alpha = self.softmax( beta )
        return alpha

    def forward(self, decoder_hidden, encoder_out):
        """
        encoder_out: (seq, batch, dim),
        decoder_hidden: (seq, batch, dim)
        """
        alpha = self.calcAlpha(decoder_hidden, encoder_out) # seq, batch, dim=1
        context = torch.sum(alpha * encoder_out, dim=0).unsqueeze(0)
        return context, alpha.permute(2, 1, 0)


# -- Step 2: Improvements ---
# Implement UNK replacement, BeamSearch, translation termination criteria here,
# you can change 'greedyDecoder' and 'translate'.
def greedyDecoder(decoder, encoder_out, encoder_hidden, maxLen):
    seq1_len, batch_size, _ = encoder_out.size()
    target_vocab_size = decoder.target_vocab_size

    outputs = torch.autograd.Variable(
        encoder_out.data.new(maxLen, batch_size, target_vocab_size))
    alphas = torch.zeros(maxLen, batch_size, seq1_len)
    # take what we need from encoder
    decoder_hidden = encoder_hidden[-decoder.n_layers:]
    # start token (ugly hack)
    output = torch.autograd.Variable(
        outputs.data.new(1, batch_size).fill_(hp.sos_idx).long())
    for t in range(maxLen):
        output, decoder_hidden, alpha = decoder(
            output, encoder_out, decoder_hidden)
        outputs[t] = output
        alphas[t] = alpha.data
        output = torch.autograd.Variable(output.data.max(dim=2)[1])
        if int(output.data) == hp.eos_idx:
            break
    return outputs, alphas.permute(1, 2, 0)


def translate(models, input_dl):
    results = []
    for i, batch in tqdm(enumerate(input_dl)):
        f, e = batch
        for i, model in enumerate(models):
            output, attention = model(f)
            if i==0:
                ens_output = output
                ens_attention = attention
            else:
                ens_output += output
                ens_attention += attention
        ens_output = ens_output.topk(1)[1]
        ens_output = models[0].tgt2txt(ens_output[:, 0].data).strip().split('<eos>')[0]
        results.append(output)
        attention = ens_attention
    return results


class BeamNode:
    def __init__(self, output, score, prev_node, len, logits, decoder_hidden) -> None:
        self.output = output
        self.logits = logits
        self.score = score
        self.prev_node = prev_node
        self.len = len
        self.token = output
        self.decoder_hidden = decoder_hidden
    def get_seq( self, place_holder):
        node = self
        sequences = torch.zeros( place_holder.shape[0] )
        for index in reversed( range( self.len ) ):
            place_holder[ index -1, : ] = node.logits
            sequences[index - 1] = node.token
            node = self.prev_node

        return place_holder , sequences
    
def beam_search(  decoder, encoder_out, encoder_hidden, maxLen, beam_width ):
    seq1_len, batch_size, _ = encoder_out.size()
    target_vocab_size = decoder.target_vocab_size
    decoder_hidden = encoder_hidden[-decoder.n_layers:]
    outputs = torch.autograd.Variable(
            encoder_out.data.new(maxLen, batch_size, target_vocab_size))
    sequences = torch.Tensor( maxLen, batch_size, 1 )
    # take what we need from encoder
    decoder_hidden = encoder_hidden[-decoder.n_layers:]
    # start token (ugly hack)
    output = torch.autograd.Variable(
        outputs.data.new(1, batch_size).fill_(hp.sos_idx).long())
    
    res = []
    for batch_id in range(batch_size):
        # nodes_cach = []
        last_nodes = []
        end_nodes = []
        new_output, decoder_hidden, alpha = decoder(
                        output.reshape(1,-1), encoder_out, decoder_hidden)
        
        node = BeamNode( output= new_output.argmax(dim = 2 ), score= torch.tensor(0., device= decoder_hidden.device ), \
                        prev_node= None, len= 1, logits= new_output, decoder_hidden= decoder_hidden)
        heappush(  last_nodes, ( -node.score, id(node), node ) )
        iterations = 0
        while True:
            new_ = []
            if len( last_nodes ) > 200:
                for item in nlargest( 200, last_nodes ):
                    heappush( new_, item )
                last_nodes = new_
            score , _ , node = heappop( last_nodes )
            decoder_hidden = node.decoder_hidden.to( decoder_hidden.device )
            output = node.output
            if node.token == hp.eos_idx or node.len > maxLen:
                end_nodes.append( node )
                if len( end_nodes) >= beam_width:
                    break
                else:
                    continue

            new_output, decoder_hidden, alpha = decoder(
                        output.reshape(-1,1), encoder_out, decoder_hidden)
            del node.decoder_hidden
            # print( 'new_output' )
            # print( new_output.shape )
            # print( torch.topk( new_output, k = 1, dim= 2 ) )
            # raise
            topk = torch.topk( new_output, k = beam_width, dim= 2, )
            # print( 'topk' )
            # print( topk )
            # raise
            probs = topk[0].squeeze((0,1))
            tokens = topk[1].squeeze((0,1))
            for item in zip( probs, tokens ):
                # print('selected token index')
                # print( item[1] )
                # print( 'token, score' )
                # print( item[1], item[0]+ node.score )
                new_node =  BeamNode(
                    output= item[1] ,
                    score= torch.log( item[0] ) - node.score ,
                    prev_node= node ,
                    len = node.len + 1 ,
                    logits= new_output,
                    decoder_hidden= decoder_hidden.to( 'cpu' ),
                )
                heappush( last_nodes, ( ( -new_node.score , id(new_node), new_node ) ) )
            iterations += 1
        
        node = sorted( end_nodes, key= lambda x: x.score )[0]
        outputs[:, batch_id, :], sequences[:,batch_id, 0] = node.get_seq( outputs[:, batch_id, :] )
        
    return outputs,  None, sequences




# ---Model Definition etc.---
# DO NOT MODIFY ANYTHING BELOW HERE


class Encoder(nn.Module):
    """
    Encoder class
    """
    def __init__(self, source_vocab_size, embed_dim, hidden_dim,
                 n_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(source_vocab_size, embed_dim,
                                  padding_idx=hp.pad_idx)
        self.rnn = nn.GRU(embed_dim,
                          hidden_dim,
                          n_layers,
                          dropout=dropout,
                          bidirectional=True)

    def forward(self, source, hidden=None):
        """
        param source: batched input indices
        param hidden: initial hidden value of self.rnn
        output (encoder_out, encoder_hidden):
            encoder_hidden: the encoder RNN states of length len(source)
            encoder_out: the final encoder states, both direction summed up
                together h^{forward} + h^{backward}
        """
        embedded = self.embed(source)  # (batch_size, seq_len, embed_dim)
        # get encoded states (encoder_hidden)
        encoder_out, encoder_hidden = self.rnn(embedded, hidden)

        # sum bidirectional outputs
        encoder_final = (encoder_out[:, :, :self.hidden_dim] +  # forward
                         encoder_out[:, :, self.hidden_dim:])   # backward

        # encoder_final:  (seq_len, batch_size, hidden_dim)
        # encoder_hidden: (n_layers * num_directions, batch_size, hidden_dim)
        return encoder_final, encoder_hidden


class Decoder(nn.Module):
    def __init__(self, target_vocab_size,
                 embed_dim, hidden_dim,
                 n_layers,
                 dropout):
        super(Decoder, self).__init__()
        self.target_vocab_size = target_vocab_size
        self.n_layers = n_layers
        self.embed = nn.Embedding(target_vocab_size,
                                  embed_dim,
                                  padding_idx=hp.pad_idx)
        self.attention = AttentionModule(hidden_dim)

        self.rnn = nn.GRU(embed_dim + hidden_dim,
                          hidden_dim,
                          n_layers,
                          dropout=dropout)

        self.out = nn.Linear(hidden_dim * 2, target_vocab_size)

    def forward(self, output, encoder_out, decoder_hidden):
        """
        decodes one output frame
        """
        embedded = self.embed(output)  # (1, batch, embed_dim)
        context, alpha = self.attention(decoder_hidden[-1:], encoder_out)
        # 1, 1, 50 (seq, batch, hidden_dim)
        rnn_output, decoder_hidden =\
            self.rnn(torch.cat([embedded, context], dim=2), decoder_hidden)
        output = self.out(torch.cat([rnn_output, context], 2))
        return output, decoder_hidden, alpha


class Seq2Seq(nn.Module):
    def __init__(self, srcLex=None, tgtLex=None, build=True):
        super(Seq2Seq, self).__init__()
        # If we are loading the model, we don't build it here
        if build is True:
            self.params = {
                'srcLex': srcLex,
                'tgtLex': tgtLex,
                'srcLexSize': len(srcLex.vocab),
                'tgtLexSize': len(tgtLex.vocab),
                'embed_dim': hp.embed_dim,
                'hidden_dim': hp.hidden_dim,
                'n_layers': hp.n_layers,
                'dropout': hp.dropout,
                'maxLen': hp.max_len,
            }
            self.build()

    def build(self):
        # self.params are loaded, start building the model accordingly
        self.encoder = Encoder(
            source_vocab_size=self.params['srcLexSize'],
            embed_dim=self.params['embed_dim'],
            hidden_dim=self.params['hidden_dim'],
            n_layers=self.params['n_layers'],
            dropout=self.params['dropout'])
        self.decoder = Decoder(
            target_vocab_size=self.params['tgtLexSize'],
            embed_dim=self.params['embed_dim'],
            hidden_dim=self.params['hidden_dim'],
            n_layers=self.params['n_layers'],
            dropout=self.params['dropout'])
        self.maxLen = self.params['maxLen']

    def forward(self, source, maxLen=None):
        """
        This method implements greedy decoding
        param source: batched input indices
        param maxLen: maximum length of generated output
        """
        if maxLen is None:
            maxLen = self.maxLen
        encoder_out, encoder_hidden = self.encoder(source)

        return greedyDecoder(self.decoder, encoder_out, encoder_hidden,
                             maxLen)

    def tgt2txt(self, tgt):
        return " ".join([self.params['tgtLex'].get_itos()[int(i)] for i in tgt])

    def save(self, file):
        torch.save((self.params, self.state_dict()), file)

    def load(self, file):
        self.params, state_dict = torch.load(file, map_location='cpu')
        self.build()
        self.load_state_dict(state_dict)

# Load Tokeniser
token_en = spacy.load("en_core_web_sm") # Load the English model to tokenize English text
token_de = spacy.load("de_core_news_sm") # Load the German model to tokenize German text


def tokenise_en(text):
    """
    Tokenize an English text and return a list of tokens
    """
    return [token.text for token in token_en.tokenizer(text)]


def tokenise_de(text):
    """
    Tokenize a German text and return a list of tokens
    """
    return [token.text for token in token_de.tokenizer(text)]


def nl_load(inFile, linesToLoad=sys.maxsize, tokeniser=None):
    if tokeniser is not None:
        return [tokeniser(e.lower().strip()) for e in open(inFile, 'r')][:linesToLoad]
    else:
        return [e.lower().strip().split() for e in open(inFile, 'r')][:linesToLoad]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, src="../data/train.tok.de", tgt="../data/train.tok.en",
                 srcLex=None, tgtLex=None, linesToLoad=sys.maxsize) -> None:
        self.source = nl_load(src, linesToLoad, tokeniser=tokenise_de)
        self.target = nl_load(tgt, linesToLoad, tokeniser=tokenise_en)
        self.srcLex = srcLex
        self.tgtLex = tgtLex
        return

    def __getitem__(self, idx) -> torch.Tensor:
        # load one sample by index, e.g like this:
        source_sample = self.source[idx]
        target_sample = self.target[idx]
        return source_sample, target_sample

    def __len__(self):
        return len(self.source)

    def build_vocab(self):
        """
        Construct vocabulary for both src and tgt using loaded data, returns said
        lex
        """
        def get_tokens(data_iter, place):
            for de, en in data_iter:
                if place == 0:
                    yield de
                else:
                    yield en
    
        self.srcLex = build_vocab_from_iterator(
            get_tokens(self, 0),
            min_freq = hp.lex_min_freq,
            specials = ['<pad>', '<sos>', '<eos>', '<unk>'],
            special_first=True
        )
        self.srcLex.set_default_index(self.srcLex['<unk>'])
        
        self.tgtLex = build_vocab_from_iterator(
            get_tokens(self, 1),
            min_freq = hp.lex_min_freq,
            specials = ['<pad>', '<sos>', '<eos>', '<unk>'],
            special_first=True
        )
        self.tgtLex.set_default_index(self.tgtLex['<unk>'])
        assert self.srcLex['<pad>'] == self.tgtLex['<pad>'] == hp.pad_idx
        assert self.srcLex['<sos>'] == self.srcLex['<sos>'] == hp.sos_idx
        assert self.srcLex['<eos>'] == self.srcLex['<eos>'] == hp.eos_idx
        assert self.srcLex['<unk>'] == self.srcLex['<unk>'] == hp.unk_idx
        return self.srcLex, self.tgtLex


def collate_batch(batch, srcLex, tgtLex):
    source, target = [], []
    for f, e in batch:
        source.append(torch.tensor([srcLex[f_tok] for f_tok in ['<sos>'] + f + ['<eos>']]))
        target.append(torch.tensor([tgtLex[e_tok] for e_tok in ['<sos>'] + e + ['<eos>']]))

    source = pad_sequence(source, padding_value=hp.pad_idx)
    target = pad_sequence(target, padding_value=hp.pad_idx)
    return source.to(hp.device), target.to(hp.device)


def loadTestData(srcFile, srcLex, device=0, linesToLoad=sys.maxsize):
    test_iter = Dataset(srcFile, srcFile, srcLex, srcLex, linesToLoad=linesToLoad)
    test_dl = DataLoader(list(test_iter), batch_size=1, shuffle=False, 
                         collate_fn=lambda batch:collate_batch(batch, srcLex, srcLex))
    return test_dl

def load_models(dir):
    models = []
    for filename in os.listdir(dir):
        if filename.endswith('.pt'):
            file_path = os.path.join(dir, filename)
            # print(file_path)
            model = Seq2Seq(build=False)
            model.load(file_path)
            model.to(hp.device)
            model.eval()
            models.append(model)
            print(f'{file_path} loaded')
    return models

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option(
        "-m", "--model", dest="model", default=os.path.join('data'), 
        help="model file")
    optparser.add_option(
        "-i", "--input", dest="input", default=os.path.join('data', 'input', 'dev.txt'),
        help="input file")
    optparser.add_option(
        "-n", "--num", dest="num", default=sys.maxsize, type='int',
        help="num of lines to load")
    (opts, _) = optparser.parse_args()

    model = load_models(opts.model)
    # model = Seq2Seq(build=False)
    # model.load(opts.model)
    # model.to(hp.device)
    # model.eval()

    # loading test dataset

    test_dl = loadTestData(opts.input, model[0].params['srcLex'],
                           device=hp.device, linesToLoad=opts.num)
    results = translate(model, test_dl)
    print("\n".join(results))

