from config import ConfigArgs as args
import os, sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import numpy as np
import pandas as pd
from model import Text2Mel, SSRN
from data import TextDataset, synth_collate_fn, load_vocab
import utils
from scipy.io.wavfile import write

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def synthesize(t2m, ssrn, data_loader, batch_size=100):
    '''
    DCTTS Architecture
    Text --> Text2Mel --> SSRN --> Wav file
    '''
    # Text2Mel
    idx2char = load_vocab()[-1]
    with torch.no_grad():
        print('='*10, ' Text2Mel ', '='*10)
        total_mel_hats = torch.zeros([len(data_loader.dataset), args.max_Ty, args.n_mels]).to(DEVICE)
        mags = torch.zeros([len(data_loader.dataset), args.max_Ty*args.r, args.n_mags]).to(DEVICE)
        for step, (texts, _, _) in enumerate(data_loader):
            texts = texts.to(DEVICE)
            prev_mel_hats = torch.zeros([len(texts), args.max_Ty, args.n_mels]).to(DEVICE)
            for t in tqdm(range(args.max_Ty-1), unit='B', ncols=70):
                mel_hats, A = t2m(texts, prev_mel_hats) # mel: (N, Ty/r, n_mels)
                prev_mel_hats[:, t+1, :] = mel_hats[:, t, :]
            total_mel_hats[step*batch_size:(step+1)*batch_size, :, :] = prev_mel_hats
            
            print('='*10, ' Alignment ', '='*10)
            alignments = A.cpu().detach().numpy()
            visual_texts = texts.cpu().detach().numpy()
            for idx in range(len(alignments)):
                text = [idx2char[ch] for ch in visual_texts[idx]]
                utils.plot_att(alignments[idx], text, args.global_step, path=os.path.join(args.sampledir, 'A'), name='{}.png'.format(idx))
            print('='*10, ' SSRN ', '='*10)
            # Mel --> Mag
            mags[step*batch_size:(step+1)*batch_size:, :, :] = \
                ssrn(total_mel_hats[step*batch_size:(step+1)*batch_size, :, :]) # mag: (N, Ty, n_mags)
            mags = mags.cpu().detach().numpy()
        print('='*10, ' Vocoder ', '='*10)
        for idx in trange(len(mags), unit='B', ncols=70):
            wav = utils.spectrogram2wav(mags[idx])
            write(os.path.join(args.sampledir, '{}.wav'.format(idx+1)), args.sr, wav)
    return None

def main():
    testset = TextDataset(args.testset)
    test_loader = DataLoader(dataset=testset, batch_size=args.test_batch, drop_last=False,
                             shuffle=False, collate_fn=synth_collate_fn, pin_memory=True)

    t2m = Text2Mel().to(DEVICE)
    ssrn = SSRN().to(DEVICE)
    
    ckpt = pd.read_csv(os.path.join(args.logdir, t2m.name, 'ckpt.csv'), sep=',', header=None)
    ckpt.columns = ['models', 'loss']
    ckpt = ckpt.sort_values(by='loss', ascending=True)
    state = torch.load(os.path.join(args.logdir, t2m.name, ckpt.models.loc[0]))
    t2m.load_state_dict(state['model'])
    args.global_step = state['global_step']

    ckpt = pd.read_csv(os.path.join(args.logdir, ssrn.name, 'ckpt.csv'), sep=',', header=None)
    ckpt.columns = ['models', 'loss']
    ckpt = ckpt.sort_values(by='loss', ascending=True)
    state = torch.load(os.path.join(args.logdir, ssrn.name, ckpt.models.loc[0]))
    ssrn.load_state_dict(state['model'])

    print('All of models are loaded.')

    t2m.eval()
    ssrn.eval()
    
    if not os.path.exists(os.path.join(args.sampledir, 'A')):
        os.makedirs(os.path.join(args.sampledir, 'A'))
    synthesize(t2m, ssrn, test_loader, args.test_batch)

if __name__ == '__main__':
    main()
