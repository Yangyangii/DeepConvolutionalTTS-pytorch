from config import ConfigArgs as args
import librosa
import numpy as np
import os, sys
from scipy import signal
import copy
import torch
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

def load_spectrogram(fpath):
    wav, sr = librosa.load(fpath, sr=args.sr)

    ## Pre-processing
    wav, _ = librosa.effects.trim(wav)
    wav = np.append(wav[0], wav[1:] - args.preemph * wav[:-1])
    # STFT
    linear = librosa.stft(y=wav,
                          n_fft=args.n_fft,
                          hop_length=args.hop_length,
                          win_length=args.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(args.sr, args.n_fft, args.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - args.ref_db + args.max_db) / args.max_db, 1e-8, 1)
    mag = np.clip((mag - args.ref_db + args.max_db) / args.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    mel, mag = padding_reduction(mel, mag)
    return mel, mag

def padding_reduction(mel, mag):
    # Padding
    t = mel.shape[0]
    n_paddings = args.r - (t % args.r) if t % args.r != 0 else 0  # for reduction
    mel = np.pad(mel, [[0, n_paddings], [0, 0]], mode="constant")
    mag = np.pad(mag, [[0, n_paddings], [0, 0]], mode="constant")
    mel = mel[::args.r, :]
    return mel, mag

def spectrogram2wav(mag):
    '''# Generate wave file from spectrogram'''
    # transpose
    mag = mag.T

    # de-normalize
    mag = (np.clip(mag, 0, 1) * args.max_db) - args.max_db + args.ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -args.preemph], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)

def griffin_lim(spectrogram):
    '''
    Applies Griffin-Lim's raw.
    '''
    X_best = copy.deepcopy(spectrogram)
    for i in range(args.gl_iter):
        X_t = librosa.istft(X_best, args.hop_length, win_length=args.win_length, window="hann")
        est = librosa.stft(X_t, args.n_fft, args.hop_length, win_length=args.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = librosa.istft(X_best, args.hop_length, win_length=args.win_length, window="hann")
    y = np.real(X_t)
    return y

def att2img(A):
    '''
    Args:
        A: (1, Tx, Ty) Tensor
    '''
    for i in range(A.shape[-1]):
        att = A[0, :, i]
        local_min, local_max = att.min(), att.max()
        A[0, :, i] = (att-local_min)/local_max
    return A


def plot_att(A, text, global_step, path='.', name=None):
    '''
    Args:
        A: (Tx, Ty) numpy array
        text: (Tx,) list
        global_step: scalar
    '''
    fig, ax = plt.subplots(figsize=(25, 25))
    im = ax.imshow(A)
    fig.colorbar(im, fraction=0.035, pad=0.02)
    fig.suptitle('{} Steps'.format(global_step), fontsize=30)
    plt.ylabel('Text', fontsize=22)
    plt.xlabel('Time', fontsize=22)
    plt.yticks(np.arange(len(text)), text)
    if name is not None:
        plt.savefig(os.path.join(path, name), format='png')
    else:
        plt.savefig(os.path.join(
            path, 'A-{}.png'.format(global_step)), format='png')
    plt.close(fig)
    
def prepro_guided_attention(N, T, g=0.2):
    W = np.zeros([args.max_Tx, args.max_Ty], dtype=np.float32)
    for tx in range(args.max_Tx):
        for ty in range(args.max_Ty):
            if ty <= T:
                W[tx, ty] = 1.0 - np.exp(-0.5 * (ty/T - tx/N)**2 / g**2)
            else:
                W[tx, ty] = 1.0 - np.exp(-0.5 * ((N-1)/N - tx/N)**2 / (g/2)**2) # forcing more at end step
    return W
