import numpy as np
import pandas as pd
import os, sys
import torch
from torch.utils.data.dataset import Dataset
import glob, re
import utils
import codecs, unicodedata
from config import ConfigArgs as args

class SpeechDataset(Dataset):
    def __init__(self, data_path, metadata, model_name, mem_mode=False, ga_mode=False):
        '''
        Args:
            data_path (str): path to dataset
            meta_path (str): path to metadata csv file
            model_name (str): {'Text2Mel', 'SSRN', 'All'}
        '''
        self.data_path = data_path
        self.model_name = model_name
        self.mem_mode = mem_mode
        self.ga_mode = ga_mode
        self.fpaths, self.texts, self.norms = read_meta(os.path.join(data_path, metadata))
        if self.mem_mode:
            self.mels = [torch.tensor(np.load(os.path.join(
                self.data_path, args.mel_dir, path))) for path in self.fpaths]
        if self.ga_mode:
            self.g_att = [torch.tensor(np.load(os.path.join(
                self.data_path, args.ga_dir, path))) for path in self.fpaths]
        
    def __getitem__(self, idx):
        text, mel, mag = None, None, None
        text = torch.tensor(self.norms[idx], dtype=torch.long)
        # Memory mode is faster
        if not self.mem_mode:
            mel_path = os.path.join(self.data_path, args.mel_dir, self.fpaths[idx])
            mel = torch.tensor(np.load(mel_path))
        else:
            mel = self.mels[idx]

        if self.model_name == 'Text2Mel':
            if not self.ga_mode:
                return (text, mel)
            else:
                # Guided attention mode
                return (text, mel, self.g_att[idx])

        mag_path = os.path.join(self.data_path, args.mag_dir, self.fpaths[idx])
        mag = torch.tensor(np.load(mag_path))
        return (text, mel, mag)

    def __len__(self):
        return len(self.fpaths)

def load_vocab():
    char2idx = {char: idx for idx, char in enumerate(args.vocab)}
    idx2char = {idx: char for idx, char in enumerate(args.vocab)}
    return char2idx, idx2char

def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                   if unicodedata.category(char) != 'Mn')  # Strip accents
    text = text.lower()
    text = re.sub(u"[^{}]".format(args.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

def read_meta(path):
    '''
    If we use pandas instead of this function, it may not cover quotes.
    Args:
        path: metadata path
    Returns:
        fpaths, texts, norms
    '''
    char2idx, _ = load_vocab()
    lines = codecs.open(path, 'r', 'utf-8').readlines()
    fpaths, texts, norms = [], [], []
    for line in lines:
        fname, text, norm = line.strip().split('|')
        fpath = fname + '.npy'
        text = text_normalize(text).strip() + u'E'  # ␃: EOS
        text = [char2idx[char] for char in text]
        norm = text_normalize(norm).strip() + u'E'  # ␃: EOS
        norm = [char2idx[char] for char in norm]
        fpaths.append(fpath)
        texts.append(text)
        norms.append(norm)
    return fpaths, texts, norms

def collate_fn(data):
    """
    Creates mini-batch tensors from the list of tuples (texts, mels, mags).
    Args:
        data: list of tuple (texts, mels, mags).
            - texts: torch tensor of shape (B, Tx).
            - mels: torch tensor of shape (B, Ty/4, n_mels).
            - mags: torch tensor of shape (B, Ty, n_mags).
    Returns:
        texts: torch tensor of shape (batch_size, padded_length).
        mels: torch tensor of shape (batch_size, padded_length, n_mels).
        mels: torch tensor of shape (batch_size, padded_length, n_mags).
    """
    # Sort a data list by text length (descending order).
    data.sort(key=lambda x: len(x[0]), reverse=True)
    texts, mels, mags = zip(*data)

    # Merge (from tuple of 1D tensor to 2D tensor).
    text_lengths = [len(text) for text in texts]
    mel_lengths = [len(mel) for mel in mels]
    mag_lengths = [len(mag) for mag in mags]
    # (number of mels, max_len, feature_dims)
    text_pads = torch.zeros(len(texts), max(text_lengths), dtype=torch.long)
    mel_pads = torch.zeros(len(mels), max(mel_lengths), mels[0].shape[-1])
    mag_pads = torch.zeros(len(mags), max(mag_lengths), mags[0].shape[-1])
    for idx in range(len(mels)):
        text_end = text_lengths[idx]
        text_pads[idx, :text_end] = texts[idx]
        mel_end = mel_lengths[idx]
        mel_pads[idx, :mel_end] = mels[idx]
        mag_end = mag_lengths[idx]
        mag_pads[idx, :mag_end] = mags[idx]
    return text_pads, mel_pads, mag_pads

def t2m_collate_fn(data):
    """
    Creates mini-batch tensors from the list of tuples (texts, mels, mags).
    Args:
        data: list of tuple (texts).
            - texts: torch tensor of shape (B, Tx).
            - mels: torch tensor of shape (B, Ty/4, n_mels).
    Returns:
        texts: torch tensor of shape (batch_size, padded_length).
        mels: torch tensor of shape (batch_size, padded_length, n_mels).
    """
    # Sort a data list by text length (descending order).
    data.sort(key=lambda x: len(x[0]), reverse=True)
    texts, mels = zip(*data)

    # Merge (from tuple of 1D tensor to 2D tensor).
    text_lengths = [len(text) for text in texts]
    mel_lengths = [len(mel) for mel in mels]
    # (number of mels, max_len, feature_dims)
    text_pads = torch.zeros(len(texts), max(text_lengths), dtype=torch.long)
    mel_pads = torch.zeros(len(mels), max(mel_lengths), mels[0].shape[-1])
    for idx in range(len(mels)):
        text_end = text_lengths[idx]
        text_pads[idx, :text_end] = texts[idx]
        mel_end = mel_lengths[idx]
        mel_pads[idx, :mel_end] = mels[idx]
    return text_pads, mel_pads, None

def t2m_ga_collate_fn(data):
    """
    Creates mini-batch tensors from the list of tuples (texts, mels, mags).
    Args:
        data: list of tuple (texts).
            - texts: torch tensor of shape (B, Tx).
            - mels: torch tensor of shape (B, Ty/4, n_mels).
            - gas: torch tensor of shape (B, max_Tx, max_Ty).
    Returns:
        texts: torch tensor of shape (B, padded_length).
        mels: torch tensor of shape (B, padded_length, n_mels).
        gas: torch tensor of shape (B, Tx, Ty/4)
    """
    # Sort a data list by text length (descending order).
    data.sort(key=lambda x: len(x[0]), reverse=True)
    texts, mels, gas = zip(*data)
    # Merge (from tuple of 1D tensor to 2D tensor).
    text_lengths = [len(text) for text in texts]
    mel_lengths = [len(mel) for mel in mels]
    # (number of mels, max_len, feature_dims)
    text_pads = torch.zeros(len(texts), max(text_lengths), dtype=torch.long)
    mel_pads = torch.zeros(len(mels), max(mel_lengths), mels[0].shape[-1])
    ga_pads = torch.zeros(len(mels), max(text_lengths), max(mel_lengths))
    for idx in range(len(mels)):
        text_end = text_lengths[idx]
        text_pads[idx, :text_end] = texts[idx]
        mel_end = mel_lengths[idx]
        mel_pads[idx, :mel_end] = mels[idx]
        ga_pads[idx] = gas[idx][:max(text_lengths), :max(mel_lengths)]
    return text_pads, mel_pads, ga_pads

class TextDataset(Dataset):
    def __init__(self, text_path):
        '''
        Args:
            text path (str): path to text set
        '''
        self.texts = read_text(text_path)

    def __getitem__(self, idx):
        text = torch.tensor(self.texts[idx], dtype=torch.long)
        return text

    def __len__(self):
        return len(self.texts)


def read_text(path):
    '''
    If we use pandas instead of this function, it may not cover quotes.
    Args:
        path: metadata path
    Returns:
        fpaths, texts, norms
    '''
    char2idx, _ = load_vocab()
    lines = codecs.open(path, 'r', 'utf-8').readlines()[1:]
    texts = []
    for line in lines:
        text = text_normalize(line.split(' ', 1)[-1]).strip() + u'E'  # ␃: EOS
        text = [char2idx[char] for char in text]
        texts.append(text)
    return texts

def synth_collate_fn(data):
    """
    Creates mini-batch tensors from the list of tuples (texts, mels, mags).
    Args:
        data: list of tuple (texts,).
            - texts: torch tensor of shape (B, Tx).
    Returns:
        texts: torch tensor of shape (batch_size, padded_length).
    """
    texts = data

    # Merge (from tuple of 1D tensor to 2D tensor).
    text_lengths = [len(text) for text in texts]
    # (number of mels, max_len, feature_dims)
    text_pads = torch.zeros(len(texts), max(text_lengths), dtype=torch.long)
    for idx in range(len(texts)):
        text_end = text_lengths[idx]
        text_pads[idx, :text_end] = texts[idx]
    return text_pads, None, None
