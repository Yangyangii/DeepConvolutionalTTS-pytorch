# -*- coding: utf-8 -*-
from config import ConfigArgs as args
from utils import load_spectrogram, prepro_guided_attention
import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import codecs
import data

NUM_JOBS = 8

def f(f_args):
    fpath, text = f_args
    mel, mag = load_spectrogram(os.path.join(args.data_path, 'wavs', fpath.replace('npy', 'wav')))
    np.save(os.path.join(args.data_path, args.ga_dir, fpath), prepro_guided_attention(len(text), len(mel), g=args.g))
    np.save(os.path.join(args.data_path, args.mel_dir, fpath), mel)
    np.save(os.path.join(args.data_path, args.mag_dir, fpath), mag)
    return None

def prepro_signal():
    print('Preprocessing signal')
    # Load data
    fpaths, texts, _ = data.read_meta(os.path.join(args.data_path, args.meta))

    # Creates folders
    if not os.path.exists(os.path.join(args.data_path, args.mel_dir)):
        os.mkdir(os.path.join(args.data_path, args.mel_dir))
    if not os.path.exists(os.path.join(args.data_path, args.mag_dir)):
        os.mkdir(os.path.join(args.data_path, args.mag_dir))
    if not os.path.exists(os.path.join(args.data_path, args.ga_dir)):
        os.mkdir(os.path.join(args.data_path, args.ga_dir))

    # Creates pool
    p = Pool(NUM_JOBS)

    total_files = len(fpaths)
    with tqdm(total=total_files) as pbar:
        for _ in tqdm(p.imap_unordered(f, list(zip(fpaths,texts)))):
            pbar.update()

def prepro_meta():
    ## train(95%)/test(5%) split for metadata
    print('Preprocessing meta')
    # Parse
    transcript = os.path.join(args.data_path, 'metadata.csv')
    train_transcript = os.path.join(args.data_path, 'meta-train.csv')
    test_transcript = os.path.join(args.data_path, 'meta-eval.csv')

    lines = codecs.open(transcript, 'r', 'utf-8').readlines()
    train_f = codecs.open(train_transcript, 'w', 'utf-8')
    test_f = codecs.open(test_transcript, 'w', 'utf-8')

    test_idx = np.load('lj_eval_idx.npy')

    for idx, line in enumerate(lines):
        if idx in test_idx:
            test_f.write(line)
        else:
            train_f.write(line)
    print('# of train set: {}, # of test set: {}'.format(1+idx-len(test_idx), len(test_idx)))
    print('Complete')

if __name__ == '__main__':
    is_signal = sys.argv[1]
    is_meta = sys.argv[2]
    print('Signal: {}, Meta: {}'.format(is_signal, is_meta))

    if is_signal in ['1', 'True']:
        prepro_signal()
    if is_meta in ['1', 'True']:
        prepro_meta()
