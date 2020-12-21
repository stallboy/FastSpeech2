# coding=utf8

import os

import numpy as np
import pyworld as pw
import tgt
import torch
from scipy.io.wavfile import read

import audio as Audio
import hparams as hp
from text import _clean_text
from utils import get_alignment


def prepare_align(in_dir):
    with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            basename = parts[0]
            text = parts[2]
            text = _clean_text(text, hp.text_cleaners)

            with open(os.path.join(in_dir, 'wavs', '{}.txt'.format(basename)), 'w') as f1:
                f1.write(text)


def build_from_path(in_dir, out_dir):
    index = 1
    train = list()
    val = list()
    f0_max = energy_max = 0
    f0_min = energy_min = 1000000
    n_frames = 0

    init_chs_dict()
    file_names = os.listdir(os.path.join(hp.preprocessed_path, "TextGrid"))
    for file_name in file_names:
        if not file_name.endswith('.TextGrid'):
            continue
        basename = file_name[:-9]

        ret = process_utterance(in_dir, out_dir, basename)
        if ret is None:
            continue
        else:
            # info, f_max, f_min, e_max, e_min, n = ret
            info = ret

        if basename[:4] in ['A11_', 'A12_', 'A13_']:
            val.append(info)
        else:
            train.append(info)

        if index % 100 == 0:
            print("Done %d" % index)
        index = index + 1

    #     f0_max = max(f0_max, f_max)
    #     f0_min = min(f0_min, f_min)
    #     energy_max = max(energy_max, e_max)
    #     energy_min = min(energy_min, e_min)
    #     n_frames += n
    #
    # with open(os.path.join(out_dir, 'stat.txt'), 'w', encoding='utf-8') as f:
    #     strs = ['Total time: {} hours'.format(n_frames * hp.hop_length / hp.sampling_rate / 3600),
    #             'Total frames: {}'.format(n_frames),
    #             'Min F0: {}'.format(f0_min),
    #             'Max F0: {}'.format(f0_max),
    #             'Min energy: {}'.format(energy_min),
    #             'Max energy: {}'.format(energy_max)]
    #     for s in strs:
    #         print(s)
    #         f.write(s + '\n')

    return [r for r in train if r is not None], [r for r in val if r is not None]


def process_utterance(in_dir, out_dir, basename):
    wav_path = os.path.join(in_dir, 'wavs', '{}.wav'.format(basename))
    tg_path = os.path.join(out_dir, 'TextGrid', '{}.TextGrid'.format(basename))

    # Get alignments
    textgrid = tgt.io.read_textgrid(tg_path)
    # phone: list<112, phone string>, 已去掉前后静音
    # duration: list<112, frames number per phone>, 每个里面是此phone持续的帧数
    # start, end: float,表示的是去掉音频文件中前后空白silence音后的区间。
    phone, duration, start, end = get_alignment(textgrid.get_tier_by_name('phones'))
    if start >= end:
        return None

    phone, duration = add_pad_between_word(phone, duration, textgrid)
    sum_duration = sum(duration)
    text = '{' + '}{'.join(phone) + '}'  # '{A}{B}{$}{C}', $ represents silent phones
    text = text.replace('{$}', ' ')  # '{A}{B} {C}'
    text = text.replace('}{', ' ')  # '{A B} {C}'

    # Read and trim wav files
    # wav ndarray<212893>
    _, wav = read(wav_path)
    wav = wav[int(hp.sampling_rate * start):int(hp.sampling_rate * end)].astype(np.float32)

    # Compute mel-scale spectrogram and energy
    # mel_spectrogram: ndarray<80, 831> 梅尔普，这里范围是0-8000HZ内，再分成80段，怎么分还不知道
    # energy: ndarray<831> 音量，这里范围是0到315
    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(torch.FloatTensor(wav))
    mel_spectrogram = mel_spectrogram.numpy().astype(np.float32)[:, :sum_duration]
    if mel_spectrogram.shape[1] >= hp.max_seq_len:
        return None

    # energy = energy.numpy().astype(np.float32)[:sum_duration]
    #
    # # Compute fundamental frequency
    # # f0 ndarray<832>
    # f0, _ = pw.dio(wav.astype(np.float64), hp.sampling_rate, frame_period=hp.hop_length / hp.sampling_rate * 1000)
    # # f0 ndarray<831> 基础频率，也可以认为是声带振动的频率，人类一般是140HZ，这里范围是70-800HZ
    # f0 = f0[:sum_duration]


    # Save alignment
    ali_filename = '{}-ali-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'alignment', ali_filename), duration, allow_pickle=False)

    # # Save fundamental prequency
    # f0_filename = '{}-f0-{}.npy'.format(hp.dataset, basename)
    # np.save(os.path.join(out_dir, 'f0', f0_filename), f0, allow_pickle=False)
    #
    # # Save energy
    # energy_filename = '{}-energy-{}.npy'.format(hp.dataset, basename)
    # np.save(os.path.join(out_dir, 'energy', energy_filename), energy, allow_pickle=False)
    #
    # # Save spectrogram
    # mel_filename = '{}-mel-{}.npy'.format(hp.dataset, basename)
    # np.save(os.path.join(out_dir, 'mel', mel_filename), mel_spectrogram.T, allow_pickle=False)

    # return '|'.join([basename, text]), max(f0), min([f for f in f0 if f != 0]), max(energy), min(energy), \
    #        mel_spectrogram.shape[1]

    return '|'.join([basename, text])


mapping = {}


def init_chs_dict():
    dictionary_path = "c:\\work\\TTS\\montreal-forced-aligner\\pretrained_models\\mandarin-lexicon.txt"
    with open(dictionary_path, encoding='utf-8') as f:
        for line in f:
            character, phones = line.rstrip().split(' ', 1)
            mapping[character] = phones


def add_pad_between_word(phone, duration, textgrid):
    tier = textgrid.get_tier_by_name('words')

    new_phone = []
    new_duration = []
    sil_phones = ['sil', 'sp', 'spn']
    idx = 0
    for t in tier._objects:
        s, e, character = t.start_time, t.end_time, t.text
        ph = mapping.get(character)
        if ph is None:
            continue

        if phone[idx] in sil_phones:
            new_phone.append(phone[idx])
            new_duration.append(duration[idx])
            idx += 1

        elif idx != 0:
            new_phone.append('_')  # 跟symbols里的pad相同
            new_duration.append(0)

        for p in ph.split():
            assert p == phone[idx]
            new_phone.append(phone[idx])
            new_duration.append(duration[idx])
            idx += 1

    while idx < len(phone):
        new_phone.append(phone[idx])
        new_duration.append(duration[idx])
        idx += 1

    return new_phone, new_duration
