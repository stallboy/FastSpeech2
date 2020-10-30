# coding=utf8

import os
import shutil
import utils
import hparams as hp
import re
import numpy as np
import torch


#######################################

def add_lab_files():
    dst_dir = "d:\\data\\data_thchs30\\per_speaker\\"
    src_dir = "d:\\data\\data_thchs30\\data\\"
    wavs = [fn for fn in os.listdir(src_dir) if fn.endswith(".wav")]
    for wav in wavs:
        basename = wav[:-4]
        speaker = basename.split("_")[0]
        speaker_dir = os.path.join(dst_dir, speaker)
        if not os.path.exists(speaker_dir):
            os.mkdir(speaker_dir)

        with open(os.path.join(src_dir, basename + ".wav.trn"), encoding='utf-8') as trn_f:
            trn_f.readline()
            pinyin = trn_f.readline()
            pinyin = pinyin.replace("nve4", "nue4")
            with open(os.path.join(speaker_dir, basename + ".lab"), "w")as lab_f:
                lab_f.write(pinyin)

        # shutil.copyfile(os.path.join(src_dir, wav), os.path.join(speaker_dir, wav))


#######################################


def flat_text_grid():
    src_dir = "..\\preprocessed\\thchs30\\TextGrid3\\"
    dst_dir = "..\\preprocessed\\thchs30\\TextGrid\\"

    for speaker in os.listdir(src_dir):
        for text_grid in os.listdir(os.path.join(src_dir, speaker)):
            shutil.move(os.path.join(src_dir, speaker, text_grid), os.path.join(dst_dir, text_grid))


#######################################

def check_texts():
    check_text_to_sequence("train.txt")
    check_text_to_sequence("val.txt")


cur_processing = ""


def check_text_to_sequence(fn):
    basename_list, text_list = utils.process_meta(os.path.join("..", hp.preprocessed_path, fn))
    for i, basename in enumerate(basename_list):
        text = text_list[i]
        global cur_processing
        cur_processing = basename
        text_to_sequence(text)
    print("check text done. fn=%s, cnt=%d" % (fn, len(basename_list)))


_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def text_to_sequence(text):
    sequence = []
    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)

        if not m:
            sequence += _symbols_to_sequence(text)
            break
        sequence += _symbols_to_sequence(m.group(1))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    return sequence


from text.symbols import symbols

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def _symbols_to_sequence(symbols):
    for s in symbols:
        if s not in _symbol_to_id:
            print(cur_processing + " [" + s + "] 不认识")
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id


#######################################

import librosa
from scipy.io.wavfile import write


def resample(in_dir, out_dir, basename):  # 这个效果不行，会有电流杂音，不要用
    wav_path = os.path.join(in_dir, 'data', '{}.wav'.format(basename))
    tgt_path = os.path.join(in_dir, 'wavs', '{}.wav'.format(basename))
    audio, sr = librosa.load(wav_path, 22050, dtype=np.float64)
    audio = (audio * 32767).astype(np.int16)
    write(tgt_path, 22050, audio)


def resample_use_ffmpeg():
    ffmpeg_file = "c:\\work\\ffmpeg-4.3.1-2020-10-01-essentials_build\\bin\\ffmpeg.exe"
    import subprocess

    src_dir = "d:\\data\\data_thchs30\\data\\"
    out_dir = "d:\\data\\data_thchs30\\wavs\\"

    wavs = [fn for fn in os.listdir(src_dir) if fn.endswith(".wav")]
    for wav in wavs:
        subprocess.run([ffmpeg_file, "-i", os.path.join(src_dir, wav), "-ar", "22050", os.path.join(out_dir, wav)])


#######################################

def check_vocoder():
    test_dir = "d:\\data\\data_thchs30\\test_vocoder\\"
    wavs = [fn for fn in os.listdir(test_dir) if fn.endswith(".wav")]

    melgan = utils.get_melgan()
    for wav in wavs:
        basename = wav[:-4]

        mel_path = os.path.join("..", hp.preprocessed_path, "mel", "{}-mel-{}.npy".format(hp.dataset, basename))
        if os.path.isfile(mel_path):
            mel_target_np = np.load(mel_path)
            mel_target = torch.from_numpy(mel_target_np).unsqueeze(0).transpose(1, 2)
            utils.melgan_infer(mel_target, melgan,
                               os.path.join(test_dir, '{}_ground-truth_melgan.wav'.format(basename)))

    # waveglow = utils.get_waveglow()
    # for wav in wavs:
    #     basename = wav[:-4]
    #
    #     mel_path = os.path.join("..", hp.preprocessed_path, "mel", "{}-mel-{}.npy".format(hp.dataset, basename))
    #     if os.path.isfile(mel_path):
    #         mel_target_np = np.load(mel_path)
    #         mel_target = torch.from_numpy(mel_target_np).unsqueeze(0).transpose(1, 2).cuda()
    #
    #         utils.waveglow_infer(mel_target, waveglow,
    #                              os.path.join(test_dir, '{}_ground-truth_waveglow.wav'.format(basename, hp.vocoder)))


if __name__ == "__main__":
    # flat_text_grid()
    # check_texts()
    check_vocoder()
    # resample_use_ffmpeg()
