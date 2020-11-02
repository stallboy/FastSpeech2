# coding=utf8

import os
import pypinyin
import pickle

import phkit


def make_data():
    src_dir = "d:\\data\\data_thchs30\\data\\"
    trns = [fn for fn in os.listdir(src_dir) if fn.endswith(".wav.trn")]

    src = []
    for trn in trns:
        basename = trn[:-8]

        with open(os.path.join(src_dir, trn), encoding='utf-8') as trn_f:
            han = trn_f.readline().strip().replace(' ', '')
            manual_pinyin = trn_f.readline().strip()
            src.append([basename, han, manual_pinyin])

    with open('g2pinyin.pkl', 'wb') as g:
        pickle.dump(src, g)


CRED = '\033[31m'
CGREEN = '\033[32m'
CGREEN_BG = '\033[42m'
CEND = '\033[0m'


def check_pypinyin():
    eq_line = 0
    ne_line = 0
    eq_pinyin = 0
    ne_pinyin = 0
    ne_pinyin_diao = 0
    ne_pinyin_pin = 0

    with open('g2pinyin.pkl', 'rb') as g:
        src = pickle.load(g)

        for basename, han, manual_pinyin in src:
            gen_pinyin_list = pypinyin.lazy_pinyin(han, style=pypinyin.Style.TONE3, neutral_tone_with_five=True)
            gen_pinyin = ' '.join(gen_pinyin_list)

            if manual_pinyin == gen_pinyin:
                eq_line += 1
                eq_pinyin += len(gen_pinyin_list)
            else:
                ne_line += 1
                need_print = ne_line < 8
                if need_print:
                    han_list = list(han)

                manual_pinyin_list = manual_pinyin.split()
                for i, gp in enumerate(gen_pinyin_list):

                    changed = gp != manual_pinyin_list[i]
                    if changed:
                        ne_pinyin += 1
                        if gp[-1] != manual_pinyin_list[i][-1]:
                            ne_pinyin_diao += 1
                        if gp[:-1] != manual_pinyin_list[i][:-1]:
                            ne_pinyin_pin += 1

                        if need_print:
                            han_list[i] = CGREEN_BG + han_list[i] + CEND
                            manual_pinyin_list[i] = CGREEN + manual_pinyin_list[i] + CEND
                            gen_pinyin_list[i] = CRED + gp + CEND

                    else:
                        eq_pinyin += 1

                if need_print:
                    print(basename, ''.join(han_list))
                    print(' '.join(manual_pinyin_list))
                    print(' '.join(gen_pinyin_list))

    print('eq line={}, ne line={}'.format(eq_line, ne_line))
    print('全行 正确率 = {}', eq_line * 100 / (eq_line + ne_line))

    print('正确拼={}, 错误拼={}（错误音={}， 错误调={}）'.format(eq_pinyin, ne_pinyin, ne_pinyin_pin, ne_pinyin_diao))
    all = eq_pinyin + ne_pinyin
    print('错误拼 错误率 = {}， 错音 = {}， 错调 = {}'.format(
          ne_pinyin * 100 / all,
          ne_pinyin_pin * 100 / all,
          ne_pinyin_diao * 100 / all))


if __name__ == "__main__":
    check_pypinyin()
