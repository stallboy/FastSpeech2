""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''

_pad = '_'
_punctuation = '!\'(),.:;? '
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_silences = ['@sp', '@spn', '@sil']

from text import cmudict

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = [_pad] + list(_special) + list(_punctuation) + \
          list(_letters) + _arpabet + _silences

import hparams as hp

# 以下是中文需要的
if hp.dataset == "thchs30":
    from text import thchs_phone

    _arpabet = ['@' + s for s in thchs_phone.valid_symbols]
    symbols = [_pad] + _arpabet + _silences
