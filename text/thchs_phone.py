# coding=utf8

# 从https://montreal-forced-aligner.readthedocs.io/en/latest/pretrained_models.html
# 的Pretrained acoustic models的Mandarin.zip里的meta.yaml里提取

valid_symbols = "a1 a2 a3 a4 a5 ai1 ai2 ai3 ai4 ai5 ao1 ao2 ao3 ao4 ao5 " \
                "b c ch d e1 e2 e3 e4 e5 ei1 ei2 ei3 ei4 ei5 f g h i1 i2 i3 i4 " \
                "i5 ia1 ia2 ia3 ia4 ia5 iao1 iao2 iao3 iao4 iao5 ie1 ie2 ie3 ie4 ie5 " \
                "ii1 ii2 ii3 ii4 ii5 io1 io2 io3 io4 io5 iou1 iou2 iou3 iou4 iu1 iu2 " \
                "iu3 iu4 iu5 j k l m n ng o1 o2 o3 o4 o5 ou1 ou2 ou3 ou4 ou5 p " \
                "q r s sh t u1 u2 u3 u4 u5 ua1 ua2 ua3 ua4 ua5 uai1 uai2 uai3 uai4 " \
                "uai5 ue1 ue2 ue3 ue4 ue5 uei1 uei2 uei3 uei4 uei5 uo1 uo2 uo3 uo4 " \
                "uo5 v1 v2 v3 v4 v5 va1 va2 va3 va4 ve1 ve2 ve3 ve4 x z zh".split()
