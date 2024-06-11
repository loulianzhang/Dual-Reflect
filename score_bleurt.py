#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

config = BleurtConfig.from_pretrained('/mgData4/loulianzhang/model/BLEURT-20')
model = BleurtForSequenceClassification.from_pretrained('/mgData4/loulianzhang/model/BLEURT-20')
tokenizer = BleurtTokenizer.from_pretrained('/mgData4/loulianzhang/model/BLEURT-20')

references = [line.strip() for line in open("/mgData4/loulianzhang/model/zh_ru/offline/mgpt/valid.zh2ru.offline.out/trans.txt")]
candidates = [line.strip() for line in open("/mgData4/loulianzhang/data/flores101_dataset/devtest/rus.devtest")]


model.eval()
model.to("cuda:4")
with torch.no_grad():
    inputs = tokenizer(references, candidates, padding='longest', truncation=True, return_tensors='pt', max_length=512)
    inputs.to("cuda:4")
    res = model(**inputs).logits.flatten().tolist()
print(sum(res)/len(res))
# [0.9990496635437012, 0.7930182218551636]
