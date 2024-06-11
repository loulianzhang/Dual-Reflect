#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sacrebleu
import json
from comet import download_model, load_from_checkpoint
    
# Choose your model from Hugging Face Hub
model_path = "/mgData4/loulianzhang/llm/Unbabel/wmt22-comet-da/checkpoints/model.ckpt"

# Load the model checkpoint:
model = load_from_checkpoint(model_path)

import pandas as pd
#df = pd.read_json("dual_reflection_agent_with_baseline_as_translator_and_LLM_as_judge.result.jsonl",lines=True)
df = pd.read_json("backup/agent_with_LLM_as_judge_4langs.result.jsonl",lines=True)
df["lang_pair"] = df["src_lang"] + "-" + df["tgt_lang"]
for lp, dt in df.groupby("lang_pair"):

    srcs, hyps, refs = [], [], []
    for _, line in dt.iterrows():
        if line["hyp"] == "ERROR!":
            continue
        srcs.append(line["src"])
        hyps.append(line["hyp"].strip("Translation: "))
        refs.append(line["tgt"])
    
    print(f"test size : {len(srcs)}")
        
    #score = sacrebleu.corpus_bleu(hyps, [refs], tokenize="13a").score
        #score = sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh").score
    #print(score)
    

    # Data must be in the following format:
    data = [{"src": s, "mt": h , "ref": r} for s,h,r in zip(srcs, hyps, refs)]

    model_output = model.predict(data, batch_size=8, gpus=1)
    print(f"language pairs: {lp}")
    print(model_output.system_score * 100) # system-level score
