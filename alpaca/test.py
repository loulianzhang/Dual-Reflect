#!/usr/bin/env python
# -*- coding: utf-8 -*-
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

model_path="/mgData1/yuxuan/outputdir/rl_test/base_checkpoint/llama-7b-hf"

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path)

model_path = "/mgData4/loulianzhang/llm/alpaca/alpaca-7b-full"

alpaca_model = AutoModelForCausalLM.from_pretrained(model_path)
alpaca_tokenizer = AutoTokenizer.from_pretrained(model_path)
