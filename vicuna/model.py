#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Use FastChat with Hugging Face generation APIs.

Usage:
python3 -m fastchat.serve.huggingface_api --model lmsys/vicuna-7b-v1.5
python3 -m fastchat.serve.huggingface_api --model lmsys/fastchat-t5-3b-v1.0
"""
import argparse

import torch

from fastchat.model import load_model, get_conversation_template, add_model_args
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class LLM(object):
    def __init__(self, model_path):
        # Load model
    # Reset default repetition penalty for T5 models.
        self.device = "cuda"
        self.temperature = 0.0
        self.repetition_penalty = 1.0
        self.max_new_tokens = 1024
        self.conv = get_conversation_template(model_path)
        
        self.model, self.tokenizer = load_model(
            model_path,
            device=self.device,
            num_gpus=2,
            max_gpu_memory="80Gib",
            load_8bit=False,
            cpu_offloading=False,
            debug=False,)

    @torch.inference_mode()
    def generate(self, prompt):
        # Build the prompt with a conversation template
        msg = prompt
        self.conv.append_message(self.conv.roles[0], msg)
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()

        # Run inference
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            do_sample=True if self.temperature > 1e-5 else False,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_new_tokens,
        )
    
        if self.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
        outputs = self.tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
    
        self.conv.update_last_message(outputs)
        # Print results
        #print(f"{conv.roles[0]}: {msg}")
        #print(f"{conv.roles[1]}: {outputs}")
        return outputs  

    def clear_history(self):
        self.conv.messages = []


if __name__ == "__main__":
    model_path = "/mgData4/loulianzhang/llm/vicuna/vicuna-7b-v1.5"
    llm = LLM(model_path)
    #res = llm.generate("Translate following Chinese text into English:\n吃掉敌人一个师")
    #res = llm.generate("Compare following two Chinese sentence in terms of meaning.\n这条街有很多可以当首饰的店铺。\n这条街有很多商店可以用来制作珠宝。")
    #print(res)
    while True:
        text = input()
        res = llm.generate(text)
        print(res)
