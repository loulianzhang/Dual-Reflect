#!/usr/bin/env python
# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

class LLM(object):
    def __init__(self, model_path):
        self.search = "beam"
        self.temperature = 0.01
        self.max_new_tokens = 256
        self.num_beams = 1
        
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.tokenizer.padding_side = "left"
        self.model.cuda()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = tokenizer.eos_token

        self.gen_config = GenerationConfig(
                        temperature=self.temperature,
                        do_sample=False if self.search=="beam" else True,
                        num_beams=self.num_beams,
                        max_new_tokens=self.max_new_tokens,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token=self.tokenizer.pad_token_id,)

    def generate(self, prompt):
        # Generate
        tokenized = self.tokenizer([prompt], padding=True, return_tensors="pt")
        input_ids = tokenized.input_ids.cuda()
        attn_mask = tokenized.attention_mask.cuda()
        input_ids = input_ids[:, :-1] if input_ids[0, -1] == self.tokenizer.eos_token_id else input_ids
        attn_mask = attn_mask[:, :-1] if input_ids[0, -1] == self.tokenizer.eos_token_id else attn_mask

        with torch.no_grad():
            generated_ids = self.model.generate(inputs=input_ids, attention_mask=attn_mask, generation_config=self.gen_config)

        gen_ids = generated_ids[0]

        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        result = gen_text.replace(prompt, "").replace("\n", "").strip()
        return result


if __name__=="__main__":
    model_path = "/mgData4/loulianzhang/llm/alpaca/alpaca-7b-full"
    llm = LLM(model_path)
    #res = llm.generate("Instruction: Translate the following Chinese text into English.\nChinese: 是否有途径处罚他\nEnglish:")
    #res = llm.generate("Instruction: Compare following two Chinese sentence in terms of meaning.\n这条街有很多可以当首饰的店铺。\n这条街有很多商店可以用来制作珠宝。\nComparison: The word \"当\" in the first sentence means to sell or mortgage. This sentence means that there are many pawn shops on this street that can mortgage jewelry; while the second sentence means that there are many pawn shops on the street that can make jewelry. Jewelry store. The meanings of these two sentences are very different.\n\nInstruction: Compare following two Chinese sentence in terms of meaning.\n男主角从小和奶奶相依为命。\n男主从小和奶奶生活在一起。\nComparison:")
    prompt = eval(open("sample.jsonl").read())["translate"]
    res = llm.generate(prompt)
    print(prompt)
    print(res)
