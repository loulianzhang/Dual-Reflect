#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用llama在wmt22测试集上使用0shot,5shot进行翻译
5shot取flores101的dev set，为了可以复现，每个语种随机取5个句对作为固定的5shot
因为在wmt21年的测试集上没有cs2uk,en2hr的语种
"""

from model import LLM # 此行必须放在第一行，不然会报错？？？
import json
from langcodes import Language
import re
import pandas as pd
from tqdm import tqdm

class FewShotPrompter():
    def __init__(self, src_lang, tgt_lang, path):
        with open(path, encoding="utf-8") as fr:
            prompt_pipeline = json.load(fr)
        self.prompt_pipeline = prompt_pipeline

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        examples = pd.read_json("examples.jsonl",lines=True)
        examples = examples[examples.src_lang==src_lang]
        examples = examples[examples.tgt_lang==tgt_lang]
        self.examples = examples

    # 5shot是将0shot的prompt重复5次
    def get_instance(self, src_text):
        prompt_pipeline = self.prompt_pipeline["translate"]
        prompts = []
        prefix = prompt_pipeline.replace("{src_lang}", self.src_lang)\
                .replace("{tgt_lang}", self.tgt_lang)
        for _, row in self.examples.iterrows():
            shot = prefix.replace("{src_text}",row["src_text"]) + row["tgt_text"]
            prompts.append(shot)
        instruct = prefix.replace("{src_text}",src_text)
        prompts.append(instruct)
        return {"translate":"\n\n\n".join(prompts)}

def few_shot_translate(src_lang, tgt_lang, writer):
    src_data = f"../dataset/wmt22.{src_lang}-{tgt_lang}.{src_lang}"
    tgt_data = f"../dataset/wmt22.{src_lang}-{tgt_lang}.{tgt_lang}"
    model_path = "/mgData4/loulianzhang/llm/alpaca/alpaca-7b-full"
    llm = LLM(model_path)
    src_lang, tgt_lang = Language.get(src_lang).display_name(), Language.get(tgt_lang).display_name()

    path = "zero_shot_prompt.json"
    prompter = FewShotPrompter(src_lang, tgt_lang, path)

    src_data = [line.strip() for line in open(src_data)]
    tgt_data = [line.strip() for line in open(tgt_data)]

    for src_text, tgt_text in tqdm(zip(src_data, tgt_data)):
        prompt = prompter.get_instance(src_text)["translate"]
        #print(prompt)
        hyp = llm.generate(prompt)
        #print(hyp)
        line = {"src_lang":src_lang, "tgt_lang":tgt_lang, "src_text":src_text, "tgt_text":tgt_text, "hyp_text":hyp, "prompt":prompt}
        writer.write(json.dumps(line,ensure_ascii=False)+"\n")

def main():
    path = "zero_shot_prompt.json"
    src_lang, tgt_lang = "en", "ja"
    src_lang, tgt_lang = Language.get(src_lang).display_name(), Language.get(tgt_lang).display_name()
    src_text = "hello"
    prompter = FewShotPrompter(src_lang, tgt_lang, path)
    prompt = prompter.get_instance(src_text)
    print(prompt["translate"])

if __name__=="__main__":
    #main()
    #create_examples_for_few_shot()
    #zero_shot_translate(src_lang, tgt_lang, writer)
    #lp = "en2de,en2ja,cs2uk,en2hr".split(",")
    lp = "en2ja,en2de,cs2uk,en2hr".split(",")
    lp = [x.split("2") for x in lp]
    writer = open("../output/wmt22_0shot_5shot_alpaca_4lp.jsonl","w")
    for src_lang, tgt_lang in lp:
        few_shot_translate(src_lang, tgt_lang, writer)
