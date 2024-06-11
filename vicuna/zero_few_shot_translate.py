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

def create_examples_for_few_shot():
    data = {}
    for name in "eng,deu,jpn,ces,ukr,hrv".split(","):
        lines = [line.strip() for line in open(f"/mgData4/loulianzhang/data/flores101_dataset/devtest/{name}.devtest")]
        lang = Language.get(name).display_name()
        data[lang] = lines
    df = pd.DataFrame(data)

    lp = "en2de,en2ja,cs2uk,en2hr".split(",")
    lp = [x.split("2") for x in lp]
    lp = [(Language.get(sl).display_name(), Language.get(tl).display_name()) for sl, tl in lp]
    examples = []
    for sl, tl in lp:
        for _, row in df[[sl,tl]].sample(n=5).iterrows():
            examples.append({"src_lang":sl, "tgt_lang":tl, "src_text":row[sl], "tgt_text":row[tl]})
    examples = pd.DataFrame(examples)
    examples.to_json("examples.jsonl",orient="records",lines=True,force_ascii=False)


class Prompter():
    def __init__(self, src_lang, tgt_lang, path):
        with open(path, encoding="utf-8") as fr:
            prompt_pipeline = json.load(fr)
        self.prompt_pipeline = prompt_pipeline

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def get_instance(self, src_text):
        prompt_pipeline = {}
        for action, prompt in self.prompt_pipeline.items():
            prompt_pipeline[action] = prompt.replace("{src_lang}", self.src_lang)\
                .replace("{tgt_lang}", self.tgt_lang).replace("{src_text}",src_text)
        return prompt_pipeline

class FewShotPrompter(Prompter):
    def __init__(self, src_lang, tgt_lang, path):
        super(FewShotPrompter, self).__init__(src_lang, tgt_lang, path)
        examples = pd.read_json("examples.jsonl",lines=True)
        examples = examples[examples.src_lang==src_lang]
        examples = examples[examples.tgt_lang==tgt_lang]
        examples = [row["src_text"]+"\t"+row["tgt_text"] for _, row in examples.iterrows()]
        self.examples = "\n".join(examples)

    def get_instance(self, src_text):
        prompt_pipeline = {}
        for action, prompt in self.prompt_pipeline.items():
            prompt_pipeline[action] = prompt.replace("{src_lang}", self.src_lang)\
                .replace("{tgt_lang}", self.tgt_lang).replace("{src_text}",src_text).replace("{examples}",self.examples)
        return prompt_pipeline

def zero_shot_translate(src_lang, tgt_lang, writer):
    src_data = f"../dataset/wmt22.{src_lang}-{tgt_lang}.{src_lang}"
    tgt_data = f"../dataset/wmt22.{src_lang}-{tgt_lang}.{tgt_lang}"
    model_path = "/mgData4/loulianzhang/llm/vicuna/vicuna-7b-v1.5"
    llm = LLM(model_path)
    src_lang, tgt_lang = Language.get(src_lang).display_name(), Language.get(tgt_lang).display_name()

    path = "zero_shot_prompt.json"
    prompter = Prompter(src_lang, tgt_lang, path)

    src_data = [line.strip() for line in open(src_data)]
    tgt_data = [line.strip() for line in open(tgt_data)]

    for src_text, tgt_text in tqdm(zip(src_data, tgt_data)):
        prompt = prompter.get_instance(src_text)["translate"]
        #print(prompt)
        hyp = llm.generate(prompt)
        #print(hyp)
        line = {"src_lang":src_lang, "tgt_lang":tgt_lang, "src_text":src_text, "tgt_text":tgt_text, "hyp_text":hyp, "prompt":prompt}
        writer.write(json.dumps(line,ensure_ascii=False)+"\n")

def few_shot_translate(src_lang, tgt_lang, writer):
    src_data = f"../dataset/wmt22.{src_lang}-{tgt_lang}.{src_lang}"
    tgt_data = f"../dataset/wmt22.{src_lang}-{tgt_lang}.{tgt_lang}"
    model_path = "/mgData4/loulianzhang/llm/vicuna/vicuna-7b-v1.5"
    llm = LLM(model_path)
    src_lang, tgt_lang = Language.get(src_lang).display_name(), Language.get(tgt_lang).display_name()

    path = "few_shot_prompt.json"
    prompter = FewShotPrompter(src_lang, tgt_lang, path)

    src_data = [line.strip() for line in open(src_data)]
    tgt_data = [line.strip() for line in open(tgt_data)]

    for src_text, tgt_text in tqdm(zip(src_data, tgt_data)):
        prompt = prompter.get_instance(src_text)["translate"]
        #print(prompt)
        hyp = llm.generate(prompt)
        llm.clear_history()
        #print(hyp)
        line = {"src_lang":src_lang, "tgt_lang":tgt_lang, "src_text":src_text, "tgt_text":tgt_text, "hyp_text":hyp, "prompt":prompt}
        writer.write(json.dumps(line,ensure_ascii=False)+"\n")

def main():
    path = "few_shot_prompt.json"
    src_lang, tgt_lang = "en", "ja"
    src_lang, tgt_lang = Language.get(src_lang).display_name(), Language.get(tgt_lang).display_name()
    src_text = "hello"
    prompter = FewShotPrompter(src_lang, tgt_lang, path)
    prompt = prompter.get_instance(src_text)
    print(prompt)

if __name__=="__main__":
    #main()
    #create_examples_for_few_shot()
    #zero_shot_translate(src_lang, tgt_lang, writer)
    #lp = "en2de,en2ja,cs2uk,en2hr".split(",")
    lp = "en2de,cs2uk,en2hr".split(",")
    lp = [x.split("2") for x in lp]
    writer = open("../output/wmt22_0shot_5shot_vicuna_4lp.jsonl","a+")
    for src_lang, tgt_lang in lp:
        few_shot_translate(src_lang, tgt_lang, writer)
