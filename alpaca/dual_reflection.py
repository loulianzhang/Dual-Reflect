#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用llama-7b-chat跑出dual-reflection在wmt22年
en2de, en2ja, en2hr, cs2uk 这4个方向的结果
baseline: zero-shot, 5-shot
ours: llama(agent) + Dual-Reflection
"""

from model import LLM
import json
from langcodes import Language
import re
from tqdm import tqdm

class Prompter():
    def __init__(self, src_lang, tgt_lang, path):
        """
        :param path: path 为每一步的prompt的存储文件，json格式，形如
        "D:\工作PCL\cross lingual understanding and translation of LLM\cross lingual understanding.json"
        """
        with open(path, encoding="utf-8") as fr:
            prompt_pipeline = json.load(fr)
        self.prompt_pipeline = prompt_pipeline

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def get_instance(self, src_text, tgt_text, backtranslation, instructions):
        prompt_pipeline = {}
        for action, prompt in self.prompt_pipeline.items():
            prompt_pipeline[action] = prompt.replace("{src_lang}", self.src_lang)\
                .replace("{tgt_lang}", self.tgt_lang).replace("{src_text}",src_text).replace("{tgt_text}",tgt_text)\
                .replace("{backtranslation}",backtranslation).replace("{instructions}",instructions)

        return prompt_pipeline 

class DualReflection(object):
    def __init__(self, src_lang, tgt_lang):
        model_path = "/mgData4/loulianzhang/llm/alpaca/alpaca-7b-full"
        llm = LLM(model_path)
        self.max_round=2
        #path = "ours_prompt.json"
        path = "../vicuna/ours_prompt_vicuna_spec.jsonl"
        self.prompter = Prompter(src_lang, tgt_lang, path)

    def get_judge(self, response):
        return 1 if response.strip(" ").startswith("OK") else 0

    def filter(self, translation):
        try:
            translation = self.regex.findall(translation)[0].replace("\"","")
        except:
            pass
        return translation.replace(" Chinese translation: ","")

    def run(self, src_text):
        tgt_text = ""
        backtranslation = ""
        instructions = ""
        i = 0
        while i < self.max_round+1:
            print(f"round {i}".center(50,"="))
            prompt = self.prompter.get_instance(src_text, tgt_text, backtranslation, instructions)
            action = "forward translate"
            print(action.center(30,"*"))
            print(prompt[action])
            translation = self.llm.generate(prompt[action])
            translation = self.filter(translation)
            print(translation)

            # 下一次即为最大的轮次，
            if i == self.max_round:
                break

            prompt = self.prompter.get_instance(src_text, translation, backtranslation, instructions)
            action = "backward translate"
            print(action.center(30,"*"))
            print(prompt[action])
            backtranslation = self.llm.generate(prompt[action])
            backtranslation = self.filter(backtranslation)
            print(backtranslation)

            prompt = self.prompter.get_instance(src_text, translation, backtranslation, instructions)
            action = "dual comparison"
            print(action.center(30,"*"))
            print(prompt[action])
            judgement = self.llm.generate(prompt[action])
            print(judgement)
            judge = self.get_judge(judgement)
            if judge:
                break
            action = "dual reflection"
            print(action.center(30,"*"))
            print(prompt[action])
            instructions = self.llm.generate(prompt[action])
            print(instructions)

            i += 1

        return i, translation

def debug_prompt():
    path = "ours_prompt.json"
    src_lang, tgt_lang = "zh", "en"
    src_lang, tgt_lang = Language.get(src_lang).display_name(), Language.get(tgt_lang).display_name()
    src_text, transaltion, backtranslation, instructions = "你好", "hello", "你好啊", "no error"
    prompter = Prompter(src_lang, tgt_lang, path)
    prompt = prompter.get_instance(src_text, transaltion, backtranslation, instructions)
    print(prompt)

def debug_df():
    model_path = "/mgData4/loulianzhang/llm/alpaca/alpaca-7b-full"
    llm = LLM(model_path)
    src_lang, tgt_lang = "zh", "en"
    src_lang, tgt_lang = Language.get(src_lang).display_name(), Language.get(tgt_lang).display_name()
    agent = DualReflection(src_lang, tgt_lang)

    src_text = "这条街有很多可以当首饰的店铺。"
    translation = agent.run(src_text)
    print("final translation:".center(50,">"))
    print(translation)

def dual_reflect_translate(src_lang, tgt_lang, writer):
    src_data = f"../dataset/wmt22.{src_lang}-{tgt_lang}.{src_lang}"
    tgt_data = f"../dataset/wmt22.{src_lang}-{tgt_lang}.{tgt_lang}"
    model_path = "/mgData4/loulianzhang/llm/alpaca/alpaca-7b-full"
    llm = LLM(model_path)
    src_lang, tgt_lang = Language.get(src_lang).display_name(), Language.get(tgt_lang).display_name()
    agent = DualReflection(src_lang, tgt_lang)

    src_data = [line.strip() for line in open(src_data)]
    tgt_data = [line.strip() for line in open(tgt_data)]

    for src_text, tgt_text in tqdm(zip(src_data, tgt_data)):
        num_round, hyp = agent.run(src_text)
        line = {"src_lang":src_lang, "tgt_lang":tgt_lang, "src_text":src_text, "tgt_text":tgt_text, "hyp_text":hyp, "num_round":num_round}
        writer.write(json.dumps(line,ensure_ascii=False)+"\n")

if __name__=="__main__":
    #main()
    #debug_prompt()
    #debug_df()
    #lp = "en2ja,en2de,cs2uk,en2hr".split(",")
    lp = "en2de,cs2uk,en2hr".split(",")
    lp = [x.split("2") for x in lp]
    writer = open("../output/wmt22_dual_reflect_vicuna_4lp.jsonl","a+")
    for src_lang, tgt_lang in lp:
        dual_reflect_translate(src_lang, tgt_lang, writer)
    writer.close()
