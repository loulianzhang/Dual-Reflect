import time
from agent_with_LLM_as_judge import Agent as Agent1
from dual_reflection_agent_with_baseline_as_translator_and_LLM_as_judge import Agent as Agent2
from tqdm import tqdm
import json
from langcodes import Language

def debug():
    #src_text = "男主居住地应该是偏南方的某个小岛上，和他的奶奶相依为命。"
    src_text = "吃掉敌人一个师。"
    #src_text = "吃了游客的鳄鱼"
    #src_text = "如果逾期，逾期记录将记录到个人信用报告中，可能会对日后买车、购房等经济生活造成不良影响。"
    model = "gpt-3.5-turbo"
    src_lang, tgt_lang = "Chinese", "English"
    agent = Agent2(model, src_lang, tgt_lang, src_text)
    res = agent.run(src_text)

def read_wmt2022(src_lang, tgt_lang):
    src_path = f"C:\\Users\\lzlou\\Desktop\\self_reflection\\wmt2022\\wmt22.{src_lang}-{tgt_lang}.{src_lang}"
    tgt_path = f"C:\\Users\\lzlou\\Desktop\\self_reflection\\wmt2022\\wmt22.{src_lang}-{tgt_lang}.{tgt_lang}"
    src_data = [line.strip() for line in open(src_path, "r", encoding="utf-8")]
    tgt_data = [line.strip() for line in open(tgt_path, "r", encoding="utf-8")]
    src_lang = Language.make(language=src_lang).display_name()
    tgt_lang = Language.make(language=tgt_lang).display_name()

    return [{"src_lang":src_lang, "tgt_lang": tgt_lang, "src":src, "tgt":tgt} for src, tgt in zip(src_data, tgt_data)]

def evaluate(save_path):
    #lang_pairs = ["en-zh", "en-ja", "ja-en", "en-hr"]
    lang_pairs = ["en-de", "de-en", "cs-uk", "uk-cs", "de-fr", "fr-de"]
    lang_pairs = [x.split("-") for x in lang_pairs]
    data = []
    for src_lang ,tgt_lang in lang_pairs:
        data.extend(read_wmt2022(src_lang, tgt_lang))
    print(f"test data size: {len(data)}")

    model = "gpt-3.5-turb" \
            "m;zz;z:o"

    # import random
    # random.shuffle(data)

    with open(save_path, "a+", encoding="utf-8") as fw:
        # for sample in tqdm(data[202:]):
        for sample in tqdm(data[3412:]):
            src = sample["src"].strip()
            tgt = sample["tgt"].strip()
            src_lang = sample["src_lang"].strip()
            tgt_lang = sample["tgt_lang"].strip()
            try:
                agent = Agent1(model, src_lang, tgt_lang, src)
                hyp = agent.run()
            except:
                hyp = "ERROR!"
            line = {"src": src.strip(), "tgt": tgt.strip(), "hyp": hyp, "src_lang":src_lang, "tgt_lang": tgt_lang}
            line = json.dumps(line, ensure_ascii=False) + "\n"
            fw.write(line)

if __name__=="__main__":
    evaluate("C:\\Users\\lzlou\\Desktop\\self_reflection\\wmt2022\\agent_with_LLM_as_judge_4langs.result.jsonl")
    #debug()