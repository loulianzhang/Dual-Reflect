from openai import OpenAI
import logging
import requests, json
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    filename='C:\\Users\\lzlou\\Desktop\\self_reflection\\log\\agent_with_QE_as_judge.eval.log',
                    level=logging.INFO,filemode="a+")

client = OpenAI(
    # This is the default and can be omitted
    api_key="<your api key>",
)

class Translator(object):
    def __init__(self, model, role="forward translator", temperature=0.0, max_tokens=2048):
        self.model = model
        self.role = role
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.message = []

    def set_meta_prompt(self, sys_prompt):
        self.message.append(sys_prompt)

    def add_user_content(self, event: str):
        """Add an new event in the memory

        Args:
            event (str): string that describe the user prompt.
        """
        self.message.append({"role": "user", "content": f"{event}"})

    def add_assist_content(self, memory: str):
        """Monologue in the memory

        Args:
            memory (str): string that generated by the model in the last round.
        """
        self.message.append({"role": "assistant", "content": f"{memory}"})

    def run(self):
        chat_completion = client.chat.completions.create(
            messages=self.message,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self.message.pop() # forward translation不记忆之前的用户输入
        return chat_completion.choices[0].message.content.replace("\n\n","\n")

class Reflector(object):
    def __init__(self, model, role="self reflector", temperature=0.0, max_tokens=2048):
        self.model = model
        self.role = role
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.message = []

    def set_meta_prompt(self, sys_prompt):
        self.message.append(sys_prompt)

    def add_user_content(self, event: str):
        """Add a new event in the memory

        Args:
            event (str): string that describe the event.
        """
        self.message.append({"role": "user", "content": f"{event}"})

    def add_assist_content(self, memory: str):
        """Monologue in the memory

        Args:
            memory (str): string that generated by the model in the last round.
        """
        self.message.append({"role": "assistant", "content": f"{memory}"})

    def QE_judge(self, source, transaltion, backtranslation):
        data = {"source": [transaltion], "target": [source], "reference": [backtranslation]}
        json_input = json.dumps(data)
        x = requests.post("https://bright.pcl.ac.cn/trans_eval", data=json_input)
        x = eval(x.text)
        return x["result"]["document_level"]

    def run(self):
        chat_completion = client.chat.completions.create(
            messages=self.message,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return chat_completion.choices[0].message.content.replace("\n\n","\n")

class Agent():
    def __init__(self, model, src_lang, tgt_lang, src_text, max_round=2):
        self.model = model
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_round = max_round # 轮次定义：前向翻译推理次数
        self.threshold = 0.9
        self.ft = Translator(model=model, temperature=0.0, max_tokens=2048)
        self.bt = Translator(model=model, temperature=0.0, max_tokens=2048)
        self.reflector = Reflector(model=model, temperature=0.0, max_tokens=2048)

        self.ft_meta_prompt = {"role": "system",
                               "content": f"Suppose you are a linguistic expert, proficient in translation from {src_lang} to {tgt_lang}. "\
                                          f"Translate following {src_lang} text provided by user into {tgt_lang} according to instructions given"}
        self.ft.set_meta_prompt(self.ft_meta_prompt)

        self.bt_sys_prompt = {"role": "system",
                              "content": f"Suppose you are a linguistic expert, proficient in translation from {tgt_lang} to {src_lang}. " \
                                         f"Translate following {tgt_lang} text provided by user into {src_lang}"}
        self.bt.set_meta_prompt(self.bt_sys_prompt)

        self.reflect_meta_prompt = {"role":"system",
                         "content":f"If you are a {src_lang} linguist, find all differences between the following two sentences provided by user"}
        self.reflector.set_meta_prompt(self.reflect_meta_prompt)

        self.src_text = src_text

    def update_input(self, feedback):
        return self.src_text + "\n" + "instructions:\n" + feedback

    def run(self):
        # start translation with self-reflection
        logging.info("*********************************start*********************************")
        feedback = ""
        i = 0
        while i < self.max_round+1:
            logging.info(f"====================round {i+1} ========================")
            #print(f"====================round {i+1} ========================")
            logging.info(f"==========forward translation===========")
            #print(f"==========forward translation===========")
            input = self.update_input(feedback)
            logging.info(f"input:\n{input}")
            #print(f"input:\n{input}")
            self.ft.add_user_content(input)
            transaltion = self.ft.run()
            logging.info(f"output:\n{transaltion}")
            #print(f"output:\n{transaltion}")

            # 下一次即为最大的轮次，
            if i == self.max_round:
                break

            logging.info(f"\n==========backward translation===========")
            #print(f"\n==========backward translation===========")
            logging.info(f"input:\n{transaltion}")
            #print(f"input:\n{transaltion}")
            self.bt.add_user_content(transaltion)
            backtranslation = self.bt.run()
            logging.info(f"output:\n{backtranslation}")
            #print(f"output:\n{backtranslation}")

            logging.info(f"\n==========self reflection============")
            #print(f"\n==========self reflection============")
            judge_value = self.reflector.QE_judge(self.src_text, transaltion, backtranslation)
            logging.info(f"QE score: {judge_value}")
            #print(f"QE score: {judge_value}")
            if judge_value > self.threshold:
                break
            input = f"{self.src_text}\n{backtranslation}"
            logging.info(f"input:\n{input}")
            #print(f"input:\n{input}")
            self.reflector.add_user_content(input)
            judgement = self.reflector.run()
            self.reflector.add_assist_content(judgement)
            logging.info(f"output\n: {judgement}")
            #print(f"output\n: {judgement}")
            input = f"Based on previous analysis, translate all words or phrases "\
                     f"in the first sentence that cause above differences into {self.tgt_lang}."
            logging.info(f"input:\n{input}")
            #print(f"input:\n{input}")
            self.reflector.add_user_content(input)
            feedback = self.reflector.run()
            logging.info(f"output:\n{feedback}")
            #print(f"output:\n{feedback}")
            self.reflector.add_assist_content(feedback)

            i += 1

        logging.info(f"====================Final translation========================")
        #print((f"====================Final translation========================"))
        logging.info(f"num rounds: {i}\ntranslation:\n{transaltion}")
        #print(f"num rounds: {i}\ntranslation:\n{transaltion}")
        logging.info("*********************************end*********************************\n")
        #print("*********************************end*********************************\n")
        return transaltion
