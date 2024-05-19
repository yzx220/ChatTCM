import os
from abc import ABC
from enum import Enum
from typing import Optional, List

import torch
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from transformers import (AutoModel, AutoConfig, AutoTokenizer)

from config import cfg

# 视情况调整，设置计算的卡编号
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 创建一个枚举类型, 不同ptuning分类
class PtuningType(Enum):
    Nothing = 0
    Classify = 1
    Keywords = 2
    NL2SQL = 3


#  ChatGLM_Ptuning类：通过传入不同PtuningType初始化可以达成单模型多训练权重的使用方式
class ChatGLM_Ptuning(LLM, ABC):
    if cfg.ONLINE:
        model_name = 'D:\work\chaGLM2-6B\model'
    else:
        model_name = 'D:\work\chaGLM2-6B\model'
        # model_name = '/home/jamepeng/git_projects/chatglm2-6b-model'

    tokenizer: AutoTokenizer = None
    model: AutoModel = None
    config: AutoConfig = None
    isClassify = False
    isKeywords = False
    isNL2SQL = False

    # 通过传入微调权重类型来加载不同的权重进行工作
    def __init__(self, ptuning_type: PtuningType):
        super().__init__()
        check_point_path = ""
        # 载入Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        if ptuning_type is PtuningType.Classify or ptuning_type is PtuningType.NL2SQL or ptuning_type is PtuningType.Keywords:
            # 如果是分类类型，加载长度为512的CLASSIFY训练权重
            if ptuning_type is PtuningType.Classify:
                self.config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True,
                                                         pre_seq_len=cfg.CLASSIFY_PTUNING_PRE_SEQ_LEN)
                check_point_path = os.path.join(cfg.CLASSIFY_CHECKPOINT_PATH, "pytorch_model.bin")
                self.model = AutoModel.from_pretrained(self.model_name, config=self.config, trust_remote_code=True)
                self.isClassify = True
            # 如果是提取关键词类型，加载长度为256的KEYWORDS训练权重
            elif ptuning_type is PtuningType.Keywords:
                self.config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True,
                                                         pre_seq_len=cfg.KEYWORDS_PTUNING_PRE_SEQ_LEN)
                check_point_path = os.path.join(cfg.KEYWORDS_CHECKPOINT_PATH, "pytorch_model.bin")
                self.model = AutoModel.from_pretrained(self.model_name, config=self.config, trust_remote_code=True)
                self.isKeywords = True
            # 如果是分类类型，加载长度为2048的NL2SQL训练权重
            elif ptuning_type is PtuningType.NL2SQL:
                self.config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True,
                                                         pre_seq_len=cfg.NL2SQL_PTUNING_PRE_SEQ_LEN)
                check_point_path = os.path.join(cfg.NL2SQL_CHECKPOINT_PATH, "pytorch_model.bin")
                self.model = AutoModel.from_pretrained(self.model_name, config=self.config, trust_remote_code=True)
                self.isNL2SQL = True
            # 装载对应路径的权重
            prefix_state_dict = torch.load(check_point_path)
            new_prefix_state_dict = {}
            for k, v in prefix_state_dict.items():
                if k.startswith("transformer.prefix_encoder."):
                    new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
            self.model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

            if cfg.CLASSIFY_PTUNING_PRE_SEQ_LEN is not None and cfg.NL2SQL_PTUNING_PRE_SEQ_LEN is not None and cfg.KEYWORDS_PTUNING_PRE_SEQ_LEN is not None:
                # P-tuning v2
                self.model.transformer.prefix_encoder.float()
        else:
            # 未识别到微调
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
            self.isClassify = self.isNL2SQL = False

        self.model.cuda().eval()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        # print(f"__call:{prompt}")
        if len(prompt) > 5120:
            prompt = prompt[:5120]
        try:
            response, _ = self.model.chat(
                self.tokenizer,
                prompt,
                history=[],
                max_length=8192,
                top_p=1, do_sample=False,
                temperature=0.001)
        except Exception as e:
            print(e)
            response = prompt
        # response = self.model.response(prompt, history=[],
        # max_length=6144, do_sample=False, top_p=1, temperature=0.001)
        # print(f"response:{response}")
        # print(f"+++++++++++++++++++++++++++++++++++")
        return response

    def _get_classify_prompt(self, question) -> str:
        classify_prompt = '''
        请问“{}”是属于下面哪个类别的问题?
        
        你只需要回答字母编号, 不要回答字母编号及选项文本外的其他内容.
        '''.format(question)
        return classify_prompt

    # 加载Classify训练权重后，来强化问题的分类能力，返回问题的类型字母编号
    def classify(self, question: str):
        if self.isClassify:
            classify_prompt = self._get_classify_prompt(question)
            response, _ = self.model.chat(
                self.tokenizer,
                classify_prompt,
                history=[],
                max_length=cfg.CLASSIFY_PTUNING_PRE_SEQ_LEN,
                top_p=1, do_sample=False,
                temperature=0.001)
            return response
        else:
            print("Error: 未装载Classify训练权重，无法继续任务")
    def _get_keywords_prompt(self, question) -> str:
        question_prompt = '''
                请帮我从以下句子中提取关键词。这些关键词是句子中最重要、最能概括句子主题的词汇。通过这些关键词，你可以更好地理解句子的内容。你只需要回答文本中的关键词,不要回答其他内容.
                用户输入：
                '''
        keywords_prompt = f"{question_prompt} {question}"
        return keywords_prompt

    # 加载Keywords训练权重后，来强化问题的提取关键词能力，返回问题的关键词
    # 查询题和计算题返回计算核心词，统计题返回符合数据库检索的字段，开放题正常返回关键词
    def keywords(self, question: str):
        if self.isKeywords:
            keywords_prompt = self._get_keywords_prompt(question)
            response, _ = self.model.chat(
                self.tokenizer,
                keywords_prompt,
                history=[],
                max_length=cfg.KEYWORDS_PTUNING_PRE_SEQ_LEN,
                top_p=1, do_sample=False,
                temperature=0.001)
            return response
        else:
            print("Error: 未装载Keywords训练权重，无法继续任务")

    @property
    def _get_nl2sql_prompt(self) -> str:
        nl2sql_prompt = '''你是一名Neo4j数据库开发人员，你精通Neo4j数据库的cql代码编写，你需要根据已知的表名、字段名和用户输入的问题编写cql代码
已知字段名：[病症,病因,处方]
要求cql代码中的字段名必须是已知字段名，不得新增字段名

示例模板：
"""
用户输入：中医治疗'脑萎缩'好吗？

cql如下：
```cql 
MATCH (n)-[:RELATED_TO]->(:Condition {name: '脑萎缩'})RETURN n"
```

用户输入：中医怎么治疗'脑萎缩'？

cql如下：
```cql 
MATCH (n)-[:RELATED_TO]->(:Condition {name: '脑萎缩'})RETURN n
```

用户输入：中医治疗'脑萎缩'的药物是什么？

cql如下：
```cql 
MATCH (n)-[:RELATED_TO]->(:Condition {name: '脑萎缩'})RETURN n
```

"""
请根据以下用户输入，输出cql代码。
用户输入：'''
        return nl2sql_prompt

    # 加载NL2SQL训练权重后，来强化问题自然语言对SQL语句的转换
    def nl2sql(self, question: str):
        if self.isNL2SQL:
            question_prompt = f"{self._get_nl2sql_prompt}\"{question}\""
            response, _ = self.model.chat(
                self.tokenizer,
                question_prompt,
                history=[],
                max_length=cfg.NL2SQL_PTUNING_MAX_LENGTH,
                top_p=1, do_sample=False,
                temperature=0.001)
            new_response = response[response.find('```sql')+7:].replace('\n```','')
            return new_response
        else:
            print("Error: 未装载NL2SQL训练权重，无法继续任务")



    # 卸载掉已经装在权重的模型
    def unload_model(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


if __name__ == '__main__':

    model = ChatGLM_Ptuning(PtuningType.Nothing)
    print(model("你好啊！"))

    model.unload_model()