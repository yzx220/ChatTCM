import os
from typing import Dict, Tuple, Union, Optional


from torch.nn import Module


from torch.utils.data import Dataset
import multiprocessing
import json
import numpy as np
import torch
import re
import random
import time
import datetime
import requests
from transformers import AutoTokenizer, AutoModel,AutoConfig
tokenizer = AutoTokenizer.from_pretrained("D:\work\ChatGLM2-6B\model", trust_remote_code=True)
model = AutoModel.from_pretrained("D:\work\ChatGLM2-6B\model",trust_remote_code=True).quantize(4).cuda()
def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    # 本文件来源于https://github.com/THUDM/ChatGLM-6B/blob/main/utils.py
    # 仅此处做少许修改以支持ChatGLM2
    device_map = {
        'transformer.embedding.word_embeddings': 0,
        'transformer.encoder.final_layernorm': 0,
        'transformer.output_layer': 0,
        'transformer.rotary_pos_emb': 0,
        'lm_head': 0
    }

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'transformer.encoder.layers.{i}'] = gpu_target
        used += 1

    return device_map


def load_model_on_gpus(checkpoint_path: Union[str, os.PathLike], num_gpus: int = 2,
                       device_map: Optional[Dict[str, int]] = None, **kwargs) -> Module:
    if num_gpus < 2 and device_map is None:
        model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half().cuda()
    else:
        from accelerate import dispatch_model

        model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half()

        if device_map is None:
            device_map = auto_configure_device_map(num_gpus)

        model = dispatch_model(model, device_map=device_map)

    return model

def shuffleDict(d):
    keys = list(d.keys())
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)

    keys = [(key, d[key]) for key in keys]
    # keys = d(keys)
    return dict(keys)

def fix_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def print_now(return_flag=0):
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    now = now.strftime('%Y/%m/%d %H:%M:%S')
    if return_flag == 0:
        print(now)
    elif return_flag == 1:
        return now
    else:
        pass

def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True
def decoder_for_chatglm2(args, input):
    time.sleep(args.api_time_interval)  # 根据args参数中的api_time_interval设置时间间隔
    answer = []
    # re = requests.post("http://127.0.0.1:8000/v1/chat/completions",
    #                    json={
    #                        "model": "string",
    #                        "messages": [
    #                            {
    #                                "role": "user",
    #                                "content": input
    #                            }
    #                        ],
    #                        "temperature": 0.95,
    #                        "top_p": 0.7,
    #                        "max_length": 2048,
    #                        "stream": "false"
    #                    }
    #                    )
    # response = json.loads(re.text)["choices"][0]["message"]["content"]

    response, history = model.chat(tokenizer, input, history=[])
    # answer.append(response)
    # with open("answer.txt", "a", encoding='utf-8') as f:  # 打开文本
    #     for it in answer:
    #         f.write(it + "\n" + "\n" + "\n")
    #返回输出结果，选择第一个候选项的文本

    return response

class Decoder():
    def __init__(self):
        # print_now() # 调用print_now函数，打印当前的日本标准时间
        pass

    def decode(self, args, input):

        response = decoder_for_chatglm2(args, input)

        return response

def data_reader(args):
    # 创建两个空列表，用于存储问题和答案
    questions = []
    answers = []
    # 创建一个JSON解码器对象，用于解析JSON格式的数据
    decoder = json.JSONDecoder()


    with open(args.dataset_path,'r',encoding='utf-8') as f:
        # 读取文件中的所有行，存储在lines列表中
        lines = f.readlines()
        # 遍历lines列表中的每一行
        for line in lines:
            json_res = decoder.raw_decode(line)[0]
            questions.append(json_res["query"].strip())
            answers.append(json_res["response"].split(",")[-1])


    return questions, answers

class MyDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.questions, self.answers = data_reader(args)
        self.len = len(self.questions)

    # 定义一个特殊方法，返回数据集的长度
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        input = self.questions[index]
        output = self.answers[index]
        return input, output


def setup_data_loader(args):

    # 修复数据加载器的随机性，以确保结果的可复现性
    fix_seed(args.random_seed)
    worker_seed = torch.initial_seed() % 2**32
    print("worker_seed : {}".format(worker_seed))
    def seed_worker(worker_id):
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(worker_seed)

    dataloader_num_workers = multiprocessing.cpu_count()
    dataloader_num_workers = min(dataloader_num_workers, args.max_num_worker)
    print("dataloader_num_workers: " + str(dataloader_num_workers))

    dataset = MyDataset(args)

    # 创建一个PyTorch的数据加载器对象，传入以下参数：
    # dataset: 数据集对象
    # shuffle: 是否打乱数据顺序
    # batch_size: 每个批次的数据大小
    # drop_last: 是否丢弃最后一个不完整的批次
    # num_workers: 工作进程数
    # worker_init_fn: 工作进程初始化函数
    # generator: 生成器对象
    # pin_memory: 是否将数据存储在锁页内存中
    dataloader = torch.utils.data.DataLoader(dataset,
                  shuffle=True,
                  batch_size=args.minibatch_size,
                  drop_last=False,
                  num_workers=dataloader_num_workers,
                  worker_init_fn=seed_worker,
                  generator=g,
                  pin_memory=True)

    return dataloader

def create_demo_text(args, cot):

    x, z, y = [], [], []
    with open(args.demo_path, encoding="utf-8") as f:
        # 把文件内容加载为json格式
        json_data = json.load(f)
        # 把json数据中的demo部分赋值给json_data
        json_data = json_data["demo"]
        # 遍历json数据中的每一行
        for line in json_data:

            x.append(line["question"])

            z.append(line["rationale"])


    index_list = list(range(len(x)))

    demo_text = ""

    for i in index_list:
        if cot:
            demo_text += x[i] + " " + z[i] + " "+ "\n\n"
            # demo_text += x[i] + " " + z[i] + " " + \
            #              args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        else:

            demo_text += x[i] + " " + args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"

    return demo_text

