import os
import platform
import signal
from transformers import AutoTokenizer, AutoModel,AutoConfig
import torch
#import readline

tokenizer = AutoTokenizer.from_pretrained("D:\work\ChatGLM2-6B\model", trust_remote_code=True)
model = AutoModel.from_pretrained("D:\work\ChatGLM2-6B\model",trust_remote_code=True).quantize(4).cuda()
# config = AutoConfig.from_pretrained(
#     "D:\work\ChatGLM2-6B\ptuning\output\\adgen-chatglm2-6b-pt-LR2\checkpoint-1500", trust_remote_code=True, pre_seq_len=128)


# prefix_state_dict = torch.load(os.path.join(
#     "D:\work\ChatGLM2-6B\ptuning\output\\adgen-chatglm2-6b-pt-LR2\checkpoint-1500", "pytorch_model.bin"))


# 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
# from utils import load_model_on_gpus
# model = load_model_on_gpus("THUDM/chatglm2-6b", num_gpus=2)

# new_prefix_state_dict = {}
# for k, v in prefix_state_dict.items():
#         if k.startswith("transformer.prefix_encoder."):
#             new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
# model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

# response, history = model.chat(tokenizer, "你好", history=[])
# print(response)
def build_prompt(history):
    prompt = "欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM2-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main():
    past_key_values, history = None, []
    global stop_stream
    print("欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            past_key_values, history = None, []
            os.system(clear_command)
            print("欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        print("\nChatGLM：", end="")
        current_length = 0
        for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True):
            if stop_stream:
                stop_stream = False
                break
            else:
                print(response[current_length:], end="", flush=True)
                current_length = len(response)
        print("")


if __name__ == "__main__":
     main()
