from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# 加载ChatGLM2-6B模型
tokenizer = AutoTokenizer.from_pretrained("D:\work\ChatGLM2-6B\model", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("D:\work\ChatGLM2-6B\model", trust_remote_code=True).quantize(4).cuda()

# content = """能推荐主治健脾和胃，益气养阴。主治脾胃气虚，元气不足。（乳腺癌化疗反应）的方剂给我么
# 2号养胃汤效果可以，抓取姜夏15克，枳实15克，陈皮15克，茯苓20克，竹茹20克，生姜20克，甘草10克，红人参15克该方水煎服，每日1剂，日服3次。
# """
# prompt = '从上文中，提取"信息"(keyword,content)，包括:"疾病"、"症状"、"药物"实体，输出json格式内容'
# input = '{}\n\n{}'.format(content, prompt)
# response, history = model.chat(tokenizer, input, history=[])
# print(response)

datasets_path = "D:\work\ChatGLM2-6B\ptuning\MED\\dev.json"

def extract_content(text,key_start,key_end):
    start = text.find(key_start)
    end = text.find(key_end, start)
    if start != -1 and end != -1:
        return text[start + 3:end].strip().replace("\n","")
    return ""

with open(datasets_path,'r',encoding="utf-8",errors='ignore') as f:

    for line in f.readlines():
        content = json.loads(line)

        prompt = '从上文中，提取只包括:"疾病"、"症状"、"药物"三个内容'
        input = '{}\n\n{}'.format(content, prompt)
        response, history = model.chat(tokenizer, input, history=[])

        out_json = {}
        with open('output.json','a', encoding="utf8",errors='ignore') as wp:
            if "：" in response:
                jibing = extract_content(response,"疾病：","症状：")
                zhuangz = extract_content(response,"症状：","药物：")
                yaowu = response[response.find("药物：")+3:].replace("\n","")
            elif ":"in response:
                jibing = extract_content(response,"疾病:","症状:")
                zhuangz = extract_content(response,"症状:","药物:")
                yaowu = response[response.find("药物:")+3:].replace("\n","")

            out_json["name"] = jibing

            out_json["symptom"] = zhuangz

            if '，' in out_json["symptom"]:
                out_json["symptom"] = zhuangz.replace('，', ';').split(';')
            elif '、' in out_json["symptom"]:
                out_json["symptom"] = zhuangz.replace('、', ';').split(';')
            elif '.' in out_json["symptom"]:
                out_json["symptom"] = zhuangz.replace('.', ';').split(';')
            else:
                out_json["symptom"] = [zhuangz]

            out_json["recommand_drug"] = [yaowu]


            print("jibing：",jibing)
            print("zhuangz：",zhuangz)
            print("yaowu：",yaowu)

            json_str = json.dumps(out_json, ensure_ascii=False)
            wp.write(json_str)
            wp.write("\n")
        print(response)

# with open(datasets_path,'r',encoding="utf-8",errors='ignore') as f:
#     for line in f.readlines():
#         content = json.loads(line)
#         prompt = '从上文中，提取只包括:"疾病"、"症状"、"药物"三个内容,并且把提取到的疾病名字填写到输出的格式为json格式{"name":"","symptom":[""],"reconmmand_drug":[""]},' \
#                  '请不要输出其他不相关的格式'
#         input = '{}\n\n{}'.format(content, prompt)
#
#         response, history = model.chat(tokenizer, input, history=[])
#
#         with open('output1.json', 'a', encoding="utf8", errors='ignore') as wp:
#             # json_str = json.dumps(response,ensure_ascii=False)
#
#             wp.write(response)
#             wp.write("\n")
#         print(response)







