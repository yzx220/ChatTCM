import argparse
import os
from utils import *
import json

def parse_arguments():

    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")

    parser.add_argument(  "--dataset", type=str, default="test", help="dataset used for experiment" )

    parser.add_argument( "--demo_path", type=str, default="demos/multiarith.json", help="pre-generated demos used for experiment")

    parser.add_argument( "--resume_id", type=int, default=0, help="resume from which question id (current line number in the output file), if the experiment fails accidently (e.g., network error)" )

    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1], help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")

    parser.add_argument("--max_num_worker", type=int, default=0, help="maximum number of workers for dataloader")

    parser.add_argument( "--model", type=str, default="chatglm2-6b", choices=["chatglm2-6b"], help="model used for decoding. Note that 'gpt3' are the smallest models."  )

    parser.add_argument(  "--method", type=str, default="cot",help="method" )

    parser.add_argument( "--output_dir", type=str, default="../multiarith.json", help="output directory" )

    parser.add_argument( "--max_length_cot", type=int, default=512, help="maximum length of output tokens by model for reasoning extraction")

    parser.add_argument( "--max_length_direct", type=int, default=512, help="maximum length of output tokens by model for answer extraction")

    parser.add_argument( "--limit_dataset_size", type=int, default=1000000, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing.")

    parser.add_argument( "--api_time_interval", type=float, default=0.0, help="sleep between runs to avoid excedding the rate limit of openai api" )

    parser.add_argument(  "--temperature", type=float, default=0.97, help="temperature for chatglm")

    parser.add_argument("--log_dir", type=str, default="./log/", help="log directory")

    parser.add_argument(
        "--task", type=str, default="multiarith_manual",help="dataset used for experiment")

    parser.add_argument(
        "--max_ra_len", type=int, default=5, help="maximum number of reasoning chains"
    )
    parser.add_argument(
        "--pred_file", type=str, default="demos/multiarith.json",
        help="use the reasoning chains generated by zero-shot-cot."
    )

    parser.add_argument(
        "--demo_save_dir", type=str, default="D:/work/auto-cot-main/multiarith.json", help="where to save the contructed demonstrations"
    )

    parser.add_argument(
        "--encoder", type=str, default="all-MiniLM-L6-v2", help="which sentence-transformer encoder for clustering"
    )
    parser.add_argument(
        "--sampling", type=str, default="center", help="whether to sample the cluster center first"
    )
    parser.add_argument(
        "--debug", type=bool, default=True, help="debug mode"
    )
    # 调用 parser 的 parse_args 方法，解析命令行参数，返回一个命名空间对象，赋值给 args
    args = parser.parse_args()
    args.cot_trigger = "请一步步进行推理并得出结论."

    if args.dataset == "test":
        args.dataset_path = "D:\work\ChatGLM2-6B\dataset\\test\ChatMed_TCM-v0.2.json"
    else:
        raise ValueError("dataset is not properly defined ...")

    return args


def main():

    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')

    fix_seed(args.random_seed)

    decoder = Decoder()

    print("setup data loader ...")
    dataloader = setup_data_loader(args)

    print_now()

    if args.method == "cot":
        demo = create_demo_text(args, cot=True)
    # 否则，如果 args.method 不是以上的任何一种，表示不需要示例文本
    else:
        # 将 demo 赋值为 None
        pass


    total = 0
    correct_list = []

    with open(args.output_dir, "a") as wp:

        for i, data in enumerate(dataloader):

            if i < args.resume_id - 1:
                # if i < 297:
                continue
            output_line = {}

            print('*************************')
            print("{}st data".format(i + 1))

            # 从 data 中取出第一个元素，也是一个列表，赋值给 x 和 y
            x, y = data

            print('*****************************')
            print("Question:", x[0])
            print('*****************************')

            # 将 x 列表中的第一个元素（问题）和 "Q: " 和 "\n" 和 "A:" 拼接起来，表示问题和答案的格式，赋值给 x
            x = "Q: " + x[0] + "\n" + "A :"
            # 将 y 列表中的第一个元素（答案）去掉两边的空格，赋值给 y
            y = y[0].strip()

            output_line["question"] = x

            output_line["answer"] = y

            x = demo + x + " " + args.cot_trigger

            z = decoder.decode(args, x)

            output_line["rationale"] = z

            print("Prompted Input:")
            print(x.replace("\n\n", "\n").strip())
            print('*****************************')

            z2 = x + z + " " + args.cot_trigger #问题加chatglm生成的答案
            #print("rationale: ", z)
            pred = decoder.decode(args, z2)
            output_line["pred_ans"] = pred

            # 将 x 赋值给 output_line 字典中的键为 "wrap_que" 的值，表示输出的问题模板
            output_line["wrap_que"] = x

            # 将 output_line 字典转换为 JSON 格式的字符串，赋值给 output_json
            output_json = json.dumps(output_line)
            # 将 output_json 写入到 wp 文件中，并换行
            wp.write(output_json + '\n')

            # 从列表中选择最频繁的答案
            # 打印 "pred : " 和 pred，表示显示预测的答案
            # print("pred : {}".format(pred))
            prompt = "你是一名中医领域的专家，请根据中医知识进行回答，并判断该问题是不是属于中医领域的，如果不是，拒绝回答并什么都不输出"
            w = decoder.decode(args, prompt + "为什么使用这些药" + y)

            print("GT : 原因是：{}\n推荐方剂是：{}\n为什么使用这个药：{}".format(z, y, w))


            print('\n')


            correct = (np.array([pred]) == np.array([y])).sum().item()

            correct_list.append(correct)


            total += 1


            if (args.limit_dataset_size != 0) and ((i + 1) >= args.limit_dataset_size):

                break

if __name__ == "__main__":
    main()