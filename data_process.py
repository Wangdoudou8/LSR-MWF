import json
import os

# 初始化一个空列表来存储所有字典
all_dicts = []

# 指定包含.json文件的文件夹路径
# folder_path = './all_data/cnndm_rationale/'
# folder_path = 'C:\\Users\\lenovo\\Desktop\\知识关联算法代码\\xsum_rationale'
folder_path = 'C:\\Users\\lenovo\\Desktop\\xsum_rationale'
# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):  # 确保处理的是.json文件
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # 读取json文件内容
            if isinstance(data, dict):  # 确保读取的内容是字典
                all_dicts.append(data)

            # 现在all_dicts列表中包含了所有json文件中的字典
print(f"共汇总了{len(all_dicts)}个字典。")
print(all_dicts[0:3])

new_data = []
cnt = 0
for idx, item in enumerate(all_dicts):
    # new_item_dict = {"document": item["article"], "original_summary": item["abstract"], "rationale": item["rationale"], "llm_sub_summaries": item["llm_sub_summaries"], "llm_summary": item["llm_summary"]}
    new_item_dict = {"document": item["article"], "original_summary": item["abstract"], "rationale": item["rationale"], "llm_summary": item["llm_summary"]}

    # 使用split方法根据"\n\n"分割字符串
    segments = item['rationale'].strip().split("\n")   # 使用strip()移除可能的首尾空白字符

    rationale_list = []
    for segment in segments:
        rationale_list.append(segment)

    aspects_txt = ""
    sentence_txt = ""
    entity_txt = ""

    flag = False
    for it in rationale_list:
        segments = it.strip().split("|")  # 使用strip()移除可能的首尾空白字符
        if len(segments) < 3 or segments[0] == "" or segments[1] == "" or segments[2] == "":
            cnt += 1
            print(cnt, it)
            continue

        aspects_txt += segments[2] + " | "
        sentence_txt += segments[1] + " | "
        entity_txt += segments[0] + " | "

    if aspects_txt == "" and sentence_txt == "" and entity_txt == "":
        # cnt += 1
        print(item)
        print("出问题了")
        continue

    if flag is False:
        new_item_dict["aspects"] = aspects_txt.strip()
        new_item_dict["sentences"] = sentence_txt.strip()
        new_item_dict["entities"] = entity_txt.strip()

        new_data.append(new_item_dict)

with open('./just_7950_new_xsum_data.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)

total_len = len(new_data)
# 按照 8:1:1 的比例划分训练集、验证集、测试集


# 按照 8:1:1 的比例划分训练集、验证集、测试集
train_len = int(total_len * 0.8)
val_len = int(total_len * 0.1)
test_len = total_len - train_len - val_len

train_data = new_data[:train_len]
val_data = new_data[train_len:train_len+val_len]
test_data = new_data[train_len+val_len:]


# # 遍历数据，将每个字典保存到单独的JSON文件中
# for index, item in enumerate(test_data):
#     # 构造文件名
#     filename = f"{index+7200}.json"
#     # 打开文件并写入数据
#     with open(f"./xsum_test/{filename}", 'w') as f:
#         json.dump(item, f)
#
# exit(1)

# 如果没有 xsum_train 的文件夹，则创建
if not os.path.exists("./xsum_train"):
    os.makedirs("./xsum_train")


# 如果没有 xsum_val 的文件夹，则创建
if not os.path.exists("./xsum_val"):
    os.makedirs("./xsum_val")


# 如果没有 xsum_test 的文件夹，则创建
if not os.path.exists("./xsum_test"):
    os.makedirs("./xsum_test")

# 遍历数据，将每个字典保存到单独的JSON文件中
for index, item in enumerate(train_data):
    # 构造文件名
    filename = f"{index}.json"
    # 打开文件并写入数据
    with open(f"./xsum_train/{filename}", 'w') as f:
        json.dump(item, f)

# 遍历数据，将每个字典保存到单独的JSON文件中
for index, item in enumerate(val_data):
    # 构造文件名
    filename = f"{index}.json"
    # 打开文件并写入数据
    with open(f"./xsum_val/{filename}", 'w') as f:
        json.dump(item, f)

# 遍历数据，将每个字典保存到单独的JSON文件中
for index, item in enumerate(test_data):
    # 构造文件名
    filename = f"{index}.json"
    # 打开文件并写入数据
    with open(f"./xsum_test/{filename}", 'w') as f:
        json.dump(item, f)


# with open('./xsum_test.source', 'w', encoding='utf-8') as f:
#     for item in test_data:
#         f.write(item['document'] + '\n')
# f.close()
#
#
# # test.target
# with open('./xsum_test.target', 'w', encoding='utf-8') as f:
#     for item in test_data:
#         f.write(item['original_summary'] + '\n')
# f.close()

print("Finish!")

