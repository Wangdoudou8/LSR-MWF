from transformers import pipeline
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import glob
import os
import re
import time
from compare_mt.rouge.rouge_scorer import RougeScorer
from datetime import datetime
import random

model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

# Load tokenizer using AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the model and do 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map="auto"
)

# Create a pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
)

# Folder path
folder_path = './cnndm_rationale'

# Gets a list of everything in the folder
items_in_folder = os.listdir(folder_path)


print(f" folder '{folder_path}' has {len(items_in_folder)} files. Start with “{len(items_in_folder)+1}.json” ")
# Read the JSON file
def read_api_keys(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        api_keys = data.get('api_keys', [])
    return api_keys


# Use functions
api_keys = read_api_keys('api_config.json')
print("api_keys:", api_keys)

random.seed(42)

rationale_template = f"""
Given a document and its summary, do the following tasks:
(1) Extract one essential aspect for each summary sentence.
(2) For the essential aspect, extract one associated sentence from the document.
(3) For the associated sentence, retrieve detailed triples of entities and relations in the format [subject entity # relation # object entity, subject entity # relation # object entity, ...].

The aspect, associated sentence and triple should be separated by "|", in the format "[aspects] | [associated sentences] | [subject entity # relation # object entity, subject entity # relation # object entity, ...]".
The aspect, associated sentence and triple should be in the same response, separated by a new line.

Example:
================Example=================
Prompt:
[Document]:
[document]

[Summary]:
[summary]

Update:
Rationale:
1. [aspects] | [associated sentences] | [subject entity # relation # object entity, subject entity # relation # object entity, ...]
2. [aspects] | [associated sentences] | [subject entity # relation # object entity, subject entity # relation # object entity, ...]
...
========================================

Prompt:
[Document]:
<document>

[Summary]:
<summary>

Update:
"""


generate_template = f"""
Given a document and its rationale for summarization.
The rationale contains (1)the essential aspects for summary; (2)the associated sentences for essential aspects; (3)the triples of entities and relations for summary sentences.
Each "[aspects] | [associated sentences] | [subject entity # relation # object entity, subject entity # relation # object entity, ...]" corresponds to one short summary sentence.

First, for each "[aspects] | [associated sentences] | [subject entity # relation # object entity, subject entity # relation # object entity, ...]", generate a multiple sub-summary.
Then, generate a total summary in one sentence with the multiple sub-summaries.

Example:
================Example=================
Prompt:
[Document]: 
[document]

[Rationale]:
1. [aspects] | [associated sentences] | [subject entity # relation # object entity, subject entity # relation # object entity, ...]
2. [aspects] | [associated sentences] | [subject entity # relation # object entity, subject entity # relation # object entity, ...]
...

Update:
Multiple sub-summaries:
[multiple sub-summaries]

Summary:
[summary]
========================================

Prompt:
[Document]: 
<document>

[Rationale]: 
<rationale>

Update: 
"""


class Llama_Decoder:
    def __init__(self):
        pass

    def decode(self, input, temperature):
        # Define message
        messages = [
            {"role": "system",
             "content": "You are a teacher responsible for the rationale process that generates a summary of a document."},
            {"role": "user", "content": f"{input}"},
        ]

        # Call pipeline for text generation
        outputs = pipe(
            messages,
            top_p=1,
            max_new_tokens=4096,
            do_sample=False,
            temperature=0
        )

        # Print the generated text
        assistant_response = outputs[0]["generated_text"][-1]["content"]

        return assistant_response


gpt_decoder = Llama_Decoder()

# Folder path
folder_path = "../data/cnndm/diverse/train"

# Get all.json files using glob
json_files = glob.glob(os.path.join(folder_path, '*.json'))
print(len(json_files))

# Filter and sort file names
def extract_number(filename):
    filename = filename.replace(".json", "")
    match = re.search(r'train\\(\d+)', filename)
    number = match.group(1)  # to get the first in the parentheses matching content, namely digital
    return int(number)


sorted_files = sorted(json_files, key=extract_number)


# Use '-' to separate time
now_custom = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
print(now_custom)
now_custom = now_custom.replace(" ", "_")
new_data_list = []
out_sub_dec_file = open(f"llama3_cnndm_output_train/{now_custom}_llama3_cnndm_sub_dec.txt", "w", encoding="utf-8")
out_dec_file = open(f"llama3_cnndm_output_train/{now_custom}_llama3_cnndm_dec.txt", "w", encoding="utf-8")
out_ref_file = open(f"llama3_cnndm_output_train/{now_custom}_llama3_cnndm_ref.txt", "w", encoding="utf-8")
scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

sub_scores_rouge_1 = 0
sub_scores_rouge_2 = 0
sub_scores_rouge_l = 0

scores_rouge_1 = 0
scores_rouge_2 = 0
scores_rouge_l = 0

idx = len(items_in_folder)+1
cnt = 1

cur_respawn_cnt = 1
respawn_cnt = 0
temperature = 0
start_time = time.time()
all_tmp_dir = []
api_num_1 = -1
api_num_2 = -1
print_output_file = open(f"llama3_cnndm_output_train/{now_custom}_llama3_cnndm_output_log.txt", "w", encoding="utf-8")
while idx <= len(sorted_files):
    fdir = sorted_files[idx-1]
    with open(fdir, "r") as f:
        data = json.load(f)
    article = data["article_untok"]
    abstract = data["abstract_untok"]
    src_txt = " ".join(article)
    tgt_txt = " ".join(abstract)
    src_txt = src_txt.replace("\n", " ")
    tgt_txt = tgt_txt.replace("\n", " ")
    rationale_template_input = rationale_template.replace("<document>", src_txt).replace("<summary>", tgt_txt)
    # print(rationale_template_input)
    rationale_start_time = time.time()
    try:
        api_num_1 = gpt_decoder.counter
        rationale = gpt_decoder.decode(input=rationale_template_input, temperature=temperature)
    except Exception as e:
        print(e, "!!!!!!!!!!!!!!!!!!!!!!! There are frequent calls 1")
        time.sleep(65)
        continue
    index_1 = rationale.find("Rationale:")
    rationale = rationale[index_1+11:]

    rationale_end_time = time.time()
    # print(rationale)
    generate_start_time = time.time()
    # print("*"*100)
    generate_template_input = generate_template.replace("<document>", src_txt).replace("<rationale>", rationale)
    try:
        api_num_2 = gpt_decoder.counter
        llame_generate_output = gpt_decoder.decode(input=generate_template_input, temperature=temperature)
    except Exception as e:
        print(e, "!!!!!!!!!!!!!!!!!!!!!!! There are frequent calls 2")
        time.sleep(61)
        continue

    generate_end_time = time.time()
    # print(llame_generate_output)
    index_2 = llame_generate_output.find("Multiple sub-summaries:")
    index_3 = llame_generate_output.find("Summary:")
    sub_summaries = llame_generate_output[index_2+24:index_3-1].replace("\n", " ").strip()
    summary = llame_generate_output[index_3+9:].replace("\n", " ").strip()

    sub_scores = scorer.score(target=tgt_txt, prediction=sub_summaries)
    sub_rouge1 = sub_scores["rouge1"].fmeasure * 100
    sub_rouge2 = sub_scores["rouge2"].fmeasure * 100
    sub_rougel = sub_scores["rougeLsum"].fmeasure * 100
    sub_scores_rouge_1 += sub_rouge1
    sub_scores_rouge_2 += sub_rouge2
    sub_scores_rouge_l += sub_rougel
    print("1 api_num: {}-{} Summary: rouge1/2/l: {:.2f}|{:.2f}|{:.2f}. ⭐AVG {:.2f}|{:.2f}|{:.2f}".format(api_num_1, api_num_2, sub_rouge1, sub_rouge2, sub_rougel, sub_scores_rouge_1/cnt, sub_scores_rouge_2/cnt, sub_scores_rouge_l/cnt))
    print_output_file.write("1 api_num: {}-{} Summary: rouge1/2/l: {:.2f}|{:.2f}|{:.2f}. ⭐AVG {:.2f}|{:.2f}|{:.2f}\n".format(api_num_1, api_num_2, sub_rouge1, sub_rouge2, sub_rougel, sub_scores_rouge_1/cnt, sub_scores_rouge_2/cnt, sub_scores_rouge_l/cnt))
    scores = scorer.score(target=tgt_txt, prediction=summary)
    rouge1 = scores["rouge1"].fmeasure * 100
    rouge2 = scores["rouge2"].fmeasure * 100
    rougel = scores["rougeLsum"].fmeasure * 100
    scores_rouge_1 += rouge1
    scores_rouge_2 += rouge2
    scores_rouge_l += rougel
    print("2 api_num: {}-{} Summary: rouge1/2/l: {:.2f}|{:.2f}|{:.2f}. ⭐AVG {:.2f}|{:.2f}|{:.2f}".format(api_num_1, api_num_2, rouge1, rouge2, rougel, scores_rouge_1/cnt, scores_rouge_2/cnt, scores_rouge_l/cnt))
    print_output_file.write("2 api_num: {}-{} Summary: rouge1/2/l: {:.2f}|{:.2f}|{:.2f}. ⭐AVG {:.2f}|{:.2f}|{:.2f}\n".format(api_num_1, api_num_2, rouge1, rouge2, rougel, scores_rouge_1/cnt, scores_rouge_2/cnt, scores_rouge_l/cnt))
    print("Processed Number :", idx," Processed Quantity :", cnt, "Inference and generation time respectively :", round(rationale_end_time - rationale_start_time, 2), round(generate_end_time - generate_start_time, 2), ", Total time spent:", round(time.time() - start_time, 2), "seconds")
    print_output_file.write("Processed number: {} processed quantity: {} reasoning and generated respectively: {}|{} a total time: {} seconds \n".format(idx, cnt, round(rationale_end_time - rationale_start_time, 2), round(generate_end_time - generate_start_time, 2), round(time.time() - start_time, 2)))
    score_1 = (1 - (sub_rouge1*sub_rouge2)/(sub_rouge1+sub_rouge2)) * 100

    if score_1 <= 85:
        sub_scores_rouge_1 -= sub_rouge1  # Backtrack
        sub_scores_rouge_2 -= sub_rouge2
        sub_scores_rouge_l -= sub_rougel
        scores_rouge_1 -= rouge1
        scores_rouge_2 -= rouge2
        scores_rouge_l -= rougel
        cnt -= 1
        # tmp_dir = {'sub_summaries': sub_summaries, 'summary': summary, 'tgt_txt': tgt_txt, 'rationale': rationale}
        # all_tmp_dir.append([tmp_dir, sub_rouge1, sub_rouge2, sub_rougel])
        # cur_respawn_cnt = cur_respawn_cnt + 1
        # respawn_cnt = respawn_cnt + 1
        print("The score is too low, don't take this example...")
        print_output_file.write("The score is too low, don't take this example...")
        # temperature = round(random.random(), 2)
        idx = idx + 1
        cnt = cnt + 1
        continue

    # cur_respawn_cnt = 1
    # all_tmp_dir = []
    # temperature = 0
    # rationale = rationale.replace("\n", " ")
    sub_summaries = sub_summaries.replace("\n", " ")
    out_sub_dec_file.write(sub_summaries + "\n")
    summary = summary.replace("\n", " ")
    out_dec_file.write(summary + "\n")
    tgt_txt = tgt_txt.replace("\n", " ")
    out_ref_file.write(tgt_txt + "\n")
    TMP_json = {"article": src_txt, "abstract": tgt_txt, "rationale": rationale, "llm_sub_summaries": sub_summaries, "llm_summary": summary}
    with open(f"./cnndm_rationale/{cnt}.json", "w", encoding="utf-8") as tmp_out_rationale_file:
        json.dump(TMP_json, tmp_out_rationale_file, ensure_ascii=False, indent=4)
    tmp_out_rationale_file.close()
    new_data_list.append(
        {"article": src_txt, "abstract": tgt_txt, "rationale": rationale, "llm_sub_summaries": sub_summaries,
         "llm_summary": summary})

    idx = idx + 1
    cnt = cnt + 1
    print("#" * 100)
    print_output_file.write("#" * 100)

with open(f"llama3_cnndm_output_train/{now_custom}_llama3_cnndm_rationale.json", "w", encoding="utf-8") as out_rationale_file:
    json.dump(new_data_list, out_rationale_file, ensure_ascii=False, indent=4)
out_sub_dec_file.close()
out_dec_file.close()
out_ref_file.close()
print_output_file.close()



