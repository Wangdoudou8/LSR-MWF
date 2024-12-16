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
tokenizer = AutoTokenizer.from_pretested(model_id)

# Load the model and do 8-bit quantization
model = AutoModelForCausalLM.from_pretested(
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

random.seed(42)

rationale_template = f"""
Given a document and its summary, do the following tasks:
(1) For the summary sentence, retrieve detailed triples of entities and relations in the format [subject entity # relation # object entity, subject entity # relation # object entity, ...].
(2) For the triples, extract one associated sentence from the document.
(3) Extract one essential aspect for the associated sentence.

The triples, associated sentences and aspects should be separated by "|", in the format "[subject entity # relation # object entity, subject entity # relation # object entity, ...] | [associated sentences] | [aspects]".
The triples, associated sentences and aspects should be in the same response, separated by a new line.
There should be only one rationale.

Example:
================Example=================
Prompt:
[Document]: 
[document]

[Summary]: 
[summary]

Update:
Rationale: 
[subject entity # relation # object entity, subject entity # relation # object entity, ...] | [associated sentences] | [aspects]

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
The rationale contains (1)the triples of entities and relations for summary sentences; (2)the associated sentences for essential aspects; (3)the essential aspects for summary.
For the "[subject entity # relation # object entity, subject entity # relation # object entity, ...] | [associated sentences] | [aspects]", generate a simple summary in one sentence.

Example:
================Example=================
Prompt:
[Document]: 
[document]

[Rationale]:
[subject entity # relation # object entity, subject entity # relation # object entity, ...] | [associated sentences] | [aspects]

Update:
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


# Folder path
folder_path = "../data/xsum/diverse/test"

# Get all *.json files using glob
json_files = glob.glob(os.path.join(folder_path, '*.json'))
print(len(json_files))


# Filter and sort file names
def extract_number(filename):
    filename = filename.replace(".json", "")
    match = re.search(r'test\/(\d+)', filename)
    number = match.group(1)  # Gets the match inside the first parenthesis, which is the number
    return int(number)


sorted_files = sorted(json_files, key=extract_number)

gpt_decoder = Llama_Decoder()

# Use '-' to separate time
now_custom = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
print(now_custom)
now_custom = now_custom.replace(" ", "_")
new_data_list = []
out_dec_file = open(f"./llama3_xsum_output_train/{now_custom}_llama3_xsum_dec.txt", "w", encoding="utf-8")
out_ref_file = open(f"./llama3_xsum_output_train/{now_custom}_llama3_xsum_ref.txt", "w", encoding="utf-8")
scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

scores_rouge_1 = 0
scores_rouge_2 = 0
scores_rouge_l = 0

# Folder path
folder_path = './xsum_rationale'

# Gets a list of everything in the folder
items_in_folder = os.listdir(folder_path)

print(f" folder '{folder_path}' has {len(items_in_folder)} files. Start with “{len(items_in_folder)+1}.json” ")
idx = len(items_in_folder) + 1
cnt = 1

cur_respawn_cnt = 1
respawn_cnt = 0
temperature = 0
start_time = time.time()
all_tmp_dir = []
api_num_1 = -1
api_num_2 = -1
print_output_file = open(f"./llama3_xsum_output_train/{now_custom}_llama3_xsum_output_log.txt", "w",
                         encoding="utf-8")
while idx <= len(sorted_files):
    fdir = sorted_files[idx - 1]
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
        rationale = gpt_decoder.decode(input=rationale_template_input, temperature=temperature)
    except Exception as e:
        print(e, "!!!!!!!!!!!!!!!!!!!!!!! There are frequent calls 1")
        time.sleep(65)
        continue
    index_1 = rationale.find("Rationale:")
    rationale = rationale[index_1 + 11:]

    rationale_end_time = time.time()
    # print(rationale)
    generate_start_time = time.time()
    # print("*"*100)
    generate_template_input = generate_template.replace("<document>", src_txt).replace("<rationale>", rationale)
    try:
        llame_generate_output = gpt_decoder.decode(input=generate_template_input, temperature=temperature)
    except Exception as e:
        print(e, "!!!!!!!!!!!!!!!!!!!!!!! There are frequent calls 2")
        time.sleep(65)
        continue

    generate_end_time = time.time()
    index_3 = llame_generate_output.find("Summary:")
    summary = llame_generate_output[index_3 + 9:].replace("\n", " ").strip()
    scores = scorer.score(target=tgt_txt, prediction=summary)
    rouge1 = scores["rouge1"].fmeasure * 100
    rouge2 = scores["rouge2"].fmeasure * 100
    rougel = scores["rougeLsum"].fmeasure * 100
    scores_rouge_1 += rouge1
    scores_rouge_2 += rouge2
    scores_rouge_l += rougel
    print("2 api_num: {}-{} Summary: rouge1/2/l: {:.2f}|{:.2f}|{:.2f}. ⭐AVG {:.2f}|{:.2f}|{:.2f}".format(api_num_1 ,
                                                                                                         api_num_2 ,
                                                                                                         rouge1, rouge2,
                                                                                                         rougel,
                                                                                                         scores_rouge_1 / cnt,
                                                                                                         scores_rouge_2 / cnt,
                                                                                                         scores_rouge_l / cnt))
    print_output_file.write(
        "2 api_num: {}-{} Summary: rouge1/2/l: {:.2f}|{:.2f}|{:.2f}. ⭐AVG {:.2f}|{:.2f}|{:.2f}\n".format(api_num_1 ,
                                                                                                         api_num_2 ,
                                                                                                         rouge1, rouge2,
                                                                                                         rougel,
                                                                                                         scores_rouge_1 / cnt,
                                                                                                         scores_rouge_2 / cnt,
                                                                                                         scores_rouge_l / cnt))
    print("Processed number:", idx, "Processed quantity:", cnt, "Inference and generation are used separately:",
          round(rationale_end_time - rationale_start_time, 2), round(generate_end_time - generate_start_time, 2),
          ", Total time spent:", round(time.time() - start_time, 2), "seconds")
    print_output_file.write(
        "Processed number: {} processed quantity: {} reasoning and generated respectively: {}|{} a total time: {} seconds \n".format(idx, cnt, round(
            rationale_end_time - rationale_start_time, 2), round(generate_end_time - generate_start_time, 2), round(
            time.time() - start_time, 2)))
    score_2 = (1 - (rouge1 + rouge2 + rougel) / 3) * 100

    if score_2 <= 65:
        scores_rouge_1 -= rouge1  # Backtrack
        scores_rouge_2 -= rouge2
        scores_rouge_l -= rougel
        cnt -= 1
        print("The score is too low. Don't take this example... {}".format(idx))
        print_output_file.write("The score is too low. Don't take this example ... {}".format(idx))
        idx = idx + 1
        cnt = cnt + 1
        continue

    summary = summary.replace("\n", " ")
    out_dec_file.write(summary + "\n")
    tgt_txt = tgt_txt.replace("\n", " ")
    out_ref_file.write(tgt_txt + "\n")
    TMP_json = {"article": src_txt, "abstract": tgt_txt, "rationale": rationale, "llm_summary": summary}
    with open(f"./xsum_rationale/{cnt}.json", "w", encoding="utf-8") as tmp_out_rationale_file:
        json.dump(TMP_json, tmp_out_rationale_file, ensure_ascii=False, indent=4)
    tmp_out_rationale_file.close()
    new_data_list.append(
        {"article": src_txt, "abstract": tgt_txt, "rationale": rationale,
         "llm_summary": summary})

    idx = idx + 1
    cnt = cnt + 1
    print("#" * 100)
    print_output_file.write("#" * 100)

with open(f"./llama3_xsum_output_train/{now_custom}_llama3_xsum_rationale.json", "w",
          encoding="utf-8") as out_rationale_file:
    json.dump(new_data_list, out_rationale_file, ensure_ascii=False, indent=4)
out_dec_file.close()
out_ref_file.close()
print_output_file.close()
