from torch.utils.data import Dataset
import os
import json
import torch
from transformers import BartTokenizer


def to_cuda(batch, gpuid):
    for n in batch:
        if n != "data":
            batch[n] = batch[n].to(gpuid)


class LSR_MWF_Dataset(Dataset):
    def __init__(self, fdir, model_type, max_len=-1, is_test=False, total_len=512, is_sorted=True, max_num=-1,
                 is_untok=True, is_pegasus=False, num=-1):
        self.isdir = os.path.isdir(fdir)
        if self.isdir:
            self.fdir = fdir
            if num > 0:
                self.num = min(len(os.listdir(fdir)), num)
            else:
                self.num = len(os.listdir(fdir))
                # self.num = 10
        else:
            with open(fdir) as f:
                self.files = [x.strip() for x in f]
            if num > 0:
                self.num = min(len(self.files), num)
            else:
                self.num = len(self.files)
        if is_pegasus:
            self.tok = BartTokenizer.from_pretrained("facebook/bart-large-xsum", verbose=False, cache_dir="pretrained_model",
                                                     local_files_only=True)
        else:
            self.tok = BartTokenizer.from_pretrained(model_type, verbose=False, cache_dir="pretrained_model",
                                                     local_files_only=True)
        self.maxlen = max_len
        self.is_test = is_test
        self.total_len = total_len
        self.sorted = is_sorted
        self.maxnum = max_num
        self.is_untok = is_untok
        self.is_pegasus = is_pegasus

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if self.isdir:
            with open(os.path.join(self.fdir, "%d.json" % idx), "r") as f:
                data = json.load(f)
        else:
            with open(self.files[idx]) as f:
                data = json.load(f)

        src_txt = data["document"]
        src = self.tok.batch_encode_plus([src_txt], max_length=self.total_len, return_tensors="pt",
                                         pad_to_max_length=False, truncation=True, padding=False)
        src_input_ids = src["input_ids"]
        src_input_ids = src_input_ids.squeeze(0)

        tgt_txt = data["original_summary"]
        tgt = self.tok.batch_encode_plus([tgt_txt], max_length=self.maxlen, return_tensors="pt",
                                         pad_to_max_length=False, truncation=True, padding=False)
        tgt_input_ids = tgt["input_ids"]
        tgt_input_ids = tgt_input_ids.squeeze(0)

        # aspects sentences entities
        asp_txt = data["aspects"]
        sen_txt = data["sentences"]
        ent_txt = data["entities"]
        asp = self.tok.batch_encode_plus([asp_txt], max_length=32, return_tensors="pt",
                                         pad_to_max_length=False, truncation=True, padding=False)
        sen = self.tok.batch_encode_plus([sen_txt], max_length=85, return_tensors="pt",
                                         pad_to_max_length=False, truncation=True, padding=False)
        ent = self.tok.batch_encode_plus([ent_txt], max_length=85, return_tensors="pt",
                                         pad_to_max_length=False, truncation=True, padding=False)
        asp_input_ids = asp["input_ids"]
        sen_input_ids = sen["input_ids"]
        ent_input_ids = ent["input_ids"]
        asp_input_ids = asp_input_ids.squeeze(0)
        sen_input_ids = sen_input_ids.squeeze(0)
        ent_input_ids = ent_input_ids.squeeze(0)

        result = {
            "src_input_ids": src_input_ids,
            "src_attn_mask": src["attention_mask"],
            "tgt_input_ids": tgt_input_ids,
            "tgt_attn_mask": tgt["attention_mask"],
            "asp_input_ids": asp_input_ids,
            "asp_attn_mask": asp["attention_mask"],
            "sen_input_ids": sen_input_ids,
            "sen_attn_mask": sen["attention_mask"],
            "ent_input_ids": ent_input_ids,
            "ent_attn_mask": ent["attention_mask"],
        }
        if self.is_test:
            result["data"] = data
        return result


def collate_mp(batch, pad_token_id, is_test=False):
    def pad(X, max_len=-1):
        if max_len < 0:
            max_len = max(x.size(0) for x in X)
        result = torch.ones(len(X), max_len, dtype=X[0].dtype) * pad_token_id
        for (i, x) in enumerate(X):
            result[i, :x.size(0)] = x
        return result

    src_input_ids = pad([x["src_input_ids"] for x in batch])
    tgt_input_ids = [x["tgt_input_ids"] for x in batch]
    tgt_input_ids = pad(tgt_input_ids)
    asp_input_ids = [x["asp_input_ids"] for x in batch]
    asp_input_ids = pad(asp_input_ids)
    sen_input_ids = [x["sen_input_ids"] for x in batch]
    sen_input_ids = pad(sen_input_ids)
    ent_input_ids = [x["ent_input_ids"] for x in batch]
    ent_input_ids = pad(ent_input_ids)

    if is_test:
        data = [x["data"] for x in batch]
    result = {
        "src_input_ids": src_input_ids,
        "src_attn_mask": batch[0]["src_attn_mask"],
        "tgt_input_ids": tgt_input_ids,
        "tgt_attn_mask": batch[0]["tgt_attn_mask"],
        "asp_input_ids": asp_input_ids,
        "asp_attn_mask": batch[0]["asp_attn_mask"],
        "sen_input_ids": sen_input_ids,
        "sen_attn_mask": batch[0]["sen_attn_mask"],
        "ent_input_ids": ent_input_ids,
        "ent_attn_mask": batch[0]["ent_attn_mask"],
    }
    if is_test:
        result["data"] = data
    return result
