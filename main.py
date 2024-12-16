import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import os
import random
from compare_mt.rouge.rouge_scorer import RougeScorer
from transformers import BartTokenizer, PegasusTokenizer
from utils import Recorder
from data_utils import to_cuda, collate_mp, LSR_MWF_Dataset
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from functools import partial
# from baseline_model import CombinationModel
from model import CombinationModel
import logging
from nltk import sent_tokenize, word_tokenize
from config import cnndm_setting, xsum_setting
from tqdm import tqdm
from transformers.models.bart.configuration_bart import BartConfig
import time 


logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_fast").setLevel(logging.ERROR)


def base_setting(args):
    args.batch_size = getattr(args, 'batch_size', 1)  # batch size on one gpu, one step
    args.epoch = getattr(args, 'epoch', 100)
    args.report_freq = getattr(args, "report_freq", 100)  # report frequency
    args.accumulate_step = getattr(args, "accumulate_step", 32)  # accumulate gradients steps
    args.model_type = getattr(args, "model_type", "facebook/bart-large-cnn")  # model type
    args.warmup_steps = getattr(args, "warmup_steps", 10000)  # warmup steps
    args.normalize = getattr(args, "normalize", True)  # normalize predicited likelihood
    args.grad_norm = getattr(args, "grad_norm", 0)  # gradient norm
    args.seed = getattr(args, "seed", 42)  # random seed 970903
    args.no_gold = getattr(args, "no_gold", False)  # whether to use gold summaries
    args.pretrained = getattr(args, "pretrained_model", "pretrained_model")  # pretrained model path
    args.max_lr = getattr(args, "max_lr", 2e-3)  # max learning rate (* 1e-2)
    args.scale = getattr(args, "scale", 1)  # scale of ranking loss
    args.datatype = getattr(args, "datatype", "diverse")  # data type
    args.dataset = getattr(args, "dataset", "cnndm")  # dataset
    args.max_len = getattr(args, "max_len", 120)  # max length of summary
    args.max_num = getattr(args, "max_num", 4)  # max number of candidate summaries
    args.smooth = getattr(args, "smooth", 0.1)  # label smoothing
    args.total_len = getattr(args, "total_len", 1024)  # total length of source article
    args.length_penalty = getattr(args, "length_penalty", 2.0)  # length penalty
    args.do_sample = getattr(args, "do_sample", True)  # whether to generaet summaries during evaluation
    args.gen_max_len = getattr(args, "gen_max_len", 140)  # max length of generated summaries
    args.gen_min_len = getattr(args, "gen_min_len", 55)  # min length of generated summaries
    args.is_pegasus = getattr(args, "is_pegasus", False)  # whether to use Pegasus as the baseline model
    args.adding = getattr(args, "adding", 0)  # used for numerical stability
    args.eval_interval = getattr(args, "eval_interval", 1000)  # evaluation intervals
    args.num_beams = getattr(args, "num_beams", 4)  # number of beams for beam search


def test(dataloader, gen_dataloader, model, args, tok, gpuid, do_sample=False):
    model.eval()
    if args.cuda:
        device = f"cuda:{gpuid}"
    else:
        device = "cpu"
    if len(args.gpuid) > 1:
        _model = model.module
    else:
        _model = model
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    rouge1, rouge2, rougeLsum = 0, 0, 0

    cnt = 0
    sample_rouge1, sample_rouge2, sample_rougeLsum = 0, 0, 0
    if do_sample:
        # generation
        def process(x):
            return sent_tokenize(" ".join(word_tokenize(x.strip())))

        with torch.no_grad():
            for (i, batch) in enumerate(gen_dataloader):
                # print("i:", i, "len(gen_dataloader)", len(gen_dataloader))
                if args.cuda:
                    to_cuda(batch, device)
                samples = batch["data"]
                slines = [x["document"] for x in samples]
                dct = tok.batch_encode_plus(slines, max_length=args.total_len, return_tensors="pt",
                                            pad_to_max_length=True, truncation=True)
                summaries = _model.generate(
                    input_ids=dct["input_ids"].to(device),
                    attention_mask=dct["attention_mask"].to(device),
                    max_length=args.gen_max_len + 2,
                    # +2 from original because we start at step=1 and stop before max_length
                    min_length=args.gen_min_len + 1,  # +1 from original because we start at step=1
                    no_repeat_ngram_size=3,
                    num_beams=args.num_beams,
                    length_penalty=args.length_penalty,
                    early_stopping=True,
                    other_input_ids=dct["input_ids"].to(device),
                    other_attention_mask=dct["attention_mask"].to(device),
                )
                dec = [tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                for (hypothesis, x) in zip(dec, samples):
                    hypothesis = hypothesis.replace("\n", " ")
                    ref = x["original_summary"]
                    x = process(ref)
                    y = process(hypothesis)
                    score = rouge_scorer.score("\n".join(x), "\n".join(y))
                    sample_rouge1 += score["rouge1"].fmeasure
                    sample_rouge2 += score["rouge2"].fmeasure
                    sample_rougeLsum += score["rougeLsum"].fmeasure
                    cnt += 1
                    # print("推理生成 cnt:", cnt)
                    # print("hyp:", hypothesis)
        sample_rouge1 = sample_rouge1 / cnt
        sample_rouge2 = sample_rouge2 / cnt
        sample_rougeLsum = sample_rougeLsum / cnt
        if len(args.gpuid) > 1:
            sample_rouge1 = torch.FloatTensor([sample_rouge1]).to(device)
            dist.all_reduce(sample_rouge1, op=dist.reduce_op.SUM)
            sample_rouge1 = sample_rouge1.item() / len(args.gpuid)
            sample_rouge2 = torch.FloatTensor([sample_rouge2]).to(device)
            dist.all_reduce(sample_rouge2, op=dist.reduce_op.SUM)
            sample_rouge2 = sample_rouge2.item() / len(args.gpuid)
            sample_rougeLsum = torch.FloatTensor([sample_rougeLsum]).to(device)
            dist.all_reduce(sample_rougeLsum, op=dist.reduce_op.SUM)
            sample_rougeLsum = sample_rougeLsum.item() / len(args.gpuid)
    print(">>>> hyp:", y)
    print(">>>> ref:", x)
    model.train()
    return {
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeLsum": rougeLsum,
        "sample_rouge1": sample_rouge1,
        "sample_rouge2": sample_rouge2,
        "sample_rougeLsum": sample_rougeLsum,
    }


def run(rank, args):
    if args.config == "cnndm":
        cnndm_setting(args)
    elif args.config == "xsum":
        xsum_setting(args)
    else:
        base_setting(args)
    print("args.config:", args.config)
    # task initialization
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    gpuid = args.gpuid[rank]
    is_master = rank == 0
    is_mp = len(args.gpuid) > 1
    world_size = len(args.gpuid)
    if is_master:
        id = len(os.listdir("./cache"))
        recorder = Recorder(id, args.log)
    # build dataloader
    # if args.is_pegasus:
    #     tok = PegasusTokenizer.from_pretrained(args.model_type)
    # else:
    print("load BartTokenizer ....... ")
    if args.config == "xsum":
        args.model_type = "facebook/bart-large-xsum"
    tok = BartTokenizer.from_pretrained(args.model_type, cache_dir="pretrained_model", local_files_only=True)

    collate_fn = partial(collate_mp, pad_token_id=tok.pad_token_id, is_test=False)
    collate_fn_val = partial(collate_mp, pad_token_id=tok.pad_token_id, is_test=True)

    train_set = LSR_MWF_Dataset(f"./data/{args.dataset}/{args.datatype}/train", args.model_type,
                            max_len=args.max_len,
                            max_num=args.max_num, total_len=args.total_len, is_pegasus=args.is_pegasus)
    val_set = LSR_MWF_Dataset(f"./data/{args.dataset}/{args.datatype}/val", args.model_type, is_test=True,
                          max_len=args.max_len,
                          is_sorted=False, max_num=args.max_num, total_len=args.total_len, is_pegasus=args.is_pegasus)
    if is_mp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=1,
                                collate_fn=collate_fn, sampler=train_sampler)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_set, num_replicas=world_size, rank=rank)
        val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn_val,
                                    sampler=val_sampler)
        val_gen_dataloader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=1, collate_fn=collate_fn_val,
                                        sampler=val_sampler)
    else:
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1,
                                collate_fn=collate_fn)
        val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn_val)
        val_gen_dataloader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=1, collate_fn=collate_fn_val)
    # build models
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    model = CombinationModel.from_pretrained(
        'facebook/bart-large-xsum',
        cache_dir="./pretrained_model",
        local_files_only=True)
    if len(args.model_pt) > 0:
        model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt), map_location=f'cuda:{gpuid}'))
    if args.cuda:
        if is_mp:
            # Using DDP
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            model = nn.parallel.DistributedDataParallel(model.to(gpuid), [gpuid], find_unused_parameters=False)
        else:
            model = model.cuda()

    # print(model)
    for name, param in model.named_parameters():  
        if ".encoder." in name:
            param.requires_grad = False

    # for name, param in model.named_parameters():  
    #     if ".encoder." in name:
    #         print(f"{name} is frozen: {not param.requires_grad}")
    
    model.train()
    # 打印模型的参数量
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    MB_params = params / 1024 / 1024
    print("model parameters: ", MB_params, "MB")
    
    # set the model to scoring mode

    # if args.smooth > 0:
    #     mle_fn = label_smoothing_loss(ignore_index=tok.pad_token_id, epsilon=args.smooth)
    # else:
    #     mle_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id)
    s_optimizer = optim.Adam(model.parameters())
    # if is_master:
    #     recorder.write_config(args, [model], __file__)
    all_step_cnt = 0
    if is_mp:
        if is_master:
            id = torch.FloatTensor([id]).to(gpuid)
        else:
            id = torch.zeros(1).to(gpuid)
        dist.all_reduce(id, op=dist.reduce_op.SUM)
        id = int(id.item())
    # define evaluation function
    if args.dataset == "xsum":
        def eval_fn(rouge1, rouge2, rougeLsum):
            return 1 - 2 * rouge1 * rouge2 / (rouge1 + rouge2)
    else:
        def eval_fn(rouge1, rouge2, rougeLsum):
            return 1 - (rouge1 * rouge2 + rougeLsum) / 3
    test_start_time = time.time()
    print("first start test ...")
    print("len(val_gen_dataloader):", len(val_gen_dataloader))
    result = test(val_dataloader, val_gen_dataloader, model, args, tok, gpuid, args.do_sample)
    print("firsttest ...", "test cost time:", time.time() - test_start_time)
    com_loss = eval_fn(result["sample_rouge1"], result["sample_rouge2"], result["sample_rougeLsum"])
    recorder.print("first val generation rouge1: %.6f, rouge2: %.6f, rougeLsum: %.6f, loss: %.6f"
            % (result["sample_rouge1"], result["sample_rouge2"], result["sample_rougeLsum"], com_loss))
    # start training
    start_time = time.time()
    minimum_com_loss = 9999999
    for epoch in range(args.epoch):
        s_optimizer.zero_grad()

        avg_com_loss = 0
        step_cnt = 0
        epoch_step = 0
        avg_loss = 0
        for (i, batch) in enumerate(dataloader):
            if args.cuda:
                to_cuda(batch, gpuid)
            step_cnt += 1
            # forward pass
            output = model(input_ids=batch['src_input_ids'],
                           attention_mask=batch['src_attn_mask'],
                           doc_labels=batch['tgt_input_ids'],
                           decoder_attention_mask=batch['tgt_attn_mask'],

                           topic_labels=batch['asp_input_ids'],
                           topic_attention_mask=batch['asp_attn_mask'],

                           sent_labels=batch['sen_input_ids'],
                           sent_attention_mask=batch['sen_attn_mask'],

                           tri_labels=batch['ent_input_ids'],
                           tri_attention_mask=batch['ent_attn_mask'],
                           )
            loss = output.loss
            logits = output.logits
            loss = loss / args.accumulate_step
            avg_loss += loss.item()
            loss.backward()
            if step_cnt == args.accumulate_step:
                # print("updating and adjust learning rate ...")
                # updating
                if args.grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                step_cnt = 0
                epoch_step += 1
                all_step_cnt += 1
                # adjust learning rate
                lr = args.max_lr * min(all_step_cnt ** (-0.5), all_step_cnt * (args.warmup_steps ** (-1.5)))
                save_lr = lr
                # lr = 3e-4
                # print("new lr:", lr)
                for param_group in s_optimizer.param_groups:
                    param_group['lr'] = lr
                # print("lr:", lr)
                s_optimizer.step()
                s_optimizer.zero_grad()
            if i % 500 == 0:
                a1_w = round( model.model.decoder.a1.item(), 6)
                a2_w = round( model.model.decoder.a2.item(), 6)
                a3_w = round( model.model.decoder.a3.item(), 6)
                a4_w = round( model.model.decoder.a4.item(), 6)
                a5_w = round( model.model.decoder.a5.item(), 6)
                a6_w = round( model.model.decoder.a6.item(), 6)
                a7_w = round( model.model.decoder.a7.item(), 6)
                a8_w = round( model.model.decoder.a8.item(), 6)
                a9_w = round( model.model.decoder.a9.item(), 6)
                a10_w = round( model.model.decoder.a10.item(), 6)
                a11_w = round( model.model.decoder.a11.item(), 6)
                a12_w = round( model.model.decoder.a12.item(), 6)

                b1_w = round( model.model.decoder.b1.item(), 6)
                b2_w = round( model.model.decoder.b2.item(), 6)
                b3_w = round( model.model.decoder.b3.item(), 6)
                b4_w = round( model.model.decoder.b4.item(), 6)
                b5_w = round( model.model.decoder.b5.item(), 6)
                b6_w = round( model.model.decoder.b6.item(), 6)
                b7_w = round( model.model.decoder.b7.item(), 6)
                b8_w = round( model.model.decoder.b8.item(), 6)
                b9_w = round( model.model.decoder.b9.item(), 6)
                b10_w = round( model.model.decoder.b10.item(), 6)
                b11_w = round( model.model.decoder.b11.item(), 6)
                b12_w = round( model.model.decoder.b12.item(), 6)

                c1_w = round( model.model.decoder.c1.item(), 6)
                c2_w = round( model.model.decoder.c2.item(), 6)
                c3_w = round( model.model.decoder.c3.item(), 6)
                c4_w = round( model.model.decoder.c4.item(), 6)
                c5_w = round( model.model.decoder.c5.item(), 6)
                c6_w = round( model.model.decoder.c6.item(), 6)
                c7_w = round( model.model.decoder.c7.item(), 6)
                c8_w = round( model.model.decoder.c8.item(), 6)
                c9_w = round( model.model.decoder.c9.item(), 6)
                c10_w = round( model.model.decoder.c10.item(), 6)
                c11_w = round( model.model.decoder.c11.item(), 6)
                c12_w = round( model.model.decoder.c12.item(), 6)
                # print(f"Train epoch: {epoch}, step: {i}/{len(dataloader)}, loss: {loss.item()}")
                print("a1-a12:", a1_w, a2_w, a3_w, a4_w, a5_w, a6_w, a7_w, a8_w, a9_w, a10_w, a11_w, a12_w)
                print("b1-b12:", b1_w, b2_w, b3_w, b4_w, b5_w, b6_w, b7_w, b8_w, b9_w, b10_w, b11_w, b12_w)
                print("c1-c12:", c1_w, c2_w, c3_w, c4_w, c5_w, c6_w, c7_w, c8_w, c9_w, c10_w, c11_w, c12_w)
                print(f"Train epoch: {epoch}, step: {i}/{len(dataloader)}, loss: {loss.item()}, avg_loss: {avg_loss/(i+1)}, cost time:{time.time() - start_time}")
                start_time = time.time()

            if epoch_step % args.report_freq == 0 and step_cnt == 0 and is_master:
                # report stats
                # print("id: %d" % id)
                # recorder.print("epoch: %d, batch: %d, avg loss: %.6f, avg mle loss: %.6f"
                #                % (
                #                    epoch + 1, epoch_step, avg_loss / args.report_freq,
                #                    avg_com_loss / args.report_freq))
                # recorder.print(f"learning rate: {lr:.6f}")
                recorder.plot("loss", {"loss": avg_loss / args.report_freq}, all_step_cnt)
                recorder.plot("com_loss", {"loss": avg_com_loss / args.report_freq}, all_step_cnt)

                recorder.print()
                avg_com_loss, avg_loss = 0, 0
            # del loss, com_loss, output

            if all_step_cnt % args.eval_interval == 0 and all_step_cnt != 0 and step_cnt == 0:
                # evaluate the model as a scorer
                print("start test ...")
                result = test(val_dataloader, val_gen_dataloader, model, args, tok, gpuid, args.do_sample)
                print("end test...")
                if args.do_sample:
                    com_loss = eval_fn(result["sample_rouge1"], result["sample_rouge2"], result["sample_rougeLsum"])
                else:
                    com_loss = result["com_loss"]
                if com_loss < minimum_com_loss and is_master:
                    minimum_com_loss = com_loss
                    if is_mp:
                        recorder.save(model.module, "model_generation.bin")
                    else:
                        recorder.save(model, "model_generation.bin")
                    recorder.print("best generation loss - epoch: %d, batch: %d" % (epoch, i / args.accumulate_step))
                if is_master:
                    recorder.print("val generation loss: %.6f" % (com_loss))
                    if args.do_sample:
                        recorder.print("val generation rouge1: %.6f, rouge2: %.6f, rougeLsum: %.6f"
                                       % (result["sample_rouge1"], result["sample_rouge2"], result["sample_rougeLsum"]))
                # save current model
                if is_master:
                    if is_mp:
                        recorder.save(model.module, "model_cur.bin")
                    else:
                        recorder.save(model, "model_cur.bin")
                    recorder.save(s_optimizer, "optimizer.bin")

            # if (i+1) % 3501 == 0:
            #     test_start_time = time.time()
            #     print("final start test ...", i)
            #     result = test(val_dataloader, val_gen_dataloader, model, args, tok, gpuid, args.do_sample)
            #     print("end final test ...", "test cost time:", time.time() - test_start_time)
            #     if args.do_sample:
            #         com_loss = eval_fn(result["sample_rouge1"], result["sample_rouge2"], result["sample_rougeLsum"])
            #     else:
            #         com_loss = result["com_loss"]
            #     if com_loss < minimum_com_loss and is_master:
            #         minimum_com_loss = com_loss
            #         if is_mp:
            #             recorder.save(model.module, "model_generation.bin")
            #         else:
            #             recorder.save(model, "model_generation.bin")
            #         recorder.print("best generation loss - epoch: %d, batch: %d" % (epoch, i / args.accumulate_step))
            #     if is_master:
            #         recorder.print("val generation loss: %.6f" % (com_loss))
            #         if args.do_sample:
            #             recorder.print("val generation rouge1: %.6f, rouge2: %.6f, rougeLsum: %.6f"
            #                         % (result["sample_rouge1"], result["sample_rouge2"], result["sample_rougeLsum"]))
            #     # save current model
            #     if is_master:
            #         if is_mp:
            #             recorder.save(model.module, "model_cur.bin")
            #         else:
            #             recorder.save(model, "model_cur.bin")
            #         recorder.save(s_optimizer, "optimizer.bin")

        if (epoch + 1) % 1 == 0:
            # if (i+1) % 3501 == 0:
            test_start_time = time.time()
            print("final start test ...", i)
            result = test(val_dataloader, val_gen_dataloader, model, args, tok, gpuid, args.do_sample)
            print("end final test ...", "test cost time:", time.time() - test_start_time)
            if args.do_sample:
                com_loss = eval_fn(result["sample_rouge1"], result["sample_rouge2"], result["sample_rougeLsum"])
            else:
                com_loss = result["com_loss"]
            if com_loss < minimum_com_loss and is_master:
                minimum_com_loss = com_loss
                if is_mp:
                    recorder.save(model.module, "model_generation.bin")
                else:
                    recorder.save(model, "model_generation.bin")
                recorder.print("best generation loss - epoch: %d, batch: %d" % (epoch, i / args.accumulate_step))
            if is_master:
                recorder.print("val generation loss: %.6f" % (com_loss))
                if args.do_sample:
                    recorder.print("val generation rouge1: %.6f, rouge2: %.6f, rougeLsum: %.6f"
                                % (result["sample_rouge1"], result["sample_rouge2"], result["sample_rougeLsum"]))
            # save current model
            if is_master:
                if is_mp:
                    recorder.save(model.module, "model_cur.bin")
                else:
                    recorder.save(model, "model_cur.bin")
                recorder.save(s_optimizer, "optimizer.bin")


def evaluation(args):
    # load data
    if args.config == "cnndm":
        cnndm_setting(args)
    elif args.config == "xsum":
        xsum_setting(args)
    else:
        base_setting(args)
    if args.is_pegasus:
        args.model_type = "facebook/bart-large-xsum"
        tok = BartTokenizer.from_pretrained(args.model_type, cache_dir="pretrained_model", local_files_only=True)
    else:
        tok = BartTokenizer.from_pretrained(args.model_type)
    # collate_fn = partial(collate_mp, pad_token_id=tok.pad_token_id, is_test=True)
    # test_set = LSR_MWF_Dataset(f"./data/{args.dataset}/{args.datatype}/test", args.model_type, is_test=True, max_len=512,
    #                        is_sorted=False, max_num=args.max_num, is_untok=True, total_len=args.total_len,
    #                        is_pegasus=args.is_pegasus)
    # batch_size = 8
    # dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=collate_fn)
    # build models
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    model = CombinationModel(BartConfig())
    if args.cuda:
        model = model.cuda()

    # model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt), map_location=f'cuda:{args.gpuid[0]}'))
    device = f'cuda:{args.gpuid[0]}'
    model.eval()
    # 输出模型的参数量
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    MB_params = params / 1024 / 1024
    print("test model parameters: ", MB_params, "MB")
    model_name = args.model_pt.split("/")[0]

    def mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    print("model_name:", model_name)
    root_dir = "./result/%s" % model_name
    mkdir(root_dir)
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

    if args.do_generation:
        print("start generation ...")
        # evaluate the model as a generator
        rouge1, rouge2, rougeLsum = 0, 0, 0
        tokenizer = tok
        count = 1
        bsz = 1
        total_num = len(os.listdir(f"./data/{args.dataset}/{args.datatype}/cnndm_test"))
        with open(f'./data/{args.dataset}/{args.datatype}/cnndm_test.source', 'r',
                  encoding='utf-8') as source, open(os.path.join(root_dir, "test.out"), 'w', encoding='utf-8') as fout:
            sline = source.readline().strip()
            slines = [sline]
            for sline in tqdm(source, total=total_num):
                if count % bsz == 0:
                    with torch.no_grad():
                        dct = tokenizer.batch_encode_plus(slines, max_length=args.total_len, return_tensors="pt",
                                                          pad_to_max_length=True, truncation=True)
                        summaries = model.generate_t(
                            input_ids=dct["input_ids"].to(device),
                            attention_mask=dct["attention_mask"].to(device),
                            max_length=args.gen_max_len + 2,
                            # +2 from original because we start at step=1 and stop before max_length
                            min_length=args.gen_min_len + 1,  # +1 from original because we start at step=1
                            no_repeat_ngram_size=3,
                            # num_beams=args.num_beams,
                            num_beams=args.num_beams,
                            length_penalty=args.length_penalty,
                            early_stopping=True,
                        )
                        dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g
                               in summaries]
                    for hypothesis in dec:
                        hypothesis = hypothesis.replace("\n", " ")
                        fout.write(hypothesis + '\n')
                        fout.flush()
                    slines = []
                sline = sline.strip()
                if len(sline) == 0:
                    sline = " "
                slines.append(sline)
                count += 1
            print("There's still something left over")
            if slines != []:
                with torch.no_grad():
                    dct = tokenizer.batch_encode_plus(slines, max_length=args.total_len, return_tensors="pt",
                                                      pad_to_max_length=True, truncation=True)
                    summaries = model.generate_t(
                        input_ids=dct["input_ids"].to(device),
                        attention_mask=dct["attention_mask"].to(device),
                        max_length=args.gen_max_len + 2,
                        # +2 from original because we start at step=1 and stop before max_length
                        min_length=args.gen_min_len + 1,  # +1 from original because we start at step=1
                        no_repeat_ngram_size=3,
                        num_beams=args.num_beams,
                        length_penalty=args.length_penalty,
                        early_stopping=True,
                    )
                    dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                           summaries]
                    for hypothesis in dec:
                        hypothesis = hypothesis.replace("\n", " ")
                        fout.write(hypothesis + '\n')
                        fout.flush()

        # calculate rouge score
        def process(x):
            return sent_tokenize(" ".join(word_tokenize(x.strip())))

        with open(os.path.join(root_dir, "test.out"), "r", encoding="utf-8") as fout, open(
                f'./data/{args.dataset}/{args.datatype}/cnndm_test.target', "r", encoding="utf-8") as target:
            for (hyp, ref) in zip(fout, target):
                hyp = hyp.strip()
                ref = ref.strip()
                hyp = process(hyp)
                ref = process(ref)
                score = rouge_scorer.score("\n".join(ref), "\n".join(hyp))
                rouge1 += score["rouge1"].fmeasure
                rouge2 += score["rouge2"].fmeasure
                rougeLsum += score["rougeLsum"].fmeasure
            rouge1 = rouge1 / total_num
            rouge2 = rouge2 / total_num
            rougeLsum = rougeLsum / total_num
            print("evaluation rouge1: %.6f, rouge2: %.6f, rougeL: %.6f" % (rouge1, rouge2, rougeLsum))


def main(args):
    # set env
    if len(args.gpuid) > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = f'{args.port}'
        mp.spawn(run, args=(args,), nprocs=len(args.gpuid), join=True)
    else:
        run(0, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--cuda", action="store_true", default=True, help="use cuda")
    parser.add_argument("--gpuid", nargs='+', type=int, default=[0], help="gpu ids")
    parser.add_argument("-e", "--evaluate", action="store_true", help="evaluate model")
    parser.add_argument("-r", "--do_reranking", action="store_true", help="do reranking evaluation")
    parser.add_argument("-g", "--do_generation", action="store_true", help="do generation evaluation")
    parser.add_argument("-l", "--log", action="store_true", help="logging")
    parser.add_argument("-p", "--port", type=int, default=8080, help="port")
    parser.add_argument("--model_pt", default="", type=str, help="model path")
    parser.add_argument("--config", default="", type=str, help="config path")
    args = parser.parse_args()
    # args.model_pt = "24-07-10-0/model_generation.bin"
    # args.config = "cnndm"
    # args.evaluate = False
    # args.log = False
    # args.do_generation = True

    if args.cuda is False:
        if args.evaluate:
            evaluation(args)
        else:
            main(args)
    else:
        if args.evaluate:
            with torch.cuda.device(args.gpuid[0]):
                evaluation(args)
        elif len(args.gpuid) == 1:
            with torch.cuda.device(args.gpuid[0]):
                main(args)
        else:
            main(args)
