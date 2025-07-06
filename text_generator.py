import os
import shutil
import argparse
import onnx
import numpy
import json
import gc
from transformers import AutoTokenizer

import ctypes
libaim = ctypes.cdll.LoadLibrary('/usr/local/lib/libaim.so')


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, type=str, help="onnx model path")
    parser.add_argument('-t', '--text', required=False, type=str, default=None, help="input text")
    parser.add_argument('-s', '--max_len', required=False, type=int, default=32, help="maximum sequence length")
    parser.add_argument('-r', '--repetition', required=False, type=int, default=1, help="num of repetition")
    parser.add_argument('-d', '--device_map', required=False, action='store_true', default=False, help="device map")
    parser.add_argument('-p', '--parallel', required=False, type=int, default=0, help="use parallel executor")
    parser.add_argument('-g', '--use_pim', required=False, action='store_true', default=False, help="use PIM")
    parser.add_argument('-o', '--optlevel', required=False, type=int, default=0, help="use specifically optimized model")
    parser.add_argument('--profile', required=False, action='store_true', default=False, help="profile results")
    # parser.add_argument('-l', '--load', required=False, type=str, default=None, help="load_list")
    parser.add_argument("--thread_num", required=False, type=int, default=1, help="number of threads to use")
    parser.add_argument('--decoupled', required=False, action='store_true', default=False, help="use Decoupled-PIM")
    parser.add_argument('--basic_optimization',
                        required=False,
                        action='store_true',
                        default=True,
                        help="Enable only basic graph optimizations. By default, all optimizations are enabled in OnnxRuntime")
    parser.add_argument('--kernel_time_only',
                        required=False,
                        action='store_true',
                        default=False,
                        help="Only include the kernel time and no fence time")
    parser.add_argument('-v', '--verbose', required=False, action='store_true', default=False)
    parser.add_argument('--samples',
                        required=False,
                        type=int,
                        default=1,
                        help="number of samples to test. Set it large enough to reduce the variance of performance result.")
    args = parser.parse_args(argv)
    return args


def text_to_token(text, tokenizer):
    token_ids = None
    token_ids = tokenizer(text, return_tensors="np", padding=True)
    return token_ids
    
    
def token_to_text(token_ids, valid_lengths, tokenizer):
    texts = [
        tokenizer.decode(ids[: int(length)], skip_special_tokens=True)
        for ids, length in zip(token_ids, valid_lengths)
    ]
    return texts
    

def sample_from_logits(logits, temperature=1.0, top_k=0):
    # Temperature scaling
    if temperature != 1.0:
        logits = logits / temperature
    # Optional top-k filtering
    if top_k > 0:
        # Sort logits for top-k filtering
        top_k_indices = numpy.argsort(logits, axis=-1)[:, -top_k:]
        mask = numpy.full_like(logits, -numpy.inf)
        for i in range(logits.shape[0]):
            mask[i, top_k_indices[i]] = logits[i, top_k_indices[i]]
        logits = mask
    # Convert logits to probabilities using softmax
    exp_logits = numpy.exp(logits - numpy.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / numpy.sum(exp_logits, axis=-1, keepdims=True)
    # Sample the next token from the probability distribution for each batch
    next_token = numpy.array([numpy.random.choice(probs.shape[-1], p=probs[i]) for i in range(probs.shape[0])])
    return next_token


def run_profile(device, onnx_model_path, use_pim, optlevel, enable_pim_kv, basic_optimization, thread_num,
                input, batch_size, max_len, tokenizer, eos_token, repetition, align_map, enable_profiling, parallelism, partition_map=None, device_map=None):
    from pim_benchmark_helper import create_onnxruntime_session

    model_path = None
    if optlevel == 0:
        model_path = onnx_model_path + "/optimized_model.onnx"
    else:
        model_path = onnx_model_path + "/step" + str(optlevel) + "_model.onnx"
    session = create_onnxruntime_session(device,
                                         model_path,
                                         use_pim,
                                         partition_map=partition_map,
                                         device_map=device_map,
                                         align_map=align_map,
                                         parallelism=parallelism,
                                         enable_all_optimization=not basic_optimization,
                                         num_threads=thread_num,
                                         enable_profiling=enable_profiling)
    total_ids = []
    valid_lengths = []
    profile_data = []
    for r in range(repetition):
        if r != 0 and enable_profiling == True:
            session.start_profiling()
        print("#################### TEXT GENERATION ####################")
        given_ids = input['input_ids']
        generated_ids = None
        input_ids = input['input_ids']
        attention_mask = input['attention_mask']
        position_ids = numpy.arange(input_ids.shape[1])[numpy.newaxis, :]
        
        finished = numpy.ones((batch_size, 1))
        input_seq_len = input_ids.shape[1]
        
        model_name = onnx_model_path.split('/')[-1]
        config_json = onnx_model_path + '/' + model_name + '_config.json'
        with open(config_json, "r") as f:
            model_config = json.load(f)
        num_layers = None
        num_attention_heads = None 
        hidden_size = None
        index_seq_len = None
        mha_model_list = ["llama2", "gpt2", "gpt2-large", "opt", "opt-1b", "opt-3b", "bloom", "bloom-1b", "phi-1b"]
        mqa_model_list = ["falcon"]
        gqa_model_list = ["falcon3-1b", "llama3-1b"] 
        # MHA (Multi-Head Attention)
        if model_name in mha_model_list:
            print("RUNNING", model_name.upper(), "/", "INPUT_SEQ_LEN =", input_seq_len, "BATCH_NUM =", batch_size)
            num_layers = model_config["num_layers"]
            num_attention_heads = model_config["num_attention_heads"]
            hidden_size = model_config["hidden_size"]
        # MQA (Multi-Query Attention)
        elif model_name in mqa_model_list:
            print("RUNNING", model_name.upper(), "/", "INPUT_SEQ_LEN =", input_seq_len, "BATCH_NUM =", batch_size)
            num_layers = model_config["num_layers"]
            num_attention_heads = 1
            hidden_size = model_config["hidden_size"] // model_config["num_attention_heads"]
        # GQA (Global-Query Attention)
        elif model_name in gqa_model_list:
            print("RUNNING", model_name.upper(), "/", "INPUT_SEQ_LEN =", input_seq_len, "BATCH_NUM =", batch_size)
            num_layers = model_config["num_layers"]
            num_attention_heads = 8
            num_in_group = 2
            hidden_size = model_config["hidden_size"] // model_config["num_attention_heads"] * num_attention_heads // num_in_group
        
        if model_name in gqa_model_list:
            past_shape = [2, batch_size, num_attention_heads // num_in_group, 0, hidden_size // (num_attention_heads // num_in_group)]
            index_seq_len = 3
        else:
            past_shape = [2, batch_size, num_attention_heads, 0, hidden_size // num_attention_heads]
            index_seq_len = 3
        past_key_values = []
        for i in range(0, num_layers):
            past_key_values.append(numpy.zeros(past_shape, dtype=numpy.float32))
        
        if use_pim is True and enable_pim_kv is True:
            past_shape[index_seq_len] += input_seq_len - 1
            libaim.InitAimCache(2*num_layers)
        
        for iter in range(max_len):    # iteration of step 1~4 
            print(">> ITERATION", iter+1, "/", max_len, end='\r', flush=True)
            # print(">> ITERATION", iter+1, "/", max_len)
            # if iter == 0:
            #     token = tokenizer.decode(given_ids[0])
            #     print(token, end="")
            # 1. Prepare inputs
            ort_inputs = None
            if model_name in ['opt', 'bloom', 'bloom-1b']:
                ort_inputs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }
            else:
                ort_inputs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'position_ids': position_ids
                }
                
            if past_key_values is not None:
                for i, past_key_value in enumerate(past_key_values):
                    ort_inputs[f'past_key_values.{i}.key'] = past_key_value[0]
                    # print(past_key_value[0].shape)
                    # if model_name in ['bloom', 'bloom-1b'] and iter == 0:
                    #     ort_inputs[f'past_key_values.{i}.value'] = numpy.swapaxes(past_key_value[1], -1, -2)
                    #     print(past_key_value[1].shape)
                    # else:
                    #     ort_inputs[f'past_key_values.{i}.value'] = past_key_value[1]
                    ort_inputs[f'past_key_values.{i}.value'] = past_key_value[1]
            # 2. Run iteration
            outputs = session.run(None, ort_inputs)
            # 3. Post-process outputs
            logits = outputs[0]
            past_key_values = []
            if use_pim is True and enable_pim_kv is True:
                past_shape[index_seq_len] += 1
                for i in range(0, num_layers):
                    past_key_values.append(numpy.zeros(past_shape, dtype=numpy.float32))
            else:
                for i in range(0, num_layers):
                    past_key_values.append([outputs[2*i+1], outputs[2*i+2]])
            last_token_logits = logits[:, -1, :]
            # print(last_token_logits)
            next_token_id = sample_from_logits(last_token_logits, temperature=1.0, top_k=1)
            # token = tokenizer.decode(next_token_id)
            # print(token, end="")
            next_token_id = numpy.reshape(next_token_id, (-1, 1))
            for b in range(batch_size):
                if next_token_id[b][0] == eos_token:
                    finished[b][1] = 0
            if iter != 0:
                generated_ids = numpy.concatenate([generated_ids, next_token_id], axis=1)
            else:
                generated_ids = next_token_id
            # 4. Prepare next input in autoregressive way
            input_ids = next_token_id
            attention_mask = numpy.concatenate([attention_mask, numpy.logical_and(numpy.ones((batch_size, 1), dtype=numpy.int64),finished)], axis=1)
            valid_lengths.append(numpy.sum(attention_mask, axis=1))
            position_ids = numpy.reshape(valid_lengths[-1]-1, (-1, 1))
        print("#########################################################")
        print("NUMBER OF GENERATED TOKENS:", max_len)
        if enable_profiling == True:
            profile_data.append(session.end_profiling())
            print("PROFILE_JSON:", profile_data[-1])
        print("#########################################################")
        total_ids.append(numpy.concatenate([given_ids, generated_ids], axis=1))
        if use_pim is True and enable_pim_kv is True:
            libaim.DeleteAimCache()
    return total_ids, valid_lengths, profile_data


def pim_helper(args):
    from onnx_model import OnnxModel
    onnx_model = None
    if args.optlevel == 0:
        onnx_model = OnnxModel(onnx.load('./models/'+ args.model + '/optimized_model.onnx', load_external_data=False))
    else:
        onnx_model = OnnxModel(onnx.load('./models/'+ args.model + '/step' + str(args.optlevel) + '_model.onnx', load_external_data=False))
    graph = onnx_model.model.graph
    align_map = {"mm_b" : [], "mm_y": [], "decoupled_registry": [], "is_trans_B": [], "is_pim_load": [],
                 "is_pim_0": [], "is_pim_1": [], "is_pim_2": [], "is_pim_3": [], "is_pim_4": [], "is_pim_5": [], "is_pim_6": [], "is_pim_7": []}
    for node in graph.node:
        if node.op_type == "MatMul":
            align_map["mm_b"].append(node.input[1])
    for node in graph.node:
        if "Gemm" in node.name:
            need_align = True
            for attr in node.attribute:
                if attr.name == "transB" and onnx.helper.get_attribute_value(attr) == 1:
                    need_align = False
                    align_map["is_trans_B"].append(node.input[1])
            if need_align:
                align_map["mm_b"].append(node.input[1])
                align_map["mm_y"].append(node.input[2])
        elif node.op_type == "FusedNode_LayerNorm_Mul_Add_Gemm_Relu_Gemm_Add":
            for inp in node.input:
                if inp.find("fc1.weight") != -1 or inp.find("fc2.weight") != -1:
                    align_map["is_trans_B"].append(inp)
    # if args.load is not None:
    #     with open(args.load, 'r') as f:
    #         align_map["is_pim_load"] = json.load(f)
    if args.device_map is True:
        initializer_device_map = None
        f_name = './models/' + args.model + '/step' + str(args.optlevel) + '_initializer_device_map.json'
        if os.path.isfile(f_name):
            with open(f_name, 'r') as f:
                initializer_device_map = json.load(f)
        for key in initializer_device_map:
            align_map[key] = initializer_device_map[key]
    return align_map


def run_cpu(args, all_inputs, batch_size, eos_token, tokenizer):
    output, valid_length, profile_data = run_profile("cpu", "./models/" + args.model,   False, 0,               False,          args.basic_optimization, args.thread_num, all_inputs, batch_size, args.max_len, tokenizer, eos_token, args.repetition, None, args.profile, args.parallel, {}, {})
    return output, valid_length, profile_data


def run_pim(args, all_inputs, batch_size, eos_token, tokenizer):
    align_map = pim_helper(args)
    # print(align_map)
    partition_map = None
    device_map    = {}
    if args.optlevel == 0:
        f_name = './models/' + args.model + '/partition_map.json'
        with open(f_name, 'r') as f:
            partition_map = json.load(f)
    elif args.optlevel != 0:
        f_name = './models/' + args.model + '/step' + str(args.optlevel) + '_partition_map.json'
        with open(f_name, 'r') as f:
            partition_map = json.load(f)
    # print(partition_map)
    if args.device_map is True:
        f_name = './models/' + args.model + '/step' + str(args.optlevel) + '_device_map.json'
        device_map_str = None
        if os.path.isfile(f_name):
            with open(f_name, 'r') as f:
                device_map_str = json.load(f)
        for id_str in device_map_str:
            device_map[int(id_str)] = device_map_str[id_str]  
    # print(device_map)
    enable_pim_kv = True
    if args.optlevel < 2:   # Enforce KV cache not allocated on AiM for optlevel 0 & 1
        enable_pim_kv = False
    output, valid_length, profile_data = run_profile("pim", "./models/" + args.model,   True, args.optlevel,    enable_pim_kv,  args.basic_optimization, args.thread_num, all_inputs, batch_size, args.max_len, tokenizer, eos_token, args.repetition, align_map, args.profile, args.parallel, partition_map, device_map)
    return output, valid_length, profile_data


def run(args):
    num_threads = args.thread_num if args.thread_num > 0 else 1

    # Set OMP environment variable before importing onnxruntime. Needed for cpu only, and no impact for onnxruntime-gpu package.
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)

    tokenizer = None
    if args.model == 'falcon3-1b':
        # tokenizer = AutoTokenizer.from_pretrained('tiiuae/Falcon3-1B-Base')
        tokenizer = AutoTokenizer.from_pretrained('falcon')
    elif args.model == 'gpt2-large':
        tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2-large')
    elif args.model == 'bloom-1b':
        tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-1b7')
    elif args.model == 'opt-1b':
        tokenizer = AutoTokenizer.from_pretrained('facebook/opt-1.3b')
    elif args.model == 'phi-1b':
        tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-1_5')
    elif args.model == 'opt-3b':
        tokenizer = AutoTokenizer.from_pretrained('facebook/opt-2.7b')
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = 0
    tokenizer.add_special_tokens({'additional_special_tokens': ['\n']})
    all_inputs = None
    profile_data = None
    
    if args.text == None:
        ##### multi-batch
        # text = [
        #     '''The founder of Meta AI is Mark Zuckerberg. ''',
        #     '''Processing-In-Memory is being focussed in the computing industry. ''',
        #     '''It is hard to decide menu for dinner today.''',
        #     '''There are some pros and cons of electric vehicles being popular. '''
        #     ]
        ##### single-batch
        # text = ['''Welcome to the world of Large Language Model!''']
        # text = ['''"Hello, there!", Obi-wan Kenobi said. General Grievous was facing Obi-wan with hundreds of battle droids.''']
        # text = ['''Processing-In-Memory is a solution for resolving memory bottleneck.''']
        # text = ['''Performance analysis of a computing device is crucial for writing a technical report.''']
        # text = ['''Welcome to our Compiler & Microarchitecture Lab of Korea University, Seoul, Korea.''']
        # text = ['''What is the motivation of developing Processing-In-Memory?''']
        # text = ['''PIM(Processing In Memory) has recently been a tempting solution for training and inferencing AI(artificial intelligence) models. Dealing AI models with traditional architectures such as GPU(Graphic Processing Unit) suffers from memory bottleneck, since computation speed exceeds the speed of memory access, and requires lots of electricity. However in PIM architecture, retrieving data from DRAM array cells to processing units requires little time, as the processing units are located in DRAM. Therefore, memory bottleneck diminishes and resource utilization of GPU also increases. Recently many DRAM makers and startups are doing their job to commercialize PIM with passion.''']
        ##### 1
        text = ['''I''']
        ##### 16
        # text = ['''I I I I I I I I I I I I I I I I''']
        ##### 64
        # text = ['''I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I''']
        ##### 992
        # text = ['''I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I''']
        ##### 1024
        # text = ['''I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I''']
    else:
        text = [args.text]
    batch_size = len(text)
    all_inputs = text_to_token(text, tokenizer)

    ######################
    #         CPU        # 
    ######################
    if args.use_pim == False:
        print("RUNNING CPU")
        output_cpu, valid_lengths, profile_data = run_cpu(args, all_inputs, batch_size, tokenizer.eos_token, tokenizer)
        # print("OUTPUT check:")
        # print("CPU:")
        # print(output_cpu, valid_lengths)
        output_text = token_to_text(output_cpu[0], valid_lengths[-1], tokenizer)
        for i in range(batch_size):
            print("BATCH ", i+1, ") ", output_text[i], sep='')

    ######################
    #         PIM        #
    ######################
    elif args.use_pim == True:
        print("RUNNING PIM")
        output_pim, valid_lengths, profile_data = run_pim(args, all_inputs, batch_size, tokenizer.eos_token, tokenizer)
        # print("OUTPUT check:")
        # print("PIM:")
        # print(output_pim)
        output_text = token_to_text(output_pim[0], valid_lengths[-1], tokenizer)
        for i in range(batch_size):
            print("BATCH ", i+1, ") ", output_text[i], sep='')

    return profile_data

if __name__ == '__main__':
    args = parse_arguments()
    # print("Arguments", args)
    profile_data = run(args)
    
    if args.profile == True:
        for profile_json in profile_data:
            print(profile_json)
            src = os.path.expanduser("./" + profile_json)
            env = None
            if profile_json[0:3] == "cpu":
                env = "_cpu"
            elif profile_json[0:3] == "pim":
                env = "_cpu+aim"
            dst = os.path.expanduser("./data/data_json/" + args.model + "/o"+ str(args.optlevel) + env)
            if not os.path.exists(dst):
                os.makedirs(dst)
            shutil.move(src, dst)
