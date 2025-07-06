import json
import argparse
from itertools import product
import onnx
import math
import os.path
import pandas as pd

CPU_EP = 'CPUExecutionProvider'
PIM_EP = 'AIMExecutionProvider'

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    # parser.add_argument('-m', '--model', required=True, type=str, help="onnx model path")
    args = parser.parse_args(argv)
    return args


def load_profile_json(json_file):
    with open(json_file, "r") as f:
        json_data = json.load(f)
    # assert isinstance(json_data, list)
    return json_data


def get_tensor_shape_and_dtype(tensor_name, graph):
    # Search in input tensors
    for input_tensor in graph.input:
        if input_tensor.name == tensor_name:
            if input_tensor.type.HasField('tensor_type'):
                shape = input_tensor.type.tensor_type.shape
                dtype = input_tensor.type.tensor_type.elem_type
                return shape, dtype
    # Search in value_info (for intermediate values and outputs)
    for value_info in graph.value_info:
        if value_info.name == tensor_name:
            if value_info.type.HasField('tensor_type'):
                shape = value_info.type.tensor_type.shape
                dtype = value_info.type.tensor_type.elem_type
                return shape, dtype
    # Search in graph outputs (e.g., final outputs)
    for output_tensor in graph.output:
        if output_tensor.name == tensor_name:
            if output_tensor.type.HasField('tensor_type'):
                shape = output_tensor.type.tensor_type.shape
                dtype = output_tensor.type.tensor_type.elem_type
                return shape, dtype
    # Search in initializers (constant tensors like weights and biases)
    for initializer in graph.initializer:
        if initializer.name == tensor_name:
            shape = initializer.dims
            dtype = initializer.data_type
            return shape, dtype
    return None, None


if __name__ == '__main__':
    args = parse_arguments()
    
    # model = onnx.load(args.model, load_external_data=False)
    model_int8 = onnx.load("./onnx_bert_int8/model.onnx", load_external_data=False)
    model_fp32 = onnx.load("./onnx_bert_fp32/model.onnx", load_external_data=False)    
    node_count = [{}, {}]
    graph = model_int8.graph
    for i, node in enumerate(graph.node):
        if node.op_type in node_count[0]:
            node_count[0][node.op_type] += 1
        else:
            node_count[0][node.op_type] = 1
            
    graph = model_fp32.graph
    for i, node in enumerate(graph.node):
        if node.op_type in node_count[1]:
            node_count[1][node.op_type] += 1
        else:
            node_count[1][node.op_type] = 1
    for node in node_count[0]:
        print(f"{node}, {node_count[0][node]}, {node_count[1][node] if node in node_count[1] else 0}")
