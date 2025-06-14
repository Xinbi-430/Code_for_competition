import torch
import numpy as np
from config.config import *
from models.models import AT_DGNN  # 根据你使用的模型名字修改
from train.prepare_data import eeg_preprocessing, build_graph  # 预处理和图构建
import argparse
import os


def load_model(args, device):
    model = AT_DGNN(args).to(device)
    model.load_state_dict(torch.load(args.load_path_final, map_location=device))
    model.eval()
    return model


def infer_single_sample(model, args, raw_eeg):
    """
    raw_eeg: numpy array of shape (channels, samples)
    """
    # Step 1: EEG 预处理
    preprocessed = eeg_preprocessing(raw_eeg, args)

    # Step 2: 构建 PLV 图结构（需要实现 PLV+TopK 建图逻辑）
    graph_data = build_graph(preprocessed, args)  # 返回应为 PyG 的 Data 对象

    # Step 3: 前向传播
    with torch.no_grad():
        graph_data = graph_data.to(args.device)
        output = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.batch)
        pred_label = torch.argmax(output, dim=1).item()

    return pred_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-path-final', type=str, default='./save/final_model.pth')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--input-shape', type=str, default="1,32,800")
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--target-rate', type=int, default=200)
    # 添加其他必要参数...
    args = parser.parse_args()
    args.input_shape = tuple(map(int, args.input_shape.split(',')))

    # 加载模型
    model = load_model(args, args.device)

    # 模拟一段 EEG 数据（真实使用中需要替换为采集的数据）
    fake_eeg = np.random.randn(args.channels, args.input_shape[-1])

    label_map = {0: "HVHA", 1: "HVLA", 2: "LVHA", 3: "LVLA"}
    result = infer_single_sample(model, args, fake_eeg)
    print(f"预测情绪象限为：{label_map[result]}")
