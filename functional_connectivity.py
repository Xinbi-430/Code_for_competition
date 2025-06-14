import numpy as np
from scipy.signal import hilbert
import torch
from utils.utils import top_k_sparsify, matrix_to_edge_index_attr  # 确保你正确导入
# 或者直接定义在当前文件中也可以

def compute_plv(eeg_data):
    # print("[DEBUG] Entered compute_plv")

    if np.any(np.isnan(eeg_data)) or np.any(np.isinf(eeg_data)):
        raise ValueError("输入 EEG 数据包含 NaN 或 Inf")

    if np.any(np.std(eeg_data, axis=1) == 0):
        raise ValueError("某些通道为全 0 或常数，PLV 计算会出错")
    n_channels = eeg_data.shape[0]
    plv_matrix = np.zeros((n_channels, n_channels))
    analytic_signal = hilbert(eeg_data)
    phase_data = np.angle(analytic_signal)
    for i in range(n_channels):
        for j in range(n_channels):
            phase_diff = phase_data[i] - phase_data[j]
            plv_matrix[i, j] = np.abs(np.sum(np.exp(1j * phase_diff))) / eeg_data.shape[1]
    return plv_matrix

def matrix_to_edge_index_attr(sparse_matrix):
    row, col = np.nonzero(sparse_matrix)
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(sparse_matrix[row, col], dtype=torch.float)
    return edge_index, edge_attr

def build_dynamic_graph(eeg_data, top_k=5):
    # 步骤1：计算功能连接矩阵（如 PLV）
    plv_matrix = compute_plv(eeg_data)
    
    # 步骤2：进行 Top-K 稀疏化
    sparse_matrix = top_k_sparsify(plv_matrix, k=top_k, symmetric=True)
    if np.any(np.isnan(sparse_matrix)):
        print("[ERROR] NaN detected in sparse_matrix after sparsify")
    # 步骤3：转换为 GNN 所需格式
    edge_index, edge_attr = matrix_to_edge_index_attr(sparse_matrix)
    if torch.isnan(edge_attr).any():
        print("[ERROR] NaN in edge_attr before training")

    return edge_index.numpy(), edge_attr.numpy()
