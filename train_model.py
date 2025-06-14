import sys
import numpy as np
import torch

from utils.utils import safe_tensor_clone
from utils.utils import *
from utils.preprocessing import preprocess_eeg
from config.config import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import roc_curve, auc
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import time

def print_graph_sparsity(edge_index, num_nodes, epoch=None, sample_id=None):
    num_edges = edge_index.shape[1]
    max_edges = num_nodes * (num_nodes - 1)
    sparsity = 1 - num_edges / max_edges
    print(f"[Graph Sparsity] Edges: {num_edges}, Nodes: {num_nodes}, Sparsity: {sparsity:.2%}")
    if epoch is not None and sample_id is not None:
        with open("sparsity_log.txt", "a") as f:
            f.write(f"Epoch {epoch}, Sample {sample_id}, Edges {num_edges}, Sparsity {sparsity:.2%}\n")



class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        logp = F.log_softmax(inputs, dim=1)
        ce_loss = F.nll_loss(logp, targets, reduction='none')
        p = torch.exp(-ce_loss)
        loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()


CUDA = torch.cuda.is_available()
_, os.environ['CUDA_VISIBLE_DEVICES'] = set_config()


def train_one_epoch(data_loader, net, loss_fn, optimizer, epoch=None):
    import time
    net.train()
    tl = Averager()
    pred_train = []
    act_train = []

    is_gnn = hasattr(net, 'accepts_graph') and net.accepts_graph

    start_time = time.time()  # ⏱ 开始计时

    for batch in data_loader:
        if is_gnn:
            x_batch, y_batch, edge_index_list, edge_attr_list = batch
        else:
            x_batch, y_batch = batch
            edge_index_list = [None] * len(x_batch)
            edge_attr_list = [None] * len(x_batch)

        batch_size = len(x_batch)

        for i in range(batch_size):
            x = x_batch[i].unsqueeze(0).float()
            y = torch.tensor([y_batch[i]]).long()

            if edge_index_list[i] is not None:
                edge_index = safe_tensor_clone(edge_index_list[i], torch.long)
                edge_attr = safe_tensor_clone(edge_attr_list[i], torch.float)

                if isinstance(edge_attr_list[i], torch.Tensor):
                    edge_attr = edge_attr_list[i].clone().detach().float()
                else:
                    edge_attr = torch.tensor(edge_attr_list[i]).float()

                if i == 0 and epoch == 1:
                    num_nodes = x.shape[2]
                    print_graph_sparsity(edge_index, num_nodes, epoch=epoch, sample_id=i)  # ✅ 打印+写入日志
            else:
                edge_index = edge_attr = None

            if CUDA:
                x = x.cuda()
                y = y.cuda()
                if edge_index is not None:
                    edge_index = edge_index.cuda()
                    edge_attr = edge_attr.cuda()

            if is_gnn:
                out = net(x, edge_index=edge_index, edge_attr=edge_attr)
            else:
                out = net(x)

            loss = loss_fn(out, y)
            _, pred = torch.max(out, 1)

            pred_train.extend(pred.cpu().data.tolist())
            act_train.extend(y.cpu().data.tolist())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            tl.add(loss.item())

    epoch_time = time.time() - start_time  # ⏱ 本轮结束
    print(f"[Timing] Epoch {epoch} took {epoch_time:.2f} seconds.")

    # ✅ 显示显存使用
    if CUDA:
        mem_used = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[GPU] Peak memory used this epoch: {mem_used:.2f} MB")

    return tl.item(), pred_train, act_train



def predict(data_loader, net, loss_fn, return_prob=False):
    net.eval()
    pred_val = []
    act_val = []
    vl = Averager()
    soft_scores = []

    is_gnn = hasattr(net, 'accepts_graph') and net.accepts_graph

    with torch.no_grad():
        for batch in data_loader:
            if is_gnn:
                x_batch, y_batch, edge_index_list, edge_attr_list = batch
            else:
                x_batch, y_batch = batch
                edge_index_list = [None] * len(x_batch)
                edge_attr_list = [None] * len(x_batch)

            batch_size = len(x_batch)

            for i in range(batch_size):
                x = x_batch[i].unsqueeze(0).float()
                y = torch.tensor([y_batch[i]]).long()

                if edge_index_list[i] is not None:
                    edge_index = safe_tensor_clone(edge_index_list[i], torch.long)
                    edge_attr = safe_tensor_clone(edge_attr_list[i], torch.float)
                else:
                    edge_index = edge_attr = None

                if CUDA:
                    x = x.cuda()
                    y = y.cuda()
                    if edge_index is not None:
                        edge_index = edge_index.cuda()
                        edge_attr = edge_attr.cuda()

                if is_gnn:
                    out = net(x, edge_index=edge_index, edge_attr=edge_attr)
                else:
                    out = net(x)

                loss = loss_fn(out, y)
                vl.add(loss.item())

                _, pred = torch.max(out, 1)
                pred_val.append(pred.item())
                act_val.append(y.item())

                if return_prob:
                    prob = F.softmax(out, dim=1).cpu().numpy()[0][1]
                    soft_scores.append(prob)

    if return_prob:
        return vl.item(), pred_val, act_val, soft_scores
    else:
        return vl.item(), pred_val, act_val








def set_up(args):
    set_gpu(args.gpu)
    ensure_path(args.save_path)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True


def train(args, data_train, label_train, graph_list_train,
          data_val, label_val, graph_list_val,
          subject, fold):
    seed_all(args.random_seed)
    save_name = '_sub' + str(subject) + '_fold' + str(fold)
    set_up(args)

    train_loader = get_dataloader(data_train, label_train, args.batch_size, graph_list_train)
    val_loader = get_dataloader(data_val, label_val, args.batch_size, graph_list_val)

    model = get_model(args)
    print("Using model:", type(model).__name__)

    if CUDA:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.use_focal:
        loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
    elif args.LS:
        loss_fn = LabelSmoothing(args.LS_rate)
    else:
        loss_fn = nn.CrossEntropyLoss()

    def save_model(name):
        previous_model = os.path.join(args.save_path, '{}.pth'.format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)
        torch.save(model.state_dict(), os.path.join(args.save_path, '{}.pth'.format(name)))

    trlog = {
        'args': vars(args),
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'max_acc': 0.0,
        'F1': 0.0
    }

    timer = Timer()
    patient = args.patient
    counter = 0

    for epoch in range(1, args.max_epoch + 1):

        loss_train, pred_train, act_train = train_one_epoch(
            data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer, epoch=epoch)

        acc_train, f1_train, _ = get_metrics(y_pred=pred_train, y_true=act_train)
        print('epoch {}, for the train set, loss={:.4f} acc={:.4f} f1={:.4f}'
              .format(epoch, loss_train, acc_train, f1_train))

        # 修改这里：predict 返回 y_score（概率）
        loss_val, pred_val, act_val, y_score = predict(
            data_loader=val_loader, net=model, loss_fn=loss_fn, return_prob=True
        )
        acc_val, f1_val, cm, auc_val = get_metrics(pred_val, act_val, return_all=True)

        print('epoch {}, for the validation set, loss={:.4f} acc={:.4f} f1={:.4f} auc={:.4f}'.
              format(epoch, loss_val, acc_val, f1_val, auc_val))

        # 可视化 ROC 曲线
        roc_path = os.path.join(args.save_path, 'roc_sub{}_fold{}.png'.format(subject, fold))
        plot_roc_curve(act_val, y_score, save_path=roc_path)

        if acc_val >= trlog['max_acc']:
            trlog['max_acc'] = acc_val
            trlog['F1'] = f1_val
            save_model('candidate')
            counter = 0
        else:
            counter += 1
            if counter >= patient:
                print('early stopping')
                break

        trlog['train_loss'].append(loss_train)
        trlog['train_acc'].append(acc_train)
        trlog['val_loss'].append(loss_val)
        trlog['val_acc'].append(acc_val)

        print('ETA:{}/{} SUB:{} FOLD:{}'.format(
            timer.measure(), timer.measure(epoch / args.max_epoch), subject, fold))

    # 保存训练日志
    save_name = 'trlog' + save_name
    experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
    save_path = os.path.join(args.save_path, experiment_setting, 'log_train')
    ensure_path(save_path)
    torch.save(trlog, os.path.join(save_path, save_name))

    return trlog['max_acc'], trlog['F1']


def test(args, data, label, graph_list, reproduce, subject, fold):
    set_up(args)
    seed_all(args.random_seed)

    # ✅ 支持图结构数据的 dataloader
    test_loader = get_dataloader(data, label, args.batch_size, graph_list)

    model = get_model(args)

    if CUDA:
        model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()

    if reproduce:
        model_name_reproduce = 'sub' + str(subject) + '_fold' + str(fold) + '.pth'
        data_type = 'model_{}_{}'.format(args.data_format, args.label_type)
        experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
        load_path_final = os.path.join(args.save_path, experiment_setting, data_type, model_name_reproduce)
        model.load_state_dict(torch.load(load_path_final))
    else:
        model.load_state_dict(torch.load(args.load_path_final))

    # ✅ 执行推理
    loss, pred, act = predict(
        data_loader=test_loader, net=model, loss_fn=loss_fn
    )

    # ✅ 获取评估指标
    acc, f1, cm = get_metrics(y_pred=pred, y_true=act, return_all=True)

    # ✅ 打印结果
    print('>>> Test:  loss={:.4f} acc={:.4f} f1={:.4f}'.format(loss, acc, f1))

    # ✅ 保存混淆矩阵图像
    plot_confusion_matrix(
        cm,
        class_names=['Positive', 'Negative'],  # 替换成你自己的类别
        save_path=os.path.join(args.save_path, 'confusion_matrix_sub{}_fold{}.png'.format(subject, fold))
    )

    return acc, pred, act


def combine_train(args, data_train, label_train, graph_list_train, subject, fold, target_acc):
    save_name = '_sub' + str(subject) + '_fold' + str(fold)
    set_up(args)
    seed_all(args.random_seed)

    train_loader = get_dataloader(data_train, label_train, args.batch_size, graph_list_train)

    model = get_model(args)
    if CUDA:
        model = model.cuda()
    model.load_state_dict(torch.load(args.load_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate * 1e-1)

    if args.LS:
        loss_fn = LabelSmoothing(args.LS_rate)
    else:
        loss_fn = nn.CrossEntropyLoss()

    def save_model(name):
        previous_model = os.path.join(args.save_path, '{}.pth'.format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)
        torch.save(model.state_dict(), os.path.join(args.save_path, '{}.pth'.format(name)))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['F1'] = 0.0

    timer = Timer()

    for epoch in range(1, args.max_epoch_cmb + 1):
        loss, pred, act = train_one_epoch(
            data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer
        )
        acc, f1, _ = get_metrics(y_pred=pred, y_true=act)
        print('Stage 2 : epoch {}, for train set loss={:.4f} acc={:.4f} f1={:.4f}'
              .format(epoch, loss, acc, f1))

        if acc >= target_acc or epoch == args.max_epoch_cmb:
            print('early stopping!')
            save_model('final_model')
            # save model here for reproduce
            model_name_reproduce = 'sub' + str(subject) + '_fold' + str(fold) + '.pth'
            data_type = 'model_{}_{}'.format(args.data_format, args.label_type)
            experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
            save_path = os.path.join(args.save_path, experiment_setting, data_type)
            ensure_path(save_path)
            model_name_reproduce = os.path.join(save_path, model_name_reproduce)
            torch.save(model.state_dict(), model_name_reproduce)
            break

        trlog['train_loss'].append(loss)
        trlog['train_acc'].append(acc)

        print('ETA:{}/{} SUB:{} TRIAL:{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch),
                                                 subject, fold))

    save_name = 'trlog_comb' + save_name
    experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
    save_path = os.path.join(args.save_path, experiment_setting, 'log_train_cmb')
    ensure_path(save_path)
    torch.save(trlog, os.path.join(save_path, save_name))

import os
import numpy as np
from utils.preprocessing import preprocess_eeg  # 确保你有这个函数

def load_dat(path, num_channels):
    data = np.fromfile(path, dtype=np.float32)
    print(f"[DEBUG] Loaded {data.shape[0]} float32 values from: {path}")

    # 修正：截断数据长度，去掉不能整除部分
    valid_len = (data.shape[0] // num_channels) * num_channels
    if valid_len != data.shape[0]:
        print(f"[WARNING] Trimming extra {data.shape[0] - valid_len} values to match {num_channels} channels")

    data = data[:valid_len]  # 截断
    num_timepoints = valid_len // num_channels
    data = data.reshape((num_channels, num_timepoints))
    return data


def run_batch_preprocessing(num_subjects=5, num_channels=32):
    preprocessed_data = {}

    for i in range(1, num_subjects + 1):
        subject_id = str(i)
        path = f"D:\MEEG\sample_{subject_id}.dat"

        print(f"\n[INFO] Processing subject {subject_id}...")

        if not os.path.exists(path):
            print(f"[WARNING] File not found: {path}")
            continue

        eeg_raw = load_dat(path, num_channels=num_channels)
        if eeg_raw is None:
            continue

        print("[INFO] EEG preprocessing...")
        eeg_clean = preprocess_eeg(eeg_raw, fs=1000, target_fs=200)

        print(f"[SUCCESS] Preprocessing completed for subject {subject_id}. Shape: {eeg_clean.shape}")
        preprocessed_data[subject_id] = eeg_clean

    return preprocessed_data


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    绘制并保存混淆矩阵图像
    cm: 混淆矩阵 (from sklearn.metrics.confusion_matrix)
    class_names: 类别标签名称列表
    save_path: 图像保存路径
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true, y_score, save_path=None):
    if np.isnan(y_score).any():
        warnings.warn("ROC Curve skipped due to NaN in scores", UndefinedMetricWarning)
        return
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC curve (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)
    plt.close()






