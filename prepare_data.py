import pickle as cPickle
from scipy import signal
from sklearn.decomposition import FastICA
from train.train_model import *
from scipy.signal import resample
import scipy.signal
from torch.utils.data import Dataset
import os
import numpy as np
import h5py
import concurrent.futures

    # x_tensor: (sample, channel, datapoint(feature)) type = torch.tensor
    # y_tensor: (sample,) type = torch.tensor
class eegDataset(Dataset):
    def __init__(self, x_tensor, y_tensor, graph_list=None):
        self.x = x_tensor
        self.y = y_tensor
        self.graph = graph_list
        assert self.x.size(0) == self.y.size(0)

    def __getitem__(self, index):
     if self.graph is None:
        return self.x[index], self.y[index]
     else:
         edge_index = torch.tensor(self.graph[index]['edge_index'], dtype=torch.long)
         edge_attr = torch.tensor(self.graph[index]['edge_attr'], dtype=torch.float32)
         return self.x[index], self.y[index], edge_index, edge_attr

    def __len__(self):
        return len(self.y)



class PrepareData:
    def __init__(self, args, external_data=None):
        # init all the parameters here
        # arg contains parameter settings
        self.args = args
        self.external_data = external_data  # ✅ 新增
        self.data = None
        self.label = None
        self.model = None
        self.data_path = args.data_path
        self.label_type = args.label_type
        self.dataset = args.dataset
        if self.dataset == 'MEEG':
            self.original_order = ['Fp1', 'Fp2', 'AF3', 'AF4', 'Fz', 'F3', 'F4', 'F7', 'F8',
                                   'FC1', 'FC2', 'FC5', 'FC6', 'Cz', 'C3', 'C4', 'T7', 'T8',
                                   'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4', 'P7', 'P8',
                                   'PO3', 'PO4', 'Oz', 'O1', 'O2']
        else:
            self.original_order = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3',
                                   'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8',
                                   'CP6',
                                   'CP2', 'P4', 'P8', 'PO4', 'O2']
        self.graph_fro_DEAP = [['Fp1', 'AF3'], ['Fp2', 'AF4'], ['F3', 'F7'], ['F4', 'F8'],
                               ['Fz'],
                               ['FC5', 'FC1'], ['FC6', 'FC2'], ['C3', 'Cz', 'C4'], ['CP5', 'CP1', 'CP2', 'CP6'],
                               ['P7', 'P3', 'Pz', 'P4', 'P8'], ['PO3', 'PO4'], ['O1', 'Oz', 'O2'],
                               ['T7'], ['T8']]
        self.graph_gen_DEAP = [['Fp1', 'Fp2'], ['AF3', 'AF4'], ['F3', 'F7', 'Fz', 'F4', 'F8'],
                               ['FC5', 'FC1', 'FC6', 'FC2'], ['C3', 'Cz', 'C4'], ['CP5', 'CP1', 'CP2', 'CP6'],
                               ['P7', 'P3', 'Pz', 'P4', 'P8'], ['PO3', 'PO4'], ['O1', 'Oz', 'O2'],
                               ['T7'], ['T8']]
        self.graph_hem_DEAP = [['Fp1', 'AF3'], ['Fp2', 'AF4'], ['F3', 'F7'], ['F4', 'F8'],
                               ['Fz', 'Cz', 'Pz', 'Oz'],
                               ['FC5', 'FC1'], ['FC6', 'FC2'], ['C3'], ['C4'], ['CP5', 'CP1'], ['CP2', 'CP6'],
                               ['P7', 'P3'], ['P4', 'P8'], ['PO3', 'O1'], ['PO4', 'O2'], ['T7'], ['T8']]
        self.graph_type = args.graph_type

    def run(self, subject_list, split, expand):
        """
        Parameters
        ----------
        subject_list: the subjects need to be processed
        split: (bool) whether to split one trial's data into shorter segment
        expand: (bool) whether to add an empty dimension for CNN

        Returns
        -------
        The processed data will be saved './data_<data_format>_<dataset>_<label_type>/sub0.hdf'
        """
        for sub in subject_list:
            data_, label_ = self.load_data_per_subject(sub)
            # select label type here
            label_ = self.label_selection(label_)

            data_, label_, graph_list = self.preprocess_data(data=data_, label=label_, split=split, expand=expand)

            print('Data and label prepared!')
            print('sample_' + str(sub + 1) + '.dat')
            print('data:' + str(data_.shape) + ' label:' + str(label_.shape))
            print('----------------------')
            self.graph_list = graph_list  # 存为属性，供 save() 使用
            self.save(data_, label_, sub)

        self.args.sampling_rate = self.args.target_rate

    def load_data_per_subject(self, sub):
        """
        This function loads the target subject's original file
        Parameters
        ----------
        sub: which subject to load

        Returns
        -------
        data: (40, 32, 7680) label: (40, 4)
        """
        if self.external_data and sub in self.external_data:
            print(f"[INFO] Using external preprocessed data for subject {sub}")
            data, label = self.external_data[sub]
            return data, label

        # 否则使用内部默认加载方式
        sub += 1
        if self.dataset == 'MEEG':
            sub_code = str('sample_' + str(sub) + '.dat')
        elif self.dataset == 'DEAP':
            if sub < 10:
                sub_code = str('s0' + str(sub) + '.dat')
            else:
                sub_code = str('s' + str(sub) + '.dat')

        subject_path = os.path.join(self.data_path, sub_code)
        subject = cPickle.load(open(subject_path, 'rb'), encoding='latin1')
        label = subject['labels']
        data = subject['data']
        data = self.reorder_channel(data=data, graph=self.graph_type)
        print(f"[INFO] Using internal loading for subject {sub}")
        print('data:' + str(data.shape) + ' label:' + str(label.shape))
        return data, label

    def reorder_channel(self, data, graph):
        """
        This function reorder the channel according to different graph designs
        Parameters
        ----------
        data: (trial, channel, data)
        graph: graph type

        Returns
        -------
        reordered data: (trial, channel, data)
        """
        if graph == 'fro':
            graph_idx = self.graph_fro_DEAP
        elif graph == 'gen':
            graph_idx = self.graph_gen_DEAP
        elif graph == 'hem':
            graph_idx = self.graph_hem_DEAP
        elif graph == 'BL':
            graph_idx = self.original_order

        idx = []
        if graph in ['BL']:
            for chan in graph_idx:
                idx.append(self.original_order.index(chan))
        else:
            num_chan_local_graph = []
            for i in range(len(graph_idx)):
                num_chan_local_graph.append(len(graph_idx[i]))
                for chan in graph_idx[i]:
                    idx.append(self.original_order.index(chan))

            # save the number of channels in local graph for building the LGG model in utils.py
            dataset = h5py.File('num_chan_local_graph_{}.hdf'.format(graph), 'w')
            dataset['data'] = num_chan_local_graph
            dataset.close()
        return data[:, idx, :]

    def label_selection(self, label):
        """
        This function: 1. selects which dimension of labels to use
                       2. create binary label
        Parameters
        ----------
        label: (trial, 4)

        Returns
        -------
        label: (trial,)
        """
        if self.label_type == 'V':
            label = label[:, 0]
        elif self.label_type == 'A':
            label = label[:, 1]
        elif self.label_type == 'D':
            label = label[:, 2]
        elif self.label_type == 'L':
            label = label[:, 3]
        if self.dataset == 'DEAP':
            label = np.where(label <= 5, 0, label)
            label = np.where(label > 5, 1, label)
        return label

    def save(self, data, label, sub):
    # """
    # This function saves the processed data and graph structure into target folder.
    # """
        import pickle
        save_path = os.getcwd()
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, self.args.dataset, self.args.label_type)
        save_path = os.path.join(save_path, data_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 构造 .hdf 文件完整路径
        file_name = f"sub{sub}.hdf"
        hdf_path = os.path.join(save_path, file_name)

        # 保存 EEG 数据和标签
        dataset = h5py.File(hdf_path, 'w')
        dataset['data'] = data
        dataset['label'] = label
        dataset.close()

        # 保存图结构为 .pkl 文件（与 .hdf 同目录）
        graph_path = os.path.join(save_path, f"graphs_sub{sub}.pkl")
        with open(graph_path, "wb") as f:
            pickle.dump(self.graph_list, f)


    # 预处理数据
    def preprocess_data(self, data, label, split, expand):
        """
        This function preprocess the data
        Parameters
        ----------
        data: (trial, channel, data)
        label: (trial,)
        split: (bool) whether to split one trial's data into shorter segment
        expand: (bool) whether to add an empty dimension for CNN
        Returns
        -------
        preprocessed
        data: (trial, channel, target_length)
        label: (trial,)
        """
        if expand:
            # expand one dimension for deep learning(CNNs)
            data = np.expand_dims(data, axis=-3)

        if self.args.dataset == 'MEEG':
            data = self.bandpass_filter(data=data, lowcut=self.args.bandpass[0], highcut=self.args.bandpass[1],
                                        fs=self.args.sampling_rate, order=5)
            data = self.notch_filter(data=data, fs=self.args.sampling_rate, Q=50)

        if self.args.sampling_rate != self.args.target_rate:
            data, label = self.downsample_data(
                data=data, label=label, sampling_rate=self.args.sampling_rate,
                target_rate=self.args.target_rate)

        if split:
            data, label, graph_list = self.split(data, label, self.args.segment, self.args.overlap, self.args.target_rate)


        return data, label, graph_list

    def split(self, data, label, segment_length, overlap, sampling_rate):
        """
        This function split one trial's data into shorter segments
        Parameters
        ----------
        data: (trial, f, channel, data)
        label: (trial,)
        segment_length: how long each segment is (e.g. 1s, 2s,...)
        overlap: overlap rate
        sampling_rate: sampling rate

        Returns
        -------
        data:(tiral, num_segment, f, channel, segment_legnth)
        label:(trial, num_segment,)
        """
        data_shape = data.shape
        step = int(segment_length * sampling_rate * (1 - overlap))
        data_segment = sampling_rate * segment_length
        data_split = []
        graph_list = []  # 存储所有 segment 的图结构

        number_segment = int((data_shape[-1] - data_segment) // step)

        # Step 1: 切分数据段
        for i in range(number_segment + 1):
            segment = data[:, :, :, (i * step):(i * step + data_segment)]  # shape: (trial, 1, channel, segment_len)
            data_split.append(segment)

        # Step 2: 构建图结构
        from utils.functional_connectivity import build_dynamic_graph

        for i in range(number_segment + 1):
            segment = data_split[i]  # shape: (trial, 1, channel, segment_len)

            trial_graphs = []
            for trial in range(segment.shape[0]):
                eeg_trial = segment[trial, 0]  # shape: (channel, time)
                edge_index, edge_attr = build_dynamic_graph(eeg_trial, top_k=8)  # 可调节稀疏度
                trial_graphs.append({'edge_index': edge_index, 'edge_attr': edge_attr})

            graph_list.append(trial_graphs)



        data_split_array = np.stack(data_split, axis=1)
        label = np.stack([np.repeat(label[i], int(number_segment + 1)) for i in range(len(label))], axis=0)
        print("The data and label are split: Data shape:" + str(data_split_array.shape) + " Label:" + str(
            label.shape))
        data = data_split_array
        assert len(data) == len(label)
        if self.args.model == 'DGCNN':
            data = self.extract_features(data, sampling_rate)
        return data, label, graph_list

    def extract_features(self, data, sfreq):
        # data 的形状应该是 (20, 14, 1, 32, 800)
        num_trials, num_segments, _, num_channels, num_samples = data.shape
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 14),
            'beta': (14, 31),
            'gamma': (31, 50)
        }
        # 初始化输出数据结构
        features = np.zeros((num_trials, num_segments, 1, num_channels, len(bands)))

        # 遍历每个频带并应用滤波器
        for i, (band_name, band_range) in enumerate(bands.items()):
            for trial in range(num_trials):
                for segment in range(num_segments):
                    for channel in range(num_channels):
                        # 获取单个通道数据
                        channel_data = data[trial, segment, 0, channel, :]
                        # 应用带通滤波器
                        filtered_data = self.bandpass_filter(channel_data, band_range[0], band_range[1], sfreq)
                        # 计算差分熵
                        features[trial, segment, 0, channel, i] = np.log(np.var(filtered_data) + 1e-8)

        # 输出数据结构应该为 (20, 14, 1, 32, 5)
        return features

    def downsample_data(self, data, label, sampling_rate, target_rate):
        """
        This function downsample the data to target length
        Parameters
        ----------
        data: (trial, channel, data)
        label: (trial,)
        sampling_rate: original sampling rate
        target_rate: target sampling rate
        Returns
        -------
        downsampled data: (trial, channel, target_length)
        label: (trial,)
        """
        target_length = int(data.shape[-1] * target_rate / sampling_rate)
        downsampled_data = resample(data, target_length, axis=-1)
        return downsampled_data, label

    # 巴特沃斯带通滤波器
    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        """
        This function applies bandpass filter to the data
        Parameters
        ----------
        data: (trial, channel, data)
        lowcut: low cut frequency
        highcut: high cut frequency
        fs: sampling rate
        order: filter order

        Returns
        -------
        filtered data: (trial, channel, data)
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='bandpass')
        filtered_data = signal.filtfilt(b, a, data, axis=-1)
        return filtered_data

    # 频率为50hz的陷波滤波器
    def notch_filter(self, data, fs, Q=50):
        """
        This function applies notch filter to the data
        Parameters
        ----------
        data: (trial, channel, data)
        fs: sampling rate
        Q: Q value for notch filter

        Returns
        -------
        filtered data: (trial, channel, data)
        """
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                self.notch_filter_per_channel(data[i, j, :], fs, Q)
        return data

    def notch_filter_per_channel(self, param, fs, Q):
        """
        This function applies notch filter to one channel
        Parameters
        ----------
        param: (data,)
        fs: sampling rate
        Q: Q value for notch filter

        Returns
        -------
        filtered data: (data,)
        """
        w0 = Q / fs
        b, a = signal.iirnotch(w0, Q)
        param = signal.filtfilt(b, a, param)
        return param
