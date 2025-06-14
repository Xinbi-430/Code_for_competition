import copy
import datetime
import pickle


from config.config import *
from sklearn.model_selection import KFold
from train.train_model import *
from utils.utils import *


ROOT = os.getcwd()
_, os.environ['CUDA_VISIBLE_DEVICES'] = set_config()


class CrossValidation:
    def __init__(self, args):
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        # Log the results per subject
        result_path = os.path.join(args.save_path, 'result')
        ensure_path(result_path)
        self.text_file = os.path.join(result_path,
                                      "results_{}.txt".format(args.dataset))
        file = open(self.text_file, 'a')
        file.write("\n" + str(datetime.datetime.now()) +
                   "\nTrain:Parameter setting for " + str(args.model) + ' on ' + str(args.dataset) +
                   "\n1)number_class:" + str(args.num_class) +
                   "\n2)random_seed:" + str(args.random_seed) +
                   "\n3)learning_rate:" + str(args.learning_rate) +
                   "\n4)training_rate:" + str(args.training_rate) +
                   "\n5)pool:" + str(args.pool) +
                   "\n6)num_epochs:" + str(args.max_epoch) +
                   "\n7)batch_size:" + str(args.batch_size) +
                   "\n8)dropout:" + str(args.dropout) +
                   "\n9)hidden_node:" + str(args.hidden) +
                   "\n10)input_shape:" + str(args.input_shape) +
                   "\n11)class:" + str(args.label_type) +
                   "\n12)T:" + str(args.T) +
                   "\n13)graph-type:" + str(args.graph_type) +
                   "\n14)patient:" + str(args.patient) +
                   "\n15)patient-cmb:" + str(args.patient_cmb) +
                   "\n16)max-epoch-cmb:" + str(args.max_epoch_cmb) +
                   "\n17)fold:" + str(args.fold) +
                   "\n18)model:" + str(args.model) +
                   "\n19)data-path:" + str(args.data_path) +
                   "\n20)balance:" + str(args.balance) +
                   "\n21)bandpass:" + str(args.bandpass) +
                   "\n22)dataset:" + str(args.dataset) +
                    "\n23)overlap:" + str(args.overlap) +
                   '\n')
        file.close()

    def load_per_subject(self, sub):
        """
        load data for sub
        param sub: which subject's data to load
        return: data and label
        """
        save_path = os.getcwd()
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, self.args.dataset, self.args.label_type)
        sub_code = 'sub' + str(sub) + '.hdf'
        path = os.path.join(save_path, data_type, sub_code)
        dataset = h5py.File(path, 'r')
        data = np.array(dataset['data'])
        label = np.array(dataset['label'])
        print('>>> Data:{} Label:{}'.format(data.shape, label.shape))
        graph_code = 'graphs_sub' + str(sub) + '.pkl'
        graph_path = os.path.join(save_path, data_type, graph_code)
        with open(graph_path, 'rb') as f:
            graph_list = pickle.load(f)

        return data, label, graph_list

    def prepare_data(self, idx_train, idx_test, data, label, graph):
        data_train = data[idx_train]
        label_train = label[idx_train]
        data_test = data[idx_test]
        label_test = label[idx_test]

        # reshape
        data_train = np.concatenate(data_train, axis=0)
        label_train = np.concatenate(label_train, axis=0)
        data_test = np.concatenate(data_test, axis=0)
        label_test = np.concatenate(label_test, axis=0)

        # ✅ graph 是 flat_graph（总长度 = trial数 × segment数），我们从 idx_train 切出 trial index，变成 segment-level index
        seg_per_trial = data.shape[1]  # e.g. 14
        segment_idx_train = np.concatenate([np.arange(i * seg_per_trial, (i + 1) * seg_per_trial) for i in idx_train])
        segment_idx_test = np.concatenate([np.arange(i * seg_per_trial, (i + 1) * seg_per_trial) for i in idx_test])

        graph_train = [graph[i] for i in segment_idx_train]
        graph_test = [graph[i] for i in segment_idx_test]

        # Normalize
        data_train, data_test = self.normalize(train=data_train, test=data_test)

        # Torch tensor
        data_train = torch.from_numpy(data_train).float()
        label_train = torch.from_numpy(label_train).long()
        data_test = torch.from_numpy(data_test).float()
        label_test = torch.from_numpy(label_test).long()

        return data_train, label_train, data_test, label_test, graph_train, graph_test


    def normalize(self, train, test):
        """
        this function do standard normalization for EEG channel by channel
        :param train: training data (sample, 1, chan, datapoint)
        :param test: testing data (sample, 1, chan, datapoint)
        :return: normalized training and testing data
        """
        # data: sample x 1 x channel x data
        for channel in range(train.shape[2]):
            mean = np.mean(train[:, :, channel, :])
            std = np.std(train[:, :, channel, :])
            train[:, :, channel, :] = (train[:, :, channel, :] - mean) / std
            test[:, :, channel, :] = (test[:, :, channel, :] - mean) / std
        return train, test

    def split_balance_class(self, data, label, train_rate, random):
        """
        Get the validation set using the same percentage of the two classe samples
        param data: training data (segment, 1, channel, data)
        param label: (segments,)
        param train_rate: the percentage of trianing data
        param random: bool, whether to shuffle the training data before get the validation data
        return: data_trian, label_train, and data_val, label_val
        """
        # Data dimension: segment x 1 x channel x data
        # Label dimension: segment x 1
        np.random.seed(0)
        # data : segments x 1 x channel x data
        # label : segments

        index_0 = np.where(label == 0)[0]
        index_1 = np.where(label == 1)[0]

        # for class 0
        index_random_0 = copy.deepcopy(index_0)

        # for class 1
        index_random_1 = copy.deepcopy(index_1)

        if random:
            np.random.shuffle(index_random_0)
            np.random.shuffle(index_random_1)

        index_train = np.concatenate((index_random_0[:int(len(index_random_0) * train_rate)],
                                      index_random_1[:int(len(index_random_1) * train_rate)]),
                                     axis=0)
        index_val = np.concatenate((index_random_0[int(len(index_random_0) * train_rate):],
                                    index_random_1[int(len(index_random_1) * train_rate):]),
                                   axis=0)

        # get validation
        val = data[index_val]
        val_label = label[index_val]

        train = data[index_train]
        train_label = label[index_train]

        return train, train_label, val, val_label

    def n_fold_CV(self, subject, fold, reproduce):
        tta, tva, ttf, tvf = [], [], [], []

        for sub in subject:
            data, label, graph = self.load_per_subject(sub)

            # ✅ 展平 graph
            flat_graph = [g for trial_graph in graph for g in trial_graph]

            va_val = Averager()
            vf_val = Averager()
            preds, acts = [], []

            kf = KFold(n_splits=fold, shuffle=True)
            for idx_fold, (idx_train, idx_test) in enumerate(kf.split(data)):
                print('Outer loop: {}-fold-CV Fold:{}'.format(fold, idx_fold))

                # ✅ 准备数据 + 图结构
                data_train, label_train, data_test, label_test, graph_train, graph_test = self.prepare_data(
                    idx_train=idx_train, idx_test=idx_test, data=data, label=label, graph=flat_graph)

                if self.args.balance:
                    data_train, label_train, data_val, label_val = self.split_balance_class(
                        data=data_train, label=label_train, train_rate=self.args.training_rate, random=True)
                else:
                    raise ValueError("You need to implement the case where balance=False")

                if reproduce:
                    acc_test, pred, act = test(
                        args=self.args,
                        data=data_test,
                        label=label_test,
                        graph_list=graph_test,  # ✅ 添加图结构
                        reproduce=reproduce,
                        subject=sub,
                        fold=idx_fold
                    )
                    acc_val = 0
                    f1_val = 0
                else:
                    print('Training:', data_train.size(), label_train.size())
                    print('Test:', data_test.size(), label_test.size())

                    acc_val, f1_val = self.first_stage(
                        data=data_train,
                        label=label_train,
                        graph=graph_train,
                        subject=sub,
                        fold=idx_fold
                    )

                    combine_train(
                        args=self.args,
                        data_train=data_train,
                        label_train=label_train,
                        graph_list_train=graph_train,  # ✅ 添加图结构
                        subject=sub,
                        fold=idx_fold,
                        target_acc=1
                    )

                    acc_test, pred, act = test(
                        args=self.args,
                        data=data_test,
                        label=label_test,
                        graph_list=graph_test,  # ✅ 添加图结构
                        reproduce=reproduce,
                        subject=sub,
                        fold=idx_fold
                    )

                va_val.add(acc_val)
                vf_val.add(f1_val)
                preds.extend(pred)
                acts.extend(act)

            tva.append(va_val.item())
            tvf.append(vf_val.item())
            acc, f1, _ = get_metrics(y_pred=preds, y_true=acts)
            tta.append(acc)
            ttf.append(f1)
            result = 'sub {}: total test accuracy {}, f1: {}'.format(sub, acc, f1)
            self.log2txt(result)

        # ✅ 汇总与打印
        tta, ttf, tva, tvf = map(np.array, (tta, ttf, tva, tvf))
        print('Final: test mean ACC:{} std:{}'.format(tta.mean(), tta.std()))
        print('Final: test mean F1:{}'.format(ttf.mean()))
        print('Final: val mean ACC:{} std:{}'.format(tva.mean(), tva.std()))
        print('Final: val mean F1:{}'.format(tvf.mean()))
        results = ('test mAcc={} std:{} mF1={} std:{} \n'
                   'val mAcc={} F1={}').format(
            tta.mean(), tta.std(), ttf.mean(), ttf.std(), tva.mean(), tvf.mean())
        self.log2txt(results)

    def first_stage(self, data, label, graph, subject, fold):
        """
        Inner loop: do 3-fold on training set to select best model
        """
        kf = KFold(n_splits=3, shuffle=True)
        va = Averager()
        vf = Averager()
        maxAcc = 0.0

        for i, (idx_train, idx_val) in enumerate(kf.split(data)):
            print('Inner 3-fold-CV Fold:{}'.format(i))
            data_train, label_train = data[idx_train], label[idx_train]
            data_val, label_val = data[idx_val], label[idx_val]
            graph_list_train = [graph[i] for i in idx_train]
            graph_list_val = [graph[i] for i in idx_val]

            acc_val, F1_val = train(args=self.args,
                        data_train=data_train,
                        label_train=label_train,
                        graph_list_train=graph_list_train,
                        data_val=data_val,
                        label_val=label_val,
                        graph_list_val=graph_list_val,
                        subject=subject,
                        fold=fold)



            va.add(acc_val)
            vf.add(F1_val)

            if acc_val >= maxAcc:
                maxAcc = acc_val
                # Save best candidate model
                old_name = os.path.join(self.args.save_path, 'candidate.pth')
                new_name = os.path.join(self.args.save_path, 'max-acc.pth')
                if os.path.exists(new_name):
                    os.remove(new_name)
                os.rename(old_name, new_name)
                print('New max ACC model saved, with the val ACC being:{}'.format(acc_val))

        return va.item(), vf.item()


    def log2txt(self, content):
        """
        This function log the content to results.txt
        param content: string, the content to log.
        """
        file = open(self.text_file, 'a')
        file.write(str(content) + '\n')
        file.close()




