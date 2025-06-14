本项目为 面向数字疗法的音乐情绪建模与脑区功能连接分析系统 的核心代码，基于改进的 AT-DGNN 模型，用于 EEG 情绪识别、脑区功能连接建模及数字疗法相关实验。

📁 项目目录结构（要进入编辑模式才能看到！）

AT-DGNN-MAIN/
├── config/
│   └── config.py                  # 参数配置文件
├── data_eeg_MEEG_A/               # 存放 MEEG 数据集的目录（需用户自行准备数据）
├── docs/                          # 项目文档目录
├── example/                       # 示例代码或实验脚本
├── models/
│   ├── models.py                  # 模型定义（如 AT-DGNN 等）
│   └── networks.py                # 神经网络相关模块
├── save/                          # 保存训练模型权重和中间结果
├── train/
│   ├── cross_validation.py        # 交叉验证模块
│   ├── prepare_data.py            # 数据预处理和准备
│   └── train_model.py             # 模型训练过程
├── utils/
│   ├── functional_connectivity.py # 功能连接计算（如 PLV）
│   ├── preprocessing.py           # 数据预处理辅助函数
│   └── utils.py                   # 工具函数
├── main.py                        # 主入口文件，用于启动训练或评估
├── num_chan_local_graph_fro.hdf   # 局部图结构信息文件
├── .gitignore                     # Git 忽略文件配置
└── LICENSE                        # 项目许可证

⚙️ 环境依赖

建议使用 Python 3.9 环境。可通过以下命令安装项目依赖：

pip install -r requirements.txt


1️⃣ 确保已准备好数据集（MEEG 数据：https://drive.google.com/drive/folders/1Tabw5sjpFiwy88yP-C-LnunNFrrre9AR）

2️⃣ 配置参数：
修改 config/config.py 中的参数以适配实验需求。

3️⃣ 启动训练：

python main.py

💡 主要功能模块说明
	•	config/：参数配置模块。
	•	train/：包含数据准备、训练、交叉验证等代码。
	•	models/：核心模型架构定义。
	•	utils/：功能连接、预处理、工具函数。
	•	save/：训练输出（如权重、日志）。
	•	main.py：主程序入口，控制训练与测试。

📝 数据说明

本项目基于 MEEG 数据集，音乐诱导情绪 EEG 数据。请根据实际研究需要准备数据，并置于 相应目录。

📌 联系与贡献

欢迎提出建议与改进。
