"""
Step_04_local_model_trainning.py

============================================================
Step 04：客户端本地模型初始化与交互式训练
============================================================

【本文件职责】

本步骤是整个 FedCALI / FedShield 流程中最核心的训练步骤之一。

它负责：

1. 根据数据集类型构建对应的 5 层 CNN；
2. 在第一轮训练前初始化统一的初始模型参数；
3. 为每个客户端维护自己的本地模型；
4. 根据上游架构分配模块给出的共享层数，将本地模型拆分为：
       Shared Architecture
       Personalized Architecture
5. 执行交互式本地训练：
       先固定共享架构，训练个性化架构；
       再固定个性化架构，训练共享架构；
6. 输出训练后的完整模型参数；
7. 显式提取共享层参数和个性化层参数，
   供后续 Step 05 / Step 06 继续处理。

============================================================
【两个数据集下的 5 层模型结构】
============================================================

虽然 CIFAR-10 和 MNIST 都使用 5 层模型，并且统一抽象为：

    layer1
    layer2
    layer3
    layer4
    layer5

但由于输入尺寸和输入通道数不同，具体网络参数存在细微差异。

------------------------------------------------------------
1. CIFAR-10：FiveLayerCNN_CIFAR10
------------------------------------------------------------

输入：
    3 × 32 × 32

网络：

    layer1:
        Conv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=5,
            padding=2
        )
        ReLU
        AvgPool2d(kernel_size=2, stride=2)

    layer2:
        Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5
        )
        ReLU
        AvgPool2d(kernel_size=2, stride=2)

    Flatten:
        16 × 6 × 6 = 576

    layer3:
        Linear(576, 120)
        ReLU

    layer4:
        Linear(120, 84)
        ReLU

    layer5:
        Linear(84, 10)

因此逻辑层映射为：

    layer1 -> conv1
    layer2 -> conv2
    layer3 -> fc1
    layer4 -> fc2
    layer5 -> fc3

------------------------------------------------------------
2. MNIST：FiveLayerCNN_MNIST
------------------------------------------------------------

输入：
    1 × 28 × 28

网络：

    layer1:
        Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            padding=2
        )
        ReLU
        AvgPool2d(kernel_size=2, stride=2)

    layer2:
        Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5
        )
        ReLU
        AvgPool2d(kernel_size=2, stride=2)

    Flatten:
        16 × 5 × 5 = 400

    layer3:
        Linear(400, 120)
        ReLU

    layer4:
        Linear(120, 84)
        ReLU

    layer5:
        Linear(84, 10)

逻辑层映射同样为：

    layer1 -> conv1
    layer2 -> conv2
    layer3 -> fc1
    layer4 -> fc2
    layer5 -> fc3

============================================================
【共享架构与个性化架构】
============================================================

Step 03 已经给出每个客户端的共享层数 m_k。

当前总模型层数固定为：

    L = 5

共享层数允许是：

    1, 2, 3, 4, 5

并且：

    Shared Layers
        = 前 m_k 层

    Personalized Layers
        = 剩余层

例如：

    m_k = 3

则：

    Shared Layers
        = [layer1, layer2, layer3]

    Personalized Layers
        = [layer4, layer5]

============================================================
【交互式本地训练】
============================================================

对于 K=1~4，本文件保留论文和原始代码中的“交互式训练”机制。

对于 K=5，由于不存在个性化层，交互式训练自然退化为普通的完整模型本地 SGD：

    Personalized Layers = []
    Shared Layers = [layer1, layer2, layer3, layer4, layer5]

此时不再执行“先个性化、后共享”的两阶段训练，
而是直接对全部 5 层参数执行本地随机梯度下降。

对于 K=1~4，每个客户端在一轮本地训练中依次执行：

阶段 1：训练个性化架构

    固定共享层参数：
        requires_grad = False

    开放个性化层参数：
        requires_grad = True

    使用客户端自己的本地训练数据优化个性化部分。

阶段 2：训练共享架构

    固定个性化层参数：
        requires_grad = False

    开放共享层参数：
        requires_grad = True

    继续使用同一客户端本地训练数据优化共享部分。

这样最终得到：

    Client k 的完整本地模型
        =
        训练后的 Shared Architecture
        +
        训练后的 Personalized Architecture

当 K = 5 时：

    Shared Architecture
        = 整个 5 层模型

    Personalized Architecture
        = 空

训练方式退化为：

    Full-model Local SGD

也就是对整个模型进行一次普通的本地 SGD 优化流程。

============================================================
【第一轮初始化】
============================================================

第一轮没有上一轮全局模型可以下发。

因此必须首先创建一个统一的初始模型：

    G^0

并让所有客户端从同一个初始模型参数开始。

本文件提供：

    initialize_global_model(...)

它会：

1. 根据 dataset_name 创建 CIFAR-10 或 MNIST 5 层 CNN；
2. 使用指定 seed 初始化模型；
3. 返回初始模型和其 state_dict。

后续每个客户端都从同一个初始 state_dict 创建自己的本地模型，
避免每个客户端各自随机初始化导致实验逻辑不一致。

在正式多轮 FL 中：

    Round 1:
        使用统一随机初始化的 G^0

    Round t > 1:
        共享层加载上一轮聚合得到的 global shared parameters；
        个性化层继续保留该客户端自己的上一轮个性化参数。

============================================================
【本步骤输入】
============================================================

正式完整流程中，Step 04 至少接收：

1. dataset_name
       "cifar10"
       或
       "mnist"

2. client_train_datasets
       来自 Step 02：

       label_result.client_train_datasets

       其中：
       - Label Flipping 客户端已经得到被修改后的训练标签；
       - 其他客户端仍然使用正常训练数据。

3. architecture_result
       来自任一兼容的架构分配模块：

       - Step_03_V_architecture_allocation.py（自适应架构）
       - Step_03_V_sp_consistent_architecture.py（统一架构）

       两者都统一提供：

       architecture_result.shared_layers[client_id]

       本 Step 04 不判断上游究竟来自哪一个 Step 03，
       只读取每个客户端最终的共享层数。

4. initial_global_state_dict
       第一轮：
           由 initialize_global_model() 生成

       后续轮：
           由上一轮 global model / aggregated shared parameters 提供

5. previous_client_state_dicts
       可选。

       第一轮没有；
       后续轮用于保留每个客户端自己的个性化参数。

============================================================
【本步骤输出——非常关键】
============================================================

本步骤最关键的输出不是 loss，也不是 accuracy，
而是：

    每个客户端训练完成后的模型参数。

这是后续安全流程真正操作的对象。

Step 05：分层模型攻击

    Gaussian / Sign Flipping

会直接修改本步骤输出的共享层模型参数。

例如：

    result.client_shared_parameters[client_id]["layer2"]

会成为模型攻击的直接输入。

------------------------------------------------------------

Step 06：参数掩码 / 加密

会继续对 Step 05 处理后的共享层模型参数执行：

    theta + xi
    theta - xi

或者后续正式设计中的密码学处理。

因此：

    Step 04 输出
        ↓
    Step 05 参数攻击
        ↓
    Step 06 参数掩码 / 加密
        ↓
    Step 07 双服务器安全计算
        ↓
    Step 08 Byzantine Detection
        ↓
    Step 09 Layer Filtering
        ↓
    Step 10 Secure Aggregation

所以本文件必须显式返回：

1. client_models
       每个客户端训练后的完整模型对象；

2. client_state_dicts
       每个客户端训练后的完整 state_dict；

3. client_shared_parameters
       每个客户端“共享层”的模型参数；

4. client_personalized_parameters
       每个客户端“个性化层”的模型参数；

5. shared_layer_names
       每个客户端实际共享哪些 layer；

6. personalized_layer_names
       每个客户端实际个性化哪些 layer；

7. training_metrics
       本轮个性化训练 loss、共享训练 loss 等辅助信息。

其中真正给 Step 05 / Step 06 使用的核心接口是：

    client_shared_parameters

============================================================
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 固定逻辑层定义
# ============================================================

ALL_LAYERS = [
    "layer1",
    "layer2",
    "layer3",
    "layer4",
    "layer5",
]

TOTAL_MODEL_LAYERS = 5
MAX_SHARED_LAYERS = 5


# ============================================================
# 模型定义
# ============================================================

class FiveLayerCNNBase(nn.Module):
    """
    CIFAR-10 / MNIST 5 层 CNN 的统一基类。

    子类只负责定义具体 conv / fc 参数。
    外部后续模块统一通过：

        layer1
        layer2
        layer3
        layer4
        layer5

    访问各逻辑层。
    """

    def get_layer_module(self, layer_name: str) -> nn.Module:
        mapping = {
            "layer1": self.conv1,
            "layer2": self.conv2,
            "layer3": self.fc1,
            "layer4": self.fc2,
            "layer5": self.fc3,
        }

        if layer_name not in mapping:
            raise ValueError(
                f"未知的层名称: {layer_name}"
            )

        return mapping[layer_name]

    def get_layer_params(
        self,
        layer_name: str,
    ) -> List[torch.nn.Parameter]:
        return list(
            self.get_layer_module(layer_name).parameters()
        )

    def get_layer_param_tensors(
        self,
        layer_name: str,
    ) -> List[torch.Tensor]:
        """
        返回某逻辑层的参数副本。

        注意：
        返回 clone，避免后续 Step 05 修改攻击参数时
        直接污染当前 model 对象。
        """
        return [
            param.detach().clone()
            for param in self.get_layer_params(layer_name)
        ]

    def set_layer_params(
        self,
        layer_name: str,
        params: List[torch.Tensor],
    ) -> None:
        target_params = self.get_layer_params(layer_name)

        if len(target_params) != len(params):
            raise ValueError(
                f"{layer_name} 参数数量不匹配："
                f"目标={len(target_params)}，输入={len(params)}"
            )

        with torch.no_grad():
            for target_param, new_param in zip(
                target_params,
                params,
            ):
                if target_param.shape != new_param.shape:
                    raise ValueError(
                        f"{layer_name} 参数形状不匹配："
                        f"目标={tuple(target_param.shape)}，"
                        f"输入={tuple(new_param.shape)}"
                    )

                target_param.copy_(
                    new_param.to(target_param.device)
                )

    def get_flattened_params(self) -> torch.Tensor:
        return torch.cat(
            [
                param.detach().reshape(-1)
                for param in self.parameters()
            ]
        )

    def _initialize_weights(self) -> None:
        """
        保留原始训练代码中的初始化策略：
        - Conv: Kaiming Normal
        - Linear: Normal(mean=0, std=0.01)
        - Linear bias: 0
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.constant_(
                        module.bias,
                        0,
                    )

            elif isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.01,
                )

                if module.bias is not None:
                    nn.init.constant_(
                        module.bias,
                        0,
                    )


class FiveLayerCNN_CIFAR10(FiveLayerCNNBase):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3,
            6,
            kernel_size=5,
            padding=2,
        )
        self.pool1 = nn.AvgPool2d(
            kernel_size=2,
            stride=2,
        )

        self.conv2 = nn.Conv2d(
            6,
            16,
            kernel_size=5,
        )
        self.pool2 = nn.AvgPool2d(
            kernel_size=2,
            stride=2,
        )

        self.fc1 = nn.Linear(
            16 * 6 * 6,
            120,
        )
        self.fc2 = nn.Linear(
            120,
            84,
        )
        self.fc3 = nn.Linear(
            84,
            num_classes,
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class FiveLayerCNN_MNIST(FiveLayerCNNBase):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.conv1 = nn.Conv2d(
            1,
            6,
            kernel_size=5,
            padding=2,
        )
        self.pool1 = nn.AvgPool2d(
            kernel_size=2,
            stride=2,
        )

        self.conv2 = nn.Conv2d(
            6,
            16,
            kernel_size=5,
        )
        self.pool2 = nn.AvgPool2d(
            kernel_size=2,
            stride=2,
        )

        self.fc1 = nn.Linear(
            16 * 5 * 5,
            120,
        )
        self.fc2 = nn.Linear(
            120,
            84,
        )
        self.fc3 = nn.Linear(
            84,
            num_classes,
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def build_model(
    dataset_name: str,
) -> FiveLayerCNNBase:
    """
    根据数据集创建对应的 5 层 CNN。
    """

    normalized_name = dataset_name.strip().lower()

    if normalized_name in {
        "cifar10",
        "cifar-10",
    }:
        return FiveLayerCNN_CIFAR10()

    if normalized_name == "mnist":
        return FiveLayerCNN_MNIST()

    raise ValueError(
        f"当前仅支持 CIFAR-10 和 MNIST，"
        f"收到 dataset_name={dataset_name!r}"
    )


# ============================================================
# 初始化
# ============================================================

def set_training_seed(
    seed: int,
) -> None:
    """
    设置本步骤需要的随机种子。

    正式 main.py 后续建议传入由 MASTER_SEED 派生出的 model_seed。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    深拷贝 state_dict 到 CPU。

    后续攻击 / 掩码模块拿到的是独立 Tensor，
    不会因为原模型继续变化而发生隐式修改。
    """
    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
    }


def initialize_global_model(
    dataset_name: str,
    seed: int = 42,
    device: str = "cpu",
) -> Tuple[FiveLayerCNNBase, Dict[str, torch.Tensor]]:
    """
    创建第一轮统一初始模型 G^0。

    所有客户端都应从该 state_dict 开始，
    而不是分别随机创建自己的模型。
    """

    set_training_seed(seed)

    model = build_model(dataset_name)
    model.to(torch.device(device))

    initial_state_dict = clone_state_dict(
        model.state_dict()
    )

    return model, initial_state_dict


# ============================================================
# 共享 / 个性化层划分
# ============================================================

def get_layer_partition(
    num_shared_layers: int,
) -> Tuple[List[str], List[str]]:
    """
    根据共享层数得到共享层和个性化层名称。

    允许 1~5：

        1~4 -> 个性化联邦学习划分
        5   -> 全模型共享，个性化层为空
    """

    if not 1 <= num_shared_layers <= MAX_SHARED_LAYERS:
        raise ValueError(
            f"共享层数必须位于 1~{MAX_SHARED_LAYERS}，"
            f"当前为 {num_shared_layers}"
        )

    shared_layers = ALL_LAYERS[:num_shared_layers]
    personalized_layers = ALL_LAYERS[
        num_shared_layers:
    ]

    return (
        shared_layers,
        personalized_layers,
    )


# ============================================================
# 输出数据结构
# ============================================================

@dataclass(frozen=True)
class ClientTrainingMetrics:
    client_id: int
    train_samples: int
    shared_layer_count: int
    personalized_layer_count: int
    personalized_loss: float
    shared_loss: float


@dataclass
class LocalTrainingResult:
    """
    Step 04 完整输出。

    其中：

        client_shared_parameters

    是 Step 05 参数攻击和 Step 06 参数掩码/加密
    最关键的输入。
    """

    dataset_name: str
    round_index: int

    client_models: Dict[int, FiveLayerCNNBase]

    client_state_dicts: Dict[
        int,
        Dict[str, torch.Tensor],
    ]

    client_shared_parameters: Dict[
        int,
        Dict[str, List[torch.Tensor]],
    ]

    client_personalized_parameters: Dict[
        int,
        Dict[str, List[torch.Tensor]],
    ]

    shared_layer_names: Dict[
        int,
        List[str],
    ]

    personalized_layer_names: Dict[
        int,
        List[str],
    ]

    training_metrics: Dict[
        int,
        ClientTrainingMetrics,
    ]


# ============================================================
# 参数提取
# ============================================================

def extract_named_layer_parameters(
    model: FiveLayerCNNBase,
    layer_names: List[str],
) -> Dict[str, List[torch.Tensor]]:
    """
    按逻辑层导出参数。

    输出格式：

        {
            "layer1": [weight, bias],
            "layer2": [weight, bias],
            ...
        }
    """

    return {
        layer_name: model.get_layer_param_tensors(
            layer_name
        )
        for layer_name in layer_names
    }


# ============================================================
# 客户端本地训练器
# ============================================================

class LocalClientTrainer:
    """
    单个客户端的一轮交互式本地训练。
    """

    def __init__(
        self,
        client_id: int,
        dataset_name: str,
        train_dataset: Dataset,
        num_shared_layers: int,
        initial_state_dict: Mapping[
            str,
            torch.Tensor,
        ],
        previous_client_state_dict: Optional[
            Mapping[str, torch.Tensor]
        ] = None,
        global_shared_parameters: Optional[
            Dict[str, List[torch.Tensor]]
        ] = None,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        batch_size: int = 64,
        device: str = "cpu",
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.client_id = client_id
        self.dataset_name = dataset_name
        self.train_dataset = train_dataset
        self.num_shared_layers = num_shared_layers

        self.shared_layers, self.personalized_layers = (
            get_layer_partition(
                num_shared_layers
            )
        )

        self.device = torch.device(device)

        self.model = build_model(
            dataset_name
        )
        self.model.to(self.device)

        # ----------------------------------------------------
        # 模型状态初始化
        # ----------------------------------------------------
        #
        # 第一轮：
        #     previous_client_state_dict = None
        #     所有客户端统一加载 initial_state_dict
        #
        # 后续轮：
        #     先恢复该客户端上一轮完整本地模型，
        #     从而保留个性化层；
        #     再用 global_shared_parameters 覆盖共享层。
        # ----------------------------------------------------

        if previous_client_state_dict is None:
            self.model.load_state_dict(
                initial_state_dict,
                strict=True,
            )
        else:
            self.model.load_state_dict(
                previous_client_state_dict,
                strict=True,
            )

        if global_shared_parameters is not None:
            self.apply_global_shared_parameters(
                global_shared_parameters
            )

        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=momentum,
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=drop_last,
        )

    def apply_global_shared_parameters(
        self,
        global_shared_parameters: Dict[
            str,
            List[torch.Tensor],
        ],
    ) -> None:
        """
        使用上一轮聚合后的全局共享参数覆盖本客户端共享层。

        个性化层不会被覆盖。
        """

        for layer_name in self.shared_layers:
            if layer_name in global_shared_parameters:
                self.model.set_layer_params(
                    layer_name,
                    global_shared_parameters[
                        layer_name
                    ],
                )

    def _set_trainable_layers(
        self,
        trainable_layers: List[str],
    ) -> None:
        """
        只开放指定逻辑层。
        """

        trainable_set = set(trainable_layers)

        for layer_name in ALL_LAYERS:
            requires_grad = (
                layer_name in trainable_set
            )

            for param in self.model.get_layer_params(
                layer_name
            ):
                param.requires_grad = requires_grad

    def _restore_all_trainable(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True

    def _train_selected_layers(
        self,
        trainable_layers: List[str],
        epochs: int,
    ) -> float:
        """
        训练指定逻辑层，其余层冻结。
        """

        if epochs <= 0:
            return 0.0

        if not trainable_layers:
            return 0.0

        self._set_trainable_layers(
            trainable_layers
        )

        self.model.train()

        total_loss = 0.0
        total_batches = 0

        for _ in range(epochs):
            for data, target in self.train_loader:
                if data.size(0) == 0:
                    continue

                data = data.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()

                output = self.model(data)
                loss = self.loss_fn(
                    output,
                    target,
                )

                loss.backward()
                self.optimizer.step()

                total_loss += float(loss.item())
                total_batches += 1

        self._restore_all_trainable()

        if total_batches == 0:
            return 0.0

        return total_loss / total_batches

    def train_personalized_layers(
        self,
        epochs: int,
    ) -> float:
        """
        阶段 1：
        固定共享层，训练个性化层。
        """
        return self._train_selected_layers(
            trainable_layers=self.personalized_layers,
            epochs=epochs,
        )

    def train_shared_layers(
        self,
        epochs: int,
    ) -> float:
        """
        阶段 2：
        固定个性化层，训练共享层。
        """
        return self._train_selected_layers(
            trainable_layers=self.shared_layers,
            epochs=epochs,
        )

    def train_full_model(
        self,
        epochs: int,
    ) -> float:
        """
        K = 5 时的完整模型本地 SGD。

        此时所有 layer1~layer5 都属于共享架构，
        个性化层为空，因此不再执行交互式两阶段训练，
        而是直接开放全部 5 层参数并进行普通 SGD。
        """
        return self._train_selected_layers(
            trainable_layers=ALL_LAYERS,
            epochs=epochs,
        )

    def train_interactively(
        self,
        personalized_epochs: int,
        shared_epochs: int,
    ) -> Tuple[float, float]:
        """
        根据共享层数自动选择本地训练方式。

        K = 1~4：
            交互式训练
            1) 固定共享层，训练个性化层
            2) 固定个性化层，训练共享层

        K = 5：
            个性化层为空，交互式训练退化为
            Full-model Local SGD。
            此时直接对全部 5 层执行普通 SGD。

        返回值仍保持：
            (personalized_loss, shared_loss)

        因此 K=5 时：
            personalized_loss = 0.0
            shared_loss = 完整模型 SGD 的平均 loss
        """

        if self.num_shared_layers == TOTAL_MODEL_LAYERS:
            personalized_loss = 0.0
            shared_loss = self.train_full_model(
                epochs=shared_epochs,
            )
            return personalized_loss, shared_loss

        personalized_loss = (
            self.train_personalized_layers(
                epochs=personalized_epochs,
            )
        )

        shared_loss = (
            self.train_shared_layers(
                epochs=shared_epochs,
            )
        )

        return (
            personalized_loss,
            shared_loss,
        )


# ============================================================
# Step 04 主入口
# ============================================================

def train_local_models(
    dataset_name: str,
    client_train_datasets: Mapping[
        int,
        Dataset,
    ],
    client_shared_layer_counts: Mapping[
        int,
        int,
    ],
    initial_global_state_dict: Mapping[
        str,
        torch.Tensor,
    ],
    previous_client_state_dicts: Optional[
        Mapping[
            int,
            Mapping[str, torch.Tensor],
        ]
    ] = None,
    global_shared_parameters: Optional[
        Dict[str, List[torch.Tensor]]
    ] = None,
    round_index: int = 1,
    personalized_epochs: int = 1,
    shared_epochs: int = 1,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    batch_size: int = 64,
    device: str = "cpu",
) -> LocalTrainingResult:
    """
    所有客户端执行一轮本地交互式训练。

    --------------------------------------------------------
    第一轮
    --------------------------------------------------------

        previous_client_state_dicts = None
        global_shared_parameters = None

    所有客户端统一从 initial_global_state_dict 开始。

    --------------------------------------------------------
    后续轮
    --------------------------------------------------------

        previous_client_state_dicts
            = 上一轮各客户端完整本地模型参数

        global_shared_parameters
            = 上一轮安全聚合后的共享参数

    每个客户端：

        先恢复自己的上一轮完整模型
        -> 保留个性化参数

        再加载上一轮 global shared parameters
        -> 更新共享层

        再执行本地训练：

        K=1~4:
            personalized training
            -> shared training

        K=5:
            full-model local SGD
    """

    if not client_train_datasets:
        raise ValueError(
            "client_train_datasets 不能为空"
        )

    if set(client_train_datasets.keys()) != set(
        client_shared_layer_counts.keys()
    ):
        raise ValueError(
            "client_train_datasets 与 "
            "client_shared_layer_counts "
            "的客户端集合必须一致"
        )

    client_models: Dict[
        int,
        FiveLayerCNNBase,
    ] = {}

    client_state_dicts: Dict[
        int,
        Dict[str, torch.Tensor],
    ] = {}

    client_shared_parameters: Dict[
        int,
        Dict[str, List[torch.Tensor]],
    ] = {}

    client_personalized_parameters: Dict[
        int,
        Dict[str, List[torch.Tensor]],
    ] = {}

    shared_layer_names: Dict[
        int,
        List[str],
    ] = {}

    personalized_layer_names: Dict[
        int,
        List[str],
    ] = {}

    training_metrics: Dict[
        int,
        ClientTrainingMetrics,
    ] = {}

    for client_id in sorted(
        client_train_datasets.keys()
    ):
        train_dataset = client_train_datasets[
            client_id
        ]

        num_shared_layers = int(
            client_shared_layer_counts[
                client_id
            ]
        )

        previous_state_dict = None

        if previous_client_state_dicts is not None:
            previous_state_dict = (
                previous_client_state_dicts.get(
                    client_id
                )
            )

        trainer = LocalClientTrainer(
            client_id=client_id,
            dataset_name=dataset_name,
            train_dataset=train_dataset,
            num_shared_layers=num_shared_layers,
            initial_state_dict=initial_global_state_dict,
            previous_client_state_dict=previous_state_dict,
            global_shared_parameters=global_shared_parameters,
            learning_rate=learning_rate,
            momentum=momentum,
            batch_size=batch_size,
            device=device,
        )

        personalized_loss, shared_loss = (
            trainer.train_interactively(
                personalized_epochs=personalized_epochs,
                shared_epochs=shared_epochs,
            )
        )

        model = trainer.model

        # 完整模型对象
        client_models[client_id] = model

        # 完整 state_dict
        client_state_dicts[client_id] = (
            clone_state_dict(
                model.state_dict()
            )
        )

        # 显式共享参数：
        # Step 05 / Step 06 的核心输入
        client_shared_parameters[client_id] = (
            extract_named_layer_parameters(
                model,
                trainer.shared_layers,
            )
        )

        # 显式个性化参数
        client_personalized_parameters[client_id] = (
            extract_named_layer_parameters(
                model,
                trainer.personalized_layers,
            )
        )

        shared_layer_names[client_id] = (
            trainer.shared_layers.copy()
        )

        personalized_layer_names[client_id] = (
            trainer.personalized_layers.copy()
        )

        training_metrics[client_id] = (
            ClientTrainingMetrics(
                client_id=client_id,
                train_samples=len(
                    train_dataset
                ),
                shared_layer_count=len(
                    trainer.shared_layers
                ),
                personalized_layer_count=len(
                    trainer.personalized_layers
                ),
                personalized_loss=personalized_loss,
                shared_loss=shared_loss,
            )
        )

    return LocalTrainingResult(
        dataset_name=dataset_name,
        round_index=round_index,
        client_models=client_models,
        client_state_dicts=client_state_dicts,
        client_shared_parameters=client_shared_parameters,
        client_personalized_parameters=(
            client_personalized_parameters
        ),
        shared_layer_names=shared_layer_names,
        personalized_layer_names=(
            personalized_layer_names
        ),
        training_metrics=training_metrics,
    )


# ============================================================
# 独立测试辅助数据集
# ============================================================

class SyntheticImageDataset(Dataset):
    """
    Step 04 裸跑测试用的极小合成数据集。

    不下载 CIFAR-10 / MNIST，
    只用于验证：

    1. 模型前向传播；
    2. 交互式训练流程；
    3. 输出参数结构；
    4. 不同共享层数的客户端能否正常训练。

    正式系统绝不会使用这个类。
    """

    def __init__(
        self,
        dataset_name: str,
        num_samples: int,
        seed: int,
    ):
        generator = torch.Generator()
        generator.manual_seed(seed)

        normalized_name = dataset_name.lower()

        if normalized_name in {
            "cifar10",
            "cifar-10",
        }:
            shape = (
                num_samples,
                3,
                32,
                32,
            )
        elif normalized_name == "mnist":
            shape = (
                num_samples,
                1,
                28,
                28,
            )
        else:
            raise ValueError(
                f"未知数据集: {dataset_name}"
            )

        self.images = torch.randn(
            shape,
            generator=generator,
        )

        self.labels = torch.randint(
            low=0,
            high=10,
            size=(num_samples,),
            generator=generator,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self,
        index: int,
    ):
        return (
            self.images[index],
            self.labels[index],
        )


# ============================================================
# 打印结果
# ============================================================

def print_local_training_result(
    result: LocalTrainingResult,
) -> None:
    print("\n" + "=" * 80)
    print("Step 04 - Local Model Training Result")
    print("=" * 80)

    print(
        f"Dataset     : {result.dataset_name}"
    )
    print(
        f"Round Index : {result.round_index}"
    )

    for client_id in sorted(
        result.training_metrics
    ):
        metric = result.training_metrics[
            client_id
        ]

        print("\n" + "-" * 80)
        print(f"Client {client_id}")
        print("-" * 80)

        print(
            f"Train Samples         : "
            f"{metric.train_samples}"
        )
        print(
            f"Shared Layer Count    : "
            f"{metric.shared_layer_count}"
        )
        print(
            f"Personalized Count    : "
            f"{metric.personalized_layer_count}"
        )
        print(
            f"Shared Layers         : "
            f"{result.shared_layer_names[client_id]}"
        )
        print(
            f"Personalized Layers   : "
            f"{result.personalized_layer_names[client_id]}"
        )
        print(
            f"Personalized Loss     : "
            f"{metric.personalized_loss:.6f}"
        )
        print(
            f"Shared Loss           : "
            f"{metric.shared_loss:.6f}"
        )

        print(
            "Shared Parameter Shapes:"
        )

        for (
            layer_name,
            params,
        ) in result.client_shared_parameters[
            client_id
        ].items():
            shapes = [
                tuple(param.shape)
                for param in params
            ]

            print(
                f"    {layer_name}: {shapes}"
            )


# ============================================================
# 这个是测试样例
# ============================================================

if __name__ == "__main__":
    # ========================================================
    # 独立运行测试
    # ========================================================
    #
    # 为了让本文件可以“裸跑”，这里不依赖 Step 01 / 02 / 03。
    #
    # 使用极小的随机图像数据，只测试 Step 04 自己：
    #
    # 1. CIFAR-10 / MNIST 模型能否正确创建；
    # 2. 第一轮统一初始化能否工作；
    # 3. 不同客户端不同共享层数能否工作；
    # 4. K=1~4 的交互式训练能否正常完成；
    # 5. K=5 是否自动退化为完整模型 Local SGD；
    # 6. 输出的共享参数能否明确按 layer1~layer5 导出。
    #
    # 正式完整流程中：
    #
    # client_train_datasets
    #     <- Step 02
    #
    # client_shared_layer_counts
    #     <- Step 03
    #
    # initial_global_state_dict
    #     <- 第一轮 initialize_global_model()
    #
    # client_shared_parameters
    #     -> Step 05 参数攻击
    #     -> Step 06 参数掩码 / 加密
    # ========================================================

    TEST_DATASET = "cifar10"
    TEST_DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # 第一轮统一初始模型
    # --------------------------------------------------------

    _, initial_state_dict = (
        initialize_global_model(
            dataset_name=TEST_DATASET,
            seed=42,
            device=TEST_DEVICE,
        )
    )

    # --------------------------------------------------------
    # 构造 3 个客户端极小训练集
    # --------------------------------------------------------

    client_train_datasets = {
        0: SyntheticImageDataset(
            TEST_DATASET,
            num_samples=32,
            seed=100,
        ),
        1: SyntheticImageDataset(
            TEST_DATASET,
            num_samples=32,
            seed=101,
        ),
        2: SyntheticImageDataset(
            TEST_DATASET,
            num_samples=32,
            seed=102,
        ),
        3: SyntheticImageDataset(
            TEST_DATASET,
            num_samples=32,
            seed=103,
        ),
    }

    # 模拟 Step 03 输出
    client_shared_layer_counts = {
        0: 4,
        1: 3,
        2: 1,
        3: 5,  # K=5：完整模型 Local SGD
    }

    result = train_local_models(
        dataset_name=TEST_DATASET,
        client_train_datasets=(
            client_train_datasets
        ),
        client_shared_layer_counts=(
            client_shared_layer_counts
        ),
        initial_global_state_dict=(
            initial_state_dict
        ),
        previous_client_state_dicts=None,
        global_shared_parameters=None,
        round_index=1,
        personalized_epochs=1,
        shared_epochs=1,
        learning_rate=0.01,
        momentum=0.9,
        batch_size=16,
        device=TEST_DEVICE,
    )

    print_local_training_result(
        result
    )

    # --------------------------------------------------------
    # 自动检查
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("Automatic Checks")
    print("=" * 80)

    # 1. 所有客户端完整 state_dict 都存在
    all_state_dicts_exist = (
        len(result.client_state_dicts)
        == len(client_train_datasets)
    )

    print(
        "所有客户端均输出完整 state_dict :",
        all_state_dicts_exist,
    )

    # 2. 共享层数量与 Step 03 输入一致
    shared_counts_match = all(
        len(
            result.shared_layer_names[
                client_id
            ]
        )
        == client_shared_layer_counts[
            client_id
        ]
        for client_id in client_train_datasets
    )

    print(
        "共享层数量与输入分配一致 :",
        shared_counts_match,
    )

    # 3. 总层数恒为 5
    total_layers_valid = all(
        (
            len(
                result.shared_layer_names[
                    client_id
                ]
            )
            +
            len(
                result.personalized_layer_names[
                    client_id
                ]
            )
        )
        == 5
        for client_id in client_train_datasets
    )

    print(
        "Shared + Personalized 恒等于 5 :",
        total_layers_valid,
    )

    # 4. Step 05 / 06 所需共享参数已经显式导出
    shared_parameters_ready = all(
        set(
            result.client_shared_parameters[
                client_id
            ].keys()
        )
        == set(
            result.shared_layer_names[
                client_id
            ]
        )
        for client_id in client_train_datasets
    )

    print(
        "Step 05 / 06 所需共享参数已显式导出 :",
        shared_parameters_ready,
    )

    # 5. K=5 客户端应为全共享、零个性化
    k5_client_id = 3
    k5_full_shared_valid = (
        result.shared_layer_names[k5_client_id] == ALL_LAYERS
        and result.personalized_layer_names[k5_client_id] == []
        and set(result.client_shared_parameters[k5_client_id].keys())
        == set(ALL_LAYERS)
        and result.client_personalized_parameters[k5_client_id] == {}
        and result.training_metrics[k5_client_id].personalized_loss == 0.0
    )

    print(
        "K=5 时自动退化为全模型共享 / Local SGD :",
        k5_full_shared_valid,
    )
