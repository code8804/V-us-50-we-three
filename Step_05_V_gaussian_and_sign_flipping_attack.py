"""
Step_05_V_gaussian_and_sign_flipping_attack.py

============================================================
Step 05：Gaussian / Sign Flipping 模型参数攻击（可视化相关）
============================================================

【本文件在整个流程中的位置】

    Step 00
    Byzantine Attack Configuration
        |
        |  AttackPlan
        |  - 哪些客户端是 Byzantine
        |  - 每个 Byzantine 客户端的 attack_type
        |  - layer_attack_ratio
        |  - layer_selection_seed
        |  - gaussian_mean / gaussian_std
        |  - sign_flip_scale
        v

    Step 01
    Data Distribution
        |
        v

    Step 02
    Label Flipping
        |
        v

    Step 03 / Step 03-sp
    Architecture Allocation
        |
        v

    Step 04
    Local Model Training
        |
        |  client_shared_parameters
        |  shared_layer_names
        v

    --------------------------------------------------------
    Step 05  <-- 本文件
    Gaussian / Sign Flipping Parameter Attack
    --------------------------------------------------------
        |
        |  attacked_client_shared_parameters
        v

    Step 06
    Parameter Masking / Encryption
        |
        v
    ...


============================================================
【本文件职责】
============================================================

本步骤只负责“模型参数层面的 Byzantine Attack”。

当前只处理：

    1. Gaussian Attack
    2. Sign Flipping Attack

Label Flipping 不在这里执行。

原因：

    Label Flipping 是数据层攻击，
    已经在 Step 02 中修改了 Byzantine 客户端训练时看到的标签。

因此：

    attack_type == LABEL_FLIPPING

的客户端到了本步骤以后，
其 Step 04 输出的模型参数直接保持不变。


============================================================
【非常重要：本步骤不重新分配攻击客户端】
============================================================

本步骤必须直接读取 Step 00 已经生成好的：

    AttackPlan

绝对不能在 Step 05 中重新执行：

    - Byzantine 客户端抽样；
    - Attack Type 抽样；
    - attacked layer 随机种子生成。

这样可以保证整场实验中：

    Step 00 页面显示的攻击计划
        ==
    Step 02 真正执行的 Label Flipping 客户端
        ==
    Step 05 真正执行的 Gaussian / Sign Flipping 客户端

不会出现前后断档。


============================================================
【Step 02 与本步骤采用同一种 AttackPlan 逻辑】
============================================================

Step 02 当前已经采用：

    attack_plan.get(client_id)

读取 Step 00 中每个客户端已经确定好的 ClientAttackConfig。

Step 02 只筛选：

    attack_type == LABEL_FLIPPING

本 Step 05 同样读取完全相同的 AttackPlan，
但只筛选：

    attack_type == GAUSSIAN
        或
    attack_type == SIGN_FLIPPING

所以：

    Step 02 和 Step 05
    不是各自重新生成一份攻击计划，

而是：

    共享同一份 Step 00 AttackPlan，
    分别在正确的攻击阶段执行不同类型的攻击。


============================================================
【攻击对象：只允许 Shared Parameters】
============================================================

Step 04 已经显式输出：

    local_training_result.client_shared_parameters

结构为：

    {
        client_id: {
            "layer1": [weight_tensor, bias_tensor],
            "layer2": [weight_tensor, bias_tensor],
            ...
        }
    }

同时还输出：

    local_training_result.shared_layer_names

例如：

    Client 3:

        shared_layer_names
            = ["layer1", "layer2"]

        personalized_layer_names
            = ["layer3", "layer4", "layer5"]

那么 Step 05 的随机攻击候选集合只能是：

    ["layer1", "layer2"]

绝对不能从完整：

    ["layer1", ..., "layer5"]

中随机抽层。

因此本步骤从数据结构上直接只接收并操作：

    client_shared_parameters

个性化参数根本不进入攻击函数，
从而避免误攻击 Personalized Layers。


============================================================
【攻击层数量：至少攻击 1 个共享层】
============================================================

对于 Gaussian / Sign Flipping Byzantine 客户端，
假设：

    shared_layer_count = S
    layer_attack_ratio = r

原始实验代码的核心思想是：

    attack_count = int(S * r)

然后：

    attack_count = max(1, attack_count)

因此例如：

    S = 1
    r = 0.5

虽然：

    int(1 * 0.5) = 0

但最终必须修正为：

    attack_count = 1

也就是说：

    只要一个客户端已经被定义为
    Gaussian / Sign Flipping Byzantine Client，

并且配置了正的 layer_attack_ratio，
它至少必须真正攻击 1 个共享层。

本工程 Step 00 的 resolve_attacked_layers(...) 已经实现了：

    attack_count = max(1, attack_count)
    attack_count = min(len(shared_layers), attack_count)

本 Step 05 不复制这套随机选层逻辑，
而是直接复用 Step 00 的：

    resolve_attacked_layers(...)

这样可以保证：

    同一个 AttackPlan
    + 同一个共享架构
    -> attacked_layers 永远一致。


============================================================
【为什么 attacked_layers 要等到 Step 05 才真正解析】
============================================================

Step 00 产生攻击计划时，
客户端的共享层数还没有由 Step 03 / Step 03-sp 决定。

例如同一个 Byzantine Client：

    Adaptive Architecture:
        Shared = [layer1, layer2]

    Consistent K=5:
        Shared = [layer1, layer2, layer3, layer4, layer5]

因此 Step 00 只能提前保存：

    layer_attack_ratio
    layer_selection_seed

到了 Step 05，
已经拿到 Step 04 实际训练后的：

    shared_layer_names

才可以把比例解析成真正的 attacked_layers。


============================================================
【本步骤不判断 Step 03 还是 Step 03-sp】
============================================================

本步骤完全不知道上游使用的是：

    Step_03_V_architecture_allocation.py

还是：

    Step_03_V_sp_consistent_architecture.py

因为 Step 05 只读取 Step 04 已经确定好的：

    local_training_result.shared_layer_names

例如：

    {
        0: ["layer1", "layer2", "layer3"],
        1: ["layer1"],
        2: ["layer1", "layer2", "layer3", "layer4", "layer5"],
    }

因此：

    Adaptive / Consistent / K=5

到了 Step 05 都采用完全相同的攻击代码。


============================================================
【Sign Flipping 的精确攻击方式】
============================================================

原始 CIFAR-10 / MNIST 实验代码中，
Sign Flipping 对被选中的参数 Tensor 执行：

    theta_attack = -theta

当前 Step 00 保留了：

    sign_flip_scale

并默认锁定：

    sign_flip_scale = -1.0

因此本步骤统一写为：

    theta_attack = sign_flip_scale * theta

默认情况下就是：

    theta_attack = -theta

注意：

    一个逻辑层通常同时包含：

        weight
        bias

例如：

    layer1:
        weight shape = (6, 3, 5, 5)
        bias shape   = (6,)

如果 layer1 被选中攻击，
则该逻辑层中的 weight 和 bias 都要一起攻击。


============================================================
【Gaussian Attack 的精确攻击方式】
============================================================

原始 CIFAR-10 / MNIST 实验代码中，
主实验所使用的 GaussianReplace 定义为：

    将被攻击参数完全替换为高斯随机参数

而不是：

    原参数 + Gaussian Noise

即：

    theta_attack ~ N(mean, std^2)

其中：

    gaussian_mean
    gaussian_std

直接读取 Step 00 的 ClientAttackConfig。

因此本步骤采用：

    Gaussian Replacement

例如：

    gaussian_mean = 0
    gaussian_std  = 1

则被选中的参数 Tensor 会被完整替换为：

    N(0, 1)

随机生成的同形状 Tensor。


============================================================
【Gaussian 随机数的可复现性】
============================================================

Step 00 的：

    layer_selection_seed

用于决定“攻击哪些共享层”。

但是 Gaussian Attack 还需要随机生成具体参数值。

因此本步骤另外从以下信息确定性派生 Gaussian Seed：

    attack_plan.master_seed
    client_id
    round_index
    layer_index
    tensor_index

这样：

    同一实验配置
    + 同一轮次
    -> Gaussian 参数攻击可复现

同时不同：

    客户端 / 层 / weight-bias Tensor / 轮次

会得到不同的 Gaussian 随机流。

注意：

    这里不会修改 Python / NumPy / Torch 的全局随机状态，
    使用独立 torch.Generator 完成随机参数生成。


============================================================
【本步骤输入】      带 # 的为主要正式输入
============================================================

# 1. attack_plan

    来自 Step 00：

        AttackPlan

    本步骤主要读取：

        attack_plan.master_seed
        attack_plan.num_clients
        attack_plan.get(client_id)

    以及每个 ClientAttackConfig 中：

        is_byzantine
        attack_type
        layer_attack_ratio
        layer_selection_seed
        gaussian_mean
        gaussian_std
        sign_flip_scale


# 2. local_training_result

    来自 Step 04：

        LocalTrainingResult

    本步骤主要读取：

        local_training_result.round_index

        local_training_result.client_shared_parameters

        local_training_result.shared_layer_names

    最关键的是：

        client_shared_parameters

    它是 Step 04 训练完成后真正要上传/进入安全流程的共享参数。


- preview_values_per_tensor

    仅用于给前端 / 调试展示少量攻击前后参数值。

    默认：

        5

    它完全不影响真实攻击。


============================================================
【本步骤输出】      带 V，需要给可视化同学使用
============================================================

统一返回：

    ParameterAttackResult


------------------------------------------------------------
1. original_client_shared_parameters
------------------------------------------------------------

攻击前的共享参数深拷贝：

    {
        client_id: {
            layer_name: [
                weight_tensor,
                bias_tensor,
            ]
        }
    }

作用：

    - 可视化攻击前参数；
    - 调试；
    - 验证 benign / label-flipping 客户端参数未变化。


------------------------------------------------------------
2. attacked_client_shared_parameters
------------------------------------------------------------

攻击后的共享参数。

这是本步骤最关键的输出。

后续 Step 06 必须使用：

    result.attacked_client_shared_parameters

而不是再使用 Step 04 原来的：

    local_training_result.client_shared_parameters

正式传递关系：

    Step 04
        client_shared_parameters
             |
             v
    Step 05
        attacked_client_shared_parameters
             |
             v
    Step 06
        Parameter Masking / Encryption


------------------------------------------------------------
3. attacked_clients
------------------------------------------------------------

本步骤真正执行“模型参数攻击”的客户端：

    Gaussian
    Sign Flipping

不包含：

    Label Flipping

因为 Label Flipping 已经在 Step 02 完成。


------------------------------------------------------------
4. skipped_label_flipping_clients
------------------------------------------------------------

AttackPlan 中属于：

    Label Flipping

的 Byzantine 客户端。

这些客户端在本步骤不会再次修改参数。


------------------------------------------------------------
5. client_attack_records
------------------------------------------------------------

每个客户端都有一条结构化记录：

    ClientParameterAttackRecord

包括：

    client_id
    is_byzantine
    attack_type
    shared_layers
    attacked_layers
    attacked_layer_count
    layer_attack_ratio
    gaussian_mean
    gaussian_std
    sign_flip_scale
    parameter_changes


------------------------------------------------------------
6. parameter_changes
------------------------------------------------------------

只对真正被攻击的逻辑层记录：

    LayerAttackVisualization

其中包括：

    layer_name
    attack_type
    tensor_changes

每个 Tensor 记录：

    tensor_index
    tensor_role
        weight / bias / param_i

    shape

    before_mean
    before_std
    before_norm

    after_mean
    after_std
    after_norm

    delta_norm

    before_preview
    after_preview


============================================================
【给可视化同学的重点】
============================================================

推荐 Step 05 页面至少展示三部分。


------------------------------------------------------------
A. 攻击计划概览
------------------------------------------------------------

例如：

    Client 0
        BENIGN

    Client 1
        BYZANTINE
        Attack = Gaussian
        Shared = [layer1, layer2, layer3]
        Attacked = [layer1, layer3]

    Client 2
        BYZANTINE
        Attack = Label Flipping
        Model Parameter Attack = SKIPPED
        Reason = Already attacked in Step 02

    Client 3
        BYZANTINE
        Attack = Sign Flipping
        Shared = [layer1]
        Attacked = [layer1]


------------------------------------------------------------
B. attacked layers 可视化
------------------------------------------------------------

建议将 5 个逻辑层画成方块：

    layer1  layer2  layer3  layer4  layer5

状态：

    Shared + attacked
    Shared + not attacked
    Personalized

注意：

    Step 05 自身只返回 Shared Layers 和 Attacked Layers。

Personalized Layers 如页面需要，
可以由：

    ALL_LAYERS - Shared Layers

得到。


------------------------------------------------------------
C. 参数攻击前后变化
------------------------------------------------------------

例如 Sign Flipping：

    layer2 / weight

    Before Mean : +0.013
    After Mean  : -0.013

    Before Norm : 1.82
    After Norm  : 1.82

Gaussian Replacement：

    layer3 / weight

    Before Norm : 0.91
    After Norm  : 263.4
    Delta Norm  : 263.5


注意：

    前端不应该直接渲染整块高维 Tensor。

只展示：

    - shape
    - mean
    - std
    - norm
    - delta norm
    - 前几个 preview values

即可。


============================================================
【本步骤明确不负责】
============================================================

- 不重新选择 Byzantine Clients；
- 不重新分配 Attack Type；
- 不执行 Label Flipping；
- 不训练模型；
- 不修改 Personalized Parameters；
- 不执行随机掩码；
- 不执行加密；
- 不执行 Byzantine Detection；
- 不执行 Layer Filtering；
- 不执行 Aggregation；
- 不判断前面使用的是 Adaptive 还是 Consistent Architecture。


============================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from Step_00_V_byzantine_config import (
    AttackPlan,
    AttackType,
    ClientAttackConfig,
    resolve_attacked_layers,
)


# ============================================================
# 固定逻辑层
# ============================================================

ALL_LAYER_NAMES = [
    "layer1",
    "layer2",
    "layer3",
    "layer4",
    "layer5",
]

GAUSSIAN_VALUE = "gaussian"
SIGN_FLIPPING_VALUE = "sign_flipping"
LABEL_FLIPPING_VALUE = "label_flipping"

DEFAULT_PREVIEW_VALUES_PER_TENSOR = 5


# ============================================================
# 输出数据结构
# ============================================================

@dataclass(frozen=True)
class TensorAttackVisualization:
    """
    单个参数 Tensor 的攻击前后统计。

    例如一个 Linear 层一般有：

        tensor_index = 0 -> weight
        tensor_index = 1 -> bias
    """

    tensor_index: int
    tensor_role: str
    shape: Tuple[int, ...]

    before_mean: float
    before_std: float
    before_norm: float

    after_mean: float
    after_std: float
    after_norm: float

    delta_norm: float

    before_preview: List[float]
    after_preview: List[float]


@dataclass(frozen=True)
class LayerAttackVisualization:
    """
    一个真正被攻击的逻辑层的可视化记录。
    """

    layer_name: str
    attack_type: str
    tensor_changes: List[TensorAttackVisualization]


@dataclass(frozen=True)
class ClientParameterAttackRecord:
    """
    每个客户端的 Step 05 攻击记录。

    即使客户端没有发生模型参数攻击，
    也会保留一条记录，方便前端完整展示所有客户端。
    """

    client_id: int
    is_byzantine: bool
    attack_type: Optional[str]

    shared_layers: List[str]
    attacked_layers: List[str]
    attacked_layer_count: int

    layer_attack_ratio: float

    gaussian_mean: Optional[float]
    gaussian_std: Optional[float]
    sign_flip_scale: Optional[float]

    status: str
    parameter_changes: List[LayerAttackVisualization]


@dataclass
class ParameterAttackResult:
    """
    Step 05 的统一输出。
    """

    round_index: int

    # 攻击前共享参数副本
    original_client_shared_parameters: Dict[
        int,
        Dict[str, List[torch.Tensor]],
    ]

    # 攻击后的共享参数
    #
    # 后续 Step 06 必须使用这个字段。
    attacked_client_shared_parameters: Dict[
        int,
        Dict[str, List[torch.Tensor]],
    ]

    # 真正在本步骤执行模型参数攻击的客户端
    attacked_clients: List[int]

    # AttackPlan 中是 Label Flipping，
    # 因此 Step 05 主动跳过模型参数攻击的客户端
    skipped_label_flipping_clients: List[int]

    # 每个客户端的完整结构化攻击记录
    client_attack_records: Dict[
        int,
        ClientParameterAttackRecord,
    ]

    @property
    def visualization_data(self) -> List[Dict[str, Any]]:
        """
        JSON-friendly 的前端数据。

        注意：
            不直接包含 Tensor，
            只包含前端真正需要的攻击元信息和统计数据。
        """

        rows: List[Dict[str, Any]] = []

        for client_id in sorted(self.client_attack_records):
            record = self.client_attack_records[client_id]

            rows.append(
                {
                    "client_id": record.client_id,
                    "is_byzantine": record.is_byzantine,
                    "attack_type": record.attack_type,
                    "status": record.status,
                    "shared_layers": record.shared_layers.copy(),
                    "attacked_layers": record.attacked_layers.copy(),
                    "attacked_layer_count": record.attacked_layer_count,
                    "layer_attack_ratio": record.layer_attack_ratio,
                    "gaussian_mean": record.gaussian_mean,
                    "gaussian_std": record.gaussian_std,
                    "sign_flip_scale": record.sign_flip_scale,
                    "parameter_changes": [
                        {
                            "layer_name": layer_record.layer_name,
                            "attack_type": layer_record.attack_type,
                            "tensor_changes": [
                                {
                                    "tensor_index": tensor_record.tensor_index,
                                    "tensor_role": tensor_record.tensor_role,
                                    "shape": list(tensor_record.shape),
                                    "before_mean": tensor_record.before_mean,
                                    "before_std": tensor_record.before_std,
                                    "before_norm": tensor_record.before_norm,
                                    "after_mean": tensor_record.after_mean,
                                    "after_std": tensor_record.after_std,
                                    "after_norm": tensor_record.after_norm,
                                    "delta_norm": tensor_record.delta_norm,
                                    "before_preview": tensor_record.before_preview.copy(),
                                    "after_preview": tensor_record.after_preview.copy(),
                                }
                                for tensor_record in layer_record.tensor_changes
                            ],
                        }
                        for layer_record in record.parameter_changes
                    ],
                }
            )

        return rows


# ============================================================
# 基础辅助函数
# ============================================================

def _attack_type_value(
    client_config: ClientAttackConfig,
) -> Optional[str]:
    attack_type = getattr(
        client_config,
        "attack_type",
        None,
    )

    if attack_type is None:
        return None

    if hasattr(attack_type, "value"):
        return str(attack_type.value)

    return str(attack_type)


def _clone_parameter_structure(
    client_shared_parameters: Mapping[
        int,
        Mapping[str, Sequence[torch.Tensor]],
    ],
) -> Dict[int, Dict[str, List[torch.Tensor]]]:
    """
    深拷贝所有共享参数到 CPU。

    这样 Step 05：
        - 不会原地修改 Step 04 的输出；
        - 可以同时保留攻击前和攻击后两个版本；
        - 后续 Step 06 拿到的是独立 Tensor。
    """

    return {
        int(client_id): {
            str(layer_name): [
                tensor.detach().cpu().clone()
                for tensor in tensors
            ]
            for layer_name, tensors in layer_map.items()
        }
        for client_id, layer_map in client_shared_parameters.items()
    }


def _layer_name_to_index(
    layer_name: str,
) -> int:
    """
    将：
        "layer3"
    转成：
        3

    Step 00 的 resolve_attacked_layers(...) 使用整数层编号。
    """

    if layer_name not in ALL_LAYER_NAMES:
        raise ValueError(
            f"非法逻辑层名称：{layer_name!r}；"
            f"当前只支持 {ALL_LAYER_NAMES}"
        )

    return ALL_LAYER_NAMES.index(layer_name) + 1


def _layer_index_to_name(
    layer_index: int,
) -> str:
    """
    将：
        3
    转成：
        "layer3"
    """

    if not 1 <= int(layer_index) <= len(ALL_LAYER_NAMES):
        raise ValueError(
            f"非法逻辑层编号：{layer_index}"
        )

    return ALL_LAYER_NAMES[int(layer_index) - 1]


def _tensor_role(
    tensor_index: int,
    tensor_count: int,
) -> str:
    """
    当前 Conv / Linear 层通常参数顺序是：

        0 -> weight
        1 -> bias

    如果未来某层参数更多，
    则使用 param_i 作为通用名称。
    """

    if tensor_count >= 1 and tensor_index == 0:
        return "weight"

    if tensor_count >= 2 and tensor_index == 1:
        return "bias"

    return f"param_{tensor_index}"


def _tensor_std(
    tensor: torch.Tensor,
) -> float:
    """
    使用 population std（unbiased=False），
    避免单元素 Tensor 得到 nan。
    """

    flat = tensor.detach().float().reshape(-1)

    if flat.numel() == 0:
        return 0.0

    return float(
        flat.std(unbiased=False).item()
    )


def _preview_values(
    tensor: torch.Tensor,
    count: int,
) -> List[float]:
    if count <= 0:
        return []

    flat = (
        tensor.detach()
        .float()
        .reshape(-1)
        .cpu()
    )

    return [
        float(value)
        for value in flat[:count].tolist()
    ]


def _make_tensor_visualization(
    tensor_index: int,
    tensor_count: int,
    before: torch.Tensor,
    after: torch.Tensor,
    preview_values_per_tensor: int,
) -> TensorAttackVisualization:
    before_float = (
        before.detach()
        .float()
        .cpu()
    )

    after_float = (
        after.detach()
        .float()
        .cpu()
    )

    delta = after_float - before_float

    return TensorAttackVisualization(
        tensor_index=tensor_index,
        tensor_role=_tensor_role(
            tensor_index,
            tensor_count,
        ),
        shape=tuple(before.shape),

        before_mean=float(
            before_float.mean().item()
        ),
        before_std=_tensor_std(
            before_float
        ),
        before_norm=float(
            torch.linalg.vector_norm(
                before_float.reshape(-1)
            ).item()
        ),

        after_mean=float(
            after_float.mean().item()
        ),
        after_std=_tensor_std(
            after_float
        ),
        after_norm=float(
            torch.linalg.vector_norm(
                after_float.reshape(-1)
            ).item()
        ),

        delta_norm=float(
            torch.linalg.vector_norm(
                delta.reshape(-1)
            ).item()
        ),

        before_preview=_preview_values(
            before_float,
            preview_values_per_tensor,
        ),
        after_preview=_preview_values(
            after_float,
            preview_values_per_tensor,
        ),
    )


# ============================================================
# Gaussian Seed
# ============================================================

def _derive_gaussian_tensor_seed(
    master_seed: int,
    client_id: int,
    round_index: int,
    layer_index: int,
    tensor_index: int,
) -> int:
    """
    为一个具体参数 Tensor 派生稳定的 Gaussian Seed。

    不使用 Python hash()，
    因为 hash() 在不同 Python 进程中可能发生随机化。

    返回范围限制在 torch.Generator 可安全使用的正整数范围内。
    """

    value = int(master_seed) & 0x7FFFFFFFFFFFFFFF

    value = (
        value * 1_000_003
        + int(client_id) * 97_409
        + int(round_index) * 1_299_709
        + int(layer_index) * 15_485_863
        + int(tensor_index) * 32_452_843
        + 0x4741555353
    )

    return int(
        value & 0x7FFFFFFFFFFFFFFF
    )


# ============================================================
# 两种模型参数攻击
# ============================================================

def _apply_sign_flipping(
    tensor: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Sign Flipping：

        theta_attack = scale * theta

    当前 Step 00 默认：

        scale = -1.0

    因此标准行为为：

        theta_attack = -theta
    """

    return (
        tensor.detach()
        .cpu()
        .clone()
        * float(scale)
    )


def _apply_gaussian_replacement(
    tensor: torch.Tensor,
    mean: float,
    std: float,
    seed: int,
) -> torch.Tensor:
    """
    Gaussian Replacement：

        theta_attack ~ N(mean, std^2)

    注意：
        不是 theta + noise，
        而是完全用 Gaussian 参数替换原参数。

    这与原始 CIFAR-10 / MNIST 实验代码中的
    GaussianReplace 定义保持一致。
    """

    if std < 0:
        raise ValueError(
            f"Gaussian std 必须 >= 0，当前为 {std}"
        )

    source = (
        tensor.detach()
        .cpu()
    )

    generator = torch.Generator(
        device="cpu"
    )
    generator.manual_seed(
        int(seed)
    )

    # torch.normal(..., generator=...) 在不同版本 PyTorch
    # 上的支持细节存在差异，因此这里使用：
    #
    #   randn * std + mean
    #
    # 数学上仍然严格得到 N(mean, std^2)。
    gaussian = torch.randn(
        source.shape,
        generator=generator,
        dtype=source.dtype,
        device="cpu",
    )

    gaussian = (
        gaussian * float(std)
        + float(mean)
    )

    return gaussian


# ============================================================
# Step 05 主攻击器
# ============================================================

class GaussianAndSignFlippingAttacker:
    """
    使用 Step 00 AttackPlan，
    对 Step 04 LocalTrainingResult 中的共享参数执行模型参数攻击。
    """

    def __init__(
        self,
        attack_plan: AttackPlan,
        local_training_result: Any,
        preview_values_per_tensor: int = (
            DEFAULT_PREVIEW_VALUES_PER_TENSOR
        ),
    ):
        self.attack_plan = attack_plan
        self.local_training_result = (
            local_training_result
        )

        self.preview_values_per_tensor = int(
            preview_values_per_tensor
        )

        self._validate_inputs()

        self.round_index = int(
            getattr(
                self.local_training_result,
                "round_index",
                1,
            )
        )

    # --------------------------------------------------------
    # 输入检查
    # --------------------------------------------------------

    def _validate_inputs(self) -> None:
        required_attack_plan_attrs = [
            "master_seed",
            "num_clients",
            "client_configs",
            "get",
        ]

        for attr in required_attack_plan_attrs:
            if not hasattr(
                self.attack_plan,
                attr,
            ):
                raise TypeError(
                    "attack_plan 缺少必要字段/方法："
                    f"{attr}"
                )

        required_local_result_attrs = [
            "client_shared_parameters",
            "shared_layer_names",
        ]

        for attr in required_local_result_attrs:
            if not hasattr(
                self.local_training_result,
                attr,
            ):
                raise TypeError(
                    "local_training_result 缺少必要字段："
                    f"{attr}"
                )

        if self.preview_values_per_tensor < 0:
            raise ValueError(
                "preview_values_per_tensor 必须 >= 0"
            )

        client_shared_parameters = (
            self.local_training_result
            .client_shared_parameters
        )

        shared_layer_names = (
            self.local_training_result
            .shared_layer_names
        )

        parameter_clients = set(
            client_shared_parameters.keys()
        )
        layer_name_clients = set(
            shared_layer_names.keys()
        )

        if parameter_clients != layer_name_clients:
            raise ValueError(
                "Step 04 的 client_shared_parameters 与 "
                "shared_layer_names 客户端集合不一致"
            )

        if len(parameter_clients) != int(
            self.attack_plan.num_clients
        ):
            raise ValueError(
                "Step 00 AttackPlan 的 num_clients 与 "
                "Step 04 输出客户端数量不一致："
                f"Step00={self.attack_plan.num_clients}, "
                f"Step04={len(parameter_clients)}"
            )

        expected_clients = set(
            range(
                int(self.attack_plan.num_clients)
            )
        )

        if parameter_clients != expected_clients:
            raise ValueError(
                "当前要求 Client ID 连续为 0..num_clients-1；"
                f"Step04 实际客户端={sorted(parameter_clients)}"
            )

        for client_id in sorted(
            parameter_clients
        ):
            layer_names = list(
                shared_layer_names[client_id]
            )

            parameter_layer_names = list(
                client_shared_parameters[
                    client_id
                ].keys()
            )

            if set(layer_names) != set(
                parameter_layer_names
            ):
                raise ValueError(
                    f"Client {client_id} 的 "
                    "shared_layer_names 与 "
                    "client_shared_parameters 层集合不一致："
                    f"names={layer_names}, "
                    f"params={parameter_layer_names}"
                )

            if not layer_names:
                raise ValueError(
                    f"Client {client_id} 没有共享层；"
                    "当前系统至少要求共享 1 层。"
                )

            for layer_name in layer_names:
                _layer_name_to_index(
                    layer_name
                )

    # --------------------------------------------------------
    # attacked layer 解析
    # --------------------------------------------------------

    def _resolve_client_attacked_layers(
        self,
        client_id: int,
    ) -> List[str]:
        """
        使用 Step 00 的 resolve_attacked_layers(...)。

        这里不复制随机选层逻辑。
        """

        cfg = self.attack_plan.get(
            client_id
        )

        attack_type = _attack_type_value(
            cfg
        )

        if (
            not bool(
                getattr(
                    cfg,
                    "is_byzantine",
                    False,
                )
            )
            or attack_type
            not in {
                GAUSSIAN_VALUE,
                SIGN_FLIPPING_VALUE,
            }
        ):
            return []

        layer_attack_ratio = float(
            getattr(
                cfg,
                "layer_attack_ratio",
                0.0,
            )
        )

        # 用户定义的语义：
        #
        # Gaussian / Sign Flipping Byzantine Client
        # 必须真正攻击至少 1 个共享层。
        #
        # 0 < ratio < 1 时，
        # Step 00 resolver 已通过 max(1, attack_count)
        # 防止 int / round 后得到 0。
        #
        # 如果 ratio 本身就是 0，
        # 则这已经不是“比例取整 bug”，而是配置矛盾，
        # 因此这里直接报错，不静默变成无攻击。
        if layer_attack_ratio <= 0:
            raise ValueError(
                f"Client {client_id} 已被配置为 "
                f"{attack_type} Byzantine Client，"
                "但 layer_attack_ratio <= 0。"
                "模型参数攻击客户端必须配置正的攻击层比例，"
                "以保证至少攻击 1 个共享层。"
            )

        shared_layer_names = list(
            self.local_training_result
            .shared_layer_names[
                client_id
            ]
        )

        shared_layer_indices = [
            _layer_name_to_index(
                layer_name
            )
            for layer_name in shared_layer_names
        ]

        attacked_indices = (
            resolve_attacked_layers(
                client_config=cfg,
                shared_layers=(
                    shared_layer_indices
                ),
            )
        )

        attacked_names = [
            _layer_index_to_name(
                layer_index
            )
            for layer_index in attacked_indices
        ]

        # ----------------------------------------------------
        # 硬约束 1：
        # 模型参数 Byzantine Client 至少攻击 1 层。
        # ----------------------------------------------------

        if not attacked_names:
            raise RuntimeError(
                f"Client {client_id} 是 "
                f"{attack_type} Byzantine Client，"
                "但没有解析出任何 attacked layer。"
            )

        # ----------------------------------------------------
        # 硬约束 2：
        # attacked layer 必须属于该客户端 Shared Layers。
        # ----------------------------------------------------

        invalid_layers = (
            set(attacked_names)
            - set(shared_layer_names)
        )

        if invalid_layers:
            raise RuntimeError(
                f"Client {client_id} 出现非法攻击层："
                f"{sorted(invalid_layers)}；"
                "模型参数攻击只能作用于 Shared Layers。"
            )

        return attacked_names

    # --------------------------------------------------------
    # 攻击单层
    # --------------------------------------------------------

    def _attack_one_layer(
        self,
        client_id: int,
        layer_name: str,
        tensors: Sequence[torch.Tensor],
        client_config: ClientAttackConfig,
    ) -> tuple[
        List[torch.Tensor],
        LayerAttackVisualization,
    ]:
        attack_type = _attack_type_value(
            client_config
        )

        attacked_tensors: List[
            torch.Tensor
        ] = []

        tensor_changes: List[
            TensorAttackVisualization
        ] = []

        layer_index = _layer_name_to_index(
            layer_name
        )

        for tensor_index, tensor in enumerate(
            tensors
        ):
            before = (
                tensor.detach()
                .cpu()
                .clone()
            )

            if attack_type == SIGN_FLIPPING_VALUE:
                scale = float(
                    getattr(
                        client_config,
                        "sign_flip_scale",
                        -1.0,
                    )
                )

                after = _apply_sign_flipping(
                    tensor=before,
                    scale=scale,
                )

            elif attack_type == GAUSSIAN_VALUE:
                mean = float(
                    getattr(
                        client_config,
                        "gaussian_mean",
                        0.0,
                    )
                )

                std = float(
                    getattr(
                        client_config,
                        "gaussian_std",
                        1.0,
                    )
                )

                seed = (
                    _derive_gaussian_tensor_seed(
                        master_seed=int(
                            self.attack_plan
                            .master_seed
                        ),
                        client_id=client_id,
                        round_index=(
                            self.round_index
                        ),
                        layer_index=layer_index,
                        tensor_index=tensor_index,
                    )
                )

                after = (
                    _apply_gaussian_replacement(
                        tensor=before,
                        mean=mean,
                        std=std,
                        seed=seed,
                    )
                )

            else:
                raise RuntimeError(
                    f"Client {client_id} / "
                    f"{layer_name} 收到不支持的 "
                    f"Step 05 attack_type={attack_type!r}"
                )

            attacked_tensors.append(
                after
            )

            tensor_changes.append(
                _make_tensor_visualization(
                    tensor_index=tensor_index,
                    tensor_count=len(tensors),
                    before=before,
                    after=after,
                    preview_values_per_tensor=(
                        self.preview_values_per_tensor
                    ),
                )
            )

        return (
            attacked_tensors,
            LayerAttackVisualization(
                layer_name=layer_name,
                attack_type=str(
                    attack_type
                ),
                tensor_changes=tensor_changes,
            ),
        )

    # --------------------------------------------------------
    # 主入口
    # --------------------------------------------------------

    def apply(
        self,
    ) -> ParameterAttackResult:
        """
        执行 Step 05。

        不原地修改 Step 04 的任何 Tensor。
        """

        step04_shared_parameters = (
            self.local_training_result
            .client_shared_parameters
        )

        original_parameters = (
            _clone_parameter_structure(
                step04_shared_parameters
            )
        )

        attacked_parameters = (
            _clone_parameter_structure(
                step04_shared_parameters
            )
        )

        attacked_clients: List[int] = []
        skipped_label_clients: List[
            int
        ] = []

        records: Dict[
            int,
            ClientParameterAttackRecord,
        ] = {}

        for client_id in range(
            int(self.attack_plan.num_clients)
        ):
            cfg = self.attack_plan.get(
                client_id
            )

            is_byzantine = bool(
                getattr(
                    cfg,
                    "is_byzantine",
                    False,
                )
            )

            attack_type = (
                _attack_type_value(
                    cfg
                )
            )

            shared_layers = list(
                self.local_training_result
                .shared_layer_names[
                    client_id
                ]
            )

            # =================================================
            # Benign Client
            # =================================================

            if not is_byzantine:
                records[
                    client_id
                ] = ClientParameterAttackRecord(
                    client_id=client_id,
                    is_byzantine=False,
                    attack_type=None,
                    shared_layers=(
                        shared_layers.copy()
                    ),
                    attacked_layers=[],
                    attacked_layer_count=0,
                    layer_attack_ratio=0.0,
                    gaussian_mean=None,
                    gaussian_std=None,
                    sign_flip_scale=None,
                    status="benign_no_parameter_attack",
                    parameter_changes=[],
                )

                continue

            # =================================================
            # Label Flipping
            # 已经在 Step 02 执行，不重复攻击模型参数
            # =================================================

            if attack_type == LABEL_FLIPPING_VALUE:
                skipped_label_clients.append(
                    client_id
                )

                records[
                    client_id
                ] = ClientParameterAttackRecord(
                    client_id=client_id,
                    is_byzantine=True,
                    attack_type=attack_type,
                    shared_layers=(
                        shared_layers.copy()
                    ),
                    attacked_layers=[],
                    attacked_layer_count=0,
                    layer_attack_ratio=0.0,
                    gaussian_mean=None,
                    gaussian_std=None,
                    sign_flip_scale=None,
                    status=(
                        "label_flipping_already_applied_"
                        "in_step_02"
                    ),
                    parameter_changes=[],
                )

                continue

            # =================================================
            # Gaussian / Sign Flipping
            # =================================================

            if attack_type not in {
                GAUSSIAN_VALUE,
                SIGN_FLIPPING_VALUE,
            }:
                raise ValueError(
                    f"Client {client_id} 是 Byzantine Client，"
                    f"但 attack_type={attack_type!r} "
                    "不是当前 Step 05 支持的类型。"
                )

            attacked_layers = (
                self._resolve_client_attacked_layers(
                    client_id
                )
            )

            parameter_changes: List[
                LayerAttackVisualization
            ] = []

            for layer_name in attacked_layers:
                original_layer_tensors = (
                    original_parameters[
                        client_id
                    ][
                        layer_name
                    ]
                )

                (
                    attacked_layer_tensors,
                    layer_visualization,
                ) = self._attack_one_layer(
                    client_id=client_id,
                    layer_name=layer_name,
                    tensors=(
                        original_layer_tensors
                    ),
                    client_config=cfg,
                )

                attacked_parameters[
                    client_id
                ][
                    layer_name
                ] = attacked_layer_tensors

                parameter_changes.append(
                    layer_visualization
                )

            attacked_clients.append(
                client_id
            )

            if attack_type == GAUSSIAN_VALUE:
                gaussian_mean = float(
                    getattr(
                        cfg,
                        "gaussian_mean",
                        0.0,
                    )
                )
                gaussian_std = float(
                    getattr(
                        cfg,
                        "gaussian_std",
                        1.0,
                    )
                )
                sign_flip_scale = None

            else:
                gaussian_mean = None
                gaussian_std = None
                sign_flip_scale = float(
                    getattr(
                        cfg,
                        "sign_flip_scale",
                        -1.0,
                    )
                )

            records[
                client_id
            ] = ClientParameterAttackRecord(
                client_id=client_id,
                is_byzantine=True,
                attack_type=attack_type,
                shared_layers=(
                    shared_layers.copy()
                ),
                attacked_layers=(
                    attacked_layers.copy()
                ),
                attacked_layer_count=len(
                    attacked_layers
                ),
                layer_attack_ratio=float(
                    getattr(
                        cfg,
                        "layer_attack_ratio",
                        0.0,
                    )
                ),
                gaussian_mean=gaussian_mean,
                gaussian_std=gaussian_std,
                sign_flip_scale=(
                    sign_flip_scale
                ),
                status=(
                    "parameter_attack_applied"
                ),
                parameter_changes=(
                    parameter_changes
                ),
            )

        return ParameterAttackResult(
            round_index=self.round_index,
            original_client_shared_parameters=(
                original_parameters
            ),
            attacked_client_shared_parameters=(
                attacked_parameters
            ),
            attacked_clients=(
                attacked_clients
            ),
            skipped_label_flipping_clients=(
                skipped_label_clients
            ),
            client_attack_records=records,
        )


# ============================================================
# 正式对外入口
# ============================================================

def apply_parameter_attacks(
    attack_plan: AttackPlan,
    local_training_result: Any,
    preview_values_per_tensor: int = (
        DEFAULT_PREVIEW_VALUES_PER_TENSOR
    ),
) -> ParameterAttackResult:
    """
    Step 05 推荐正式入口。

    正式 main.py：

        attack_result = apply_parameter_attacks(
            attack_plan=attack_plan,
            local_training_result=local_training_result,
        )

    后续 Step 06：

        attacked_shared_parameters = (
            attack_result
            .attacked_client_shared_parameters
        )
    """

    attacker = (
        GaussianAndSignFlippingAttacker(
            attack_plan=attack_plan,
            local_training_result=(
                local_training_result
            ),
            preview_values_per_tensor=(
                preview_values_per_tensor
            ),
        )
    )

    return attacker.apply()


# ============================================================
# 控制台 / 可视化调试输出
# ============================================================

def print_parameter_attack_result(
    result: ParameterAttackResult,
) -> None:
    print(
        "\n"
        + "=" * 82
    )

    print(
        "Step 05 - Gaussian / Sign Flipping Parameter Attack"
    )

    print(
        "=" * 82
    )

    print(
        f"Round Index                    : "
        f"{result.round_index}"
    )

    print(
        f"Parameter Attacked Clients     : "
        f"{result.attacked_clients}"
    )

    print(
        f"Skipped Label-Flipping Clients : "
        f"{result.skipped_label_flipping_clients}"
    )

    for client_id in sorted(
        result.client_attack_records
    ):
        record = (
            result.client_attack_records[
                client_id
            ]
        )

        print(
            "\n"
            + "-" * 82
        )

        print(
            f"Client {client_id:02d}"
        )

        print(
            "-" * 82
        )

        print(
            f"Is Byzantine       : "
            f"{record.is_byzantine}"
        )

        print(
            f"Attack Type         : "
            f"{record.attack_type}"
        )

        print(
            f"Status              : "
            f"{record.status}"
        )

        print(
            f"Shared Layers       : "
            f"{record.shared_layers}"
        )

        print(
            f"Attacked Layers     : "
            f"{record.attacked_layers}"
        )

        print(
            f"Attack Layer Count  : "
            f"{record.attacked_layer_count}"
        )

        if (
            record.status
            == "parameter_attack_applied"
        ):
            print(
                f"Layer Attack Ratio  : "
                f"{record.layer_attack_ratio:.4f}"
            )

            if (
                record.attack_type
                == GAUSSIAN_VALUE
            ):
                print(
                    f"Gaussian Mean       : "
                    f"{record.gaussian_mean}"
                )
                print(
                    f"Gaussian Std        : "
                    f"{record.gaussian_std}"
                )

            elif (
                record.attack_type
                == SIGN_FLIPPING_VALUE
            ):
                print(
                    f"Sign Flip Scale     : "
                    f"{record.sign_flip_scale}"
                )

            for layer_record in (
                record.parameter_changes
            ):
                print(
                    f"\n  {layer_record.layer_name}"
                )

                for tensor_record in (
                    layer_record.tensor_changes
                ):
                    print(
                        "    "
                        f"{tensor_record.tensor_role} "
                        f"shape={tensor_record.shape}"
                    )

                    print(
                        "      "
                        f"before_mean="
                        f"{tensor_record.before_mean:.6f} | "
                        f"after_mean="
                        f"{tensor_record.after_mean:.6f}"
                    )

                    print(
                        "      "
                        f"before_norm="
                        f"{tensor_record.before_norm:.6f} | "
                        f"after_norm="
                        f"{tensor_record.after_norm:.6f} | "
                        f"delta_norm="
                        f"{tensor_record.delta_norm:.6f}"
                    )

                    print(
                        "      "
                        f"before_preview="
                        f"{tensor_record.before_preview}"
                    )

                    print(
                        "      "
                        f"after_preview="
                        f"{tensor_record.after_preview}"
                    )


# ============================================================
# 这个是测试样例
# ============================================================

if __name__ == "__main__":
    # ========================================================
    # 独立测试目的
    # ========================================================
    #
    # 1. 构造 Step 00 AttackPlan；
    # 2. 构造一个最小的“Step 04 风格输出”；
    # 3. 执行 Step 05；
    # 4. 自动检查：
    #
    #    - Benign Client 参数完全不变；
    #    - Label Flipping Client 参数完全不变；
    #    - Gaussian / Sign Flipping 至少攻击 1 个共享层；
    #    - attacked_layers 全部属于 Shared Layers；
    #    - 未攻击共享层参数保持不变；
    #    - Sign Flipping 确实等于 scale * theta；
    #    - Gaussian Replacement 可复现；
    #    - Step 04 原始参数未被原地修改。
    #
    # 注意：
    # 这里使用随机小 Tensor，只验证 Step 05 自己。
    # 正式系统中输入必须来自真实 Step 04。
    # ========================================================

    from dataclasses import dataclass as _dataclass

    from Step_00_V_byzantine_config import (
        ByzantineConfigGenerator,
    )

    @_dataclass
    class _FakeLocalTrainingResult:
        round_index: int

        client_shared_parameters: Dict[
            int,
            Dict[str, List[torch.Tensor]],
        ]

        shared_layer_names: Dict[
            int,
            List[str],
        ]

    # --------------------------------------------------------
    # 为了确保三种客户端状态都有，
    # 这里手工构造 ClientAttackConfig，
    # 不依赖按比例分配恰好得到某种类型。
    # --------------------------------------------------------

    fake_plan = AttackPlan(
        master_seed=42,
        num_clients=4,
        byzantine_ratio=0.75,
        client_configs={
            0: ClientAttackConfig(
                client_id=0,
                is_byzantine=False,
            ),

            1: ClientAttackConfig(
                client_id=1,
                is_byzantine=True,
                attack_type=(
                    AttackType.LABEL_FLIPPING
                ),
                layer_attack_ratio=0.0,
                layer_selection_seed=None,
            ),

            2: ClientAttackConfig(
                client_id=2,
                is_byzantine=True,
                attack_type=(
                    AttackType.SIGN_FLIPPING
                ),
                layer_attack_ratio=0.25,

                # 共享层只有 1 层，
                # 0.25 层不能变成 0 层，
                # 必须最终攻击 layer1。
                layer_selection_seed=12345,
                sign_flip_scale=-1.0,
            ),

            3: ClientAttackConfig(
                client_id=3,
                is_byzantine=True,
                attack_type=(
                    AttackType.GAUSSIAN
                ),
                layer_attack_ratio=0.40,
                layer_selection_seed=67890,
                gaussian_mean=0.0,
                gaussian_std=1.0,
            ),
        },
    )

    generator = torch.Generator()
    generator.manual_seed(7)

    fake_shared_parameters = {
        # Client 0：Benign，3 层共享
        0: {
            "layer1": [
                torch.randn(
                    3,
                    2,
                    generator=generator,
                ),
                torch.randn(
                    3,
                    generator=generator,
                ),
            ],
            "layer2": [
                torch.randn(
                    2,
                    3,
                    generator=generator,
                ),
                torch.randn(
                    2,
                    generator=generator,
                ),
            ],
            "layer3": [
                torch.randn(
                    4,
                    2,
                    generator=generator,
                ),
                torch.randn(
                    4,
                    generator=generator,
                ),
            ],
        },

        # Client 1：Label Flipping，2 层共享
        1: {
            "layer1": [
                torch.randn(
                    3,
                    2,
                    generator=generator,
                ),
                torch.randn(
                    3,
                    generator=generator,
                ),
            ],
            "layer2": [
                torch.randn(
                    2,
                    3,
                    generator=generator,
                ),
                torch.randn(
                    2,
                    generator=generator,
                ),
            ],
        },

        # Client 2：Sign Flipping，只有 1 个共享层
        2: {
            "layer1": [
                torch.randn(
                    3,
                    2,
                    generator=generator,
                ),
                torch.randn(
                    3,
                    generator=generator,
                ),
            ],
        },

        # Client 3：Gaussian，K=5 风格全共享
        3: {
            "layer1": [
                torch.randn(
                    3,
                    2,
                    generator=generator,
                ),
                torch.randn(
                    3,
                    generator=generator,
                ),
            ],
            "layer2": [
                torch.randn(
                    2,
                    3,
                    generator=generator,
                ),
                torch.randn(
                    2,
                    generator=generator,
                ),
            ],
            "layer3": [
                torch.randn(
                    4,
                    2,
                    generator=generator,
                ),
                torch.randn(
                    4,
                    generator=generator,
                ),
            ],
            "layer4": [
                torch.randn(
                    2,
                    4,
                    generator=generator,
                ),
                torch.randn(
                    2,
                    generator=generator,
                ),
            ],
            "layer5": [
                torch.randn(
                    2,
                    2,
                    generator=generator,
                ),
                torch.randn(
                    2,
                    generator=generator,
                ),
            ],
        },
    }

    fake_result = _FakeLocalTrainingResult(
        round_index=1,
        client_shared_parameters=(
            fake_shared_parameters
        ),
        shared_layer_names={
            client_id: list(
                layer_map.keys()
            )
            for client_id, layer_map in (
                fake_shared_parameters.items()
            )
        },
    )

    # 保存 Step 04 风格原参数副本，
    # 用于检查没有被 Step 05 原地污染。
    step04_before = (
        _clone_parameter_structure(
            fake_result
            .client_shared_parameters
        )
    )

    result = apply_parameter_attacks(
        attack_plan=fake_plan,
        local_training_result=(
            fake_result
        ),
        preview_values_per_tensor=3,
    )

    print_parameter_attack_result(
        result
    )

    print(
        "\n"
        + "=" * 82
    )

    print(
        "Automatic Checks"
    )

    print(
        "=" * 82
    )

    def _layer_equal(
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> bool:
        return (
            len(a) == len(b)
            and all(
                torch.equal(x, y)
                for x, y in zip(a, b)
            )
        )

    def _client_equal(
        a: Mapping[
            str,
            Sequence[torch.Tensor],
        ],
        b: Mapping[
            str,
            Sequence[torch.Tensor],
        ],
    ) -> bool:
        return (
            set(a.keys()) == set(b.keys())
            and all(
                _layer_equal(
                    a[layer_name],
                    b[layer_name],
                )
                for layer_name in a
            )
        )

    # --------------------------------------------------------
    # 1. Benign 参数保持不变
    # --------------------------------------------------------

    benign_unchanged = _client_equal(
        result.original_client_shared_parameters[
            0
        ],
        result.attacked_client_shared_parameters[
            0
        ],
    )

    print(
        "Benign Client 参数保持不变 :",
        benign_unchanged,
    )

    # --------------------------------------------------------
    # 2. Label Flipping Client 在 Step 05 参数保持不变
    # --------------------------------------------------------

    label_client_unchanged = (
        _client_equal(
            result.original_client_shared_parameters[
                1
            ],
            result.attacked_client_shared_parameters[
                1
            ],
        )
    )

    print(
        "Label Flipping Client 在 Step 05 参数保持不变 :",
        label_client_unchanged,
    )

    # --------------------------------------------------------
    # 3. Parameter Byzantine Client 至少攻击 1 层
    # --------------------------------------------------------

    at_least_one_layer = all(
        len(
            result.client_attack_records[
                client_id
            ].attacked_layers
        )
        >= 1
        for client_id in [
            2,
            3,
        ]
    )

    print(
        "Gaussian / Sign Flipping 至少攻击 1 个共享层 :",
        at_least_one_layer,
    )

    # --------------------------------------------------------
    # 4. attacked layers 全部属于 Shared Layers
    # --------------------------------------------------------

    attacked_only_shared = all(
        set(
            result.client_attack_records[
                client_id
            ].attacked_layers
        ).issubset(
            set(
                result.client_attack_records[
                    client_id
                ].shared_layers
            )
        )
        for client_id in [
            2,
            3,
        ]
    )

    print(
        "所有 attacked layers 均属于 Shared Layers :",
        attacked_only_shared,
    )

    # --------------------------------------------------------
    # 5. Client 2 只有 1 个共享层，ratio=0.25，
    #    最终必须仍攻击 layer1
    # --------------------------------------------------------

    one_layer_minimum_rule = (
        result.client_attack_records[
            2
        ].attacked_layers
        == ["layer1"]
    )

    print(
        "1 个共享层 × 0.25 时仍至少攻击 layer1 :",
        one_layer_minimum_rule,
    )

    # --------------------------------------------------------
    # 6. Sign Flipping = -theta
    # --------------------------------------------------------

    sign_correct = all(
        torch.equal(
            attacked_tensor,
            -original_tensor,
        )
        for original_tensor, attacked_tensor in zip(
            result.original_client_shared_parameters[
                2
            ][
                "layer1"
            ],
            result.attacked_client_shared_parameters[
                2
            ][
                "layer1"
            ],
        )
    )

    print(
        "Sign Flipping 默认 scale=-1 实现正确 :",
        sign_correct,
    )

    # --------------------------------------------------------
    # 7. Gaussian 重跑可复现
    # --------------------------------------------------------

    repeat_result = apply_parameter_attacks(
        attack_plan=fake_plan,
        local_training_result=(
            fake_result
        ),
        preview_values_per_tensor=3,
    )

    gaussian_reproducible = (
        _client_equal(
            result.attacked_client_shared_parameters[
                3
            ],
            repeat_result.attacked_client_shared_parameters[
                3
            ],
        )
    )

    print(
        "同一 AttackPlan / Round 下 Gaussian 参数攻击可复现 :",
        gaussian_reproducible,
    )

    # --------------------------------------------------------
    # 8. Step 04 原参数没有被原地修改
    # --------------------------------------------------------

    step04_not_mutated = all(
        _client_equal(
            step04_before[client_id],
            fake_result.client_shared_parameters[
                client_id
            ],
        )
        for client_id in (
            fake_result
            .client_shared_parameters
        )
    )

    print(
        "Step 04 原始共享参数未被 Step 05 原地修改 :",
        step04_not_mutated,
    )

    # --------------------------------------------------------
    # 9. Step 06 所需输出已准备好
    # --------------------------------------------------------

    step06_ready = (
        set(
            result
            .attacked_client_shared_parameters
            .keys()
        )
        == set(
            fake_result
            .client_shared_parameters
            .keys()
        )
    )

    print(
        "Step 06 所需 attacked_client_shared_parameters 已准备好 :",
        step06_ready,
    )
