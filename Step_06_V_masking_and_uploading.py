"""
Step_06_V_masking_and_uploading.py

============================================================
Step 06：基于零和随机掩码的模型参数保护与逻辑上传（可视化相关）
============================================================

【本步骤在完整流程中的位置】

    Step 04 Local Model Training
        |
        |  clean client_shared_parameters
        |  （如果当前实验没有模型参数攻击，也可以直接进入 Step 06）
        v

    Step 05 Gaussian / Sign Flipping Attack
        |
        |  attacked_client_shared_parameters
        |  （如果当前实验启用了模型参数攻击，优先使用这一份）
        v

    --------------------------------------------------------
    Step 06  <-- 本文件
    Random-Mask Protection + Logical Uploading
    --------------------------------------------------------
        |
        |  TS / DS1 / DS2 / DS3 四份受掩码保护参数
        |  对应四份零和 Random Mask
        v

    Step 07+
    后续安全计算 / Byzantine Detection / Aggregation


============================================================
【本步骤核心数学逻辑】
============================================================

对于某个客户端、某个共享层、某个参数 Tensor：

    theta

这里 theta 可以是：

    1. Step 04 的正常本地训练参数；
    2. Step 05 已经经过 Gaussian / Sign Flipping 后的参数。

本步骤固定拆成 4 份：

    TS
    DS1
    DS2
    DS3

首先生成四个随机掩码：

    r_TS
    r_DS1
    r_DS2
    r_DS3

要求：

    r_TS + r_DS1 + r_DS2 + r_DS3 = 0

实现时随机生成前三个：

    r_TS
    r_DS1
    r_DS2

然后令：

    r_DS3 = -(r_TS + r_DS1 + r_DS2)

因此严格满足“零和掩码”。


随后构造四份受掩码保护的模型参数：

    omega_TS  = theta + r_TS
    omega_DS1 = theta + r_DS1
    omega_DS2 = theta + r_DS2
    omega_DS3 = theta + r_DS3

将四份受掩码保护参数相加：

    omega_TS + omega_DS1 + omega_DS2 + omega_DS3

  = theta + theta + theta + theta
    + r_TS + r_DS1 + r_DS2 + r_DS3

  = 4 * theta + 0

  = 4 * theta

因此：

    sum(mask_i)  = 0
    sum(masked_parameter_i) = 4 * theta

这就是本步骤希望在可视化页面中重点展示的两个“不变量”。


============================================================
【为什么这里使用 theta + r_i】
============================================================

如果写成：

    theta + r_i

并且四个 r_i 零和，

由于四个 Mask 零和，四份受掩码保护参数求和得到：

    4 * theta

若后续协议需要由四份数据恢复 theta，可再除以 4。

这里不采用 theta/4 + r_i 的加法份额形式。
本步骤定位为“基于随机掩码的秘密共享技术”：
每个逻辑服务器获得一份 theta + r_i 形式的受掩码保护参数，
而不是 Shamir (n,t) 门限秘密共享份额。


============================================================
【重要术语说明】
============================================================

本文件把：

    r_TS, r_DS1, r_DS2, r_DS3

统一称为：

    Mask / Random Mask / 随机掩码

它们不是传统密码系统中的：

    Encryption Key / Secret Key / 密钥

因此前端最好显示：

    “随机掩码 Mask”

而不是直接写：

    “密钥 Key”

如果比赛展示阶段为了通俗说明需要写“秘密随机量”，也可以；
但代码和技术文档中建议继续使用 Mask。

另外：

    masks_by_server

虽然作为 Step 06 的输出保留，
主要用于：
    - 后续算法研究；
    - 调试；
    - 可视化演示“零和”性质。

真实系统中不能把某个服务器自己的：

    omega_i
    mask_i

同时公开给该服务器作为长期可见信息，
否则它可以计算：

    omega_i - mask_i = theta

从而恢复完整 theta。

因此：

    受掩码保护参数 omega_i 是逻辑上传给对应服务器的对象；
    Mask 是客户端侧/协议侧秘密随机量。

本 Demo 可以为了讲解而展示 Mask，
但页面应明确标注：

    “仅教学/协议可视化展示，实际服务器不可同时获得对应 Mask”。


============================================================
【逻辑服务器，而不是强制部署 4 台机器】
============================================================

本工程将四个接收角色表示为：

    TS
    DS1
    DS2
    DS3

这里它们可以只是：

    4 个逻辑计算/存储角色

不要求真的启动四台物理服务器，也不要求本步骤模拟：

    socket
    HTTP
    RPC
    网络时延
    连接失败

本步骤的重点是：

    参数如何拆分
    ↓
    每个逻辑节点拿到什么
    ↓
    四份随机掩码如何保持零和，以及受掩码参数之间的求和关系

因此这里的“uploading”表示：

    将每一份受掩码保护参数放入对应逻辑服务器的输出容器

而不是实现真实网络通信。


============================================================
【输入方式：正式流程与独立 Demo 共用同一核心算法】
============================================================

本步骤核心函数：

    mask_and_split_shared_parameters(...)

只需要：

    client_shared_parameters
    master_seed
    round_index

它完全不要求自己调用 Step 04 或 Step 05。

因此：

------------------------------------------------------------
A. 正式完整流程：从 Step 05 输入
------------------------------------------------------------

如果当前实验执行了模型参数攻击：

    Step 05:
        attack_result.attacked_client_shared_parameters

    Step 06:
        mask_and_upload_from_step05(
            parameter_attack_result=attack_result,
            master_seed=attack_plan.master_seed,
        )

此时 theta 就是：

    “可能已经被攻击过”的共享参数。


------------------------------------------------------------
B. 正式完整流程：直接从 Step 04 输入
------------------------------------------------------------

如果当前实验不需要 Step 05 的模型参数攻击：

    Step 04:
        local_training_result.client_shared_parameters

    Step 06:
        mask_and_upload_from_step04(
            local_training_result=local_training_result,
            master_seed=attack_plan.master_seed,
        )


------------------------------------------------------------
C. 独立机制 Demo / 可视化输入
------------------------------------------------------------

也可以直接构造：

    client_shared_parameters = {
        0: {
            "layer1": [tensor1, tensor2],
        }
    }

然后直接调用核心函数。

所以：

    Step 06 的算法本身
    不依赖“前面步骤必须真的运行”。

只是完整实验的 main.py 会把前面真实输出串起来。


============================================================
【本步骤正式输入】
============================================================

# 1. client_shared_parameters

结构与 Step 04 / Step 05 保持一致：

    {
        client_id: {
            "layer1": [
                weight_tensor,
                bias_tensor,
            ],
            "layer2": [
                weight_tensor,
                bias_tensor,
            ],
            ...
        }
    }

注意：

    这里只处理 Shared Parameters。

Personalized Parameters：

    不上传
    不拆分
    不掩码


# 2. master_seed

用于确定性生成随机掩码。

正式完整流程中建议直接沿用：

    attack_plan.master_seed

本步骤会继续结合：

    client_id
    round_index
    layer_index
    tensor_index
    server_index

派生稳定随机种子。


# 3. round_index

当前联邦学习轮次。

来自：

    Step 04 LocalTrainingResult.round_index

或：

    Step 05 ParameterAttackResult.round_index

不同轮次必须产生不同 Mask，
因此 round_index 会进入 Mask Seed 派生过程。


# 4. mask_std

随机掩码前三份的高斯标准差。

默认：

    1.0

前三个独立 Mask：

    r_TS, r_DS1, r_DS2
        ~ N(0, mask_std^2)

第四个：

    r_DS3 = -(r_TS + r_DS1 + r_DS2)

因此第四个不是独立采样，
而是为了保证四个 Mask 严格零和。

说明：

    当前代码是在 PyTorch 浮点 Tensor 上模拟随机掩码。
    如果未来比赛材料需要严格论证“信息论秘密共享”，
    可以再将底层实现升级为定点编码 + 有限域加法共享。

当前 Step 06 的职责是工程模拟与后续算法接口，
不会在这里扩展有限域密码工程。


============================================================
【本步骤输出：MaskingAndUploadingResult】
============================================================

------------------------------------------------------------
1. original_client_shared_parameters
------------------------------------------------------------

Step 06 接收到的原始 theta 参数副本。

如果从 Step 05 进入：

    就是“可能经过攻击”的 theta。

如果从 Step 04 直接进入：

    就是正常训练后的 theta。


------------------------------------------------------------
2. masked_parameters_by_server
------------------------------------------------------------

四个逻辑服务器分别得到的受掩码保护参数：

    {
        "TS": {
            client_id: {
                "layer1": [masked_weight, masked_bias],
                ...
            }
        },

        "DS1": {...},
        "DS2": {...},
        "DS3": {...},
    }

后续模块应该优先使用这个字段。

对同一个 theta，四份受掩码保护参数满足：

    omega_TS + omega_DS1 + omega_DS2 + omega_DS3 = 4 * theta


------------------------------------------------------------
3. masks_by_server
------------------------------------------------------------

四个逻辑服务器对应的 Mask：

    {
        "TS":  {... r_TS ...},
        "DS1": {... r_DS1 ...},
        "DS2": {... r_DS2 ...},
        "DS3": {... r_DS3 ...},
    }

满足：

    r_TS + r_DS1 + r_DS2 + r_DS3 = 0

再次强调：

    这是协议内部 / 调试 / 可视化数据。

真实服务器的“接收数据”应视为：

    masked_parameters_by_server

而不是：

    share + mask 两者都明文交给服务器。


------------------------------------------------------------
4. upload_packets
------------------------------------------------------------

为后续模拟逻辑上传提供的结构化描述：

    server_name
    client_id
    layer_names

这里不复制整块 Tensor，
避免为了日志/可视化重复占用大量内存。


------------------------------------------------------------
5. tensor_records
------------------------------------------------------------

对每个：

    Client
    Layer
    Tensor

保留一条：

    TensorMaskingRecord

其中包含：

    original_shape
    original_mean
    original_std
    original_norm

    mask_norms
    masked_parameter_norms

    mask_sum_norm
    sum_relation_error_norm

    original_preview
    mask_previews
    masked_parameter_previews
    summed_masked_preview

这是给可视化同学最方便的结构化数据。


------------------------------------------------------------
6. visualization_data
------------------------------------------------------------

JSON-friendly 输出。

不直接塞高维 Tensor，
只提供页面应该展示的统计量和 preview。


============================================================
【给可视化同学的页面备注】
============================================================

Step 06 建议重点展示：

------------------------------------------------------------
A. 原始模型参数 theta
------------------------------------------------------------

不要直接显示全部 Tensor。

例如显示：

    Client 3
    layer2
    weight
    shape = [16, 6, 5, 5]

    original preview:
        [0.013, -0.021, 0.004, ...]

    norm:
        1.842


------------------------------------------------------------
B. 四份受掩码保护后的模型参数
------------------------------------------------------------

页面可以并排展示：

    TS
        omega_TS = theta + r_TS

    DS1
        omega_DS1 = theta + r_DS1

    DS2
        omega_DS2 = theta + r_DS2

    DS3
        omega_DS3 = theta + r_DS3

每一份显示：

    preview
    norm

即可。


------------------------------------------------------------
C. 对应四个随机掩码 Mask
------------------------------------------------------------

并排展示：

    r_TS
    r_DS1
    r_DS2
    r_DS3

每份只展示：

    preview
    norm

同时注明：

    “Mask 仅用于协议过程和可视化演示，
     真实逻辑服务器不会同时获得自己的受掩码保护参数与对应 Mask。”


------------------------------------------------------------
D. 零和掩码不变量
------------------------------------------------------------

页面突出展示：

    r_TS + r_DS1 + r_DS2 + r_DS3 = 0

并展示数值检查：

    ||sum(mask_i)||

理论上为 0；
浮点计算下应接近 0。


------------------------------------------------------------
E. 参数求和不变量
------------------------------------------------------------

页面突出展示：

    omega_TS + omega_DS1 + omega_DS2 + omega_DS3

      = 4 * theta
        + (r_TS + r_DS1 + r_DS2 + r_DS3)

      = 4 * theta

展示：

    sum_relation_error
        = ||sum(omega_i) - 4*theta||

应接近 0。


============================================================
【本步骤不负责】
============================================================

- 不执行本地训练；
- 不重新实施 Gaussian / Sign Flipping；
- 不重新选择 Byzantine Client；
- 不决定 Shared / Personalized Architecture；
- 不执行真实网络传输；
- 不执行后续 Byzantine Detection；
- 不执行 Layer Filtering；
- 不执行最终安全聚合；
- 不负责真正前端绘图；
- 不把 Mask 误称为传统意义上的 Encryption Key。

============================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch


# ============================================================
# 固定四个逻辑服务器：1 个 TS + 3 个 DS
# ============================================================

SERVER_NAMES: Tuple[str, str, str, str] = (
    "TS",
    "DS1",
    "DS2",
    "DS3",
)

NUM_SERVERS = 4

DEFAULT_MASK_STD = 1.0
DEFAULT_PREVIEW_VALUES_PER_TENSOR = 5

ALL_LAYER_NAMES = (
    "layer1",
    "layer2",
    "layer3",
    "layer4",
    "layer5",
)


# ============================================================
# 输出数据结构
# ============================================================

@dataclass(frozen=True)
class TensorMaskingRecord:
    """
    一个参数 Tensor 的随机掩码保护与可视化记录。

    注意：
        这里全部是统计数据/preview，
        真正高维 Tensor 保存在：

            masked_parameters_by_server
            masks_by_server
    """

    client_id: int
    layer_name: str
    tensor_index: int
    tensor_role: str

    shape: Tuple[int, ...]

    original_mean: float
    original_std: float
    original_norm: float

    original_preview: List[float]

    mask_norms: Dict[str, float]
    masked_parameter_norms: Dict[str, float]

    mask_previews: Dict[str, List[float]]
    masked_parameter_previews: Dict[str, List[float]]

    mask_sum_norm: float
    sum_relation_error_norm: float

    summed_masked_preview: List[float]


@dataclass(frozen=True)
class LogicalUploadPacket:
    """
    一个逻辑服务器收到某个客户端受掩码保护参数的描述。

    为避免重复复制高维 Tensor，
    这里只记录元数据。

    真正受掩码保护参数：
        result.masked_parameters_by_server[server_name][client_id]
    """

    server_name: str
    client_id: int
    layer_names: List[str]
    tensor_count: int


@dataclass
class MaskingAndUploadingResult:
    """
    Step 06 统一输出。
    """

    round_index: int
    master_seed: int
    mask_std: float

    server_names: Tuple[str, str, str, str]

    # Step 06 接收到的 theta 副本。
    original_client_shared_parameters: Dict[
        int,
        Dict[str, List[torch.Tensor]],
    ]

    # 逻辑上传给四个服务器的 4 份受掩码保护参数。
    #
    # 后续安全计算的核心输入。
    masked_parameters_by_server: Dict[
        str,
        Dict[
            int,
            Dict[str, List[torch.Tensor]],
        ],
    ]

    # 四个对应的零和随机 Mask。
    #
    # 主要用于协议内部 / 调试 / 可视化。
    masks_by_server: Dict[
        str,
        Dict[
            int,
            Dict[str, List[torch.Tensor]],
        ],
    ]

    upload_packets: List[LogicalUploadPacket]

    tensor_records: List[TensorMaskingRecord]

    @property
    def visualization_data(self) -> List[Dict[str, Any]]:
        """
        JSON-friendly 可视化数据。

        不直接把 Tensor 暴露给前端。
        """

        rows: List[Dict[str, Any]] = []

        for record in self.tensor_records:
            rows.append(
                {
                    "client_id": record.client_id,
                    "layer_name": record.layer_name,
                    "tensor_index": record.tensor_index,
                    "tensor_role": record.tensor_role,
                    "shape": list(record.shape),

                    "original_mean": record.original_mean,
                    "original_std": record.original_std,
                    "original_norm": record.original_norm,
                    "original_preview": record.original_preview.copy(),

                    "mask_norms": dict(record.mask_norms),
                    "masked_parameter_norms": dict(record.masked_parameter_norms),

                    "mask_previews": {
                        server_name: values.copy()
                        for server_name, values
                        in record.mask_previews.items()
                    },

                    "masked_parameter_previews": {
                        server_name: values.copy()
                        for server_name, values
                        in record.masked_parameter_previews.items()
                    },

                    "mask_sum_norm": record.mask_sum_norm,
                    "sum_relation_error_norm": (
                        record.sum_relation_error_norm
                    ),

                    "summed_masked_preview": (
                        record.summed_masked_preview.copy()
                    ),

                    # 前端可以直接显示这两个公式文本。
                    "mask_equation": (
                        "r_TS + r_DS1 + r_DS2 + r_DS3 = 0"
                    ),

                    "masked_parameter_equation": (
                        "omega_TS + omega_DS1 + omega_DS2 + omega_DS3 = 4 * theta"
                    ),
                }
            )

        return rows


# ============================================================
# 基础辅助函数
# ============================================================

def _clone_parameter_structure(
    client_shared_parameters: Mapping[
        int,
        Mapping[str, Sequence[torch.Tensor]],
    ],
) -> Dict[int, Dict[str, List[torch.Tensor]]]:
    """
    将输入共享参数完整克隆到 CPU。

    Step 06 不允许原地污染 Step 04 / Step 05 输出。
    """

    return {
        int(client_id): {
            str(layer_name): [
                tensor.detach().cpu().clone()
                for tensor in tensors
            ]
            for layer_name, tensors in layer_map.items()
        }
        for client_id, layer_map
        in client_shared_parameters.items()
    }


def _empty_server_parameter_structure() -> Dict[
    str,
    Dict[int, Dict[str, List[torch.Tensor]]],
]:
    return {
        server_name: {}
        for server_name in SERVER_NAMES
    }


def _layer_name_to_index(
    layer_name: str,
) -> int:
    if layer_name not in ALL_LAYER_NAMES:
        raise ValueError(
            f"非法逻辑层名称：{layer_name!r}；"
            f"当前只支持 {list(ALL_LAYER_NAMES)}"
        )

    return ALL_LAYER_NAMES.index(layer_name) + 1


def _tensor_role(
    tensor_index: int,
    tensor_count: int,
) -> str:
    if tensor_count >= 1 and tensor_index == 0:
        return "weight"

    if tensor_count >= 2 and tensor_index == 1:
        return "bias"

    return f"param_{tensor_index}"


def _tensor_std(
    tensor: torch.Tensor,
) -> float:
    flat = tensor.detach().float().reshape(-1)

    if flat.numel() == 0:
        return 0.0

    return float(
        flat.std(unbiased=False).item()
    )


def _tensor_norm(
    tensor: torch.Tensor,
) -> float:
    flat = tensor.detach().float().reshape(-1)

    if flat.numel() == 0:
        return 0.0

    return float(
        torch.linalg.vector_norm(flat).item()
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


# ============================================================
# Mask Seed 派生
# ============================================================

def _derive_mask_seed(
    master_seed: int,
    client_id: int,
    round_index: int,
    layer_index: int,
    tensor_index: int,
    independent_mask_index: int,
) -> int:
    """
    为前三份独立随机 Mask 派生稳定 Seed。

    注意：
        DS3 的 Mask 不独立采样，
        因此不会调用本函数生成 DS3 seed。

    不使用 Python hash()，
    避免不同 Python 进程中的 hash randomization。
    """

    value = int(master_seed) & 0x7FFFFFFFFFFFFFFF

    value = (
        value * 1_000_003
        + int(client_id) * 97_409
        + int(round_index) * 1_299_709
        + int(layer_index) * 15_485_863
        + int(tensor_index) * 32_452_843
        + int(independent_mask_index) * 49_979_687
        + 0x4D41534B34
    )

    return int(
        value & 0x7FFFFFFFFFFFFFFF
    )


# ============================================================
# 四服务器随机掩码保护核心算法
# ============================================================

def _sample_independent_mask(
    reference_tensor: torch.Tensor,
    mask_std: float,
    seed: int,
) -> torch.Tensor:
    """
    在 CPU 上生成与 reference_tensor 同形状、同 dtype 的随机 Mask。

    mask ~ N(0, mask_std^2)
    """

    if mask_std < 0:
        raise ValueError(
            f"mask_std 必须 >= 0，当前为 {mask_std}"
        )

    reference = (
        reference_tensor.detach()
        .cpu()
    )

    if not (
        reference.is_floating_point()
        or reference.is_complex()
    ):
        raise TypeError(
            "模型参数 Tensor 必须是浮点/复数类型，"
            f"当前 dtype={reference.dtype}"
        )

    if reference.is_complex():
        raise TypeError(
            "当前 Step 06 暂不处理复数模型参数 Tensor。"
        )

    generator = torch.Generator(
        device="cpu"
    )
    generator.manual_seed(
        int(seed)
    )

    mask = torch.randn(
        reference.shape,
        generator=generator,
        dtype=reference.dtype,
        device="cpu",
    )

    return mask * float(mask_std)


def _mask_tensor_for_four_servers(
    theta: torch.Tensor,
    master_seed: int,
    client_id: int,
    round_index: int,
    layer_index: int,
    tensor_index: int,
    mask_std: float,
) -> tuple[
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
]:
    """
    对单个 theta 构造四份受掩码保护参数：

        omega_i = theta + r_i

    并满足：

        sum(r_i) = 0
        sum(omega_i) = 4 * theta
    """

    theta_cpu = (
        theta.detach()
        .cpu()
        .clone()
    )

    # --------------------------------------------------------
    # 前三个 Mask 独立随机生成
    # --------------------------------------------------------

    independent_masks: List[
        torch.Tensor
    ] = []

    for independent_mask_index in range(3):
        seed = _derive_mask_seed(
            master_seed=master_seed,
            client_id=client_id,
            round_index=round_index,
            layer_index=layer_index,
            tensor_index=tensor_index,
            independent_mask_index=(
                independent_mask_index
            ),
        )

        independent_masks.append(
            _sample_independent_mask(
                reference_tensor=theta_cpu,
                mask_std=mask_std,
                seed=seed,
            )
        )

    r_ts = independent_masks[0]
    r_ds1 = independent_masks[1]
    r_ds2 = independent_masks[2]

    # --------------------------------------------------------
    # 第四份 Mask 强制补成零和
    # --------------------------------------------------------

    r_ds3 = -(
        r_ts
        + r_ds1
        + r_ds2
    )

    masks = {
        "TS": r_ts,
        "DS1": r_ds1,
        "DS2": r_ds2,
        "DS3": r_ds3,
    }

    # --------------------------------------------------------
    # 每个服务器得到完整 theta 加上其对应随机掩码
    # --------------------------------------------------------

    masked_parameters = {
        server_name: (
            theta_cpu
            + masks[server_name]
        )
        for server_name in SERVER_NAMES
    }

    return masked_parameters, masks


# ============================================================
# 统计 / 可视化记录
# ============================================================

def _build_tensor_record(
    client_id: int,
    layer_name: str,
    tensor_index: int,
    tensor_count: int,
    theta: torch.Tensor,
    masked_parameters: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    preview_values_per_tensor: int,
) -> TensorMaskingRecord:
    mask_sum = sum(
        (
            masks[server_name]
            for server_name in SERVER_NAMES
        ),
        torch.zeros_like(theta),
    )

    summed_masked = sum(
        (
            masked_parameters[server_name]
            for server_name in SERVER_NAMES
        ),
        torch.zeros_like(theta),
    )

    sum_relation_error = (
        summed_masked
        - float(NUM_SERVERS) * theta
    )

    return TensorMaskingRecord(
        client_id=client_id,
        layer_name=layer_name,
        tensor_index=tensor_index,
        tensor_role=_tensor_role(
            tensor_index,
            tensor_count,
        ),
        shape=tuple(theta.shape),

        original_mean=float(
            theta.detach()
            .float()
            .mean()
            .item()
        ),
        original_std=_tensor_std(theta),
        original_norm=_tensor_norm(theta),

        original_preview=_preview_values(
            theta,
            preview_values_per_tensor,
        ),

        mask_norms={
            server_name: _tensor_norm(
                masks[server_name]
            )
            for server_name in SERVER_NAMES
        },

        masked_parameter_norms={
            server_name: _tensor_norm(
                masked_parameters[server_name]
            )
            for server_name in SERVER_NAMES
        },

        mask_previews={
            server_name: _preview_values(
                masks[server_name],
                preview_values_per_tensor,
            )
            for server_name in SERVER_NAMES
        },

        masked_parameter_previews={
            server_name: _preview_values(
                masked_parameters[server_name],
                preview_values_per_tensor,
            )
            for server_name in SERVER_NAMES
        },

        mask_sum_norm=_tensor_norm(
            mask_sum
        ),

        sum_relation_error_norm=(
            _tensor_norm(
                sum_relation_error
            )
        ),

        summed_masked_preview=(
            _preview_values(
                summed_masked,
                preview_values_per_tensor,
            )
        ),
    )


# ============================================================
# 输入验证
# ============================================================

def _validate_client_shared_parameters(
    client_shared_parameters: Mapping[
        int,
        Mapping[str, Sequence[torch.Tensor]],
    ],
) -> None:
    if not client_shared_parameters:
        raise ValueError(
            "client_shared_parameters 不能为空"
        )

    for client_id, layer_map in (
        client_shared_parameters.items()
    ):
        if int(client_id) < 0:
            raise ValueError(
                f"client_id 不能为负数：{client_id}"
            )

        if not layer_map:
            raise ValueError(
                f"Client {client_id} 没有 Shared Parameters"
            )

        for layer_name, tensors in (
            layer_map.items()
        ):
            _layer_name_to_index(
                str(layer_name)
            )

            if not tensors:
                raise ValueError(
                    f"Client {client_id} / {layer_name} "
                    "没有参数 Tensor"
                )

            for tensor_index, tensor in enumerate(
                tensors
            ):
                if not isinstance(
                    tensor,
                    torch.Tensor,
                ):
                    raise TypeError(
                        f"Client {client_id} / {layer_name} / "
                        f"tensor {tensor_index} 不是 torch.Tensor"
                    )


# ============================================================
# Step 06 核心正式入口
# ============================================================

def mask_and_split_shared_parameters(
    client_shared_parameters: Mapping[
        int,
        Mapping[str, Sequence[torch.Tensor]],
    ],
    master_seed: int,
    round_index: int,
    mask_std: float = DEFAULT_MASK_STD,
    preview_values_per_tensor: int = (
        DEFAULT_PREVIEW_VALUES_PER_TENSOR
    ),
) -> MaskingAndUploadingResult:
    """
    Step 06 最核心、最通用的函数。

    它不关心输入来自：

        Step 04
        Step 05
        独立 Demo
        可视化适配器

    只要结构满足：

        client_shared_parameters

    就执行完全相同的四服务器随机掩码保护算法。
    """

    _validate_client_shared_parameters(
        client_shared_parameters
    )

    if int(round_index) < 0:
        raise ValueError(
            "round_index 必须 >= 0"
        )

    if float(mask_std) < 0:
        raise ValueError(
            "mask_std 必须 >= 0"
        )

    if int(preview_values_per_tensor) < 0:
        raise ValueError(
            "preview_values_per_tensor 必须 >= 0"
        )

    original_parameters = (
        _clone_parameter_structure(
            client_shared_parameters
        )
    )

    masked_parameters_by_server = (
        _empty_server_parameter_structure()
    )

    masks_by_server = (
        _empty_server_parameter_structure()
    )

    tensor_records: List[
        TensorMaskingRecord
    ] = []

    upload_packets: List[
        LogicalUploadPacket
    ] = []

    # ========================================================
    # Client -> Layer -> Tensor
    # ========================================================

    for client_id in sorted(
        original_parameters
    ):
        layer_map = (
            original_parameters[
                client_id
            ]
        )

        # 先为四个逻辑服务器建立该客户端容器
        for server_name in SERVER_NAMES:
            masked_parameters_by_server[
                server_name
            ][
                client_id
            ] = {}

            masks_by_server[
                server_name
            ][
                client_id
            ] = {}

        client_tensor_count = 0

        for layer_name, tensors in (
            layer_map.items()
        ):
            layer_index = (
                _layer_name_to_index(
                    layer_name
                )
            )

            # 为四个服务器建立该层参数列表
            for server_name in SERVER_NAMES:
                masked_parameters_by_server[
                    server_name
                ][
                    client_id
                ][
                    layer_name
                ] = []

                masks_by_server[
                    server_name
                ][
                    client_id
                ][
                    layer_name
                ] = []

            for tensor_index, theta in enumerate(
                tensors
            ):
                theta_cpu = (
                    theta.detach()
                    .cpu()
                    .clone()
                )

                masked_parameters, masks = (
                    _mask_tensor_for_four_servers(
                        theta=theta_cpu,
                        master_seed=int(
                            master_seed
                        ),
                        client_id=int(
                            client_id
                        ),
                        round_index=int(
                            round_index
                        ),
                        layer_index=(
                            layer_index
                        ),
                        tensor_index=(
                            tensor_index
                        ),
                        mask_std=float(
                            mask_std
                        ),
                    )
                )

                for server_name in SERVER_NAMES:
                    masked_parameters_by_server[
                        server_name
                    ][
                        client_id
                    ][
                        layer_name
                    ].append(
                        masked_parameters[
                            server_name
                        ].clone()
                    )

                    masks_by_server[
                        server_name
                    ][
                        client_id
                    ][
                        layer_name
                    ].append(
                        masks[
                            server_name
                        ].clone()
                    )

                tensor_records.append(
                    _build_tensor_record(
                        client_id=int(
                            client_id
                        ),
                        layer_name=(
                            layer_name
                        ),
                        tensor_index=(
                            tensor_index
                        ),
                        tensor_count=len(
                            tensors
                        ),
                        theta=theta_cpu,
                        masked_parameters=masked_parameters,
                        masks=masks,
                        preview_values_per_tensor=int(
                            preview_values_per_tensor
                        ),
                    )
                )

                client_tensor_count += 1

        # 每个服务器都“收到”该客户端的一份完整受掩码保护共享架构参数
        for server_name in SERVER_NAMES:
            upload_packets.append(
                LogicalUploadPacket(
                    server_name=(
                        server_name
                    ),
                    client_id=int(
                        client_id
                    ),
                    layer_names=list(
                        layer_map.keys()
                    ),
                    tensor_count=(
                        client_tensor_count
                    ),
                )
            )

    return MaskingAndUploadingResult(
        round_index=int(
            round_index
        ),
        master_seed=int(
            master_seed
        ),
        mask_std=float(
            mask_std
        ),
        server_names=SERVER_NAMES,

        original_client_shared_parameters=(
            original_parameters
        ),

        masked_parameters_by_server=(
            masked_parameters_by_server
        ),

        masks_by_server=(
            masks_by_server
        ),

        upload_packets=(
            upload_packets
        ),

        tensor_records=(
            tensor_records
        ),
    )


# ============================================================
# Step 04 适配入口
# ============================================================

def mask_and_upload_from_step04(
    local_training_result: Any,
    master_seed: int,
    mask_std: float = DEFAULT_MASK_STD,
    preview_values_per_tensor: int = (
        DEFAULT_PREVIEW_VALUES_PER_TENSOR
    ),
) -> MaskingAndUploadingResult:
    """
    当 Step 06 直接接 Step 04 时使用。

    要求 local_training_result 至少有：

        client_shared_parameters
        round_index
    """

    if not hasattr(
        local_training_result,
        "client_shared_parameters",
    ):
        raise TypeError(
            "local_training_result 缺少 "
            "client_shared_parameters"
        )

    if not hasattr(
        local_training_result,
        "round_index",
    ):
        raise TypeError(
            "local_training_result 缺少 round_index"
        )

    return mask_and_split_shared_parameters(
        client_shared_parameters=(
            local_training_result
            .client_shared_parameters
        ),
        master_seed=int(
            master_seed
        ),
        round_index=int(
            local_training_result
            .round_index
        ),
        mask_std=float(
            mask_std
        ),
        preview_values_per_tensor=int(
            preview_values_per_tensor
        ),
    )


# ============================================================
# Step 05 适配入口
# ============================================================

def mask_and_upload_from_step05(
    parameter_attack_result: Any,
    master_seed: int,
    mask_std: float = DEFAULT_MASK_STD,
    preview_values_per_tensor: int = (
        DEFAULT_PREVIEW_VALUES_PER_TENSOR
    ),
) -> MaskingAndUploadingResult:
    """
    当 Step 06 接 Step 05 时使用。

    要求 parameter_attack_result 至少有：

        attacked_client_shared_parameters
        round_index

    此时被拆分的 theta 就是：

        Step 05 攻击完成后的共享模型参数。
    """

    if not hasattr(
        parameter_attack_result,
        "attacked_client_shared_parameters",
    ):
        raise TypeError(
            "parameter_attack_result 缺少 "
            "attacked_client_shared_parameters"
        )

    if not hasattr(
        parameter_attack_result,
        "round_index",
    ):
        raise TypeError(
            "parameter_attack_result 缺少 round_index"
        )

    return mask_and_split_shared_parameters(
        client_shared_parameters=(
            parameter_attack_result
            .attacked_client_shared_parameters
        ),
        master_seed=int(
            master_seed
        ),
        round_index=int(
            parameter_attack_result
            .round_index
        ),
        mask_std=float(
            mask_std
        ),
        preview_values_per_tensor=int(
            preview_values_per_tensor
        ),
    )


# ============================================================
# 数学一致性检查函数
# ============================================================

def sum_masked_client_parameters(
    result: MaskingAndUploadingResult,
) -> Dict[
    int,
    Dict[str, List[torch.Tensor]],
]:
    """
    仅用于测试 / 验证：

        4 * theta
            = omega_TS
            + omega_DS1
            + omega_DS2
            + omega_DS3

    后续安全协议不应该为了方便而处处直接调用这个函数，
    否则会破坏“分份计算”的设计意义。
    """

    summed_masked: Dict[
        int,
        Dict[str, List[torch.Tensor]],
    ] = {}

    original = (
        result.original_client_shared_parameters
    )

    for client_id, layer_map in (
        original.items()
    ):
        summed_masked[
            client_id
        ] = {}

        for layer_name, tensors in (
            layer_map.items()
        ):
            summed_masked[
                client_id
            ][
                layer_name
            ] = []

            for tensor_index, theta in enumerate(
                tensors
            ):
                summed_masked_tensor = sum(
                    (
                        result.masked_parameters_by_server[
                            server_name
                        ][
                            client_id
                        ][
                            layer_name
                        ][
                            tensor_index
                        ]
                        for server_name
                        in SERVER_NAMES
                    ),
                    torch.zeros_like(
                        theta
                    ),
                )

                summed_masked[
                    client_id
                ][
                    layer_name
                ].append(
                    summed_masked_tensor
                )

    return summed_masked


def sum_client_masks(
    result: MaskingAndUploadingResult,
) -> Dict[
    int,
    Dict[str, List[torch.Tensor]],
]:
    """
    仅用于测试 / 可视化验证：

        mask_sum
            = r_TS + r_DS1 + r_DS2 + r_DS3

    理论值：

        0
    """

    mask_sums: Dict[
        int,
        Dict[str, List[torch.Tensor]],
    ] = {}

    original = (
        result.original_client_shared_parameters
    )

    for client_id, layer_map in (
        original.items()
    ):
        mask_sums[
            client_id
        ] = {}

        for layer_name, tensors in (
            layer_map.items()
        ):
            mask_sums[
                client_id
            ][
                layer_name
            ] = []

            for tensor_index, theta in enumerate(
                tensors
            ):
                mask_sum = sum(
                    (
                        result.masks_by_server[
                            server_name
                        ][
                            client_id
                        ][
                            layer_name
                        ][
                            tensor_index
                        ]
                        for server_name
                        in SERVER_NAMES
                    ),
                    torch.zeros_like(
                        theta
                    ),
                )

                mask_sums[
                    client_id
                ][
                    layer_name
                ].append(
                    mask_sum
                )

    return mask_sums


# ============================================================
# 控制台输出
# ============================================================

def print_masking_and_uploading_result(
    result: MaskingAndUploadingResult,
    max_records: int = 8,
) -> None:
    """
    裸跑测试 / 调试输出。

    高维 Tensor 不整块打印，
    只展示关键统计和 preview。
    """

    print(
        "\n"
        + "=" * 88
    )

    print(
        "Step 06 - Random-Mask Protection and Logical Uploading"
    )

    print(
        "=" * 88
    )

    print(
        f"Round Index   : {result.round_index}"
    )

    print(
        f"Master Seed   : {result.master_seed}"
    )

    print(
        f"Mask Std      : {result.mask_std}"
    )

    print(
        f"Servers       : {list(result.server_names)}"
    )

    print(
        f"Tensor Records: {len(result.tensor_records)}"
    )

    print(
        "\nCore Equations:"
    )

    print(
        "  r_TS + r_DS1 + r_DS2 + r_DS3 = 0"
    )

    print(
        "  omega_TS + omega_DS1 + omega_DS2 + omega_DS3 = 4 * theta"
    )

    for record in (
        result.tensor_records[
            :max_records
        ]
    ):
        print(
            "\n"
            + "-" * 88
        )

        print(
            f"Client {record.client_id:02d} | "
            f"{record.layer_name} | "
            f"{record.tensor_role} | "
            f"shape={record.shape}"
        )

        print(
            f"Original norm              : "
            f"{record.original_norm:.8f}"
        )

        print(
            f"Mask sum norm              : "
            f"{record.mask_sum_norm:.12e}"
        )

        print(
            f"Sum relation error norm  : "
            f"{record.sum_relation_error_norm:.12e}"
        )

        print(
            f"Original preview           : "
            f"{record.original_preview}"
        )

        for server_name in SERVER_NAMES:
            print(
                f"{server_name:>3} mask preview          : "
                f"{record.mask_previews[server_name]}"
            )

            print(
                f"{server_name:>3} masked preview         : "
                f"{record.masked_parameter_previews[server_name]}"
            )

        print(
            f"Summed masked preview      : "
            f"{record.summed_masked_preview}"
        )


# ============================================================
# 测试样例
# ============================================================

if __name__ == "__main__":
    # ========================================================
    # 本文件可以直接独立验证 Step 06 核心逻辑。
    #
    # 正式完整实验中可以链式：
    #
    #     Step 00
    #       -> Step 01
    #       -> Step 02
    #       -> Step 03
    #       -> Step 04
    #       -> Step 05
    #       -> Step 06
    #
    # 但 Step 06 核心算法本身只依赖：
    #
    #     client_shared_parameters
    #     master_seed
    #     round_index
    #
    # 因此这里使用手工构造的小 Tensor 做快速单元测试，
    # 不需要真的启动完整 FL 训练。
    # ========================================================

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        2026
    )

    fake_parameters = {
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
        },

        1: {
            "layer1": [
                torch.randn(
                    4,
                    3,
                    generator=generator,
                ),
                torch.randn(
                    4,
                    generator=generator,
                ),
            ],
        },
    }

    # 保存输入副本，
    # 检查 Step 06 不会原地污染上游参数。
    before = (
        _clone_parameter_structure(
            fake_parameters
        )
    )

    result = (
        mask_and_split_shared_parameters(
            client_shared_parameters=(
                fake_parameters
            ),
            master_seed=42,
            round_index=1,
            mask_std=1.0,
            preview_values_per_tensor=3,
        )
    )

    print_masking_and_uploading_result(
        result,
        max_records=10,
    )

    print(
        "\n"
        + "=" * 88
    )

    print(
        "Automatic Checks"
    )

    print(
        "=" * 88
    )

    # --------------------------------------------------------
    # Helper
    # --------------------------------------------------------

    def _allclose_structure(
        a: Mapping[
            int,
            Mapping[
                str,
                Sequence[torch.Tensor],
            ],
        ],
        b: Mapping[
            int,
            Mapping[
                str,
                Sequence[torch.Tensor],
            ],
        ],
        atol: float = 1e-6,
        rtol: float = 1e-6,
    ) -> bool:
        if set(a.keys()) != set(b.keys()):
            return False

        for client_id in a:
            if (
                set(a[client_id].keys())
                != set(b[client_id].keys())
            ):
                return False

            for layer_name in a[client_id]:
                a_tensors = (
                    a[client_id][layer_name]
                )

                b_tensors = (
                    b[client_id][layer_name]
                )

                if len(a_tensors) != len(
                    b_tensors
                ):
                    return False

                for x, y in zip(
                    a_tensors,
                    b_tensors,
                ):
                    if not torch.allclose(
                        x,
                        y,
                        atol=atol,
                        rtol=rtol,
                    ):
                        return False

        return True

    # --------------------------------------------------------
    # 1. 四个服务器名称固定正确
    # --------------------------------------------------------

    server_names_correct = (
        result.server_names
        == (
            "TS",
            "DS1",
            "DS2",
            "DS3",
        )
    )

    print(
        "服务器逻辑角色为 TS/DS1/DS2/DS3 :",
        server_names_correct,
    )

    # --------------------------------------------------------
    # 2. Mask 零和
    # --------------------------------------------------------

    mask_sums = (
        sum_client_masks(
            result
        )
    )

    all_masks_zero_sum = True

    for client_id, layer_map in (
        mask_sums.items()
    ):
        for layer_name, tensors in (
            layer_map.items()
        ):
            for tensor in tensors:
                if not torch.allclose(
                    tensor,
                    torch.zeros_like(
                        tensor
                    ),
                    atol=1e-6,
                    rtol=1e-6,
                ):
                    all_masks_zero_sum = False
                    break

    print(
        "四份 Mask 求和约等于 0 :",
        all_masks_zero_sum,
    )

    # --------------------------------------------------------
    # 3. 四份受掩码保护参数求和得到 4 * theta
    # --------------------------------------------------------

    summed_masked = (
        sum_masked_client_parameters(
            result
        )
    )

    reconstruction_correct = (
        _allclose_structure(
            summed_masked,
            {
                cid: {
                    lname: [float(NUM_SERVERS) * t for t in tensors]
                    for lname, tensors in layers.items()
                }
                for cid, layers in result.original_client_shared_parameters.items()
            },
            atol=1e-5,
            rtol=1e-5,
        )
    )

    print(
        "四份受掩码保护参数求和得到 4 * theta :",
        reconstruction_correct,
    )

    # --------------------------------------------------------
    # 4. 上游输入没有被原地修改
    # --------------------------------------------------------

    input_not_mutated = (
        _allclose_structure(
            before,
            fake_parameters,
            atol=0.0,
            rtol=0.0,
        )
    )

    print(
        "上游模型参数未被原地修改 :",
        input_not_mutated,
    )

    # --------------------------------------------------------
    # 5. 相同输入 / seed / round 可复现
    # --------------------------------------------------------

    repeat = (
        mask_and_split_shared_parameters(
            client_shared_parameters=(
                fake_parameters
            ),
            master_seed=42,
            round_index=1,
            mask_std=1.0,
            preview_values_per_tensor=3,
        )
    )

    reproducible = True

    for server_name in SERVER_NAMES:
        if not _allclose_structure(
            result.masked_parameters_by_server[
                server_name
            ],
            repeat.masked_parameters_by_server[
                server_name
            ],
            atol=0.0,
            rtol=0.0,
        ):
            reproducible = False
            break

        if not _allclose_structure(
            result.masks_by_server[
                server_name
            ],
            repeat.masks_by_server[
                server_name
            ],
            atol=0.0,
            rtol=0.0,
        ):
            reproducible = False
            break

    print(
        "同一 seed / round 下 Mask 与受掩码保护参数可复现 :",
        reproducible,
    )

    # --------------------------------------------------------
    # 6. 换 round 后 Mask 应变化
    # --------------------------------------------------------

    next_round = (
        mask_and_split_shared_parameters(
            client_shared_parameters=(
                fake_parameters
            ),
            master_seed=42,
            round_index=2,
            mask_std=1.0,
            preview_values_per_tensor=3,
        )
    )

    one_mask_round1 = (
        result.masks_by_server[
            "TS"
        ][
            0
        ][
            "layer1"
        ][
            0
        ]
    )

    one_mask_round2 = (
        next_round.masks_by_server[
            "TS"
        ][
            0
        ][
            "layer1"
        ][
            0
        ]
    )

    round_changes_mask = (
        not torch.equal(
            one_mask_round1,
            one_mask_round2,
        )
    )

    print(
        "不同 round 会产生不同 Mask :",
        round_changes_mask,
    )

    # --------------------------------------------------------
    # 7. visualization_data 不包含高维 Tensor
    # --------------------------------------------------------

    visualization_ready = (
        len(
            result.visualization_data
        )
        == len(
            result.tensor_records
        )
    )

    print(
        "visualization_data 已准备好 :",
        visualization_ready,
    )
