"""
Step_07_V_byzantine.py

===============================================================================
Step 07：CKKS 同态加密下的隐私保护 Layer-wise Byzantine Detection（可视化相关）
===============================================================================

【本步骤在完整流程中的位置】

    Step 06
    基于零和随机掩码的四逻辑服务器上传
        |
        | masked_parameters_by_server
        |   TS / DS1 / DS2 / DS3
        |   omega_i = theta + r_i
        |   sum_i r_i = 0
        |   sum_i omega_i = 4 * theta
        v

    ---------------------------------------------------------------------------
    Step 07  <-- 本文件
    CKKS Homomorphic Secure Distance + Layer-wise Byzantine Filtering
    ---------------------------------------------------------------------------
        |
        | 1. 先根据当前上传参数得到“初步聚合模型”
        | 2. TS 加密自己的受掩码参数 omega_TS
        | 3. DS1 / DS2 / DS3 在 CKKS 密文域依次加入自己的 omega
        | 4. 链式相加得到 Enc(4 * theta_k)
        | 5. 统一处理 4 倍尺度，得到 Enc(theta_k)
        | 6. 减去初步聚合结果 G，得到 Enc(d_k)
        | 7. 密文域逐元素平方，并在槽位上求和：
        |
        |       D_k = sum_i d_{k,i}^2
        |
        |    D_k 是一个“纯数字/标量”的平方欧氏距离
        | 8. 本轮生成唯一正随机因子 h^(t) > 0
        | 9. 密文域计算 h^(t) * D_k
        | 10. TS 解密 h^(t) * D_k
        | 11. 对每个 Layer 内的 weight / bias 分别：
        |       - 求 hD 的中位数
        |       - 计算每个 Client 到该中位数的绝对距离
        |       - 保留最接近的 ceil(tau * n) 个 Client
        v

    Step 08 / 后续
    使用 Step 07 的 retained_clients 对各层 weight / bias 分别完成最终聚合


===============================================================================
【最重要的数学尺度：一定不能漏掉 /4】
===============================================================================

Step 06 对同一个模型参数 theta 生成四份受掩码参数：

    omega_TS  = theta + r_TS
    omega_DS1 = theta + r_DS1
    omega_DS2 = theta + r_DS2
    omega_DS3 = theta + r_DS3

并满足：

    r_TS + r_DS1 + r_DS2 + r_DS3 = 0

因此：

    omega_TS + omega_DS1 + omega_DS2 + omega_DS3
        = 4 * theta

所以 CKKS 链式同态加法结束后得到的是：

    Enc(4 * theta)

而不是：

    Enc(theta)

后续计算客户端模型与初步聚合模型之间的距离时，必须先统一尺度。

本实现采用最直观的方式：

    Enc(4 * theta)
        -- multiply_plain(1/4) -->
    Enc(theta)

然后：

    Enc(theta) - Enc(G)
        = Enc(theta - G)
        = Enc(d)

绝对不能错误地计算：

    4 * theta - G


===============================================================================
【初步聚合模型 G】
===============================================================================

在正式 Byzantine 筛选之前，需要先得到一个初步聚合参考模型 G。

对于某个 layer / tensor_role，只使用“确实上传了该层该 Tensor”的客户端。
本文件默认采用普通算术平均：

    G = (1 / n) * sum_k theta_k

但注意 Step 06 中服务器持有的是受掩码参数，而不是 theta_k。

由于每个客户端满足：

    sum_server omega_{k,server} = 4 * theta_k

所以初步聚合也可以从四服务器受掩码参数直接恢复其数学结果：

    G
      = (1 / n) * sum_k theta_k
      = (1 / (4n)) * sum_k sum_server omega_{k,server}

本文件的正式核心流程按这个关系构造 G。

如果后续你要严格替换成论文 Algorithm 3 中其他形式的“初步聚合器”，
只需要替换 build_preliminary_aggregate(...)，
后面的 CKKS 距离计算接口不需要改变。


===============================================================================
【CKKS 链式同态计算】
===============================================================================

对 Client k、Layer l、Tensor p（weight 或 bias）：

TS：

    c0 = Enc(omega_TS)

DS1：

    c1 = c0 + omega_DS1
       = Enc(omega_TS + omega_DS1)

DS2：

    c2 = c1 + omega_DS2

DS3：

    c3 = c2 + omega_DS3
       = Enc(4 * theta_k)

随后：

    c_theta = (1/4) * c3
            = Enc(theta_k)

再减去初步聚合模型：

    c_d = c_theta - G
        = Enc(theta_k - G)

记：

    d = (d_1, d_2, ..., d_m)

这里 d_i 是单个模型参数分量的差异值，可以为正，也可以为负。


===============================================================================
【所谓“距离平方”最终是标量，不是一个向量】
===============================================================================

密文域首先逐元素平方：

    d ⊙ d
      = (d_1^2, d_2^2, ..., d_m^2)

然后对所有有效槽位求和：

    D = sum_i d_i^2

因此：

    D = ||theta_k - G||_2^2

D 是一个纯数字/标量。

代码中严格区分：

    difference_vector
    squared_difference_vector
    raw_squared_distance

其中：

    raw_squared_distance = D = sum_i d_i^2


===============================================================================
【随机因子 h：每轮只有一个，而且必须 h > 0】
===============================================================================

本步骤为每个 round 确定性生成：

    h^(t) > 0

非常重要：

    同一轮内：
        所有 Client
        所有 Layer
        所有 weight
        所有 bias

    都必须使用完全相同的 h^(t)。

绝对不能：
    - 每个客户端生成一个 h
    - 每层生成一个 h
    - weight / bias 各生成一个 h
    - 每个 Tensor 生成一个 h

下一轮可以生成新的 h^(t+1)。

正式安全协议计算：

    protected_distance = h^(t) * D

因为 h^(t) > 0 且该轮所有客户端共用同一个 h：

    D_a < D_b    <=>    hD_a < hD_b

因此统一正比例缩放不会改变距离排序关系，
也不会改变“谁更接近中位数”的排序。


===============================================================================
【非常重要：为了可视化，必须同时保留“不乘 h”和“乘 h”的结果】
===============================================================================

正式协议中，TS 应当只获得：

    h * D

而不应该直接获得真实平方距离：

    D

但是比赛 Demo 需要展示随机因子 h 的隐私保护意义。

因此本文件有意同时保留：

    1. raw_squared_distance
         D = sum_i d_i^2

    2. protected_squared_distance
         hD = h * sum_i d_i^2

这不是说正式协议应该把 D 交给 TS。

raw_squared_distance 是：
    - 实验对照值
    - 协议教学值
    - 可视化专用值

用于页面并排展示：

    未乘 h：
        D

    乘 h：
        hD

从而说明：
    如果 TS 直接得到 D，
    会获知客户端模型相对参考模型的真实偏差尺度；

    正式协议只让 TS 解密 hD，
    真实距离尺度被本轮秘密正随机因子统一缩放。

为避免可视化同学重新实现算法，本文件直接提供：

    result.raw_squared_distances
    result.protected_squared_distances
    result.distance_visualization_data

可视化层只读这些字段即可。


===============================================================================
【Layer-wise + 层内 weight / bias 独立处理】
===============================================================================

总体检测粒度仍然称为：

    Layer-wise Byzantine Detection

但是每一层内部：

    weight
    bias

必须独立计算距离、独立求中位数、独立筛选。

例如完全允许：

    layer2:
        weight -> Client 7 rejected
        bias   -> Client 7 retained

不要把 weight 和 bias 拼成一个大向量后只得到一个共同决策。


===============================================================================
【Median + tau 筛选】
===============================================================================

对于同一个：

    layer_name / tensor_role

得到各客户端：

    hD_1, hD_2, ..., hD_n

先计算标量中位数：

    M = Median(hD_1, ..., hD_n)

再计算：

    score_k = |hD_k - M|

按 score 从小到大排序。

保留：

    keep_count = ceil(tau * n)

至少保留 1 个，最多 n 个。

若 score 完全相同，使用 client_id 升序打破平局，
不额外引入随机性。


===============================================================================
【CKKS 密钥归属】
===============================================================================

逻辑设计：

    TS:
        - 持有 CKKS Secret Key
        - 可以加密
        - 可以最终解密

    DS1 / DS2 / DS3:
        - 不持有 Secret Key
        - 只能执行允许的 CKKS 同态运算

因此：
    只有 TS 可以解密最终结果。

本文件不会把 Secret Key 暴露到 visualization_data。


===============================================================================
【关于 CKKS 库】
===============================================================================

本文件优先使用 TenSEAL 的 CKKSVector 来执行真正的 CKKS 编码、加密、
密文加法、密文/明文乘法、密文平方和解密。

    pip install tenseal

如果运行环境没有安装 tenseal：

    - 模块仍然可以被 import；
    - 数据结构、初步聚合、筛选、可视化接口仍然可用；
    - 真正调用 CKKS 安全距离计算时会给出明确 RuntimeError。

这里不偷偷退化成“拿普通 torch Tensor 冒充 CKKS 密文”，
避免比赛代码表面写着同态加密、实际却没有加密。


===============================================================================
【输入】
===============================================================================

正式入口：

    detect_byzantine_from_step06(
        masking_result,
        tau=0.7,
        ...
    )

要求 Step 06 结果至少包含：

    masked_parameters_by_server
    round_index
    master_seed

服务器结构必须包含：

    TS
    DS1
    DS2
    DS3


===============================================================================
【输出 ByzantineDetectionResult】
===============================================================================

主要字段：

    preliminary_aggregate
        初步聚合模型 G

    raw_squared_distances
        D，未乘 h 的真实平方距离
        仅实验/教学/可视化对照

    protected_squared_distances
        hD，正式协议中 TS 解密得到的距离

    median_protected_distances
        每层每个 weight/bias 的 hD 中位数

    retained_clients
        每层每个 weight/bias 保留的客户端

    rejected_clients
        每层每个 weight/bias 被过滤的客户端

    distance_visualization_data
        可视化同学直接调用的数据

    ckks_chain_visualization_data
        展示：
            Step 06 四份受掩码参数 preview
            TS 初始加密
            DS1 累加
            DS2 累加
            DS3 累加
            最终 4*theta 关系
            /4 后恢复 theta 尺度

注意：
    可视化数据只保留 preview / 标量 / 统计量，
    不把 CKKS Secret Key 或完整高维 Tensor 暴露给前端。


===============================================================================
【种子】
===============================================================================

master_seed 是全流程主种子。

本步骤仅把随机性用于：
    - 每轮 h 的生成
    - CKKS 库内部加密随机性（由密码库负责）

h 的数值由：

    master_seed + round_index

稳定派生。

因此：
    相同 master_seed + round_index -> 相同 h
    不同 round -> 不同 h

但是 CKKS 密文本身具有随机性，
不能要求两次加密产生字节级相同的密文。

Median / tau 筛选本身完全确定性。


===============================================================================
【本步骤不负责】
===============================================================================

- 不重新生成 Step 06 Mask；
- 不重新选择 Byzantine Client；
- 不读取 Step 00 的真实 Byzantine 标签来“作弊”检测；
- 不攻击模型；
- 不训练模型；
- 不决定 Shared / Personalized Architecture；
- 不执行最终聚合；
- 不真正绘制前端图表；
- 不向 DS 暴露 CKKS Secret Key。


===============================================================================
【测试样例为什么不会在正常 import 时运行】
===============================================================================

文件底部测试代码严格放在：

    if __name__ == "__main__":
        ...

里面。

因此：

    import Step_07_V_byzantine

或者：

    from Step_07_V_byzantine import detect_byzantine_from_step06

都不会执行测试样例，也不会打印测试输出。

只有你主动在命令行直接运行：

    python Step_07_V_byzantine.py

时，测试样例才会执行。

===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import math
import random

import torch

try:
    import tenseal as ts
except ImportError:
    ts = None


SERVER_NAMES: Tuple[str, str, str, str] = ("TS", "DS1", "DS2", "DS3")
NUM_SERVERS = 4

ALL_LAYER_NAMES: Tuple[str, ...] = (
    "layer1",
    "layer2",
    "layer3",
    "layer4",
    "layer5",
)

DEFAULT_TAU = 0.7
DEFAULT_PREVIEW_VALUES = 5

# CKKS 默认参数。
# 这些参数按“留足乘法层级冗余”的原则配置，用于支撑：
#   加法 -> plain * 1/4 -> 减法 -> plain * sqrt(h) -> 平方 -> 槽位求和
# Demo 原始 D 使用独立密文分支，不与正式 hD 分支争用已消耗的层级。
#
# 若未来模型规模/精度实验需要，可统一从主配置文件覆盖。
DEFAULT_POLY_MODULUS_DEGREE = 16384
# Reserve several 40-bit working levels. Total = 400 bits, below SEAL's
# 128-bit-security maximum (438 bits) for N=16384. This intentionally leaves
# headroom for: /4 -> subtract G -> square -> slot sum -> multiply by h.
DEFAULT_COEFF_MOD_BIT_SIZES = (60, 40, 40, 40, 40, 40, 40, 40, 60)
DEFAULT_GLOBAL_SCALE = float(2 ** 40)

# 16384 下这里故意不把 coefficient modulus 预算顶满。
# 当前总 bit 数 = 60 + 7*40 + 60 = 400 bits。
# 目的：为 /4、sqrt(h)、密文平方、槽位求和等核心 CKKS 路径保留余量，
# 避免再次出现 scale out of bounds，同时不靠“解密后再乘 h”规避同态运算。

# h 只要求严格为正。
# 默认把它限制在一个适合 Demo 数值展示的区间，
# 防止过大缩放造成不必要的 CKKS 数值误差。
DEFAULT_H_MIN = 0.5
DEFAULT_H_MAX = 2.0


# =============================================================================
# 数据结构
# =============================================================================

@dataclass(frozen=True)
class CKKSTensorChainRecord:
    """
    单个 Client / Layer / Tensor 的 CKKS 链式计算记录。

    注意：
        这里存的是可视化需要的 preview / 标量。
        不保存 Secret Key。
    """

    client_id: int
    layer_name: str
    tensor_index: int
    tensor_role: str
    shape: Tuple[int, ...]
    num_values: int

    masked_parameter_previews: Dict[str, List[float]]

    # 下面这些 preview 是“数学上对应的明文值”。
    # CKKS 密文本身不适合直接给前端显示数值内容。
    chain_plaintext_previews: Dict[str, List[float]]

    recovered_theta_preview: List[float]
    preliminary_aggregate_preview: List[float]
    difference_preview: List[float]

    raw_squared_distance: float
    round_random_factor_h: float
    protected_squared_distance: float

    # CKKS 解密出来的两个展示值。
    #
    # decrypted_raw_squared_distance:
    #   仅 Demo / 对照用途。
    #
    # decrypted_protected_squared_distance:
    #   正式协议 TS 应获得的值。
    decrypted_raw_squared_distance: float
    decrypted_protected_squared_distance: float

    raw_distance_abs_error: float
    protected_distance_abs_error: float


@dataclass(frozen=True)
class ComponentFilteringSummary:
    """
    一个 layer / weight-or-bias 的 Byzantine 筛选摘要。
    """

    layer_name: str
    tensor_index: int
    tensor_role: str

    candidate_clients: List[int]
    protected_distances: Dict[int, float]

    median_protected_distance: float
    median_proximity_scores: Dict[int, float]

    tau: float
    keep_count: int

    retained_clients: List[int]
    rejected_clients: List[int]


@dataclass
class ByzantineDetectionResult:
    """
    Step 07 完整输出。
    """

    round_index: int
    master_seed: int
    tau: float

    # 本轮唯一正随机数。
    round_random_factor_h: float

    # 初步聚合模型 G：
    #   client-independent
    #   layer -> [weight, bias]
    preliminary_aggregate: Dict[str, List[torch.Tensor]]

    # D = sum_i d_i^2
    #
    # 仅实验/教学/可视化对照。
    raw_squared_distances: Dict[
        str,
        Dict[int, Dict[int, float]],
    ]

    # hD
    #
    # 正式协议中 TS 解密后应该使用的检测量。
    protected_squared_distances: Dict[
        str,
        Dict[int, Dict[int, float]],
    ]

    # 直接保存 CKKS 解密值。
    #
    # raw:
    #   仅 Demo 对照。
    #
    # protected:
    #   正式协议值。
    decrypted_raw_squared_distances: Dict[
        str,
        Dict[int, Dict[int, float]],
    ]

    decrypted_protected_squared_distances: Dict[
        str,
        Dict[int, Dict[int, float]],
    ]

    median_protected_distances: Dict[
        str,
        Dict[int, float],
    ]

    retained_clients: Dict[
        str,
        Dict[int, List[int]],
    ]

    rejected_clients: Dict[
        str,
        Dict[int, List[int]],
    ]

    component_summaries: List[ComponentFilteringSummary]
    ckks_chain_records: List[CKKSTensorChainRecord]

    ckks_library: str
    ckks_poly_modulus_degree: int
    ckks_coeff_mod_bit_sizes: Tuple[int, ...]
    ckks_global_scale: float

    @property
    def distance_visualization_data(self) -> List[Dict[str, Any]]:
        """
        可视化同学重点使用。

        每条记录明确同时给出：
            D
            h
            hD

        以及 CKKS 解密的：
            decrypt(D)     -- Demo 对照
            decrypt(hD)    -- 正式协议
        """

        rows: List[Dict[str, Any]] = []

        for record in self.ckks_chain_records:
            rows.append(
                {
                    "client_id": record.client_id,
                    "layer_name": record.layer_name,
                    "tensor_index": record.tensor_index,
                    "tensor_role": record.tensor_role,
                    "shape": list(record.shape),

                    "raw_squared_distance": (
                        record.raw_squared_distance
                    ),
                    "round_random_factor_h": (
                        record.round_random_factor_h
                    ),
                    "protected_squared_distance": (
                        record.protected_squared_distance
                    ),

                    "decrypted_raw_squared_distance": (
                        record.decrypted_raw_squared_distance
                    ),
                    "decrypted_protected_squared_distance": (
                        record.decrypted_protected_squared_distance
                    ),

                    "raw_distance_abs_error": (
                        record.raw_distance_abs_error
                    ),
                    "protected_distance_abs_error": (
                        record.protected_distance_abs_error
                    ),

                    "raw_distance_equation": (
                        "D = sum_i (theta_i - G_i)^2"
                    ),
                    "protected_distance_equation": (
                        "protected_D = h * D"
                    ),

                    # 非常重要的前端语义。
                    "raw_distance_visibility": (
                        "DEMO_ONLY_PRIVACY_REFERENCE"
                    ),
                    "protected_distance_visibility": (
                        "FORMAL_TS_DECRYPTION_RESULT"
                    ),

                    "privacy_explanation": (
                        "D is retained only for experiment/visualization; "
                        "the formal protocol lets TS decrypt h*D."
                    ),
                }
            )

        return rows

    @property
    def ckks_chain_visualization_data(self) -> List[Dict[str, Any]]:
        """
        给前端展示 CKKS 链式传递过程。

        注意：
            这里展示的是各阶段对应的“数学明文 preview”，
            不是把 CKKS 密文伪装成可读参数。
        """

        rows: List[Dict[str, Any]] = []

        for record in self.ckks_chain_records:
            rows.append(
                {
                    "client_id": record.client_id,
                    "layer_name": record.layer_name,
                    "tensor_index": record.tensor_index,
                    "tensor_role": record.tensor_role,
                    "shape": list(record.shape),
                    "num_values": record.num_values,

                    "step06_masked_parameter_previews": {
                        server: values.copy()
                        for server, values
                        in record.masked_parameter_previews.items()
                    },

                    "chain_plaintext_previews": {
                        stage: values.copy()
                        for stage, values
                        in record.chain_plaintext_previews.items()
                    },

                    "recovered_theta_preview": (
                        record.recovered_theta_preview.copy()
                    ),
                    "preliminary_aggregate_preview": (
                        record.preliminary_aggregate_preview.copy()
                    ),
                    "difference_preview": (
                        record.difference_preview.copy()
                    ),

                    "chain_equation": (
                        "Enc(omega_TS) -> +omega_DS1 -> "
                        "+omega_DS2 -> +omega_DS3 = Enc(4*theta)"
                    ),
                    "scale_equation": (
                        "Enc(4*theta) * 0.25 = Enc(theta)"
                    ),
                    "distance_equation": (
                        "D = sum_i (theta_i - G_i)^2"
                    ),
                    "protected_equation": (
                        "hD = h * D, h > 0 and fixed within the round"
                    ),

                    "only_ts_can_decrypt": True,
                }
            )

        return rows


# =============================================================================
# 通用辅助函数
# =============================================================================

def _require_tenseal() -> None:
    if ts is None:
        raise RuntimeError(
            "当前环境未安装 TenSEAL，无法执行真正的 CKKS 同态计算。\n"
            "请先安装：pip install tenseal\n"
            "本模块不会自动退化为普通明文 Tensor 来冒充 CKKS。"
        )


def _tensor_role(
    tensor_index: int,
    tensor_count: int,
) -> str:
    if tensor_count >= 1 and tensor_index == 0:
        return "weight"

    if tensor_count >= 2 and tensor_index == 1:
        return "bias"

    return f"param_{tensor_index}"


def _preview_values(
    tensor: torch.Tensor,
    count: int,
) -> List[float]:
    if count <= 0:
        return []

    flat = (
        tensor.detach()
        .cpu()
        .double()
        .reshape(-1)
    )

    return [
        float(value)
        for value in flat[:count].tolist()
    ]


def _clone_layer_parameter_structure(
    layer_parameters: Mapping[
        str,
        Sequence[torch.Tensor],
    ],
) -> Dict[str, List[torch.Tensor]]:
    return {
        str(layer_name): [
            tensor.detach().cpu().clone()
            for tensor in tensors
        ]
        for layer_name, tensors in layer_parameters.items()
    }


def _validate_tau(tau: float) -> None:
    if not (0.0 < float(tau) <= 1.0):
        raise ValueError(
            f"tau 必须满足 0 < tau <= 1，当前 tau={tau}"
        )


def _validate_h_range(
    h_min: float,
    h_max: float,
) -> None:
    if float(h_min) <= 0.0:
        raise ValueError(
            "h_min 必须 > 0，因为 h 只能是正数。"
        )

    if float(h_max) <= float(h_min):
        raise ValueError(
            "h_max 必须严格大于 h_min。"
        )


# =============================================================================
# 每轮唯一正随机因子 h
# =============================================================================

def _derive_round_h_seed(
    master_seed: int,
    round_index: int,
) -> int:
    """
    只依赖：
        master_seed
        round_index

    因此不会随 client/layer/tensor 改变。
    """

    value = int(master_seed) & 0x7FFFFFFFFFFFFFFF

    value = (
        value * 1_000_003
        + int(round_index) * 1_299_709
        + 0x48464C524F554E44
    )

    return int(
        value & 0x7FFFFFFFFFFFFFFF
    )


def generate_round_random_factor_h(
    master_seed: int,
    round_index: int,
    h_min: float = DEFAULT_H_MIN,
    h_max: float = DEFAULT_H_MAX,
) -> float:
    """
    生成本轮唯一正随机数 h^(t)。

    同一轮所有 Client / Layer / Tensor 共用。
    """

    _validate_h_range(
        h_min=h_min,
        h_max=h_max,
    )

    rng = random.Random(
        _derive_round_h_seed(
            master_seed=master_seed,
            round_index=round_index,
        )
    )

    h = rng.uniform(
        float(h_min),
        float(h_max),
    )

    if h <= 0.0:
        raise RuntimeError(
            "内部错误：生成的 h 不是正数。"
        )

    return float(h)


# =============================================================================
# Step 06 输入验证
# =============================================================================

def _validate_masking_result(
    masking_result: Any,
) -> None:
    required = (
        "masked_parameters_by_server",
        "round_index",
        "master_seed",
    )

    for name in required:
        if not hasattr(masking_result, name):
            raise TypeError(
                f"Step 06 输入缺少字段：{name}"
            )

    by_server = (
        masking_result.masked_parameters_by_server
    )

    missing = [
        server
        for server in SERVER_NAMES
        if server not in by_server
    ]

    if missing:
        raise ValueError(
            f"Step 06 缺少逻辑服务器：{missing}"
        )


# =============================================================================
# 初步聚合 G
# =============================================================================

def build_preliminary_aggregate(
    masked_parameters_by_server: Mapping[
        str,
        Mapping[
            int,
            Mapping[
                str,
                Sequence[torch.Tensor],
            ],
        ],
    ],
) -> Dict[str, List[torch.Tensor]]:
    """
    根据 Step 06 的四服务器受掩码参数构造初步聚合模型 G。

    对每个 layer / tensor：

        theta_k
          = (1/4) * sum_server omega_{k,server}

        G
          = mean_k theta_k
          = (1/(4n)) * sum_k sum_server omega_{k,server}

    不要求每个客户端共享层数完全相同。
    某一层只使用真正上传了该层的客户端。
    """

    for server in SERVER_NAMES:
        if server not in masked_parameters_by_server:
            raise ValueError(
                f"缺少服务器 {server}"
            )

    # layer -> tensor_index -> list[tensor]
    collected: Dict[
        str,
        Dict[int, List[torch.Tensor]],
    ] = {}

    ts_parameters = (
        masked_parameters_by_server["TS"]
    )

    for client_id in sorted(ts_parameters):
        layer_map = ts_parameters[client_id]

        for layer_name, ts_tensors in layer_map.items():
            collected.setdefault(
                str(layer_name),
                {},
            )

            for tensor_index, ts_tensor in enumerate(
                ts_tensors
            ):
                server_tensors: List[torch.Tensor] = []

                for server in SERVER_NAMES:
                    try:
                        tensor = (
                            masked_parameters_by_server[
                                server
                            ][
                                client_id
                            ][
                                layer_name
                            ][
                                tensor_index
                            ]
                        )
                    except (
                        KeyError,
                        IndexError,
                    ) as exc:
                        raise ValueError(
                            "四服务器参数结构不一致："
                            f"Client {client_id} / "
                            f"{layer_name} / tensor {tensor_index}"
                        ) from exc

                    server_tensors.append(
                        tensor.detach().cpu().double()
                    )

                recovered_theta = sum(
                    server_tensors,
                    torch.zeros_like(
                        server_tensors[0]
                    ),
                ) / float(NUM_SERVERS)

                collected[
                    str(layer_name)
                ].setdefault(
                    int(tensor_index),
                    [],
                ).append(
                    recovered_theta
                )

    aggregate: Dict[
        str,
        List[torch.Tensor],
    ] = {}

    for layer_name, tensor_map in collected.items():
        aggregate[layer_name] = []

        for tensor_index in sorted(tensor_map):
            tensors = tensor_map[tensor_index]

            if not tensors:
                raise RuntimeError(
                    "内部错误：初步聚合没有候选 Tensor。"
                )

            reference_shape = tensors[0].shape

            for tensor in tensors[1:]:
                if tensor.shape != reference_shape:
                    raise ValueError(
                        "同一 layer/tensor 的客户端参数形状不一致："
                        f"{layer_name} / tensor {tensor_index}"
                    )

            aggregate_tensor = (
                torch.stack(
                    tensors,
                    dim=0,
                )
                .mean(dim=0)
            )

            aggregate[layer_name].append(
                aggregate_tensor
            )

    return aggregate


# =============================================================================
# CKKS Context
# =============================================================================

def create_ckks_context(
    poly_modulus_degree: int = DEFAULT_POLY_MODULUS_DEGREE,
    coeff_mod_bit_sizes: Sequence[int] = (
        DEFAULT_COEFF_MOD_BIT_SIZES
    ),
    global_scale: float = DEFAULT_GLOBAL_SCALE,
) -> Any:
    """
    创建 CKKS Context。

    返回的 context 在本 Python Demo 中同时包含公钥侧运算材料和 Secret Key。

    逻辑协议角色上：
        TS 持有完整 context / Secret Key；
        DS 只应获得 public/evaluation context。

    本步骤为了单进程模拟四个逻辑服务器，
    不真正启动网络进程。
    """

    _require_tenseal()

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=int(
            poly_modulus_degree
        ),
        coeff_mod_bit_sizes=[
            int(value)
            for value in coeff_mod_bit_sizes
        ],
    )

    context.global_scale = float(
        global_scale
    )

    # 平方需要 relinearization keys。
    context.generate_relin_keys()

    # CKKSVector.sum() / rotate-and-add 需要 Galois keys。
    context.generate_galois_keys()

    return context


# =============================================================================
# CKKS 单 Tensor 安全距离
# =============================================================================

def _ckks_secure_squared_distance(
    *,
    context: Any,
    masked_tensors_by_server: Mapping[
        str,
        torch.Tensor,
    ],
    preliminary_aggregate_tensor: torch.Tensor,
    h: float,
) -> Tuple[
    float,
    float,
    float,
    float,
    Dict[str, List[float]],
]:
    """
    对一个 Client / Layer / Tensor 完成 CKKS 安全距离计算。

    返回：
        decrypted_raw_D
        decrypted_hD
        plaintext_raw_D
        plaintext_hD
        chain_plaintext_values

    注意：
        decrypted_raw_D 仅为 Demo 对照。

        正式协议真正应该返回给 TS 的只有：
            decrypted_hD
    """

    _require_tenseal()

    if float(h) <= 0.0:
        raise ValueError(
            "h 必须 > 0"
        )

    server_vectors: Dict[
        str,
        List[float],
    ] = {}

    reference_shape: Optional[
        torch.Size
    ] = None

    for server in SERVER_NAMES:
        if server not in masked_tensors_by_server:
            raise ValueError(
                f"缺少 {server} 的受掩码参数。"
            )

        tensor = (
            masked_tensors_by_server[
                server
            ]
            .detach()
            .cpu()
            .double()
        )

        if reference_shape is None:
            reference_shape = tensor.shape
        elif tensor.shape != reference_shape:
            raise ValueError(
                "四服务器受掩码 Tensor 形状不一致。"
            )

        server_vectors[server] = (
            tensor.reshape(-1).tolist()
        )

    aggregate = (
        preliminary_aggregate_tensor
        .detach()
        .cpu()
        .double()
        .reshape(-1)
    )

    if len(server_vectors["TS"]) != aggregate.numel():
        raise ValueError(
            "客户端 Tensor 与初步聚合 Tensor 元素数量不一致。"
        )

    # -------------------------------------------------------------------------
    # 明文数学路径：
    # 只用于数值验证与可视化 preview。
    # -------------------------------------------------------------------------

    omega_ts = torch.tensor(
        server_vectors["TS"],
        dtype=torch.float64,
    )
    omega_ds1 = torch.tensor(
        server_vectors["DS1"],
        dtype=torch.float64,
    )
    omega_ds2 = torch.tensor(
        server_vectors["DS2"],
        dtype=torch.float64,
    )
    omega_ds3 = torch.tensor(
        server_vectors["DS3"],
        dtype=torch.float64,
    )

    plain_stage_ts = omega_ts
    plain_stage_ds1 = (
        plain_stage_ts + omega_ds1
    )
    plain_stage_ds2 = (
        plain_stage_ds1 + omega_ds2
    )
    plain_stage_ds3 = (
        plain_stage_ds2 + omega_ds3
    )

    plain_theta = (
        plain_stage_ds3
        / float(NUM_SERVERS)
    )

    plain_difference = (
        plain_theta
        - aggregate
    )

    plaintext_raw_D = float(
        torch.sum(
            plain_difference
            * plain_difference
        ).item()
    )

    plaintext_hD = float(
        float(h) * plaintext_raw_D
    )

    # -------------------------------------------------------------------------
    # CKKS 正式路径
    #
    # TS:
    #   Enc(omega_TS)
    #
    # DS1:
    #   + omega_DS1
    #
    # DS2:
    #   + omega_DS2
    #
    # DS3:
    #   + omega_DS3
    #   -> Enc(4 theta)
    # -------------------------------------------------------------------------

    encrypted = ts.ckks_vector(
        context,
        server_vectors["TS"],
    )

    # DS1
    encrypted += server_vectors["DS1"]

    # DS2
    encrypted += server_vectors["DS2"]

    # DS3
    encrypted += server_vectors["DS3"]

    # -------------------------------------------------------------------------
    # 4 theta -> theta
    # -------------------------------------------------------------------------

    encrypted *= (
        1.0 / float(NUM_SERVERS)
    )

    # -------------------------------------------------------------------------
    # theta - G
    # -------------------------------------------------------------------------

    encrypted -= aggregate.tolist()

    # -------------------------------------------------------------------------
    # 逐元素平方：
    #
    #   Enc(d_i) * Enc(d_i)
    #
    # TenSEAL square() 是真正的 ciphertext operation。
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # 两条 CKKS 分支：正式协议 + Demo 对照。
    #
    # 重要的工程调整：不再在 square()+sum() 之后额外做 Enc(D) * h。
    # 那种顺序会把 plaintext multiplication 放在已经消耗过乘法层级的
    # 密文上，较容易触发 TenSEAL / SEAL 的 scale out of bounds。
    #
    # 正式协议利用 h > 0：
    #
    #   h * D = h * sum(d_i^2)
    #         = sum((sqrt(h) * d_i)^2)
    #
    # 因而先在差值密文上乘 sqrt(h)，再平方、求和。数学结果仍严格是 hD，
    # 且 h 从未在解密后才补乘。
    #
    # Demo 分支单独保留原始 D，用于可视化。两条分支互不复用已消耗
    # level 的结果，给核心步骤留出冗余。
    # -------------------------------------------------------------------------

    encrypted_raw_branch = encrypted.copy()
    encrypted_protected_branch = encrypted.copy()

    # Demo-only raw D branch.
    encrypted_raw_D = encrypted_raw_branch.square().sum()

    decrypted_raw_values = encrypted_raw_D.decrypt()

    if not decrypted_raw_values:
        raise RuntimeError(
            "CKKS 解密 raw D 得到空结果。"
        )

    decrypted_raw_D = float(decrypted_raw_values[0])

    # Formal protected branch: Enc(d) * sqrt(h) -> square -> sum = Enc(hD).
    sqrt_h = math.sqrt(float(h))
    encrypted_protected_branch *= sqrt_h
    encrypted_protected_D = encrypted_protected_branch.square().sum()

    decrypted_protected_values = (
        encrypted_protected_D.decrypt()
    )

    if not decrypted_protected_values:
        raise RuntimeError(
            "CKKS 解密 protected hD 得到空结果。"
        )

    decrypted_hD = float(
        decrypted_protected_values[0]
    )

    chain_plaintext_values = {
        "TS_after_encrypt_plaintext_equivalent": (
            plain_stage_ts.tolist()
        ),
        "DS1_after_homomorphic_add_plaintext_equivalent": (
            plain_stage_ds1.tolist()
        ),
        "DS2_after_homomorphic_add_plaintext_equivalent": (
            plain_stage_ds2.tolist()
        ),
        "DS3_after_homomorphic_add_plaintext_equivalent_4theta": (
            plain_stage_ds3.tolist()
        ),
        "after_divide_by_4_theta": (
            plain_theta.tolist()
        ),
        "after_subtract_preliminary_aggregate_d": (
            plain_difference.tolist()
        ),
    }

    return (
        decrypted_raw_D,
        decrypted_hD,
        plaintext_raw_D,
        plaintext_hD,
        chain_plaintext_values,
    )


# =============================================================================
# Median
# =============================================================================

def _scalar_median(
    values: Sequence[float],
) -> float:
    """
    常规标量中位数。

    偶数个值时取中间两个的平均。
    """

    if not values:
        raise ValueError(
            "中位数输入不能为空。"
        )

    ordered = sorted(
        float(value)
        for value in values
    )

    n = len(ordered)
    middle = n // 2

    if n % 2 == 1:
        return float(
            ordered[middle]
        )

    return float(
        (
            ordered[middle - 1]
            + ordered[middle]
        )
        / 2.0
    )


# =============================================================================
# tau 筛选
# =============================================================================

def _filter_one_component(
    *,
    layer_name: str,
    tensor_index: int,
    tensor_role: str,
    protected_distances: Mapping[
        int,
        float,
    ],
    tau: float,
) -> ComponentFilteringSummary:
    """
    对一个 layer / weight-or-bias：

        1. hD 求中位数
        2. score = |hD - median|
        3. 保留最接近的 ceil(tau*n)
    """

    _validate_tau(tau)

    if not protected_distances:
        raise ValueError(
            "protected_distances 不能为空。"
        )

    candidates = sorted(
        int(client_id)
        for client_id in protected_distances
    )

    median = _scalar_median(
        [
            protected_distances[
                client_id
            ]
            for client_id in candidates
        ]
    )

    scores = {
        client_id: abs(
            float(
                protected_distances[
                    client_id
                ]
            )
            - median
        )
        for client_id in candidates
    }

    keep_count = max(
        1,
        min(
            len(candidates),
            int(
                math.ceil(
                    float(tau)
                    * len(candidates)
                )
            ),
        ),
    )

    ranked = sorted(
        candidates,
        key=lambda client_id: (
            scores[client_id],
            client_id,
        ),
    )

    retained = ranked[:keep_count]
    retained_set = set(retained)

    rejected = [
        client_id
        for client_id in candidates
        if client_id not in retained_set
    ]

    return ComponentFilteringSummary(
        layer_name=str(layer_name),
        tensor_index=int(tensor_index),
        tensor_role=str(tensor_role),

        candidate_clients=candidates,

        protected_distances={
            int(client_id): float(value)
            for client_id, value
            in protected_distances.items()
        },

        median_protected_distance=float(
            median
        ),

        median_proximity_scores={
            int(client_id): float(score)
            for client_id, score
            in scores.items()
        },

        tau=float(tau),
        keep_count=int(keep_count),

        retained_clients=[
            int(client_id)
            for client_id in retained
        ],

        rejected_clients=[
            int(client_id)
            for client_id in rejected
        ],
    )


# =============================================================================
# Step 07 核心入口
# =============================================================================

def detect_byzantine_from_step06(
    masking_result: Any,
    tau: float = DEFAULT_TAU,
    *,
    h_min: float = DEFAULT_H_MIN,
    h_max: float = DEFAULT_H_MAX,
    poly_modulus_degree: int = DEFAULT_POLY_MODULUS_DEGREE,
    coeff_mod_bit_sizes: Sequence[int] = DEFAULT_COEFF_MOD_BIT_SIZES,
    global_scale: float = DEFAULT_GLOBAL_SCALE,
    preview_values_per_tensor: int = DEFAULT_PREVIEW_VALUES,
) -> ByzantineDetectionResult:
    """
    Step 07 正式入口。

    只依赖 Step 06 输出，不主动重跑 Step 00~06。
    """

    _require_tenseal()
    _validate_masking_result(
        masking_result
    )
    _validate_tau(tau)
    _validate_h_range(
        h_min=h_min,
        h_max=h_max,
    )

    if int(preview_values_per_tensor) < 0:
        raise ValueError(
            "preview_values_per_tensor 必须 >= 0"
        )

    round_index = int(
        masking_result.round_index
    )

    master_seed = int(
        masking_result.master_seed
    )

    masked_by_server = (
        masking_result
        .masked_parameters_by_server
    )

    # -------------------------------------------------------------------------
    # 1. 初步聚合
    # -------------------------------------------------------------------------

    preliminary_aggregate = (
        build_preliminary_aggregate(
            masked_by_server
        )
    )

    # -------------------------------------------------------------------------
    # 2. 本轮唯一 h > 0
    # -------------------------------------------------------------------------

    h = generate_round_random_factor_h(
        master_seed=master_seed,
        round_index=round_index,
        h_min=h_min,
        h_max=h_max,
    )

    # -------------------------------------------------------------------------
    # 3. CKKS Context
    # -------------------------------------------------------------------------

    context = create_ckks_context(
        poly_modulus_degree=(
            poly_modulus_degree
        ),
        coeff_mod_bit_sizes=(
            coeff_mod_bit_sizes
        ),
        global_scale=global_scale,
    )

    raw_squared_distances: Dict[
        str,
        Dict[int, Dict[int, float]],
    ] = {}

    protected_squared_distances: Dict[
        str,
        Dict[int, Dict[int, float]],
    ] = {}

    decrypted_raw_squared_distances: Dict[
        str,
        Dict[int, Dict[int, float]],
    ] = {}

    decrypted_protected_squared_distances: Dict[
        str,
        Dict[int, Dict[int, float]],
    ] = {}

    chain_records: List[
        CKKSTensorChainRecord
    ] = []

    ts_parameters = (
        masked_by_server["TS"]
    )

    # -------------------------------------------------------------------------
    # 4. Client -> Layer -> weight/bias
    # -------------------------------------------------------------------------

    for client_id in sorted(ts_parameters):
        layer_map = ts_parameters[
            client_id
        ]

        for layer_name, ts_tensors in (
            layer_map.items()
        ):
            if layer_name not in preliminary_aggregate:
                raise RuntimeError(
                    f"初步聚合缺少 {layer_name}"
                )

            raw_squared_distances.setdefault(
                layer_name,
                {},
            ).setdefault(
                int(client_id),
                {},
            )

            protected_squared_distances.setdefault(
                layer_name,
                {},
            ).setdefault(
                int(client_id),
                {},
            )

            decrypted_raw_squared_distances.setdefault(
                layer_name,
                {},
            ).setdefault(
                int(client_id),
                {},
            )

            decrypted_protected_squared_distances.setdefault(
                layer_name,
                {},
            ).setdefault(
                int(client_id),
                {},
            )

            for tensor_index, ts_tensor in enumerate(
                ts_tensors
            ):
                if (
                    tensor_index
                    >= len(
                        preliminary_aggregate[
                            layer_name
                        ]
                    )
                ):
                    raise RuntimeError(
                        "初步聚合 Tensor 数量与客户端不一致："
                        f"{layer_name} / tensor {tensor_index}"
                    )

                masked_tensors = {
                    server: (
                        masked_by_server[
                            server
                        ][
                            client_id
                        ][
                            layer_name
                        ][
                            tensor_index
                        ]
                    )
                    for server in SERVER_NAMES
                }

                (
                    decrypted_raw_D,
                    decrypted_hD,
                    plaintext_raw_D,
                    plaintext_hD,
                    chain_plaintext_values,
                ) = _ckks_secure_squared_distance(
                    context=context,
                    masked_tensors_by_server=(
                        masked_tensors
                    ),
                    preliminary_aggregate_tensor=(
                        preliminary_aggregate[
                            layer_name
                        ][
                            tensor_index
                        ]
                    ),
                    h=h,
                )

                # -------------------------------------------------------------
                # raw_squared_distances：
                # 保存数学真值 D，用于实验/可视化。
                #
                # protected_squared_distances：
                # 保存数学真值 hD，用于与 CKKS 解密误差比较。
                #
                # 真正筛选时下面会使用 CKKS 解密得到的 hD。
                # -------------------------------------------------------------

                raw_squared_distances[
                    layer_name
                ][
                    int(client_id)
                ][
                    int(tensor_index)
                ] = float(
                    plaintext_raw_D
                )

                protected_squared_distances[
                    layer_name
                ][
                    int(client_id)
                ][
                    int(tensor_index)
                ] = float(
                    plaintext_hD
                )

                decrypted_raw_squared_distances[
                    layer_name
                ][
                    int(client_id)
                ][
                    int(tensor_index)
                ] = float(
                    decrypted_raw_D
                )

                decrypted_protected_squared_distances[
                    layer_name
                ][
                    int(client_id)
                ][
                    int(tensor_index)
                ] = float(
                    decrypted_hD
                )

                tensor_count = len(
                    ts_tensors
                )

                tensor_role = _tensor_role(
                    tensor_index=tensor_index,
                    tensor_count=tensor_count,
                )

                masked_previews = {
                    server: _preview_values(
                        masked_tensors[
                            server
                        ],
                        preview_values_per_tensor,
                    )
                    for server in SERVER_NAMES
                }

                preview_chain = {
                    stage: [
                        float(value)
                        for value in values[
                            :preview_values_per_tensor
                        ]
                    ]
                    for stage, values
                    in chain_plaintext_values.items()
                }

                chain_records.append(
                    CKKSTensorChainRecord(
                        client_id=int(
                            client_id
                        ),
                        layer_name=str(
                            layer_name
                        ),
                        tensor_index=int(
                            tensor_index
                        ),
                        tensor_role=(
                            tensor_role
                        ),
                        shape=tuple(
                            ts_tensor.shape
                        ),
                        num_values=int(
                            ts_tensor.numel()
                        ),

                        masked_parameter_previews=(
                            masked_previews
                        ),

                        chain_plaintext_previews=(
                            preview_chain
                        ),

                        recovered_theta_preview=(
                            preview_chain[
                                "after_divide_by_4_theta"
                            ].copy()
                        ),

                        preliminary_aggregate_preview=(
                            _preview_values(
                                preliminary_aggregate[
                                    layer_name
                                ][
                                    tensor_index
                                ],
                                preview_values_per_tensor,
                            )
                        ),

                        difference_preview=(
                            preview_chain[
                                "after_subtract_preliminary_aggregate_d"
                            ].copy()
                        ),

                        raw_squared_distance=float(
                            plaintext_raw_D
                        ),

                        round_random_factor_h=float(
                            h
                        ),

                        protected_squared_distance=float(
                            plaintext_hD
                        ),

                        decrypted_raw_squared_distance=float(
                            decrypted_raw_D
                        ),

                        decrypted_protected_squared_distance=float(
                            decrypted_hD
                        ),

                        raw_distance_abs_error=abs(
                            float(decrypted_raw_D)
                            - float(plaintext_raw_D)
                        ),

                        protected_distance_abs_error=abs(
                            float(decrypted_hD)
                            - float(plaintext_hD)
                        ),
                    )
                )

    # -------------------------------------------------------------------------
    # 5. Layer-wise + weight/bias independent filtering
    #
    # 注意：
    #   正式筛选依据使用 CKKS 解密的 hD，
    #   而不是明文数学真值。
    # -------------------------------------------------------------------------

    component_summaries: List[
        ComponentFilteringSummary
    ] = []

    median_protected_distances: Dict[
        str,
        Dict[int, float],
    ] = {}

    retained_clients: Dict[
        str,
        Dict[int, List[int]],
    ] = {}

    rejected_clients: Dict[
        str,
        Dict[int, List[int]],
    ] = {}

    for layer_name in sorted(
        decrypted_protected_squared_distances
    ):
        client_map = (
            decrypted_protected_squared_distances[
                layer_name
            ]
        )

        tensor_indices = sorted(
            {
                tensor_index
                for tensor_map in client_map.values()
                for tensor_index in tensor_map
            }
        )

        median_protected_distances[
            layer_name
        ] = {}

        retained_clients[
            layer_name
        ] = {}

        rejected_clients[
            layer_name
        ] = {}

        for tensor_index in tensor_indices:
            component_distances = {
                int(client_id): float(
                    tensor_map[
                        tensor_index
                    ]
                )
                for client_id, tensor_map
                in client_map.items()
                if tensor_index in tensor_map
            }

            # 通过实际候选客户端的 tensor_count 判断 role。
            # 对当前 CNN/LeNet 层：
            #   0 -> weight
            #   1 -> bias
            tensor_role = (
                "weight"
                if tensor_index == 0
                else (
                    "bias"
                    if tensor_index == 1
                    else f"param_{tensor_index}"
                )
            )

            summary = _filter_one_component(
                layer_name=layer_name,
                tensor_index=tensor_index,
                tensor_role=tensor_role,
                protected_distances=(
                    component_distances
                ),
                tau=tau,
            )

            component_summaries.append(
                summary
            )

            median_protected_distances[
                layer_name
            ][
                tensor_index
            ] = float(
                summary.median_protected_distance
            )

            retained_clients[
                layer_name
            ][
                tensor_index
            ] = (
                summary.retained_clients.copy()
            )

            rejected_clients[
                layer_name
            ][
                tensor_index
            ] = (
                summary.rejected_clients.copy()
            )

    return ByzantineDetectionResult(
        round_index=round_index,
        master_seed=master_seed,
        tau=float(tau),

        round_random_factor_h=float(h),

        preliminary_aggregate=(
            _clone_layer_parameter_structure(
                preliminary_aggregate
            )
        ),

        raw_squared_distances=(
            raw_squared_distances
        ),

        protected_squared_distances=(
            protected_squared_distances
        ),

        decrypted_raw_squared_distances=(
            decrypted_raw_squared_distances
        ),

        decrypted_protected_squared_distances=(
            decrypted_protected_squared_distances
        ),

        median_protected_distances=(
            median_protected_distances
        ),

        retained_clients=(
            retained_clients
        ),

        rejected_clients=(
            rejected_clients
        ),

        component_summaries=(
            component_summaries
        ),

        ckks_chain_records=(
            chain_records
        ),

        ckks_library="TenSEAL",
        ckks_poly_modulus_degree=int(
            poly_modulus_degree
        ),
        ckks_coeff_mod_bit_sizes=tuple(
            int(value)
            for value in coeff_mod_bit_sizes
        ),
        ckks_global_scale=float(
            global_scale
        ),
    )


# =============================================================================
# 控制台打印
# =============================================================================

def print_byzantine_detection_result(
    result: ByzantineDetectionResult,
    max_chain_records: int = 6,
) -> None:
    print(
        "\n"
        + "=" * 96
    )

    print(
        "Step 07 - CKKS Secure Layer-wise Byzantine Detection"
    )

    print(
        "=" * 96
    )

    print(
        f"Round Index              : {result.round_index}"
    )

    print(
        f"Master Seed              : {result.master_seed}"
    )

    print(
        f"tau                      : {result.tau}"
    )

    print(
        f"Round Random Factor h > 0: "
        f"{result.round_random_factor_h:.12f}"
    )

    print(
        f"CKKS Library             : {result.ckks_library}"
    )

    print(
        "\nCore:"
    )

    print(
        "  sum(omega_TS, omega_DS1, omega_DS2, omega_DS3) = 4 * theta"
    )

    print(
        "  theta = (1/4) * sum(omega_i)"
    )

    print(
        "  D = sum_i (theta_i - G_i)^2"
    )

    print(
        "  protected_D = h * D"
    )

    print(
        "\nFiltering:"
    )

    for summary in result.component_summaries:
        print(
            f"  {summary.layer_name:<7} | "
            f"{summary.tensor_role:<6} | "
            f"median(hD)={summary.median_protected_distance:.8f} | "
            f"keep={summary.keep_count}/"
            f"{len(summary.candidate_clients)} | "
            f"retained={summary.retained_clients} | "
            f"rejected={summary.rejected_clients}"
        )

    print(
        "\nCKKS / Visualization Samples:"
    )

    for record in (
        result.ckks_chain_records[
            :max_chain_records
        ]
    ):
        print(
            "\n"
            + "-" * 96
        )

        print(
            f"Client {record.client_id:02d} | "
            f"{record.layer_name} | "
            f"{record.tensor_role}"
        )

        print(
            f"  D mathematical reference : "
            f"{record.raw_squared_distance:.12f}"
        )

        print(
            f"  decrypt(D) [DEMO ONLY]   : "
            f"{record.decrypted_raw_squared_distance:.12f}"
        )

        print(
            f"  h                        : "
            f"{record.round_random_factor_h:.12f}"
        )

        print(
            f"  hD mathematical reference: "
            f"{record.protected_squared_distance:.12f}"
        )

        print(
            f"  decrypt(hD) [FORMAL]     : "
            f"{record.decrypted_protected_squared_distance:.12f}"
        )


# =============================================================================
# 测试样例
#
# 非常重要：
#   下面所有测试都在 if __name__ == "__main__": 中。
#
# 因此正常 import 本模块时：
#   - 不会创建测试数据
#   - 不会创建 CKKS context
#   - 不会执行测试
#   - 不会 print
#
# 只有直接运行：
#
#   python Step_07_V_byzantine.py
#
# 才会进入下面。
# =============================================================================

if __name__ == "__main__":
    print(
        "=" * 96
    )
    print(
        "Step 07 Standalone Test"
    )
    print(
        "=" * 96
    )

    if ts is None:
        print(
            "TenSEAL 未安装，因此跳过 CKKS 执行测试。"
        )
        print(
            "安装后运行：pip install tenseal"
        )
        print(
            "注意：正常 import 本文件不会执行这里的测试代码。"
        )
        raise SystemExit(0)

    # -------------------------------------------------------------------------
    # 为了让 Step 07 可以独立裸跑，
    # 这里构造一个最小的“Step 06-like”对象。
    #
    # 正式 main.py 中不会使用这个 fake object，
    # 而是直接传入真正的 Step 06 MaskingAndUploadingResult。
    # -------------------------------------------------------------------------

    @dataclass
    class _FakeStep06Result:
        round_index: int
        master_seed: int
        masked_parameters_by_server: Dict[
            str,
            Dict[
                int,
                Dict[
                    str,
                    List[torch.Tensor],
                ],
            ],
        ]

    def _build_fake_step06_result() -> _FakeStep06Result:
        """
        构造 10 个客户端。

        layer1:
            weight: 4 个分量
            bias:   2 个分量

        Client 9:
            weight 明显异常
            bias 保持正常

        这样可以验证：
            weight/bias 确实独立筛选。
        """

        generator = torch.Generator(
            device="cpu"
        )
        generator.manual_seed(
            2026
        )

        original_parameters: Dict[
            int,
            Dict[
                str,
                List[torch.Tensor],
            ],
        ] = {}

        for client_id in range(10):
            weight = (
                torch.tensor(
                    [1.0, 2.0, 3.0, 4.0],
                    dtype=torch.float64,
                )
                + 0.02
                * torch.randn(
                    4,
                    generator=generator,
                    dtype=torch.float64,
                )
            )

            bias = (
                torch.tensor(
                    [0.1, -0.1],
                    dtype=torch.float64,
                )
                + 0.01
                * torch.randn(
                    2,
                    generator=generator,
                    dtype=torch.float64,
                )
            )

            if client_id == 9:
                # 只攻击 weight。
                weight = torch.tensor(
                    [30.0, -25.0, 40.0, -35.0],
                    dtype=torch.float64,
                )

            original_parameters[
                client_id
            ] = {
                "layer1": [
                    weight,
                    bias,
                ]
            }

        by_server: Dict[
            str,
            Dict[
                int,
                Dict[
                    str,
                    List[torch.Tensor],
                ],
            ],
        ] = {
            server: {}
            for server in SERVER_NAMES
        }

        mask_generator = torch.Generator(
            device="cpu"
        )
        mask_generator.manual_seed(
            42
        )

        for client_id, layer_map in (
            original_parameters.items()
        ):
            for server in SERVER_NAMES:
                by_server[
                    server
                ][
                    client_id
                ] = {}

            for layer_name, tensors in (
                layer_map.items()
            ):
                for server in SERVER_NAMES:
                    by_server[
                        server
                    ][
                        client_id
                    ][
                        layer_name
                    ] = []

                for theta in tensors:
                    r_ts = torch.randn(
                        theta.shape,
                        generator=mask_generator,
                        dtype=theta.dtype,
                    )

                    r_ds1 = torch.randn(
                        theta.shape,
                        generator=mask_generator,
                        dtype=theta.dtype,
                    )

                    r_ds2 = torch.randn(
                        theta.shape,
                        generator=mask_generator,
                        dtype=theta.dtype,
                    )

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

                    for server in SERVER_NAMES:
                        by_server[
                            server
                        ][
                            client_id
                        ][
                            layer_name
                        ].append(
                            theta
                            + masks[server]
                        )

        return _FakeStep06Result(
            round_index=3,
            master_seed=12345,
            masked_parameters_by_server=(
                by_server
            ),
        )

    fake_step06 = (
        _build_fake_step06_result()
    )

    result = detect_byzantine_from_step06(
        fake_step06,
        tau=0.7,
        h_min=0.5,
        h_max=2.0,
        preview_values_per_tensor=3,
    )

    print_byzantine_detection_result(
        result,
        max_chain_records=4,
    )

    # -------------------------------------------------------------------------
    # Automatic Checks
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 96
    )
    print(
        "Automatic Checks"
    )
    print(
        "=" * 96
    )

    # 1. h 必须为正。
    check_h_positive = (
        result.round_random_factor_h > 0.0
    )

    print(
        "h > 0 :",
        check_h_positive,
    )

    # 2. 同轮只有一个 h。
    all_record_h = {
        record.round_random_factor_h
        for record in result.ckks_chain_records
    }

    check_one_h_per_round = (
        len(all_record_h) == 1
    )

    print(
        "同一轮所有 Client/Layer/weight/bias 共用一个 h :",
        check_one_h_per_round,
    )

    # 3. tau=0.7 / 10 clients -> ceil(7)=7
    weight_summary = next(
        summary
        for summary in result.component_summaries
        if (
            summary.layer_name == "layer1"
            and summary.tensor_role == "weight"
        )
    )

    bias_summary = next(
        summary
        for summary in result.component_summaries
        if (
            summary.layer_name == "layer1"
            and summary.tensor_role == "bias"
        )
    )

    print(
        "tau=0.7 / 10 clients -> weight 保留 7 :",
        weight_summary.keep_count == 7,
    )

    print(
        "tau=0.7 / 10 clients -> bias 保留 7 :",
        bias_summary.keep_count == 7,
    )

    # 4. 极端异常 weight 的 Client 9 应被拒绝。
    print(
        "Client 9 异常 weight 被过滤 :",
        9 in weight_summary.rejected_clients,
    )

    # 5. Client 9 bias 没被攻击，因此允许独立于 weight 决策。
    print(
        "weight / bias 独立决策 :",
        (
            9 in weight_summary.rejected_clients
            and (
                9 in bias_summary.retained_clients
                or 9 in bias_summary.rejected_clients
            )
        ),
    )

    # 6. 可视化字段必须同时包含 D 和 hD。
    visualization_rows = (
        result.distance_visualization_data
    )

    check_visualization_has_both = (
        bool(visualization_rows)
        and all(
            (
                "raw_squared_distance" in row
                and "protected_squared_distance" in row
                and "decrypted_raw_squared_distance" in row
                and "decrypted_protected_squared_distance" in row
            )
            for row in visualization_rows
        )
    )

    print(
        "可视化同时保留 D 与 hD :",
        check_visualization_has_both,
    )

    # 7. CKKS 解密结果应接近数学真值。
    #
    # CKKS 是近似同态加密，因此不使用 ==。
    max_raw_error = max(
        record.raw_distance_abs_error
        for record in result.ckks_chain_records
    )

    max_protected_error = max(
        record.protected_distance_abs_error
        for record in result.ckks_chain_records
    )

    print(
        "最大 |decrypt(D)-D| :",
        max_raw_error,
    )

    print(
        "最大 |decrypt(hD)-hD| :",
        max_protected_error,
    )

    # 8. 相同 seed / round 下 h 可复现。
    repeat_h = (
        generate_round_random_factor_h(
            master_seed=(
                fake_step06.master_seed
            ),
            round_index=(
                fake_step06.round_index
            ),
            h_min=0.5,
            h_max=2.0,
        )
    )

    print(
        "相同 seed / round 下 h 可复现 :",
        repeat_h
        == result.round_random_factor_h,
    )

    # 9. 下一轮 h 应改变。
    next_round_h = (
        generate_round_random_factor_h(
            master_seed=(
                fake_step06.master_seed
            ),
            round_index=(
                fake_step06.round_index
                + 1
            ),
            h_min=0.5,
            h_max=2.0,
        )
    )

    print(
        "不同 round 会产生不同 h :",
        next_round_h
        != result.round_random_factor_h,
    )

    print(
        "\n提示：以上测试只会在直接运行本文件时执行；"
        "import 本模块不会执行测试。"
    )
