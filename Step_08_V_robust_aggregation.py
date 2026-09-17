"""
Step_08_V_robust_aggregation.py

===============================================================================
Step 08：基于 Step 07 筛选结果的层级鲁棒聚合（可视化相关）
===============================================================================

【本步骤的核心职责】

Step 07 已经完成 Byzantine 检测，并输出：

    retained_clients[layer_name][tensor_index]
    rejected_clients[layer_name][tensor_index]

其中：
    tensor_index = 0 -> weight
    tensor_index = 1 -> bias

Step 08 不重新检测 Byzantine，也不读取 Step 00 的真实 Byzantine 标签。
本步骤只相信 Step 07 的检测结果，并对“被保留的参数组件”执行最终鲁棒聚合。

因此聚合粒度严格保持为：

    Layer-wise + Intra-layer Fine-grained Aggregation

也就是说：

    layer1 / weight -> 使用自己的 retained client set
    layer1 / bias   -> 使用自己的 retained client set
    layer2 / weight -> 使用自己的 retained client set
    layer2 / bias   -> 使用自己的 retained client set
    ...

绝不能把某一层 weight 的 retained set 强行用于同层 bias。

-------------------------------------------------------------------------------
【支持两条 Step 06 / Step 07 分支】
-------------------------------------------------------------------------------

1. SP / Plaintext 分支

    Step_06_V_sp_uploading.py
        -> Step_07_V_sp_byzantine.py
        -> Step_08_V_robust_aggregation.py

TS 已经持有客户端明文共享参数 theta，因此对某层某组件：

    G_robust = (1 / |B|) * sum_{k in B} theta_k

其中 B 为 Step 07 给出的 retained client set。

2. Random Masking + CKKS Byzantine Detection 分支

    Step_06_V_masking_and_uploading.py
        -> Step_07_V_byzantine.py
        -> Step_08_V_robust_aggregation.py

Step 06 中四个逻辑服务器持有：

    omega_TS  = theta + r_TS
    omega_DS1 = theta + r_DS1
    omega_DS2 = theta + r_DS2
    omega_DS3 = theta + r_DS3

并满足：

    r_TS + r_DS1 + r_DS2 + r_DS3 = 0

所以：

    omega_TS + omega_DS1 + omega_DS2 + omega_DS3 = 4 * theta

对 Step 07 保留下来的同一个客户端集合 B，各服务器先分别聚合：

    A_server = (1 / |B|) * sum_{k in B} omega_{k,server}

然后恢复最终鲁棒聚合结果：

    G_robust
        = (1 / 4) * (
              A_TS + A_DS1 + A_DS2 + A_DS3
          )
        = (1 / |B|) * sum_{k in B} theta_k

这样 Step 08 不需要先恢复每一个客户端的原始 theta 再聚合。

-------------------------------------------------------------------------------
【V：给可视化模块预留的数据】
-------------------------------------------------------------------------------

本文件不负责画图。

“V”的含义是：算法执行时主动生成 visualization-ready variables，
让后续前端/可视化模块直接读取，不需要重新解析 Tensor 或重复计算。

核心变量：

    result.aggregation_visualization_data

这是一个 List[Dict]，每一项对应：

    一个 layer + 一个 tensor component(weight / bias)

其中包含：

    layer_name
    layer_index
    tensor_index
    tensor_role

    participating_client_ids
    retained_client_ids
    rejected_client_ids

    participant_count
    retained_count
    rejected_count
    retention_ratio
    rejection_ratio

    aggregation_mode

    naive_aggregate_preview
    robust_aggregate_preview
    aggregate_delta_preview

    naive_aggregate_norm
    robust_aggregate_norm
    aggregate_difference_norm
    aggregate_difference_relative_norm

    robust_aggregate_mean
    robust_aggregate_std
    robust_aggregate_min
    robust_aggregate_max

    naive_aggregate_mean
    naive_aggregate_std
    naive_aggregate_min
    naive_aggregate_max

    parameter_shape
    parameter_numel

    server_aggregate_previews
    reconstruction_consistency_error

其中：
    naive aggregate  = 不做 Byzantine 过滤、直接对所有参与者平均
    robust aggregate = 只对 Step 07 retained clients 平均

因此前端可以直接展示：

    “过滤前聚合结果 vs 过滤后鲁棒聚合结果”
    “哪些客户端被保留 / 排除”
    “鲁棒聚合前后参数变化量”
    “每层 weight / bias 的独立聚合情况”

注意：
    preview / norm / mean / std 等仅用于展示。
    真正传给下一轮训练的是完整 Tensor：

        result.robust_aggregated_parameters

-------------------------------------------------------------------------------
【正式输出】
-------------------------------------------------------------------------------

    RobustAggregationResult

其中最重要的两个字段：

1.
    result.robust_aggregated_parameters

结构：

    {
        "layer1": [aggregated_weight, aggregated_bias],
        "layer2": [aggregated_weight, aggregated_bias],
        ...
    }

这是下一轮共享模型真正使用的完整聚合参数。

2.
    result.aggregation_visualization_data

这是可视化模块直接消费的数据。

-------------------------------------------------------------------------------
【与联邦学习训练轮次的关系】
-------------------------------------------------------------------------------

本项目中，Step 03 与 Step 04~09 的职责必须严格区分。

Step 03：架构分配阶段，只在联邦训练开始前执行一次。
一旦完成分配，各客户端“哪些层共享、哪些层个性化”的结构在后续全部
联邦训练轮次中保持不变，不会在每一轮重新执行架构分配。

真正按轮重复执行的是：

    Step 04 -> Step 05 -> Step 06 -> Step 07 -> Step 08 -> Step 09
       ^                                                        |
       |                                                        |
       +------------ 下一轮共享参数 / 客户端状态 ---------------+

其中：
    Step 04：客户端基于当前轮共享参数执行本地训练；
    Step 05：实施本轮预先配置的 Byzantine 攻击；
    Step 06：执行安全上传 / 明文上传；
    Step 07：执行 Layer-wise Byzantine 检测与 weight/bias 独立筛选；
    Step 08：仅使用 Step 07 retained 参数执行层级鲁棒聚合；
    Step 09：在 Step 08 聚合之后，对真实良性客户端执行 local-test evaluation。

第 t 轮 Step 08 得到：

    G_robust^(t) = result.robust_aggregated_parameters

Step 09 使用 G_robust^(t) 与各良性客户端本轮保留的 personalized layers
构造聚合后的完整个性化模型，并在各自 local test set 上计算准确率。
Step 09 只负责评估，不修改模型参数。

Step 09 完成后：
    G_robust^(t)
        -> 第 t+1 轮 Step 04 的 global_shared_parameters

同时：
    第 t 轮 Step 04 的 client_state_dicts
        -> 第 t+1 轮 Step 04 的 previous_client_state_dicts
           （用于继续保留各客户端自己的 personalized layers）

因此：
    Step 03    = 一次性的架构初始化阶段
    Step 04~08 = 每轮训练、安全处理、检测与鲁棒聚合
    Step 09    = 每轮聚合后的模型性能评估
    Step 04~09 = 完整的 round loop

本 Step 08 只负责当前轮鲁棒聚合。
外层主训练流程必须先调用 Step 09 完成本轮评估，再进入下一轮 Step 04。

-------------------------------------------------------------------------------
【测试约定】
-------------------------------------------------------------------------------

正式核心函数：
    只消费 Step 06 + Step 07 的结果，不主动重跑 Step 00~07。

文件底部：
    if __name__ == "__main__":
        ...

只放独立测试样例。
因此 import 本模块不会自动执行测试。
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import math

import torch


SERVER_NAMES: Tuple[str, str, str, str] = (
    "TS",
    "DS1",
    "DS2",
    "DS3",
)

NUM_SERVERS = 4
DEFAULT_PREVIEW_VALUES_PER_TENSOR = 5


# =============================================================================
# 数据结构
# =============================================================================

@dataclass(frozen=True)
class AggregationComponentSummary:
    """
    单个 Layer / Tensor component 的鲁棒聚合摘要。

    完整 Tensor 不塞进本 summary；
    完整聚合结果统一保存在 result.robust_aggregated_parameters 中。
    """

    layer_name: str
    layer_index: int
    tensor_index: int
    tensor_role: str

    participating_client_ids: List[int]
    retained_client_ids: List[int]
    rejected_client_ids: List[int]

    participant_count: int
    retained_count: int
    rejected_count: int
    retention_ratio: float
    rejection_ratio: float

    aggregation_mode: str

    parameter_shape: Tuple[int, ...]
    parameter_numel: int

    naive_aggregate_preview: List[float]
    robust_aggregate_preview: List[float]
    aggregate_delta_preview: List[float]

    naive_aggregate_norm: float
    robust_aggregate_norm: float
    aggregate_difference_norm: float
    aggregate_difference_relative_norm: float

    naive_aggregate_mean: float
    naive_aggregate_std: float
    naive_aggregate_min: float
    naive_aggregate_max: float

    robust_aggregate_mean: float
    robust_aggregate_std: float
    robust_aggregate_min: float
    robust_aggregate_max: float

    # Full masking 分支中用于展示四服务器各自聚合结果。
    # SP 分支为空字典。
    server_aggregate_previews: Dict[str, List[float]]

    # Full masking 分支：
    # 用四服务器恢复的 robust aggregate 与直接数学参考聚合之间的最大绝对误差。
    #
    # SP 分支没有四服务器恢复过程，因此为 0.0。
    reconstruction_consistency_error: float

    @property
    def visualization_dict(self) -> Dict[str, Any]:
        """
        转成前端可直接读取的普通 Dict。
        """
        return {
            "layer_name": self.layer_name,
            "layer_index": self.layer_index,
            "tensor_index": self.tensor_index,
            "tensor_role": self.tensor_role,

            "participating_client_ids": (
                self.participating_client_ids.copy()
            ),
            "retained_client_ids": (
                self.retained_client_ids.copy()
            ),
            "rejected_client_ids": (
                self.rejected_client_ids.copy()
            ),

            "participant_count": self.participant_count,
            "retained_count": self.retained_count,
            "rejected_count": self.rejected_count,
            "retention_ratio": self.retention_ratio,
            "rejection_ratio": self.rejection_ratio,

            "aggregation_mode": self.aggregation_mode,

            "parameter_shape": self.parameter_shape,
            "parameter_numel": self.parameter_numel,

            "naive_aggregate_preview": (
                self.naive_aggregate_preview.copy()
            ),
            "robust_aggregate_preview": (
                self.robust_aggregate_preview.copy()
            ),
            "aggregate_delta_preview": (
                self.aggregate_delta_preview.copy()
            ),

            "naive_aggregate_norm": self.naive_aggregate_norm,
            "robust_aggregate_norm": self.robust_aggregate_norm,
            "aggregate_difference_norm": (
                self.aggregate_difference_norm
            ),
            "aggregate_difference_relative_norm": (
                self.aggregate_difference_relative_norm
            ),

            "naive_aggregate_mean": self.naive_aggregate_mean,
            "naive_aggregate_std": self.naive_aggregate_std,
            "naive_aggregate_min": self.naive_aggregate_min,
            "naive_aggregate_max": self.naive_aggregate_max,

            "robust_aggregate_mean": self.robust_aggregate_mean,
            "robust_aggregate_std": self.robust_aggregate_std,
            "robust_aggregate_min": self.robust_aggregate_min,
            "robust_aggregate_max": self.robust_aggregate_max,

            "server_aggregate_previews": {
                server_name: preview.copy()
                for server_name, preview
                in self.server_aggregate_previews.items()
            },

            "reconstruction_consistency_error": (
                self.reconstruction_consistency_error
            ),
        }


@dataclass
class RobustAggregationResult:
    """
    Step 08 统一输出。

    robust_aggregated_parameters：
        正式算法输出，供下一轮训练更新共享模型。

    aggregation_visualization_data：
        专门给可视化模块使用。
    """

    round_index: int
    master_seed: int
    aggregation_mode: str

    # Layer -> List[Tensor]
    robust_aggregated_parameters: Dict[
        str, List[torch.Tensor]
    ]

    # 不过滤时的直接平均，仅作为对照/可视化数据。
    naive_aggregated_parameters: Dict[
        str, List[torch.Tensor]
    ]

    # Layer -> tensor_index -> retained/rejected ids
    retained_clients: Dict[
        str, Dict[int, List[int]]
    ]
    rejected_clients: Dict[
        str, Dict[int, List[int]]
    ]

    component_summaries: List[
        AggregationComponentSummary
    ]

    @property
    def aggregation_visualization_data(
        self,
    ) -> List[Dict[str, Any]]:
        """
        可视化同学的主要对接变量。
        """
        return [
            summary.visualization_dict
            for summary in self.component_summaries
        ]

    @property
    def visualization_data(
        self,
    ) -> List[Dict[str, Any]]:
        """
        简短别名，兼容后续统一前端接口。
        """
        return self.aggregation_visualization_data

    @property
    def aggregation_visualization_summary(
        self,
    ) -> Dict[str, Any]:
        """
        页面顶部/总览卡片可以直接使用的汇总变量。
        """
        total_components = len(
            self.component_summaries
        )

        total_participations = sum(
            summary.participant_count
            for summary in self.component_summaries
        )

        total_retained = sum(
            summary.retained_count
            for summary in self.component_summaries
        )

        total_rejected = sum(
            summary.rejected_count
            for summary in self.component_summaries
        )

        mean_difference_norm = (
            sum(
                summary.aggregate_difference_norm
                for summary in self.component_summaries
            )
            / total_components
            if total_components > 0
            else 0.0
        )

        return {
            "round_index": self.round_index,
            "master_seed": self.master_seed,
            "aggregation_mode": self.aggregation_mode,

            "component_count": total_components,
            "total_parameter_participations": (
                total_participations
            ),
            "total_retained_participations": (
                total_retained
            ),
            "total_rejected_participations": (
                total_rejected
            ),

            "overall_retention_ratio": (
                total_retained
                / total_participations
                if total_participations > 0
                else 0.0
            ),

            "overall_rejection_ratio": (
                total_rejected
                / total_participations
                if total_participations > 0
                else 0.0
            ),

            "mean_aggregate_difference_norm": (
                mean_difference_norm
            ),

            "layer_names": sorted(
                self.robust_aggregated_parameters
            ),
        }


# =============================================================================
# 基础工具
# =============================================================================

def _tensor_role(
    tensor_index: int,
) -> str:
    if tensor_index == 0:
        return "weight"
    if tensor_index == 1:
        return "bias"
    return f"param_{tensor_index}"


def _layer_index(
    layer_name: str,
) -> int:
    if (
        layer_name.startswith("layer")
        and layer_name[5:].isdigit()
    ):
        return int(layer_name[5:])
    return -1


def _preview_values(
    tensor: torch.Tensor,
    count: int,
) -> List[float]:
    if count <= 0:
        return []

    flat = (
        tensor.detach()
        .cpu()
        .reshape(-1)
    )

    return [
        float(value)
        for value in flat[:count].tolist()
    ]


def _tensor_norm(
    tensor: torch.Tensor,
) -> float:
    return float(
        torch.linalg.vector_norm(
            tensor.detach()
            .double()
            .cpu()
            .reshape(-1)
        ).item()
    )


def _tensor_stats(
    tensor: torch.Tensor,
) -> Tuple[float, float, float, float]:
    flat = (
        tensor.detach()
        .double()
        .cpu()
        .reshape(-1)
    )

    if flat.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0

    mean_value = float(
        torch.mean(flat).item()
    )

    # unbiased=False，避免只有 1 个元素时得到 nan。
    std_value = float(
        torch.std(
            flat,
            unbiased=False,
        ).item()
    )

    min_value = float(
        torch.min(flat).item()
    )

    max_value = float(
        torch.max(flat).item()
    )

    return (
        mean_value,
        std_value,
        min_value,
        max_value,
    )


def _mean_tensors(
    tensors: Sequence[torch.Tensor],
) -> torch.Tensor:
    """
    对同 shape Tensor 求算术平均。

    计算阶段统一转 CPU float64，最后恢复首个 Tensor 的 dtype。
    这样既减少聚合累计误差，又保持下游模型参数 dtype 兼容。
    """
    if not tensors:
        raise ValueError(
            "不能对空 Tensor 集合执行聚合"
        )

    reference = tensors[0]

    reference_shape = tuple(
        reference.shape
    )

    for tensor in tensors:
        if tuple(tensor.shape) != reference_shape:
            raise ValueError(
                "参与同一参数组件聚合的 Tensor shape 不一致："
                f"{reference_shape} vs {tuple(tensor.shape)}"
            )

    stacked = torch.stack(
        [
            tensor.detach()
            .double()
            .cpu()
            for tensor in tensors
        ],
        dim=0,
    )

    mean_tensor = torch.mean(
        stacked,
        dim=0,
    )

    return mean_tensor.to(
        dtype=reference.dtype
    )


def _clone_client_sets(
    source: Mapping[
        str, Mapping[int, Sequence[int]]
    ],
) -> Dict[str, Dict[int, List[int]]]:
    return {
        str(layer_name): {
            int(tensor_index): [
                int(client_id)
                for client_id in client_ids
            ]
            for tensor_index, client_ids
            in tensor_map.items()
        }
        for layer_name, tensor_map
        in source.items()
    }


def _validate_detection_result(
    detection_result: Any,
) -> None:
    if not hasattr(
        detection_result,
        "retained_clients",
    ):
        raise TypeError(
            "Step 07 输出缺少 retained_clients"
        )

    if not hasattr(
        detection_result,
        "rejected_clients",
    ):
        raise TypeError(
            "Step 07 输出缺少 rejected_clients"
        )


def _resolve_round_metadata(
    step06_result: Any,
    step07_result: Any,
) -> Tuple[int, int]:
    round_index = int(
        getattr(
            step07_result,
            "round_index",
            getattr(
                step06_result,
                "round_index",
                0,
            ),
        )
    )

    master_seed = int(
        getattr(
            step07_result,
            "master_seed",
            getattr(
                step06_result,
                "master_seed",
                0,
            ),
        )
    )

    return round_index, master_seed


def _build_summary(
    *,
    layer_name: str,
    tensor_index: int,
    participating_client_ids: Sequence[int],
    retained_client_ids: Sequence[int],
    rejected_client_ids: Sequence[int],
    aggregation_mode: str,
    naive_aggregate: torch.Tensor,
    robust_aggregate: torch.Tensor,
    preview_values_per_tensor: int,
    server_aggregate_previews: Dict[
        str, List[float]
    ],
    reconstruction_consistency_error: float,
) -> AggregationComponentSummary:

    delta = (
        robust_aggregate.detach()
        .double()
        .cpu()
        - naive_aggregate.detach()
        .double()
        .cpu()
    )

    naive_norm = _tensor_norm(
        naive_aggregate
    )
    robust_norm = _tensor_norm(
        robust_aggregate
    )
    difference_norm = _tensor_norm(
        delta
    )

    relative_difference = (
        difference_norm
        / max(
            naive_norm,
            1e-12,
        )
    )

    (
        naive_mean,
        naive_std,
        naive_min,
        naive_max,
    ) = _tensor_stats(
        naive_aggregate
    )

    (
        robust_mean,
        robust_std,
        robust_min,
        robust_max,
    ) = _tensor_stats(
        robust_aggregate
    )

    participant_count = len(
        participating_client_ids
    )
    retained_count = len(
        retained_client_ids
    )
    rejected_count = len(
        rejected_client_ids
    )

    return AggregationComponentSummary(
        layer_name=str(layer_name),
        layer_index=_layer_index(
            str(layer_name)
        ),
        tensor_index=int(
            tensor_index
        ),
        tensor_role=_tensor_role(
            int(tensor_index)
        ),

        participating_client_ids=[
            int(client_id)
            for client_id
            in participating_client_ids
        ],
        retained_client_ids=[
            int(client_id)
            for client_id
            in retained_client_ids
        ],
        rejected_client_ids=[
            int(client_id)
            for client_id
            in rejected_client_ids
        ],

        participant_count=participant_count,
        retained_count=retained_count,
        rejected_count=rejected_count,

        retention_ratio=(
            retained_count
            / participant_count
            if participant_count > 0
            else 0.0
        ),

        rejection_ratio=(
            rejected_count
            / participant_count
            if participant_count > 0
            else 0.0
        ),

        aggregation_mode=str(
            aggregation_mode
        ),

        parameter_shape=tuple(
            robust_aggregate.shape
        ),
        parameter_numel=int(
            robust_aggregate.numel()
        ),

        naive_aggregate_preview=(
            _preview_values(
                naive_aggregate,
                preview_values_per_tensor,
            )
        ),

        robust_aggregate_preview=(
            _preview_values(
                robust_aggregate,
                preview_values_per_tensor,
            )
        ),

        aggregate_delta_preview=(
            _preview_values(
                delta,
                preview_values_per_tensor,
            )
        ),

        naive_aggregate_norm=naive_norm,
        robust_aggregate_norm=robust_norm,
        aggregate_difference_norm=(
            difference_norm
        ),
        aggregate_difference_relative_norm=(
            relative_difference
        ),

        naive_aggregate_mean=naive_mean,
        naive_aggregate_std=naive_std,
        naive_aggregate_min=naive_min,
        naive_aggregate_max=naive_max,

        robust_aggregate_mean=robust_mean,
        robust_aggregate_std=robust_std,
        robust_aggregate_min=robust_min,
        robust_aggregate_max=robust_max,

        server_aggregate_previews={
            str(server_name): list(preview)
            for server_name, preview
            in server_aggregate_previews.items()
        },

        reconstruction_consistency_error=float(
            reconstruction_consistency_error
        ),
    )


# =============================================================================
# SP / Plaintext 分支
# =============================================================================

def robust_aggregate_from_sp(
    plain_uploading_result: Any,
    detection_result: Any,
    *,
    preview_values_per_tensor: int = (
        DEFAULT_PREVIEW_VALUES_PER_TENSOR
    ),
) -> RobustAggregationResult:
    """
    Step 06 SP + Step 07 SP -> Step 08。

    正式核心函数：
        只吃上游输出，不主动重跑前置步骤。
    """

    if int(
        preview_values_per_tensor
    ) < 0:
        raise ValueError(
            "preview_values_per_tensor 必须 >= 0"
        )

    _validate_detection_result(
        detection_result
    )

    if not hasattr(
        plain_uploading_result,
        "ts_parameters",
    ):
        raise TypeError(
            "SP Step 06 输出缺少 ts_parameters"
        )

    plaintext_parameters = (
        plain_uploading_result
        .ts_parameters
    )

    retained_clients = (
        _clone_client_sets(
            detection_result
            .retained_clients
        )
    )

    rejected_clients = (
        _clone_client_sets(
            detection_result
            .rejected_clients
        )
    )

    round_index, master_seed = (
        _resolve_round_metadata(
            plain_uploading_result,
            detection_result,
        )
    )

    robust_parameters: Dict[
        str, List[torch.Tensor]
    ] = {}

    naive_parameters: Dict[
        str, List[torch.Tensor]
    ] = {}

    summaries: List[
        AggregationComponentSummary
    ] = []

    # Step 07 的 retained_clients 本身就是
    # Layer -> tensor_index -> client ids，
    # 因此直接以它作为 Step 08 的聚合任务清单。
    for layer_name in sorted(
        retained_clients
    ):
        robust_parameters[
            layer_name
        ] = []

        naive_parameters[
            layer_name
        ] = []

        for tensor_index in sorted(
            retained_clients[
                layer_name
            ]
        ):
            retained_ids = (
                retained_clients[
                    layer_name
                ][
                    tensor_index
                ]
            )

            rejected_ids = (
                rejected_clients
                .get(
                    layer_name,
                    {},
                )
                .get(
                    tensor_index,
                    [],
                )
            )

            if not retained_ids:
                raise RuntimeError(
                    f"{layer_name} / tensor {tensor_index} "
                    "没有 retained client，无法聚合"
                )

            participating_ids = sorted(
                set(
                    retained_ids
                )
                | set(
                    rejected_ids
                )
            )

            if not participating_ids:
                raise RuntimeError(
                    f"{layer_name} / tensor {tensor_index} "
                    "没有参与客户端"
                )

            def get_tensor(
                client_id: int,
            ) -> torch.Tensor:
                if client_id not in plaintext_parameters:
                    raise KeyError(
                        f"SP 参数缺少 Client {client_id}"
                    )

                if (
                    layer_name
                    not in plaintext_parameters[
                        client_id
                    ]
                ):
                    raise KeyError(
                        f"Client {client_id} 缺少 {layer_name}"
                    )

                tensors = (
                    plaintext_parameters[
                        client_id
                    ][
                        layer_name
                    ]
                )

                if tensor_index >= len(
                    tensors
                ):
                    raise IndexError(
                        f"Client {client_id} / "
                        f"{layer_name} 缺少 tensor_index="
                        f"{tensor_index}"
                    )

                return tensors[
                    tensor_index
                ]

            naive_aggregate = (
                _mean_tensors(
                    [
                        get_tensor(
                            client_id
                        )
                        for client_id
                        in participating_ids
                    ]
                )
            )

            robust_aggregate = (
                _mean_tensors(
                    [
                        get_tensor(
                            client_id
                        )
                        for client_id
                        in retained_ids
                    ]
                )
            )

            naive_parameters[
                layer_name
            ].append(
                naive_aggregate.clone()
            )

            robust_parameters[
                layer_name
            ].append(
                robust_aggregate.clone()
            )

            summaries.append(
                _build_summary(
                    layer_name=layer_name,
                    tensor_index=tensor_index,
                    participating_client_ids=(
                        participating_ids
                    ),
                    retained_client_ids=(
                        retained_ids
                    ),
                    rejected_client_ids=(
                        rejected_ids
                    ),
                    aggregation_mode=(
                        "SP_PLAINTEXT_ROBUST_MEAN"
                    ),
                    naive_aggregate=(
                        naive_aggregate
                    ),
                    robust_aggregate=(
                        robust_aggregate
                    ),
                    preview_values_per_tensor=(
                        preview_values_per_tensor
                    ),
                    server_aggregate_previews={},
                    reconstruction_consistency_error=(
                        0.0
                    ),
                )
            )

    return RobustAggregationResult(
        round_index=round_index,
        master_seed=master_seed,
        aggregation_mode=(
            "SP_PLAINTEXT_ROBUST_MEAN"
        ),
        robust_aggregated_parameters=(
            robust_parameters
        ),
        naive_aggregated_parameters=(
            naive_parameters
        ),
        retained_clients=retained_clients,
        rejected_clients=rejected_clients,
        component_summaries=summaries,
    )


# =============================================================================
# Full Random-Masking 分支
# =============================================================================

def robust_aggregate_from_masking(
    masking_result: Any,
    detection_result: Any,
    *,
    preview_values_per_tensor: int = (
        DEFAULT_PREVIEW_VALUES_PER_TENSOR
    ),
) -> RobustAggregationResult:
    """
    Step 06 Random Masking + Step 07 CKKS Detection -> Step 08。

    对每个 layer / tensor component：

        1. 读取 Step 07 retained set；
        2. TS / DS1 / DS2 / DS3 分别只聚合 retained clients；
        3. 四服务器聚合结果相加；
        4. 除以 4；
        5. 得到最终 robust aggregate。

    注意：
        不先逐客户端恢复 theta。
    """

    if int(
        preview_values_per_tensor
    ) < 0:
        raise ValueError(
            "preview_values_per_tensor 必须 >= 0"
        )

    _validate_detection_result(
        detection_result
    )

    if not hasattr(
        masking_result,
        "masked_parameters_by_server",
    ):
        raise TypeError(
            "Full Step 06 输出缺少 "
            "masked_parameters_by_server"
        )

    masked_by_server = (
        masking_result
        .masked_parameters_by_server
    )

    missing_servers = [
        server_name
        for server_name in SERVER_NAMES
        if server_name
        not in masked_by_server
    ]

    if missing_servers:
        raise KeyError(
            "Full Step 06 缺少服务器参数："
            f"{missing_servers}"
        )

    retained_clients = (
        _clone_client_sets(
            detection_result
            .retained_clients
        )
    )

    rejected_clients = (
        _clone_client_sets(
            detection_result
            .rejected_clients
        )
    )

    round_index, master_seed = (
        _resolve_round_metadata(
            masking_result,
            detection_result,
        )
    )

    robust_parameters: Dict[
        str, List[torch.Tensor]
    ] = {}

    naive_parameters: Dict[
        str, List[torch.Tensor]
    ] = {}

    summaries: List[
        AggregationComponentSummary
    ] = []

    for layer_name in sorted(
        retained_clients
    ):
        robust_parameters[
            layer_name
        ] = []

        naive_parameters[
            layer_name
        ] = []

        for tensor_index in sorted(
            retained_clients[
                layer_name
            ]
        ):
            retained_ids = (
                retained_clients[
                    layer_name
                ][
                    tensor_index
                ]
            )

            rejected_ids = (
                rejected_clients
                .get(
                    layer_name,
                    {},
                )
                .get(
                    tensor_index,
                    [],
                )
            )

            if not retained_ids:
                raise RuntimeError(
                    f"{layer_name} / tensor {tensor_index} "
                    "没有 retained client，无法聚合"
                )

            participating_ids = sorted(
                set(
                    retained_ids
                )
                | set(
                    rejected_ids
                )
            )

            if not participating_ids:
                raise RuntimeError(
                    f"{layer_name} / tensor {tensor_index} "
                    "没有参与客户端"
                )

            def get_masked_tensor(
                server_name: str,
                client_id: int,
            ) -> torch.Tensor:
                server_map = (
                    masked_by_server[
                        server_name
                    ]
                )

                if client_id not in server_map:
                    raise KeyError(
                        f"{server_name} 缺少 Client "
                        f"{client_id}"
                    )

                if (
                    layer_name
                    not in server_map[
                        client_id
                    ]
                ):
                    raise KeyError(
                        f"{server_name} / Client "
                        f"{client_id} 缺少 {layer_name}"
                    )

                tensors = (
                    server_map[
                        client_id
                    ][
                        layer_name
                    ]
                )

                if tensor_index >= len(
                    tensors
                ):
                    raise IndexError(
                        f"{server_name} / Client "
                        f"{client_id} / {layer_name} "
                        f"缺少 tensor_index={tensor_index}"
                    )

                return tensors[
                    tensor_index
                ]

            # -------------------------------------------------------------
            # A. Naive aggregate：所有参与客户端，不做过滤。
            #    仅用于可视化对照。
            # -------------------------------------------------------------
            naive_server_aggregates: Dict[
                str, torch.Tensor
            ] = {}

            for server_name in SERVER_NAMES:
                naive_server_aggregates[
                    server_name
                ] = _mean_tensors(
                    [
                        get_masked_tensor(
                            server_name,
                            client_id,
                        )
                        for client_id
                        in participating_ids
                    ]
                )

            naive_aggregate = (
                sum(
                    (
                        tensor.detach()
                        .double()
                        .cpu()
                        for tensor
                        in naive_server_aggregates.values()
                    ),
                    torch.zeros_like(
                        next(
                            iter(
                                naive_server_aggregates
                                .values()
                            )
                        ),
                        dtype=torch.float64,
                        device="cpu",
                    ),
                )
                / float(NUM_SERVERS)
            ).to(
                dtype=next(
                    iter(
                        naive_server_aggregates
                        .values()
                    )
                ).dtype
            )

            # -------------------------------------------------------------
            # B. Robust aggregate：
            #    每台服务器只聚合 retained clients。
            # -------------------------------------------------------------
            robust_server_aggregates: Dict[
                str, torch.Tensor
            ] = {}

            for server_name in SERVER_NAMES:
                robust_server_aggregates[
                    server_name
                ] = _mean_tensors(
                    [
                        get_masked_tensor(
                            server_name,
                            client_id,
                        )
                        for client_id
                        in retained_ids
                    ]
                )

            robust_aggregate = (
                sum(
                    (
                        tensor.detach()
                        .double()
                        .cpu()
                        for tensor
                        in robust_server_aggregates.values()
                    ),
                    torch.zeros_like(
                        next(
                            iter(
                                robust_server_aggregates
                                .values()
                            )
                        ),
                        dtype=torch.float64,
                        device="cpu",
                    ),
                )
                / float(NUM_SERVERS)
            ).to(
                dtype=next(
                    iter(
                        robust_server_aggregates
                        .values()
                    )
                ).dtype
            )

            # -------------------------------------------------------------
            # C. 一致性检查：
            #
            # 仅用于工程验证/可视化。
            # 根据：
            #   theta = (1/4) * sum_server omega_server
            # 先数学恢复 retained clients 的 theta，再直接平均，
            # 应与“各服务器先平均再 /4”的结果一致。
            #
            # 正式 robust_aggregate 使用的是上面的服务器聚合路径。
            # -------------------------------------------------------------
            recovered_retained_tensors: List[
                torch.Tensor
            ] = []

            for client_id in retained_ids:
                recovered_theta = (
                    sum(
                        (
                            get_masked_tensor(
                                server_name,
                                client_id,
                            )
                            .detach()
                            .double()
                            .cpu()
                            for server_name
                            in SERVER_NAMES
                        ),
                        torch.zeros_like(
                            get_masked_tensor(
                                "TS",
                                client_id,
                            ),
                            dtype=torch.float64,
                            device="cpu",
                        ),
                    )
                    / float(NUM_SERVERS)
                )

                recovered_retained_tensors.append(
                    recovered_theta
                )

            direct_reference = (
                _mean_tensors(
                    recovered_retained_tensors
                )
                .double()
                .cpu()
            )

            reconstruction_error = float(
                torch.max(
                    torch.abs(
                        robust_aggregate
                        .double()
                        .cpu()
                        - direct_reference
                    )
                ).item()
            )

            naive_parameters[
                layer_name
            ].append(
                naive_aggregate.clone()
            )

            robust_parameters[
                layer_name
            ].append(
                robust_aggregate.clone()
            )

            summaries.append(
                _build_summary(
                    layer_name=layer_name,
                    tensor_index=tensor_index,
                    participating_client_ids=(
                        participating_ids
                    ),
                    retained_client_ids=(
                        retained_ids
                    ),
                    rejected_client_ids=(
                        rejected_ids
                    ),
                    aggregation_mode=(
                        "FOUR_SERVER_MASKED_ROBUST_MEAN"
                    ),
                    naive_aggregate=(
                        naive_aggregate
                    ),
                    robust_aggregate=(
                        robust_aggregate
                    ),
                    preview_values_per_tensor=(
                        preview_values_per_tensor
                    ),
                    server_aggregate_previews={
                        server_name: (
                            _preview_values(
                                robust_server_aggregates[
                                    server_name
                                ],
                                preview_values_per_tensor,
                            )
                        )
                        for server_name
                        in SERVER_NAMES
                    },
                    reconstruction_consistency_error=(
                        reconstruction_error
                    ),
                )
            )

    return RobustAggregationResult(
        round_index=round_index,
        master_seed=master_seed,
        aggregation_mode=(
            "FOUR_SERVER_MASKED_ROBUST_MEAN"
        ),
        robust_aggregated_parameters=(
            robust_parameters
        ),
        naive_aggregated_parameters=(
            naive_parameters
        ),
        retained_clients=retained_clients,
        rejected_clients=rejected_clients,
        component_summaries=summaries,
    )


# =============================================================================
# 自动分支入口
# =============================================================================

def robust_aggregate_from_step06_step07(
    step06_result: Any,
    step07_result: Any,
    *,
    preview_values_per_tensor: int = (
        DEFAULT_PREVIEW_VALUES_PER_TENSOR
    ),
) -> RobustAggregationResult:
    """
    Step 08 推荐给 main.py 使用的统一入口。

    自动识别：
        - SP plaintext branch
        - Full four-server masking branch
    """

    has_masked = hasattr(
        step06_result,
        "masked_parameters_by_server",
    )

    has_plaintext = hasattr(
        step06_result,
        "ts_parameters",
    )

    if has_masked:
        return robust_aggregate_from_masking(
            masking_result=step06_result,
            detection_result=step07_result,
            preview_values_per_tensor=(
                preview_values_per_tensor
            ),
        )

    if has_plaintext:
        return robust_aggregate_from_sp(
            plain_uploading_result=step06_result,
            detection_result=step07_result,
            preview_values_per_tensor=(
                preview_values_per_tensor
            ),
        )

    raise TypeError(
        "无法识别 Step 06 输出类型："
        "既没有 masked_parameters_by_server，"
        "也没有 ts_parameters"
    )


# =============================================================================
# 输出打印（仅用于命令行测试/调试）
# =============================================================================

def print_aggregation_summary(
    result: RobustAggregationResult,
) -> None:
    print(
        "=" * 96
    )
    print(
        "Step 08 - Robust Layer-wise Aggregation"
    )
    print(
        "=" * 96
    )

    print(
        f"Round Index      : {result.round_index}"
    )
    print(
        f"Master Seed      : {result.master_seed}"
    )
    print(
        f"Aggregation Mode : {result.aggregation_mode}"
    )

    print()

    for summary in result.component_summaries:
        print(
            f"{summary.layer_name:<8} | "
            f"{summary.tensor_role:<6} | "
            f"participants={summary.participant_count:<3} | "
            f"retained={summary.retained_count:<3} | "
            f"rejected={summary.rejected_count:<3} | "
            f"delta_norm="
            f"{summary.aggregate_difference_norm:.8f}"
        )

        print(
            f"    retained ids : "
            f"{summary.retained_client_ids}"
        )

        print(
            f"    rejected ids : "
            f"{summary.rejected_client_ids}"
        )

        print(
            f"    naive preview : "
            f"{summary.naive_aggregate_preview}"
        )

        print(
            f"    robust preview: "
            f"{summary.robust_aggregate_preview}"
        )

        if (
            summary.aggregation_mode
            == "FOUR_SERVER_MASKED_ROBUST_MEAN"
        ):
            print(
                "    reconstruction consistency error: "
                f"{summary.reconstruction_consistency_error:.12e}"
            )


# =============================================================================
# 独立测试样例
# =============================================================================

if __name__ == "__main__":
    from dataclasses import dataclass

    print(
        "=" * 96
    )
    print(
        "Step 08 Standalone Test"
    )
    print(
        "=" * 96
    )

    # -------------------------------------------------------------------------
    # 构造 4 个客户端：
    # Client 3 的 weight 为明显异常值；
    # bias 则保留不同客户端集合，用来验证 weight/bias 独立聚合。
    # -------------------------------------------------------------------------

    client_parameters = {
        0: {
            "layer1": [
                torch.tensor(
                    [1.0, 2.0, 3.0],
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [0.10, 0.20],
                    dtype=torch.float32,
                ),
            ]
        },
        1: {
            "layer1": [
                torch.tensor(
                    [1.1, 2.1, 3.1],
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [0.11, 0.19],
                    dtype=torch.float32,
                ),
            ]
        },
        2: {
            "layer1": [
                torch.tensor(
                    [0.9, 1.9, 2.9],
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [0.09, 0.21],
                    dtype=torch.float32,
                ),
            ]
        },
        3: {
            "layer1": [
                torch.tensor(
                    [20.0, -20.0, 30.0],
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [0.12, 0.18],
                    dtype=torch.float32,
                ),
            ]
        },
    }

    @dataclass
    class _FakeSPUpload:
        round_index: int
        master_seed: int
        ts_parameters: Dict[
            int,
            Dict[str, List[torch.Tensor]],
        ]

    @dataclass
    class _FakeDetection:
        round_index: int
        master_seed: int
        retained_clients: Dict[
            str, Dict[int, List[int]]
        ]
        rejected_clients: Dict[
            str, Dict[int, List[int]]
        ]

    detection = _FakeDetection(
        round_index=3,
        master_seed=12345,
        retained_clients={
            "layer1": {
                0: [0, 1, 2],
                1: [0, 1, 3],
            }
        },
        rejected_clients={
            "layer1": {
                0: [3],
                1: [2],
            }
        },
    )

    # -------------------------------------------------------------------------
    # Test A: SP
    # -------------------------------------------------------------------------

    sp_upload = _FakeSPUpload(
        round_index=3,
        master_seed=12345,
        ts_parameters=client_parameters,
    )

    sp_result = (
        robust_aggregate_from_step06_step07(
            sp_upload,
            detection,
        )
    )

    print()
    print(
        "[A] SP / Plaintext Branch"
    )
    print_aggregation_summary(
        sp_result
    )

    expected_weight = torch.tensor(
        [1.0, 2.0, 3.0],
        dtype=torch.float32,
    )

    expected_bias = torch.mean(
        torch.stack(
            [
                client_parameters[0]["layer1"][1],
                client_parameters[1]["layer1"][1],
                client_parameters[3]["layer1"][1],
            ],
            dim=0,
        ),
        dim=0,
    )

    print()
    print(
        "SP weight robust aggregate correct:",
        torch.allclose(
            sp_result
            .robust_aggregated_parameters[
                "layer1"
            ][0],
            expected_weight,
            atol=1e-6,
        ),
    )

    print(
        "SP bias independent retained set correct:",
        torch.allclose(
            sp_result
            .robust_aggregated_parameters[
                "layer1"
            ][1],
            expected_bias,
            atol=1e-6,
        ),
    )

    # -------------------------------------------------------------------------
    # Test B: Full 4-server masking
    # -------------------------------------------------------------------------

    generator = torch.Generator(
        device="cpu"
    )
    generator.manual_seed(
        20260916
    )

    masked_by_server = {
        server_name: {}
        for server_name in SERVER_NAMES
    }

    for client_id, layer_map in (
        client_parameters.items()
    ):
        for server_name in SERVER_NAMES:
            masked_by_server[
                server_name
            ][
                client_id
            ] = {
                "layer1": []
            }

        for tensor in layer_map[
            "layer1"
        ]:
            r_ts = torch.randn(
                tensor.shape,
                generator=generator,
                dtype=tensor.dtype,
            )
            r_ds1 = torch.randn(
                tensor.shape,
                generator=generator,
                dtype=tensor.dtype,
            )
            r_ds2 = torch.randn(
                tensor.shape,
                generator=generator,
                dtype=tensor.dtype,
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

            for server_name in SERVER_NAMES:
                masked_by_server[
                    server_name
                ][
                    client_id
                ][
                    "layer1"
                ].append(
                    tensor
                    + masks[
                        server_name
                    ]
                )

    @dataclass
    class _FakeMaskingUpload:
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

    full_upload = _FakeMaskingUpload(
        round_index=3,
        master_seed=12345,
        masked_parameters_by_server=(
            masked_by_server
        ),
    )

    full_result = (
        robust_aggregate_from_step06_step07(
            full_upload,
            detection,
        )
    )

    print()
    print(
        "[B] Four-server Masking Branch"
    )
    print_aggregation_summary(
        full_result
    )

    print()
    print(
        "Full weight robust aggregate correct:",
        torch.allclose(
            full_result
            .robust_aggregated_parameters[
                "layer1"
            ][0],
            expected_weight,
            atol=1e-5,
        ),
    )

    print(
        "Full bias independent retained set correct:",
        torch.allclose(
            full_result
            .robust_aggregated_parameters[
                "layer1"
            ][1],
            expected_bias,
            atol=1e-5,
        ),
    )

    print(
        "SP / Full robust weight consistent:",
        torch.allclose(
            sp_result
            .robust_aggregated_parameters[
                "layer1"
            ][0],
            full_result
            .robust_aggregated_parameters[
                "layer1"
            ][0],
            atol=1e-5,
        ),
    )

    print(
        "Visualization variable exists:",
        bool(
            full_result
            .aggregation_visualization_data
        ),
    )

    print(
        "Visualization summary exists:",
        bool(
            full_result
            .aggregation_visualization_summary
        ),
    )

    max_reconstruction_error = max(
        summary.reconstruction_consistency_error
        for summary
        in full_result.component_summaries
    )

    print(
        "Max reconstruction consistency error:",
        max_reconstruction_error,
    )

    print()
    print(
        "提示：以上测试只会在直接运行本文件时执行；"
        "import 本模块不会执行测试。"
    )
