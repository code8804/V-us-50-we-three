"""
Step_07_V_sp_byzantine.py

============================================================
Step 07 特殊分支：明文参数下的层级 Byzantine 检测（可视化相关）
============================================================

【本步骤对应的上游】
    Step_06_V_sp_uploading.py

Step 06 SP 分支没有执行随机掩码、秘密共享或加密：
    Client shared parameter theta  --->  TS

因此 Step 07 SP 分支中，TS 可以直接读取每个客户端上传的完整共享模型参数。

这带来两个同时存在的效果：

1. Byzantine 检测非常直接：
   TS 可以直接对明文模型参数计算中位数、偏差，并选择最接近中位数的 tau 比例参数。

2. 隐私泄露风险非常直观：
   TS 持有并能够查看每个客户端的原始共享模型参数。
   本文件特意保留 plaintext_exposure_records / privacy_visualization_data，
   供前端展示“服务器能够看到客户端原始参数”这一隐私问题。

============================================================
【检测粒度：Layer-wise + Intra-layer Fine-grained Filtering】
============================================================

整体方法仍然以“层”为组织单位：

    Layer 1
    Layer 2
    ...

但在每一层内部，weight 与 bias 分开执行检测：

    layer2 / weight -> 独立计算中位数、独立计算距离、独立选择 tau
    layer2 / bias   -> 独立计算中位数、独立计算距离、独立选择 tau

因此允许出现：

    Client 3 / layer2 / weight -> rejected
    Client 3 / layer2 / bias   -> retained

这样不会因为某层 weight 异常，就连该层仍然正常的 bias 一起丢弃。

============================================================
【核心算法】
============================================================

对于某个共享层 l 的某个参数组件 p（例如 weight 或 bias），
收集所有“拥有该共享层”的客户端明文参数：

    theta_1, theta_2, ..., theta_n

第一步：逐元素计算中位数参数：

    median_theta = elementwise_median(theta_1, ..., theta_n)

第二步：对每个客户端计算其参数与中位数参数的平方 L2 偏差：

    score_k = ||theta_k - median_theta||_2^2

第三步：按照 score 从小到大排序。

第四步：保留 tau 比例、最接近中位数的客户端参数：

    retain_count = ceil(tau * n)

并至少保留 1 个。

注意：
- weight 与 bias 的 score 不混用；
- 不同 layer 的 score 不混用；
- 不同形状的参数不会放在一起比较；
- 相同 score 时按 client_id 升序稳定打破平局，不使用随机数。

============================================================
【关于 Seed】
============================================================

本步骤的检测算法本身是确定性的：
    median
    squared L2 distance
    sorting
都不需要随机采样。

因此 master_seed 仅作为“完整实验上下文”保存在输出结果中，
方便 main.py 统一追踪实验种子，但不会人为加入随机 tie-break。

同样输入 + 同样 tau，无论 seed 是多少，检测结果都应完全一致。

这比为了“用上 seed”而强行加入随机性更合理，也更利于复现实验。

============================================================
【本步骤不负责】
============================================================

- 不重新实施 Gaussian / Sign Flipping；
- 不重新决定 Byzantine Client；
- 不执行随机掩码；
- 不执行加密；
- 不执行秘密共享；
- 不执行最终模型聚合；
- 不修改 Step 06 的上传结果；
- 不根据 Step 00 的真实攻击标签“作弊式”判断异常。

真实攻击标签只能在后续评估 detection accuracy 时使用，
不能进入本步骤的中位数检测逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch


TS_NAME = "TS"
ALL_LAYER_NAMES = ("layer1", "layer2", "layer3", "layer4", "layer5")
DEFAULT_TAU = 0.7
DEFAULT_PREVIEW_VALUES_PER_TENSOR = 5


@dataclass(frozen=True)
class PlaintextExposureRecord:
    """
    用于可视化“TS 能直接查看客户端明文模型参数”。

    complete plaintext Tensor 不塞进 record，完整 Tensor 已经存在于：
        result.plaintext_parameters

    这里保留 shape / preview / norm 等页面展示信息。
    """

    client_id: int
    layer_name: str
    tensor_index: int
    tensor_role: str
    shape: Tuple[int, ...]
    plaintext_preview: List[float]
    plaintext_norm: float
    exposure_message: str


@dataclass(frozen=True)
class ByzantineTensorDecision:
    """
    一个 Client / Layer / Tensor component 的检测结果。
    """

    client_id: int
    layer_name: str
    tensor_index: int
    tensor_role: str

    score: float
    rank: int
    retained: bool

    parameter_preview: List[float]
    median_preview: List[float]


@dataclass(frozen=True)
class TensorComponentDetectionSummary:
    """
    某一 Layer 内某个参数组件（weight / bias）的整体检测摘要。
    """

    layer_name: str
    tensor_index: int
    tensor_role: str
    participating_client_ids: List[int]

    tau: float
    retain_count: int

    retained_client_ids: List[int]
    rejected_client_ids: List[int]

    median_shape: Tuple[int, ...]
    median_preview: List[float]

    scores_by_client: Dict[int, float]


@dataclass
class PlainByzantineDetectionResult:
    """
    Step 07 SP 明文 Byzantine 检测统一输出。
    """

    round_index: int
    master_seed: int
    tau: float

    # TS 可以直接看到的完整明文共享模型参数。
    # 这是本 SP 分支最关键的“隐私泄露”证据变量。
    plaintext_parameters: Dict[
        int, Dict[str, List[torch.Tensor]]
    ]

    # 每层每个组件的 element-wise median Tensor。
    median_parameters: Dict[
        str, Dict[int, torch.Tensor]
    ]

    # Layer -> tensor_index -> retained client ids
    retained_clients: Dict[
        str, Dict[int, List[int]]
    ]

    # Layer -> tensor_index -> rejected client ids
    rejected_clients: Dict[
        str, Dict[int, List[int]]
    ]

    component_summaries: List[TensorComponentDetectionSummary]
    decisions: List[ByzantineTensorDecision]
    plaintext_exposure_records: List[PlaintextExposureRecord]

    @property
    def privacy_leakage_present(self) -> bool:
        """
        SP 直接上传场景下恒为 True：
        TS 能直接访问每个客户端完整的共享模型参数。
        """
        return True

    @property
    def visualization_data(self) -> List[Dict[str, Any]]:
        """
        Byzantine 检测页面使用的数据。
        """
        rows: List[Dict[str, Any]] = []

        for d in self.decisions:
            rows.append(
                {
                    "client_id": d.client_id,
                    "layer_name": d.layer_name,
                    "tensor_index": d.tensor_index,
                    "tensor_role": d.tensor_role,
                    "score": d.score,
                    "rank": d.rank,
                    "retained": d.retained,
                    "decision": "retained" if d.retained else "rejected",
                    "parameter_preview": d.parameter_preview.copy(),
                    "median_preview": d.median_preview.copy(),
                    "tau": self.tau,
                }
            )

        return rows

    @property
    def privacy_visualization_data(self) -> List[Dict[str, Any]]:
        """
        隐私风险页面使用的数据。

        页面可以明确展示：
            TS -> Client k -> Layer l -> weight/bias -> plaintext preview

        从而说明无隐私保护时，服务器可以查看客户端原始模型参数。
        """
        return [
            {
                "server": TS_NAME,
                "client_id": r.client_id,
                "layer_name": r.layer_name,
                "tensor_index": r.tensor_index,
                "tensor_role": r.tensor_role,
                "shape": list(r.shape),
                "plaintext_preview": r.plaintext_preview.copy(),
                "plaintext_norm": r.plaintext_norm,
                "is_plaintext_visible_to_server": True,
                "privacy_status": "EXPOSED",
                "message": r.exposure_message,
            }
            for r in self.plaintext_exposure_records
        ]


def _clone_parameter_structure(
    parameters: Mapping[
        int, Mapping[str, Sequence[torch.Tensor]]
    ],
) -> Dict[int, Dict[str, List[torch.Tensor]]]:
    return {
        int(client_id): {
            str(layer_name): [
                tensor.detach().cpu().clone()
                for tensor in tensors
            ]
            for layer_name, tensors in layer_map.items()
        }
        for client_id, layer_map in parameters.items()
    }


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
        .float()
        .reshape(-1)
        .cpu()
    )

    return [
        float(value)
        for value in flat[:count].tolist()
    ]


def _tensor_norm(
    tensor: torch.Tensor,
) -> float:
    flat = tensor.detach().float().reshape(-1)

    if flat.numel() == 0:
        return 0.0

    return float(
        torch.linalg.vector_norm(flat).item()
    )


def _validate_tau(tau: float) -> None:
    if not (0.0 < float(tau) <= 1.0):
        raise ValueError(
            f"tau 必须满足 0 < tau <= 1，当前 tau={tau}"
        )


def _validate_plaintext_parameters(
    parameters: Mapping[
        int, Mapping[str, Sequence[torch.Tensor]]
    ],
) -> None:
    if not parameters:
        raise ValueError("plaintext parameters 不能为空")

    for client_id, layer_map in parameters.items():
        if int(client_id) < 0:
            raise ValueError(
                f"client_id 不能为负数：{client_id}"
            )

        if not layer_map:
            raise ValueError(
                f"Client {client_id} 没有 Shared Parameters"
            )

        for layer_name, tensors in layer_map.items():
            if str(layer_name) not in ALL_LAYER_NAMES:
                raise ValueError(
                    f"非法层名称：{layer_name!r}"
                )

            if not tensors:
                raise ValueError(
                    f"Client {client_id} / {layer_name} "
                    "没有参数 Tensor"
                )

            for tensor_index, tensor in enumerate(tensors):
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(
                        f"Client {client_id} / {layer_name} / "
                        f"tensor {tensor_index} 不是 torch.Tensor"
                    )

                if not tensor.is_floating_point():
                    raise TypeError(
                        "模型参数必须为浮点 Tensor，"
                        f"当前 dtype={tensor.dtype}"
                    )


def _collect_layer_tensor_groups(
    plaintext_parameters: Mapping[
        int, Mapping[str, Sequence[torch.Tensor]]
    ],
) -> Dict[
    str,
    Dict[
        int,
        List[Tuple[int, torch.Tensor, str]],
    ],
]:
    """
    按：
        Layer -> tensor_index(weight/bias) -> clients
    重新组织数据。

    自适应共享架构下，不同客户端拥有的 shared layers 数量可能不同，
    因此某层只比较真正拥有并上传该层的客户端。
    """

    groups: Dict[
        str,
        Dict[
            int,
            List[Tuple[int, torch.Tensor, str]],
        ],
    ] = {}

    for client_id in sorted(plaintext_parameters):
        layer_map = plaintext_parameters[client_id]

        for layer_name, tensors in layer_map.items():
            groups.setdefault(layer_name, {})

            for tensor_index, tensor in enumerate(tensors):
                role = _tensor_role(
                    tensor_index=tensor_index,
                    tensor_count=len(tensors),
                )

                groups[layer_name].setdefault(
                    tensor_index,
                    [],
                ).append(
                    (
                        int(client_id),
                        tensor.detach().cpu().clone(),
                        role,
                    )
                )

    return groups


def _elementwise_median(
    client_tensors: Sequence[Tuple[int, torch.Tensor, str]],
) -> torch.Tensor:
    """
    对同一 Layer / 同一参数组件的多个客户端 Tensor 逐元素取中位数。

    torch.median 在偶数样本时返回两个中间值中的较小者。
    为得到通常统计意义上的中位数，这里使用 quantile(0.5, interpolation='midpoint')，
    即偶数客户端时取两个中间值的平均。
    """

    if not client_tensors:
        raise ValueError("client_tensors 不能为空")

    reference_shape = tuple(client_tensors[0][1].shape)

    for client_id, tensor, _ in client_tensors:
        if tuple(tensor.shape) != reference_shape:
            raise ValueError(
                "同一 Layer / Tensor component 的参数形状必须一致；"
                f"Client {client_id} shape={tuple(tensor.shape)}, "
                f"reference={reference_shape}"
            )

    stacked = torch.stack(
        [
            tensor.detach().float().cpu()
            for _, tensor, _ in client_tensors
        ],
        dim=0,
    )

    return torch.quantile(
        stacked,
        q=0.5,
        dim=0,
        interpolation="midpoint",
    )


def _squared_l2_distance(
    tensor: torch.Tensor,
    median_tensor: torch.Tensor,
) -> float:
    """
    score = ||theta - median_theta||_2^2
    """

    difference = (
        tensor.detach().float().cpu()
        - median_tensor.detach().float().cpu()
    )

    return float(
        torch.sum(
            difference * difference
        ).item()
    )


def detect_plaintext_byzantine_parameters(
    plaintext_parameters: Mapping[
        int, Mapping[str, Sequence[torch.Tensor]]
    ],
    round_index: int,
    master_seed: int,
    tau: float = DEFAULT_TAU,
    preview_values_per_tensor: int = (
        DEFAULT_PREVIEW_VALUES_PER_TENSOR
    ),
) -> PlainByzantineDetectionResult:
    """
    Step 07 SP 的核心算法。

    重要：
    - 输入是 TS 已经能够直接读取的客户端明文共享模型参数；
    - 以 Layer 为总体组织单位；
    - Layer 内 weight / bias 独立计算 median / score / retained set；
    - 不使用真实 Byzantine 标签；
    - 不使用随机检测。
    """

    _validate_tau(tau)
    _validate_plaintext_parameters(
        plaintext_parameters
    )

    if int(round_index) < 0:
        raise ValueError(
            "round_index 必须 >= 0"
        )

    if int(preview_values_per_tensor) < 0:
        raise ValueError(
            "preview_values_per_tensor 必须 >= 0"
        )

    plaintext = _clone_parameter_structure(
        plaintext_parameters
    )

    groups = _collect_layer_tensor_groups(
        plaintext
    )

    median_parameters: Dict[
        str, Dict[int, torch.Tensor]
    ] = {}

    retained_clients: Dict[
        str, Dict[int, List[int]]
    ] = {}

    rejected_clients: Dict[
        str, Dict[int, List[int]]
    ] = {}

    component_summaries: List[
        TensorComponentDetectionSummary
    ] = []

    decisions: List[
        ByzantineTensorDecision
    ] = []

    # ========================================================
    # Layer -> Tensor component(weight / bias)
    # ========================================================
    for layer_name in ALL_LAYER_NAMES:
        if layer_name not in groups:
            continue

        median_parameters[layer_name] = {}
        retained_clients[layer_name] = {}
        rejected_clients[layer_name] = {}

        for tensor_index in sorted(groups[layer_name]):
            client_tensors = groups[
                layer_name
            ][
                tensor_index
            ]

            if not client_tensors:
                continue

            tensor_role = client_tensors[0][2]

            median_tensor = _elementwise_median(
                client_tensors
            )

            median_parameters[
                layer_name
            ][
                tensor_index
            ] = median_tensor.clone()

            scored_clients: List[
                Tuple[float, int, torch.Tensor]
            ] = []

            for client_id, tensor, _ in client_tensors:
                score = _squared_l2_distance(
                    tensor=tensor,
                    median_tensor=median_tensor,
                )

                scored_clients.append(
                    (
                        score,
                        int(client_id),
                        tensor,
                    )
                )

            # 确定性排序：
            # 先按 deviation score，完全相同时按 client_id。
            scored_clients.sort(
                key=lambda item: (
                    item[0],
                    item[1],
                )
            )

            participant_count = len(
                scored_clients
            )

            retain_count = max(
                1,
                min(
                    participant_count,
                    int(
                        ceil(
                            float(tau)
                            * participant_count
                        )
                    ),
                ),
            )

            retained_set = {
                client_id
                for _, client_id, _
                in scored_clients[:retain_count]
            }

            retained_ids = [
                client_id
                for _, client_id, _
                in scored_clients
                if client_id in retained_set
            ]

            rejected_ids = [
                client_id
                for _, client_id, _
                in scored_clients
                if client_id not in retained_set
            ]

            retained_clients[
                layer_name
            ][
                tensor_index
            ] = retained_ids.copy()

            rejected_clients[
                layer_name
            ][
                tensor_index
            ] = rejected_ids.copy()

            scores_by_client: Dict[
                int, float
            ] = {}

            median_preview = _preview_values(
                median_tensor,
                preview_values_per_tensor,
            )

            for rank_index, (
                score,
                client_id,
                tensor,
            ) in enumerate(
                scored_clients,
                start=1,
            ):
                scores_by_client[
                    client_id
                ] = float(score)

                decisions.append(
                    ByzantineTensorDecision(
                        client_id=client_id,
                        layer_name=layer_name,
                        tensor_index=tensor_index,
                        tensor_role=tensor_role,
                        score=float(score),
                        rank=rank_index,
                        retained=(
                            client_id in retained_set
                        ),
                        parameter_preview=(
                            _preview_values(
                                tensor,
                                preview_values_per_tensor,
                            )
                        ),
                        median_preview=(
                            median_preview.copy()
                        ),
                    )
                )

            component_summaries.append(
                TensorComponentDetectionSummary(
                    layer_name=layer_name,
                    tensor_index=tensor_index,
                    tensor_role=tensor_role,
                    participating_client_ids=[
                        client_id
                        for _, client_id, _
                        in scored_clients
                    ],
                    tau=float(tau),
                    retain_count=retain_count,
                    retained_client_ids=(
                        retained_ids.copy()
                    ),
                    rejected_client_ids=(
                        rejected_ids.copy()
                    ),
                    median_shape=tuple(
                        median_tensor.shape
                    ),
                    median_preview=(
                        median_preview.copy()
                    ),
                    scores_by_client=(
                        dict(scores_by_client)
                    ),
                )
            )

    # ========================================================
    # 明文泄露可视化记录
    # ========================================================
    exposure_records: List[
        PlaintextExposureRecord
    ] = []

    for client_id in sorted(plaintext):
        for layer_name, tensors in (
            plaintext[client_id].items()
        ):
            for tensor_index, tensor in enumerate(
                tensors
            ):
                tensor_role = _tensor_role(
                    tensor_index,
                    len(tensors),
                )

                exposure_records.append(
                    PlaintextExposureRecord(
                        client_id=client_id,
                        layer_name=layer_name,
                        tensor_index=tensor_index,
                        tensor_role=tensor_role,
                        shape=tuple(tensor.shape),
                        plaintext_preview=(
                            _preview_values(
                                tensor,
                                preview_values_per_tensor,
                            )
                        ),
                        plaintext_norm=(
                            _tensor_norm(tensor)
                        ),
                        exposure_message=(
                            "TS can directly inspect the "
                            "client's plaintext shared-model "
                            "parameter."
                        ),
                    )
                )

    return PlainByzantineDetectionResult(
        round_index=int(round_index),
        master_seed=int(master_seed),
        tau=float(tau),
        plaintext_parameters=plaintext,
        median_parameters=median_parameters,
        retained_clients=retained_clients,
        rejected_clients=rejected_clients,
        component_summaries=component_summaries,
        decisions=decisions,
        plaintext_exposure_records=(
            exposure_records
        ),
    )


def detect_from_step06_sp(
    plain_uploading_result: Any,
    master_seed: int,
    tau: float = DEFAULT_TAU,
    preview_values_per_tensor: int = (
        DEFAULT_PREVIEW_VALUES_PER_TENSOR
    ),
) -> PlainByzantineDetectionResult:
    """
    正式接口：
        Step_06_V_sp_uploading.py
            -> Step_07_V_sp_byzantine.py

    优先读取：
        plain_uploading_result.ts_parameters

    因为 SP 上传方案中 TS 持有完整明文参数。
    """

    if not hasattr(
        plain_uploading_result,
        "ts_parameters",
    ):
        raise TypeError(
            "plain_uploading_result 缺少 "
            "ts_parameters；请确认输入来自 "
            "Step_06_V_sp_uploading.py"
        )

    if not hasattr(
        plain_uploading_result,
        "round_index",
    ):
        raise TypeError(
            "plain_uploading_result 缺少 round_index"
        )

    return detect_plaintext_byzantine_parameters(
        plaintext_parameters=(
            plain_uploading_result.ts_parameters
        ),
        round_index=int(
            plain_uploading_result.round_index
        ),
        master_seed=int(master_seed),
        tau=float(tau),
        preview_values_per_tensor=int(
            preview_values_per_tensor
        ),
    )


def print_plain_byzantine_result(
    result: PlainByzantineDetectionResult,
) -> None:
    """
    控制台调试输出。
    """

    print("\n" + "=" * 92)
    print(
        "Step 07 SP - Plaintext Layer-wise Byzantine Detection"
    )
    print("=" * 92)

    print(f"Round Index : {result.round_index}")
    print(f"Master Seed : {result.master_seed}")
    print(f"Tau         : {result.tau}")
    print(
        "Privacy     : EXPOSED "
        "(TS can inspect plaintext client parameters)"
    )

    for summary in result.component_summaries:
        print("\n" + "-" * 92)
        print(
            f"{summary.layer_name} / "
            f"{summary.tensor_role}"
        )
        print(
            f"Participants : "
            f"{summary.participating_client_ids}"
        )
        print(
            f"Retain Count : "
            f"{summary.retain_count}"
        )
        print(
            f"Retained     : "
            f"{summary.retained_client_ids}"
        )
        print(
            f"Rejected     : "
            f"{summary.rejected_client_ids}"
        )
        print(
            f"Median Prev. : "
            f"{summary.median_preview}"
        )
        print("Scores:")
        for client_id, score in sorted(
            summary.scores_by_client.items()
        ):
            print(
                f"  Client {client_id:02d}: "
                f"{score:.8f}"
            )


if __name__ == "__main__":
    # ============================================================
    # 独立测试样例
    #
    # 目标：
    # 1. 验证 tau=0.7 时选择最接近 median 的客户端；
    # 2. 验证 weight / bias 独立检测；
    # 3. 验证同一层可以出现：
    #       某客户端 weight 被拒绝，但 bias 被保留；
    # 4. 验证 TS 明文泄露变量存在；
    # 5. 验证算法与 seed 无关，保持确定性。
    # ============================================================

    # 这里构造 10 个客户端。
    # tau=0.7 -> ceil(10 * 0.7) = 7 个被保留。
    #
    # Client 9:
    #   layer1 weight = 极端异常
    #   layer1 bias   = 正常
    #
    # 因此希望看到：
    #   Client 9 / layer1 / weight -> rejected
    #   Client 9 / layer1 / bias   -> retained
    #
    # 这正是“层级检测 + 层内细粒度过滤”的示例。

    generator = torch.Generator(
        device="cpu"
    )
    generator.manual_seed(2026)

    fake_plaintext_parameters: Dict[
        int,
        Dict[str, List[torch.Tensor]],
    ] = {}

    for client_id in range(10):
        # 正常 weight 以 1.0 附近为中心。
        weight = (
            torch.ones(2, 2)
            + torch.randn(
                2,
                2,
                generator=generator,
            ) * 0.03
        )

        # 正常 bias 以 0.1 附近为中心。
        bias = (
            torch.full((2,), 0.1)
            + torch.randn(
                2,
                generator=generator,
            ) * 0.01
        )

        fake_plaintext_parameters[
            client_id
        ] = {
            "layer1": [
                weight,
                bias,
            ]
        }

    # Client 9 只攻击 weight。
    # bias 保持其原先正常值。
    fake_plaintext_parameters[
        9
    ][
        "layer1"
    ][
        0
    ] = torch.full(
        (2, 2),
        50.0,
    )

    result = (
        detect_plaintext_byzantine_parameters(
            plaintext_parameters=(
                fake_plaintext_parameters
            ),
            round_index=1,
            master_seed=42,
            tau=0.7,
            preview_values_per_tensor=4,
        )
    )

    print_plain_byzantine_result(
        result
    )

    print("\n" + "=" * 92)
    print("Automatic Checks")
    print("=" * 92)

    weight_retained = set(
        result.retained_clients[
            "layer1"
        ][
            0
        ]
    )

    bias_retained = set(
        result.retained_clients[
            "layer1"
        ][
            1
        ]
    )

    print(
        "tau=0.7 / 10 clients -> retain 7 :",
        len(weight_retained) == 7
        and len(bias_retained) == 7,
    )

    print(
        "Client 9 abnormal weight rejected :",
        9 not in weight_retained,
    )

    print(
        "Client 9 normal bias can be retained:",
        9 in bias_retained,
    )

    print(
        "Plaintext privacy exposure recorded :",
        result.privacy_leakage_present
        and len(
            result.privacy_visualization_data
        ) > 0
        and all(
            row[
                "is_plaintext_visible_to_server"
            ]
            for row in (
                result.privacy_visualization_data
            )
        ),
    )

    # ------------------------------------------------------------
    # Seed consistency check
    # ------------------------------------------------------------
    # 本算法不使用随机检测，因此换 seed 后结果也必须一致。
    result_other_seed = (
        detect_plaintext_byzantine_parameters(
            plaintext_parameters=(
                fake_plaintext_parameters
            ),
            round_index=1,
            master_seed=999999,
            tau=0.7,
            preview_values_per_tensor=4,
        )
    )

    same_decisions = (
        result.retained_clients
        == result_other_seed.retained_clients
        and result.rejected_clients
        == result_other_seed.rejected_clients
    )

    print(
        "Detection is deterministic across seeds:",
        same_decisions,
    )
