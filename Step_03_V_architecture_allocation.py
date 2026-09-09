"""
Step_03_V_architecture_allocation.py

============================================================
Step 03：基于客户端本地数据量的自适应共享架构分配
============================================================

【本文件职责】

本步骤接收 Step 01 数据分发阶段产生的“各客户端本地数据量”，
根据 FedCALI 的自适应共享架构分配思想，为每个客户端确定：

1. 本地数据量 N_k；
2. 共享层数 m_k；
3. 个性化层数 L - m_k；
4. 哪些层属于共享层；
5. 哪些层属于个性化层。

本项目当前固定采用 5 层模型：

    Layer 1
    Layer 2
    Layer 3
    Layer 4
    Layer 5

其中：
- 总模型层数 L = 5；
- 最大共享层数为 L - 1 = 4；
- 因此共享层数只能取 1、2、3、4；
- 至少保留 1 个个性化层；
- 前 m_k 层为共享层；
- 剩余 L - m_k 层为个性化层。

例如：

    m_k = 3

则：

    Shared Layers       = [1, 2, 3]
    Personalized Layers = [4, 5]

------------------------------------------------------------
【为什么数据越少，共享层越多】

FedCALI 的设计思想是：

- 本地数据量较少的客户端，本地知识不足，
  因而需要更多依赖其他客户端提供的全局知识，
  所以分配更多共享层；

- 本地数据量较多的客户端，有更充分的本地数据支撑，
  因而可以保留更多个性化结构，
  所以分配更少共享层。

因此，共享层数量与客户端本地数据量呈反向关系。

------------------------------------------------------------
【共享层数计算】

设：

    N_k   = Client k 的本地数据量
    N_max = 所有客户端中的最大本地数据量
    N_min = 所有客户端中的最小本地数据量
    L     = 总模型层数，当前固定为 5

首先计算对数归一化比例：

                  log(N_max) - log(N_k)
    ratio_k = --------------------------------
                log(N_max) - log(N_min)

然后映射到可共享的 L - 1 个层：

    m_k = max(
        1,
        ceil(ratio_k * (L - 1))
    )

并最终限制在：

    1 <= m_k <= L - 1

当前 L = 5，因此：

    1 <= m_k <= 4

特别地：

- N_k = N_min 时，ratio = 1，得到 4 个共享层；
- N_k = N_max 时，ratio = 0，经至少共享 1 层约束后得到 1 个共享层。

------------------------------------------------------------
【本步骤为什么带 V】

文件名中的 V 表示：

    本步骤的输出结果需要直接提供给可视化模块展示。

前端/可视化模块至少应能直接获得并展示：

    Client 0
    Local Data Size      : 850
    Shared Layer Count   : 4
    Personalized Count   : 1
    Shared Layers        : [1, 2, 3, 4]
    Personalized Layers  : [5]

因此本文件不会只返回一个内部使用的 m_k，
而是显式返回结构化 ArchitectureAllocationResult，
方便后续：

1. 前端直接可视化；
2. Step 04 Local Training 确定共享/个性化层；
3. Gaussian / Sign Flipping 根据客户端实际共享层解析攻击层；
4. 后续 Masking / Detection / Aggregation 知道每个客户端参与哪些共享层。

------------------------------------------------------------
【与 Step 01 的接口】

Step 01 当前输出：

    data_result.client_sizes

其中 client_sizes 表示每个客户端持有的本地总数据量
（Train + Local Test）。

论文中的 N_k = |D_k| 表示客户端本地数据量，因此本步骤正式流程
默认使用 client_sizes，而不是只使用 client_train_sizes。

正式完整流程：

    Step 01:
        data_result = distributor.distribute()

    Step 03:
        architecture_result = allocate_architecture_from_data_result(
            data_result
        )

同时，为了支持机制演示，本文件也允许直接手工输入：

    client_sizes = {
        0: 500,
        1: 1000,
        2: 3000,
        3: 8000,
    }

两种入口最终调用完全相同的核心算法。
============================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log
from typing import Any, Dict, List, Mapping, Sequence, Union


TOTAL_MODEL_LAYERS = 5
MIN_SHARED_LAYERS = 1
MAX_SHARED_LAYERS = TOTAL_MODEL_LAYERS - 1


@dataclass(frozen=True)
class ClientArchitectureAllocation:
    client_id: int
    local_data_size: int
    shared_layer_count: int
    personalized_layer_count: int
    shared_layers: List[int]
    personalized_layers: List[int]
    normalized_ratio: float


@dataclass(frozen=True)
class ArchitectureAllocationResult:
    total_model_layers: int
    min_shared_layers: int
    max_shared_layers: int
    n_min: int
    n_max: int

    client_sizes: Dict[int, int]
    shared_layers: Dict[int, int]
    personalized_layers: Dict[int, int]
    client_allocations: Dict[int, ClientArchitectureAllocation]

    @property
    def visualization_data(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        for client_id in sorted(self.client_allocations):
            item = self.client_allocations[client_id]

            rows.append(
                {
                    "client_id": item.client_id,
                    "local_data_size": item.local_data_size,
                    "shared_layer_count": item.shared_layer_count,
                    "personalized_layer_count": item.personalized_layer_count,
                    "shared_layers": item.shared_layers.copy(),
                    "personalized_layers": item.personalized_layers.copy(),
                    "normalized_ratio": item.normalized_ratio,
                }
            )

        return rows

    def get(self, client_id: int) -> ClientArchitectureAllocation:
        if client_id not in self.client_allocations:
            raise ValueError(f"客户端 {client_id} 不存在")

        return self.client_allocations[client_id]


ClientSizesInput = Union[
    Mapping[int, int],
    Sequence[int],
]


def _normalize_client_sizes(
    client_sizes: ClientSizesInput,
) -> Dict[int, int]:
    if isinstance(client_sizes, Mapping):
        normalized = {
            int(client_id): int(size)
            for client_id, size in client_sizes.items()
        }
    else:
        normalized = {
            client_id: int(size)
            for client_id, size in enumerate(client_sizes)
        }

    if not normalized:
        raise ValueError("client_sizes 不能为空")

    for client_id, size in normalized.items():
        if client_id < 0:
            raise ValueError(
                f"client_id 不能为负数：{client_id}"
            )

        if size <= 0:
            raise ValueError(
                f"Client {client_id} 的本地数据量必须大于 0，"
                f"当前为 {size}"
            )

    return dict(sorted(normalized.items()))


def calculate_shared_layer_count(
    local_data_size: int,
    n_min: int,
    n_max: int,
    total_model_layers: int = TOTAL_MODEL_LAYERS,
) -> tuple[int, float]:
    """
    计算单个客户端共享层数。

    正常公式：

                     log(N_max) - log(N_k)
        ratio_k = --------------------------------
                    log(N_max) - log(N_min)

        m_k = max(1, ceil(ratio_k * (L - 1)))

    如果 N_max == N_min，说明所有客户端数据量完全一致，
    原公式分母为 0。此时统一采用 ratio = 0.5，
    对当前 5 层模型得到 2 个共享层。
    """

    if total_model_layers < 2:
        raise ValueError(
            "total_model_layers 至少为 2"
        )

    if local_data_size <= 0 or n_min <= 0 or n_max <= 0:
        raise ValueError("所有数据量必须大于 0")

    if not (n_min <= local_data_size <= n_max):
        raise ValueError(
            "local_data_size 必须位于 [n_min, n_max] 范围内"
        )

    max_shared_layers = total_model_layers - 1

    if n_max == n_min:
        ratio = 0.5
    else:
        denominator = log(n_max) - log(n_min)

        ratio = (
            log(n_max) - log(local_data_size)
        ) / denominator

    ratio = min(max(float(ratio), 0.0), 1.0)

    shared_count = max(
        MIN_SHARED_LAYERS,
        ceil(ratio * max_shared_layers),
    )

    shared_count = min(
        shared_count,
        max_shared_layers,
    )

    return int(shared_count), ratio


def allocate_architecture(
    client_sizes: ClientSizesInput,
    total_model_layers: int = TOTAL_MODEL_LAYERS,
) -> ArchitectureAllocationResult:
    """
    根据所有客户端的数据量统一分配共享/个性化架构。
    """

    if total_model_layers != 5:
        raise ValueError(
            "当前项目架构固定为 5 层模型，"
            "Step 03 暂不允许修改 total_model_layers。"
        )

    normalized_sizes = _normalize_client_sizes(client_sizes)

    n_min = min(normalized_sizes.values())
    n_max = max(normalized_sizes.values())

    shared_layers: Dict[int, int] = {}
    personalized_layers: Dict[int, int] = {}
    client_allocations: Dict[
        int,
        ClientArchitectureAllocation,
    ] = {}

    for client_id, local_data_size in normalized_sizes.items():
        shared_count, ratio = calculate_shared_layer_count(
            local_data_size=local_data_size,
            n_min=n_min,
            n_max=n_max,
            total_model_layers=total_model_layers,
        )

        personalized_count = (
            total_model_layers - shared_count
        )

        shared_layer_indices = list(
            range(1, shared_count + 1)
        )

        personalized_layer_indices = list(
            range(
                shared_count + 1,
                total_model_layers + 1,
            )
        )

        shared_layers[client_id] = shared_count
        personalized_layers[client_id] = personalized_count

        client_allocations[client_id] = (
            ClientArchitectureAllocation(
                client_id=client_id,
                local_data_size=local_data_size,
                shared_layer_count=shared_count,
                personalized_layer_count=personalized_count,
                shared_layers=shared_layer_indices,
                personalized_layers=personalized_layer_indices,
                normalized_ratio=ratio,
            )
        )

    return ArchitectureAllocationResult(
        total_model_layers=total_model_layers,
        min_shared_layers=MIN_SHARED_LAYERS,
        max_shared_layers=total_model_layers - 1,
        n_min=n_min,
        n_max=n_max,
        client_sizes=normalized_sizes,
        shared_layers=shared_layers,
        personalized_layers=personalized_layers,
        client_allocations=client_allocations,
    )


def allocate_architecture_from_data_result(
    data_result: Any,
) -> ArchitectureAllocationResult:
    """
    直接接收 Step 01 的 DataDistributionResult。

    只要求传入对象具有：
        data_result.client_sizes
    """

    if not hasattr(data_result, "client_sizes"):
        raise TypeError(
            "data_result 必须具有 client_sizes 属性；"
            "请传入 Step 01 的 DataDistributionResult"
        )

    return allocate_architecture(
        client_sizes=data_result.client_sizes,
        total_model_layers=TOTAL_MODEL_LAYERS,
    )


def print_architecture_allocation_result(
    result: ArchitectureAllocationResult,
) -> None:
    print("\n" + "=" * 78)
    print("Step 03 - Adaptive Architecture Allocation")
    print("=" * 78)

    print(
        f"Total Model Layers : {result.total_model_layers}"
    )
    print(
        f"Shared Layer Range : "
        f"{result.min_shared_layers} - "
        f"{result.max_shared_layers}"
    )
    print(f"N_min              : {result.n_min}")
    print(f"N_max              : {result.n_max}")

    for client_id in sorted(result.client_allocations):
        item = result.client_allocations[client_id]

        print("\n" + "-" * 78)
        print(f"Client {client_id}")
        print("-" * 78)

        print(
            f"Local Data Size       : "
            f"{item.local_data_size}"
        )
        print(
            f"Normalized Ratio      : "
            f"{item.normalized_ratio:.6f}"
        )
        print(
            f"Shared Layer Count    : "
            f"{item.shared_layer_count}"
        )
        print(
            f"Personalized Count    : "
            f"{item.personalized_layer_count}"
        )
        print(
            f"Shared Layers         : "
            f"{item.shared_layers}"
        )
        print(
            f"Personalized Layers   : "
            f"{item.personalized_layers}"
        )


if __name__ == "__main__":
    # ========================================================
    # 独立运行测试
    # ========================================================
    #
    # Step 03 真正依赖的输入只有：
    #     每个客户端持有多少数据
    #
    # 因此独立测试时直接手工构造 client_sizes。
    #
    # 正式完整 FL 流程中：
    #
    #     result = allocate_architecture_from_data_result(
    #         data_result
    #     )
    #
    # 两种方式调用完全相同的核心算法。
    # ========================================================

    test_client_sizes = {
        0: 500,
        1: 900,
        2: 1800,
        3: 4000,
        4: 8000,
    }

    result = allocate_architecture(
        client_sizes=test_client_sizes,
    )

    print_architecture_allocation_result(result)

    print("\n" + "=" * 78)
    print("Visualization Data")
    print("=" * 78)

    for row in result.visualization_data:
        print(row)

    print("\n" + "=" * 78)
    print("Automatic Checks")
    print("=" * 78)

    valid_shared_range = all(
        1 <= count <= 4
        for count in result.shared_layers.values()
    )

    print(
        "所有客户端共享层数均位于 1~4 :",
        valid_shared_range,
    )

    valid_total_layers = all(
        (
            item.shared_layer_count
            + item.personalized_layer_count
        ) == 5
        for item in result.client_allocations.values()
    )

    print(
        "Shared + Personalized 恒等于 5 :",
        valid_total_layers,
    )

    sorted_by_size = sorted(
        result.client_allocations.values(),
        key=lambda item: item.local_data_size,
    )

    monotonic = all(
        sorted_by_size[i].shared_layer_count
        >= sorted_by_size[i + 1].shared_layer_count
        for i in range(len(sorted_by_size) - 1)
    )

    print(
        "数据量越少 -> 共享层数不少于数据更多客户端 :",
        monotonic,
    )

    min_data_client = min(
        result.client_allocations.values(),
        key=lambda item: item.local_data_size,
    )

    print(
        "最小数据量客户端共享 4 层 :",
        min_data_client.shared_layer_count == 4,
    )

    max_data_client = max(
        result.client_allocations.values(),
        key=lambda item: item.local_data_size,
    )

    print(
        "最大数据量客户端共享 1 层 :",
        max_data_client.shared_layer_count == 1,
    )
