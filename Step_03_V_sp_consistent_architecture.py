"""
Step_03_V_sp_consistent_architecture.py

============================================================
Step 03 变体：统一共享-个性化架构分配
============================================================

【本文件职责】

本文件是：

    Step_03_V_architecture_allocation.py

的“统一架构变体”。

两者在整个系统中的位置完全相同，输入/输出接口也尽量保持一致，
区别只在于“共享层数如何决定”。

------------------------------------------------------------
原 Step 03：FedCALI 自适应架构
------------------------------------------------------------

原版本根据每个客户端的数据量 N_k 自适应计算共享层数 m_k：

    数据量较少
        -> 共享层更多
        -> 个性化层更少

    数据量较多
        -> 共享层更少
        -> 个性化层更多

因此不同客户端可以拥有不同的共享/个性化架构。

例如：

    Client 0: Shared 4 / Personalized 1
    Client 1: Shared 3 / Personalized 2
    Client 2: Shared 1 / Personalized 4


------------------------------------------------------------
本变体：Consistent Architecture
------------------------------------------------------------

本文件不根据客户端数据量决定共享层数。

而是额外输入一个统一参数：

    K

所有客户端都采用完全相同的模型划分：

    Shared Layers
        = 前 K 层

    Personalized Layers
        = 后 5-K 层

其中：

    K ∈ {1, 2, 3, 4, 5}

例如 K = 3：

    所有客户端：

        Shared Layers
            = [1, 2, 3]

        Personalized Layers
            = [4, 5]

因此无论不同客户端持有多少本地数据，
它们的共享/个性化架构划分都完全一致。


============================================================
【K = 5 的特殊意义】
============================================================

当：

    K = 5

时：

    Shared Layers
        = [1, 2, 3, 4, 5]

    Personalized Layers
        = []

也就是整个 5 层模型全部共享，不再存在个性化层。

从“模型架构划分”这个维度看：

    K = 5

对应于 FedAvg 的全共享模型结构。

需要注意：

这里只说明“架构形式”与 FedAvg 一致。

如果后续仍然使用：

    Byzantine Detection
    Layer Filtering
    Secure Aggregation

那么整个系统并不等同于标准 FedAvg。

若要完整复现普通 FedAvg，
还需要在后续流程中关闭安全过滤并采用普通参数平均。


============================================================
【5 层模型统一逻辑】
============================================================

本项目总模型层数固定为：

    L = 5

并统一抽象成：

    layer1
    layer2
    layer3
    layer4
    layer5

本 Step 03 只负责“逻辑层划分”，
不关心 CIFAR-10 和 MNIST 每一层内部的具体参数维度。

具体网络结构由 Step 04 负责。

无论 CIFAR-10 还是 MNIST，
本步骤都只输出：

    前 K 层 -> Shared
    后 5-K 层 -> Personalized


============================================================
【为什么本文件仍然接收 client_sizes】
============================================================

虽然统一架构模式不再根据客户端数据量计算 K，
但本文件仍然保留与原 Step 03 一致的输入结构：

    client_sizes

原因有三个：

1. 保持两种 Step 03 的调用接口尽量一致；
2. 可视化仍然需要同时展示：
       Client ID
       Local Data Size
       Shared Layer Count
       Personalized Layer Count
3. 后续 main.py 可以非常方便地切换两种架构策略，
   而不需要改变 Step 01 的数据流。

因此：

    client_sizes

在本文件中只用于：

    - 确认有哪些客户端；
    - 保留各客户端本地数据量；
    - 提供给可视化展示；

它不会参与共享层数 K 的计算。


============================================================
【与原 Step 03 的输出兼容】
============================================================

为了让 Step 04 以及后续模块无需关心使用了哪一种架构策略，
本文件保持与原 Step 03 相同的核心输出结构：

    ClientArchitectureAllocation

    ArchitectureAllocationResult

其中继续提供：

    result.client_sizes

    result.shared_layers

    result.personalized_layers

    result.client_allocations

    result.visualization_data


例如：

    result.shared_layers = {
        0: 3,
        1: 3,
        2: 3,
    }

表示所有客户端统一共享 3 层。


------------------------------------------------------------
关于 normalized_ratio
------------------------------------------------------------

原自适应版本中：

    normalized_ratio

表示客户端数据量经过对数归一化后的架构分配比例。

本统一架构版本中，
数据量不再参与架构划分，因此该值已经不具有原来的算法意义。

为了保持输出字段完全兼容，
本文件仍然保留 normalized_ratio 字段，
并统一设置为：

    K / 5

它在这里表示“模型共享比例”，仅用于兼容和展示，
不能再解释为原 FedCALI 的数据量归一化比例。


============================================================
【本步骤为什么带 V】
============================================================

文件名中的 V 表示：

    本模块的输出需要直接提供给可视化模块展示。

前端可以直接展示：

    Architecture Mode:
        Consistent

    Fixed Shared Layers K:
        3

    Client 0
        Local Data Size      : 500
        Shared Layer Count   : 3
        Personalized Count   : 2
        Shared Layers        : [1, 2, 3]
        Personalized Layers  : [4, 5]

    Client 1
        Local Data Size      : 900
        Shared Layer Count   : 3
        Personalized Count   : 2
        Shared Layers        : [1, 2, 3]
        Personalized Layers  : [4, 5]

无论本地数据量不同，
所有客户端都保持相同架构。


============================================================
【与后续 Step 04 的关系】
============================================================

本步骤的核心输出：

    result.shared_layers[client_id]

仍然是每个客户端的共享层数量。

因此 Step 04 不需要知道：

    这个 K 是自适应算出来的，
    还是统一指定的。

Step 04 只需要读取：

    client_shared_layer_counts

然后按照对应的共享层数进行本地训练。

特别注意：

    本变体允许 K = 5。

因此 Step 04 必须允许：

    shared_layer_count = 5

此时：

    personalized_layers = []

    shared_layers = [
        layer1,
        layer2,
        layer3,
        layer4,
        layer5
    ]

也就是跳过“个性化层训练”，
直接训练完整共享模型。


============================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Union


# ============================================================
# 固定模型配置
# ============================================================

TOTAL_MODEL_LAYERS = 5
MIN_SHARED_LAYERS = 1

# 与原自适应 Step 03 不同：
# 统一架构模式允许 K = 5。
MAX_SHARED_LAYERS = TOTAL_MODEL_LAYERS


# ============================================================
# 与原 Step 03 保持兼容的数据结构
# ============================================================

@dataclass(frozen=True)
class ClientArchitectureAllocation:
    client_id: int
    local_data_size: int
    shared_layer_count: int
    personalized_layer_count: int
    shared_layers: List[int]
    personalized_layers: List[int]

    # 保留原 Step 03 的字段名称以维持接口兼容。
    #
    # 在本变体中：
    # normalized_ratio = K / 5
    #
    # 它表示共享层比例，
    # 不再表示原 FedCALI 的数据量对数归一化比例。
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
        """
        直接提供给前端 / 可视化模块的结构化数据。

        字段与原 Step 03 保持一致。
        """

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

    def get(
        self,
        client_id: int,
    ) -> ClientArchitectureAllocation:
        if client_id not in self.client_allocations:
            raise ValueError(
                f"客户端 {client_id} 不存在"
            )

        return self.client_allocations[client_id]


ClientSizesInput = Union[
    Mapping[int, int],
    Sequence[int],
]


# ============================================================
# 输入标准化
# ============================================================

def _normalize_client_sizes(
    client_sizes: ClientSizesInput,
) -> Dict[int, int]:
    """
    将：

        {client_id: size}

    或：

        [size0, size1, size2, ...]

    统一转成：

        Dict[int, int]

    这一部分与原 Step 03 保持相同逻辑。
    """

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
        raise ValueError(
            "client_sizes 不能为空"
        )

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

    return dict(
        sorted(normalized.items())
    )


def _validate_fixed_shared_layers(
    fixed_shared_layers: int,
    total_model_layers: int,
) -> int:
    """
    检查统一共享层数 K。

    当前项目：
        L = 5
        K ∈ [1, 5]
    """

    try:
        fixed_shared_layers = int(
            fixed_shared_layers
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "fixed_shared_layers 必须是整数"
        ) from exc

    if not (
        MIN_SHARED_LAYERS
        <= fixed_shared_layers
        <= total_model_layers
    ):
        raise ValueError(
            "统一共享层数 K 必须位于 "
            f"{MIN_SHARED_LAYERS}~{total_model_layers}，"
            f"当前为 {fixed_shared_layers}"
        )

    return fixed_shared_layers


# ============================================================
# 核心架构划分
# ============================================================

def calculate_consistent_layer_partition(
    fixed_shared_layers: int,
    total_model_layers: int = TOTAL_MODEL_LAYERS,
) -> tuple[
    int,
    int,
    List[int],
    List[int],
    float,
]:
    """
    计算统一架构划分。

    输入：
        fixed_shared_layers = K

    输出：
        shared_count
        personalized_count
        shared_layer_indices
        personalized_layer_indices
        shared_ratio

    例如：

        K = 3

    返回：

        shared_count = 3
        personalized_count = 2
        shared_layers = [1, 2, 3]
        personalized_layers = [4, 5]
        shared_ratio = 3 / 5 = 0.6
    """

    if total_model_layers != TOTAL_MODEL_LAYERS:
        raise ValueError(
            "当前项目总模型层数固定为 5，"
            "暂不允许修改 total_model_layers。"
        )

    k = _validate_fixed_shared_layers(
        fixed_shared_layers,
        total_model_layers,
    )

    shared_count = k

    personalized_count = (
        total_model_layers - k
    )

    shared_layer_indices = list(
        range(
            1,
            k + 1,
        )
    )

    personalized_layer_indices = list(
        range(
            k + 1,
            total_model_layers + 1,
        )
    )

    shared_ratio = (
        shared_count
        / total_model_layers
    )

    return (
        shared_count,
        personalized_count,
        shared_layer_indices,
        personalized_layer_indices,
        float(shared_ratio),
    )


def allocate_consistent_architecture(
    client_sizes: ClientSizesInput,
    fixed_shared_layers: int,
    total_model_layers: int = TOTAL_MODEL_LAYERS,
) -> ArchitectureAllocationResult:
    """
    为所有客户端分配完全一致的共享/个性化架构。

    --------------------------------------------------------
    输入
    --------------------------------------------------------

    client_sizes:
        与原 Step 03 相同。

        用于：
        - 确认客户端集合；
        - 保留本地数据量；
        - 可视化展示。

        不参与架构计算。

    fixed_shared_layers:
        统一共享层数 K。

        K ∈ {1, 2, 3, 4, 5}

    --------------------------------------------------------
    输出
    --------------------------------------------------------

    与原 Step 03 保持相同的：

        ArchitectureAllocationResult

    因而后续模块仍然可以直接读取：

        result.shared_layers
        result.personalized_layers
        result.client_allocations
        result.visualization_data
    """

    if total_model_layers != TOTAL_MODEL_LAYERS:
        raise ValueError(
            "当前项目架构固定为 5 层模型，"
            "暂不允许修改 total_model_layers。"
        )

    normalized_sizes = _normalize_client_sizes(
        client_sizes
    )

    k = _validate_fixed_shared_layers(
        fixed_shared_layers,
        total_model_layers,
    )

    (
        shared_count,
        personalized_count,
        shared_layer_indices,
        personalized_layer_indices,
        shared_ratio,
    ) = calculate_consistent_layer_partition(
        fixed_shared_layers=k,
        total_model_layers=total_model_layers,
    )

    # 虽然本变体不使用 N_min / N_max 计算架构，
    # 仍然保留这两个输出字段以与原 Step 03 接口一致。
    n_min = min(
        normalized_sizes.values()
    )
    n_max = max(
        normalized_sizes.values()
    )

    shared_layers: Dict[int, int] = {}
    personalized_layers: Dict[int, int] = {}

    client_allocations: Dict[
        int,
        ClientArchitectureAllocation,
    ] = {}

    for client_id, local_data_size in (
        normalized_sizes.items()
    ):
        # 所有客户端使用完全相同的 K。
        shared_layers[
            client_id
        ] = shared_count

        personalized_layers[
            client_id
        ] = personalized_count

        client_allocations[
            client_id
        ] = ClientArchitectureAllocation(
            client_id=client_id,
            local_data_size=local_data_size,
            shared_layer_count=shared_count,
            personalized_layer_count=personalized_count,
            shared_layers=(
                shared_layer_indices.copy()
            ),
            personalized_layers=(
                personalized_layer_indices.copy()
            ),
            normalized_ratio=shared_ratio,
        )

    return ArchitectureAllocationResult(
        total_model_layers=total_model_layers,
        min_shared_layers=MIN_SHARED_LAYERS,
        max_shared_layers=MAX_SHARED_LAYERS,
        n_min=n_min,
        n_max=n_max,
        client_sizes=normalized_sizes,
        shared_layers=shared_layers,
        personalized_layers=personalized_layers,
        client_allocations=client_allocations,
    )


# ============================================================
# 与原 Step 03 相同风格的通用函数名
# ============================================================

def allocate_architecture(
    client_sizes: ClientSizesInput,
    fixed_shared_layers: int,
    total_model_layers: int = TOTAL_MODEL_LAYERS,
) -> ArchitectureAllocationResult:
    """
    与原 Step_03_V_architecture_allocation.py 的
    allocate_architecture() 保持相似调用风格。

    区别是本变体需要额外给出：

        fixed_shared_layers = K

    其余输出结构保持一致。
    """

    return allocate_consistent_architecture(
        client_sizes=client_sizes,
        fixed_shared_layers=fixed_shared_layers,
        total_model_layers=total_model_layers,
    )


def allocate_architecture_from_data_result(
    data_result: Any,
    fixed_shared_layers: int,
) -> ArchitectureAllocationResult:
    """
    直接接收 Step 01 的 DataDistributionResult。

    正式流程：

        architecture_result = (
            allocate_architecture_from_data_result(
                data_result=data_result,
                fixed_shared_layers=K,
            )
        )

    与原自适应 Step 03 一样，
    只要求 data_result 具有：

        client_sizes

    属性。
    """

    if not hasattr(
        data_result,
        "client_sizes",
    ):
        raise TypeError(
            "data_result 必须具有 client_sizes 属性；"
            "请传入 Step 01 的 DataDistributionResult"
        )

    return allocate_consistent_architecture(
        client_sizes=data_result.client_sizes,
        fixed_shared_layers=fixed_shared_layers,
        total_model_layers=TOTAL_MODEL_LAYERS,
    )


# ============================================================
# 输出打印
# ============================================================

def print_architecture_allocation_result(
    result: ArchitectureAllocationResult,
) -> None:
    """
    控制台展示。

    因为所有客户端采用统一架构，
    可以直接从任一客户端读取 K。
    """

    first_client_id = min(
        result.client_allocations
    )

    fixed_k = (
        result.client_allocations[
            first_client_id
        ].shared_layer_count
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "Step 03 Variant - "
        "Consistent Shared/Personalized Architecture"
    )

    print(
        "=" * 78
    )

    print(
        "Architecture Mode  : Consistent"
    )

    print(
        f"Total Model Layers : "
        f"{result.total_model_layers}"
    )

    print(
        f"Fixed Shared K     : "
        f"{fixed_k}"
    )

    print(
        f"Personalized Count : "
        f"{result.total_model_layers - fixed_k}"
    )

    print(
        f"Allowed K Range    : "
        f"{result.min_shared_layers} - "
        f"{result.max_shared_layers}"
    )

    print(
        f"N_min              : "
        f"{result.n_min}"
    )

    print(
        f"N_max              : "
        f"{result.n_max}"
    )

    for client_id in sorted(
        result.client_allocations
    ):
        item = result.client_allocations[
            client_id
        ]

        print(
            "\n"
            + "-" * 78
        )

        print(
            f"Client {client_id}"
        )

        print(
            "-" * 78
        )

        print(
            f"Local Data Size       : "
            f"{item.local_data_size}"
        )

        print(
            f"Shared Ratio          : "
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


# ============================================================
# 这个是测试样例
# ============================================================

if __name__ == "__main__":
    # ========================================================
    # 独立运行测试
    # ========================================================
    #
    # 本变体测试需要两个核心输入：
    #
    # 1. client_sizes
    #       保持与原 Step 03 一样；
    #
    # 2. FIXED_SHARED_LAYERS = K
    #       所有客户端统一共享 K 层。
    #
    # 正式完整 FL 流程中可写成：
    #
    #     result = allocate_architecture_from_data_result(
    #         data_result=data_result,
    #         fixed_shared_layers=K,
    #     )
    #
    # 后续 Step 04 仍然读取：
    #
    #     result.shared_layers
    #
    # 不需要知道架构是 adaptive 还是 consistent。
    # ========================================================

    test_client_sizes = {
        0: 500,
        1: 900,
        2: 1800,
        3: 4000,
        4: 8000,
    }

    # --------------------------------------------------------
    # 修改这里即可测试 K = 1~5
    # --------------------------------------------------------
    FIXED_SHARED_LAYERS = 3

    result = allocate_architecture(
        client_sizes=test_client_sizes,
        fixed_shared_layers=FIXED_SHARED_LAYERS,
    )

    print_architecture_allocation_result(
        result
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "Visualization Data"
    )

    print(
        "=" * 78
    )

    for row in result.visualization_data:
        print(row)

    print(
        "\n"
        + "=" * 78
    )

    print(
        "Automatic Checks"
    )

    print(
        "=" * 78
    )

    # --------------------------------------------------------
    # 1. 所有客户端共享层数都必须等于 K
    # --------------------------------------------------------

    all_clients_use_same_k = all(
        count == FIXED_SHARED_LAYERS
        for count in result.shared_layers.values()
    )

    print(
        "所有客户端共享层数均等于 K :",
        all_clients_use_same_k,
    )

    # --------------------------------------------------------
    # 2. K 合法范围为 1~5
    # --------------------------------------------------------

    valid_shared_range = all(
        1 <= count <= 5
        for count in result.shared_layers.values()
    )

    print(
        "所有客户端共享层数均位于 1~5 :",
        valid_shared_range,
    )

    # --------------------------------------------------------
    # 3. Shared + Personalized 恒等于 5
    # --------------------------------------------------------

    total_layers_valid = all(
        (
            result.shared_layers[client_id]
            +
            result.personalized_layers[client_id]
        )
        == TOTAL_MODEL_LAYERS
        for client_id in result.client_sizes
    )

    print(
        "Shared + Personalized 恒等于 5 :",
        total_layers_valid,
    )

    # --------------------------------------------------------
    # 4. 每个客户端共享层索引都完全一致
    # --------------------------------------------------------

    expected_shared_layers = list(
        range(
            1,
            FIXED_SHARED_LAYERS + 1,
        )
    )

    shared_indices_valid = all(
        result.client_allocations[
            client_id
        ].shared_layers
        == expected_shared_layers
        for client_id in result.client_sizes
    )

    print(
        "所有客户端 Shared Layers 完全一致 :",
        shared_indices_valid,
    )

    # --------------------------------------------------------
    # 5. K = 5 的全共享结构专项检查
    # --------------------------------------------------------

    fedavg_style_result = (
        allocate_architecture(
            client_sizes=test_client_sizes,
            fixed_shared_layers=5,
        )
    )

    k5_is_fully_shared = all(
        (
            item.shared_layer_count == 5
            and item.personalized_layer_count == 0
            and item.shared_layers == [1, 2, 3, 4, 5]
            and item.personalized_layers == []
        )
        for item in (
            fedavg_style_result
            .client_allocations
            .values()
        )
    )

    print(
        "K=5 时为完整 5 层共享 / 0 层个性化 :",
        k5_is_fully_shared,
    )
