"""
Step_06_V_sp_uploading.py

============================================================
Step 06 特殊分支：无隐私保护的原始共享模型参数直接上传（可视化相关）
============================================================

【作用】
本文件与 Step_06_V_masking_and_uploading.py 构成 Step 06 的两种可选方案：

    A. Privacy / Masking:
       Step_06_V_masking_and_uploading.py
       对共享模型参数进行随机掩码处理，再分别交给 TS / DS1 / DS2 / DS3。

    B. Plain / No Protection:
       Step_06_V_sp_uploading.py
       不执行任何掩码、拆分、加密或秘密共享处理，
       直接把客户端完整的原始共享模型参数上传给 TS。

本分支用于：
    1. 展示“不采用隐私保护”时的基线流程；
    2. 与随机掩码方案进行可视化对比；
    3. 后续构造无隐私保护的对照实验。

【重要】
- Step 05 是否真正修改参数，由 Step 00 的攻击策略自动决定。
- Step 06 不重新决定攻击类型，也不重新实施攻击。
- 如果接 Step 05，则上传 Step 05 输出的 attacked_client_shared_parameters；
  对于没有参数攻击的客户端，这些参数自然保持原值。
- 如果实验流程直接接 Step 04，也支持上传 client_shared_parameters。
- 本文件只处理 Shared Parameters；Personalized Parameters 不上传。
- “直接上传”是逻辑上传：这里只把参数放入 TS 对应的数据结构，
  不模拟 HTTP / RPC / socket 等真实网络通信。

【核心关系】
对于 Client k、共享层 l、层内某个参数 Tensor theta：

    TS receives theta directly.

即：
    uploaded_theta = theta

没有：
    Mask
    Share splitting
    Encryption
    Secret sharing

因此 DS1 / DS2 / DS3 在本分支中不参与参数接收。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch


TS_NAME = "TS"
SERVER_NAMES: Tuple[str, ...] = (TS_NAME,)
ALL_LAYER_NAMES = ("layer1", "layer2", "layer3", "layer4", "layer5")
DEFAULT_PREVIEW_VALUES_PER_TENSOR = 5


@dataclass(frozen=True)
class PlainTensorUploadRecord:
    """单个共享参数 Tensor 的直接上传记录，主要用于可视化。"""

    client_id: int
    layer_name: str
    tensor_index: int
    tensor_role: str
    shape: Tuple[int, ...]
    mean: float
    std: float
    norm: float
    preview: List[float]


@dataclass(frozen=True)
class PlainUploadPacket:
    """TS 收到某个客户端完整共享架构参数的逻辑上传描述。"""

    server_name: str
    client_id: int
    layer_names: List[str]
    tensor_count: int


@dataclass
class PlainUploadingResult:
    """Step 06 无隐私保护分支的统一输出。"""

    round_index: int
    server_names: Tuple[str, ...]

    # Step 06 接收到的共享参数副本。
    original_client_shared_parameters: Dict[
        int, Dict[str, List[torch.Tensor]]
    ]

    # 无任何处理，直接上传给 TS。
    parameters_by_server: Dict[
        str, Dict[int, Dict[str, List[torch.Tensor]]]
    ]

    upload_packets: List[PlainUploadPacket]
    tensor_records: List[PlainTensorUploadRecord]

    @property
    def ts_parameters(self) -> Dict[int, Dict[str, List[torch.Tensor]]]:
        """后续无保护流程可直接使用 TS 持有的完整参数。"""
        return self.parameters_by_server[TS_NAME]

    @property
    def visualization_data(self) -> List[Dict[str, Any]]:
        """JSON-friendly：只给 preview / 统计量，不把完整高维 Tensor 塞给前端。"""
        return [
            {
                "client_id": r.client_id,
                "layer_name": r.layer_name,
                "tensor_index": r.tensor_index,
                "tensor_role": r.tensor_role,
                "shape": list(r.shape),
                "original_mean": r.mean,
                "original_std": r.std,
                "original_norm": r.norm,
                "original_preview": r.preview.copy(),
                "upload_mode": "plain",
                "destination_server": TS_NAME,
                "processing": "none",
                "upload_equation": "uploaded_theta = theta",
            }
            for r in self.tensor_records
        ]


def _clone_parameter_structure(
    client_shared_parameters: Mapping[
        int, Mapping[str, Sequence[torch.Tensor]]
    ],
) -> Dict[int, Dict[str, List[torch.Tensor]]]:
    """完整克隆到 CPU，避免 Step 06 原地修改 Step 04 / Step 05 输出。"""
    return {
        int(client_id): {
            str(layer_name): [
                tensor.detach().cpu().clone() for tensor in tensors
            ]
            for layer_name, tensors in layer_map.items()
        }
        for client_id, layer_map in client_shared_parameters.items()
    }


def _tensor_role(tensor_index: int, tensor_count: int) -> str:
    if tensor_count >= 1 and tensor_index == 0:
        return "weight"
    if tensor_count >= 2 and tensor_index == 1:
        return "bias"
    return f"param_{tensor_index}"


def _tensor_std(tensor: torch.Tensor) -> float:
    flat = tensor.detach().float().reshape(-1)
    return 0.0 if flat.numel() == 0 else float(flat.std(unbiased=False).item())


def _tensor_norm(tensor: torch.Tensor) -> float:
    flat = tensor.detach().float().reshape(-1)
    return 0.0 if flat.numel() == 0 else float(torch.linalg.vector_norm(flat).item())


def _preview_values(tensor: torch.Tensor, count: int) -> List[float]:
    if count <= 0:
        return []
    flat = tensor.detach().float().reshape(-1).cpu()
    return [float(v) for v in flat[:count].tolist()]


def _validate_client_shared_parameters(
    client_shared_parameters: Mapping[
        int, Mapping[str, Sequence[torch.Tensor]]
    ],
) -> None:
    if not client_shared_parameters:
        raise ValueError("client_shared_parameters 不能为空")

    for client_id, layer_map in client_shared_parameters.items():
        if int(client_id) < 0:
            raise ValueError(f"client_id 不能为负数：{client_id}")
        if not layer_map:
            raise ValueError(f"Client {client_id} 没有 Shared Parameters")

        for layer_name, tensors in layer_map.items():
            if str(layer_name) not in ALL_LAYER_NAMES:
                raise ValueError(
                    f"非法逻辑层名称：{layer_name!r}；"
                    f"当前只支持 {list(ALL_LAYER_NAMES)}"
                )
            if not tensors:
                raise ValueError(
                    f"Client {client_id} / {layer_name} 没有参数 Tensor"
                )
            for tensor_index, tensor in enumerate(tensors):
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(
                        f"Client {client_id} / {layer_name} / "
                        f"tensor {tensor_index} 不是 torch.Tensor"
                    )


def upload_shared_parameters_directly(
    client_shared_parameters: Mapping[
        int, Mapping[str, Sequence[torch.Tensor]]
    ],
    round_index: int,
    preview_values_per_tensor: int = DEFAULT_PREVIEW_VALUES_PER_TENSOR,
) -> PlainUploadingResult:
    """
    Step 06 无隐私保护分支的核心函数。

    输入什么共享参数，就把什么共享参数完整交给 TS。
    不加 Mask，不拆 Share，不加密，不做额外数值变换。
    """
    _validate_client_shared_parameters(client_shared_parameters)

    if int(round_index) < 0:
        raise ValueError("round_index 必须 >= 0")
    if int(preview_values_per_tensor) < 0:
        raise ValueError("preview_values_per_tensor 必须 >= 0")

    original = _clone_parameter_structure(client_shared_parameters)

    # 再克隆一份作为 TS 实际“收到”的数据。
    # 这样 TS 容器与 original 不是同一个 Tensor 引用。
    ts_parameters = _clone_parameter_structure(original)

    parameters_by_server = {TS_NAME: ts_parameters}
    upload_packets: List[PlainUploadPacket] = []
    tensor_records: List[PlainTensorUploadRecord] = []

    # Client -> Shared Layer -> Tensor(weight/bias)
    for client_id in sorted(original):
        layer_map = original[client_id]
        tensor_count_total = 0

        for layer_name, tensors in layer_map.items():
            for tensor_index, theta in enumerate(tensors):
                tensor_records.append(
                    PlainTensorUploadRecord(
                        client_id=int(client_id),
                        layer_name=layer_name,
                        tensor_index=tensor_index,
                        tensor_role=_tensor_role(tensor_index, len(tensors)),
                        shape=tuple(theta.shape),
                        mean=float(theta.detach().float().mean().item()),
                        std=_tensor_std(theta),
                        norm=_tensor_norm(theta),
                        preview=_preview_values(
                            theta, int(preview_values_per_tensor)
                        ),
                    )
                )
                tensor_count_total += 1

        upload_packets.append(
            PlainUploadPacket(
                server_name=TS_NAME,
                client_id=int(client_id),
                layer_names=list(layer_map.keys()),
                tensor_count=tensor_count_total,
            )
        )

    return PlainUploadingResult(
        round_index=int(round_index),
        server_names=SERVER_NAMES,
        original_client_shared_parameters=original,
        parameters_by_server=parameters_by_server,
        upload_packets=upload_packets,
        tensor_records=tensor_records,
    )


def upload_from_step04(
    local_training_result: Any,
    preview_values_per_tensor: int = DEFAULT_PREVIEW_VALUES_PER_TENSOR,
) -> PlainUploadingResult:
    """Step 04 -> Step 06 无保护上传。"""
    if not hasattr(local_training_result, "client_shared_parameters"):
        raise TypeError(
            "local_training_result 缺少 client_shared_parameters"
        )
    if not hasattr(local_training_result, "round_index"):
        raise TypeError("local_training_result 缺少 round_index")

    return upload_shared_parameters_directly(
        client_shared_parameters=local_training_result.client_shared_parameters,
        round_index=int(local_training_result.round_index),
        preview_values_per_tensor=int(preview_values_per_tensor),
    )


def upload_from_step05(
    parameter_attack_result: Any,
    preview_values_per_tensor: int = DEFAULT_PREVIEW_VALUES_PER_TENSOR,
) -> PlainUploadingResult:
    """
    Step 05 -> Step 06 无保护上传。

    Step 05 是否产生实际参数攻击由 Step 00 的 AttackPlan 控制；
    Step 06 不做第二次攻击判断，只接收 Step 05 最终参数。
    """
    if not hasattr(parameter_attack_result, "attacked_client_shared_parameters"):
        raise TypeError(
            "parameter_attack_result 缺少 attacked_client_shared_parameters"
        )
    if not hasattr(parameter_attack_result, "round_index"):
        raise TypeError("parameter_attack_result 缺少 round_index")

    return upload_shared_parameters_directly(
        client_shared_parameters=(
            parameter_attack_result.attacked_client_shared_parameters
        ),
        round_index=int(parameter_attack_result.round_index),
        preview_values_per_tensor=int(preview_values_per_tensor),
    )


def print_plain_uploading_result(
    result: PlainUploadingResult,
    max_records: int = 8,
) -> None:
    """裸跑测试 / 调试输出。"""
    print("\n" + "=" * 88)
    print("Step 06 - Plain Shared-Parameter Uploading")
    print("=" * 88)
    print(f"Round Index   : {result.round_index}")
    print(f"Servers Used  : {list(result.server_names)}")
    print("Protection    : NONE")
    print("Destination   : TS only")
    print("Core Equation : uploaded_theta = theta")
    print(f"Tensor Records: {len(result.tensor_records)}")

    for record in result.tensor_records[:max_records]:
        print("\n" + "-" * 88)
        print(
            f"Client {record.client_id:02d} | "
            f"{record.layer_name} | {record.tensor_role} | "
            f"shape={record.shape}"
        )
        print(f"Original/Uploaded norm    : {record.norm:.8f}")
        print(f"Original/Uploaded preview : {record.preview}")


if __name__ == "__main__":
    # ============================================================
    # 独立快速测试
    #
    # 正式完整流程可以：
    #   Step 00 -> 01 -> 02 -> 03 -> 04 -> 05 -> Step 06
    #
    # 其中 Step 05 是否真正攻击参数由 Step 00 自动控制。
    # Step 06 再根据实验配置选择：
    #
    #   masking branch -> Step_06_V_masking_and_uploading.py
    #   plain branch   -> 本文件
    #
    # 本测试只验证“原参数原样进入 TS”这一核心性质。
    # ============================================================

    generator = torch.Generator(device="cpu")
    generator.manual_seed(2026)

    fake_parameters = {
        0: {
            "layer1": [
                torch.randn(3, 2, generator=generator),
                torch.randn(3, generator=generator),
            ],
            "layer2": [
                torch.randn(2, 3, generator=generator),
                torch.randn(2, generator=generator),
            ],
        },
        1: {
            "layer1": [
                torch.randn(4, 3, generator=generator),
                torch.randn(4, generator=generator),
            ],
        },
    }

    before = _clone_parameter_structure(fake_parameters)

    result = upload_shared_parameters_directly(
        client_shared_parameters=fake_parameters,
        round_index=1,
        preview_values_per_tensor=3,
    )

    print_plain_uploading_result(result, max_records=10)

    # 自动检查：TS 收到的每个 Tensor 必须与输入完全一致。
    all_equal = True
    for client_id, layer_map in before.items():
        for layer_name, tensors in layer_map.items():
            uploaded = result.ts_parameters[client_id][layer_name]
            for original_tensor, uploaded_tensor in zip(tensors, uploaded):
                if not torch.equal(original_tensor, uploaded_tensor):
                    all_equal = False

    # 自动检查：调用后上游输入不应被修改。
    input_not_mutated = True
    for client_id, layer_map in before.items():
        for layer_name, tensors in layer_map.items():
            current = fake_parameters[client_id][layer_name]
            for old_tensor, current_tensor in zip(tensors, current):
                if not torch.equal(old_tensor, current_tensor):
                    input_not_mutated = False

    print("\n" + "=" * 88)
    print("Automatic Checks")
    print("=" * 88)
    print("仅使用 TS                         :", result.server_names == ("TS",))
    print("TS 收到的参数与输入完全一致       :", all_equal)
    print("上游模型参数未被原地修改           :", input_not_mutated)
    print(
        "visualization_data 已准备好       :",
        len(result.visualization_data) == len(result.tensor_records),
    )
