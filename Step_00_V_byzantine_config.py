"""
Step_00_V_byzantine_config.py

联邦学习实验第 0 步：拜占庭攻击配置模块（可视化相关）。

职责：
1. 在完整实验开始前，确定哪些客户端是 Byzantine 客户端；
2. 为每个 Byzantine 客户端确定攻击类型；
3. 对 Gaussian / Sign Flipping 这类模型参数攻击，只确定“攻击层比例”；
4. 为后续“具体攻击哪些共享层”预先固定一个 layer_selection_seed；
5. 为 Label Flipping 保存标签翻转策略；
6. 输出一份固定的 AttackPlan，供后续所有阶段直接读取，避免“后面重新随机”导致断档。

为什么第 0 步不直接确定 attacked_layers：
- 每个客户端的共享层数要到第 2 步“架构分配”后才能确定；
- Gaussian / Sign Flipping 应只作用于该客户端真正参与共享/上传的层；
- 因此第 0 步只能确定攻击比例，不能提前假设所有客户端都有相同的共享层集合。

明确不负责：
- 不实际修改客户端数据；
- 不实际修改模型参数；
- 不决定客户端共享层数；
- 不提前确定具体 attacked_layers；
- 不执行本地训练；
- 不执行检测与聚合。

重点看这里的输入！！！

输入：     带 # 的为需要读取作为输入的内容

# - num_clients：客户端总数 （ 建议是 10 个 ，FL训练比较慢）

# - byzantine_ratio：拜占庭客户端比例 （我的论文是 0 ，0.1 ，0.2 ，0.3 ，建议是只提供这些选项）

# - attack_type_weights：攻击类型配置，例如全部 Sign Flipping，或者三种攻击混合
    （请注意，这里的 “输入” 是类似一个数组的东西，是三种攻击的分别比例，但是实际上可视化输入的大概率
    是一个字符串/选项，你得把这个输入的东西调整成对应的数组，而不要动这个代码里的东西） （大概？）

# - layer_attack_ratio：对于 Gaussian / Sign Flipping，
    攻击“该客户端共享层”的多少比例 （攻击的层数比例，等第 2 步决定好架构后进一步决定是哪些层）

# - gaussian_mean、gaussian_std：高斯攻击参数 （中心值和标准差）

# - master_seed：本次实验主随机种子 （可调控）

- sign_flip_scale：符号翻转攻击强度 （锁死为 -1 ）

- label_flip_map：标签翻转规则 （ None 占位符）
    （实际上到第 1 步后面才会实施，得先发数据再说，实际上不需要作为可视化输入）

输出：（需要可视化的话，得注意看）
- AttackPlan
  - Byzantine 客户端列表
  - Benign 客户端列表
  - 每个客户端的 ClientAttackConfig
    - client_id
    - is_byzantine
    - attack_type
    - layer_attack_ratio
    - layer_selection_seed
    - 对应攻击强度参数
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import floor
from typing import Dict, List, Optional, Sequence
import random


DEFAULT_MASTER_SEED = 42


class AttackType(str, Enum):
    LABEL_FLIPPING = "label_flipping"
    GAUSSIAN = "gaussian"
    SIGN_FLIPPING = "sign_flipping"


@dataclass(frozen=True)
class ClientAttackConfig:
    client_id: int
    is_byzantine: bool
    attack_type: Optional[AttackType] = None

    # 对模型参数攻击而言，这是“共享层攻击比例”
    layer_attack_ratio: float = 0.0

    # 第 0 步只固定用于后续选择具体共享攻击层的随机种子
    layer_selection_seed: Optional[int] = None

    gaussian_mean: float = 0.0
    gaussian_std: float = 1.0

    sign_flip_scale: float = -1.0

    label_flip_map: Optional[Dict[int, int]] = None


@dataclass(frozen=True)
class AttackPlan:
    master_seed: int
    num_clients: int
    byzantine_ratio: float
    client_configs: Dict[int, ClientAttackConfig]

    @property
    def byzantine_clients(self) -> List[int]:
        return [
            client_id
            for client_id, cfg in self.client_configs.items()
            if cfg.is_byzantine
        ]

    @property
    def benign_clients(self) -> List[int]:
        return [
            client_id
            for client_id, cfg in self.client_configs.items()
            if not cfg.is_byzantine
        ]

    def get(self, client_id: int) -> ClientAttackConfig:
        if client_id not in self.client_configs:
            raise ValueError(f"客户端 {client_id} 不存在")
        return self.client_configs[client_id]


class ByzantineConfigGenerator:
    def __init__(
        self,
        num_clients: int,
        byzantine_ratio: float,
        attack_type_weights: Dict[AttackType | str, float],
        layer_attack_ratio: float = 1.0,
        gaussian_mean: float = 0.0,
        gaussian_std: float = 1.0,
        sign_flip_scale: float = -1.0,
        label_flip_map: Optional[Dict[int, int]] = None,
        master_seed: int = DEFAULT_MASTER_SEED,
    ):
        self.num_clients = num_clients
        self.byzantine_ratio = byzantine_ratio
        self.layer_attack_ratio = layer_attack_ratio

        self.gaussian_mean = gaussian_mean
        self.gaussian_std = gaussian_std
        self.sign_flip_scale = sign_flip_scale
        self.label_flip_map = label_flip_map

        self.master_seed = master_seed
        self.rng = random.Random(master_seed)

        self.attack_type_weights: Dict[AttackType, float] = {}
        for attack_type, weight in attack_type_weights.items():
            attack_enum = (
                attack_type
                if isinstance(attack_type, AttackType)
                else AttackType(attack_type)
            )
            self.attack_type_weights[attack_enum] = float(weight)

        self._validate_config()

    def _validate_config(self) -> None:
        if self.num_clients <= 0:
            raise ValueError("num_clients 必须 > 0")

        if not (0.0 <= self.byzantine_ratio <= 1.0):
            raise ValueError("byzantine_ratio 必须位于 [0, 1]")

        if not (0.0 <= self.layer_attack_ratio <= 1.0):
            raise ValueError("layer_attack_ratio 必须位于 [0, 1]")

        if not self.attack_type_weights and self.byzantine_ratio > 0:
            raise ValueError(
                "存在 Byzantine 客户端时，attack_type_weights 不能为空"
            )

        if any(weight < 0 for weight in self.attack_type_weights.values()):
            raise ValueError("attack_type_weights 中的权重不能为负数")

        if (
            self.byzantine_ratio > 0
            and sum(self.attack_type_weights.values()) <= 0
        ):
            raise ValueError("attack_type_weights 权重总和必须 > 0")

        if self.gaussian_std < 0:
            raise ValueError("gaussian_std 必须 >= 0")

    def _calculate_byzantine_count(self) -> int:
        if self.byzantine_ratio == 0:
            return 0

        count = int(round(self.num_clients * self.byzantine_ratio))
        return min(self.num_clients, max(1, count))

    def _allocate_attack_type_counts(
        self,
        byzantine_count: int,
    ) -> Dict[AttackType, int]:
        if byzantine_count == 0:
            return {
                attack_type: 0
                for attack_type in self.attack_type_weights
            }

        total_weight = sum(self.attack_type_weights.values())

        raw_counts = {
            attack_type: byzantine_count * weight / total_weight
            for attack_type, weight in self.attack_type_weights.items()
        }

        counts = {
            attack_type: floor(raw)
            for attack_type, raw in raw_counts.items()
        }

        remainder = byzantine_count - sum(counts.values())

        if remainder > 0:
            order = sorted(
                raw_counts.keys(),
                key=lambda attack_type: (
                    raw_counts[attack_type] - counts[attack_type]
                ),
                reverse=True,
            )

            for attack_type in order[:remainder]:
                counts[attack_type] += 1

        return counts

    def _new_layer_selection_seed(self) -> int:
        return self.rng.randrange(0, 2**32)

    def generate(self) -> AttackPlan:
        byzantine_count = self._calculate_byzantine_count()
        all_clients = list(range(self.num_clients))

        byzantine_clients = sorted(
            self.rng.sample(all_clients, byzantine_count)
        )

        type_counts = self._allocate_attack_type_counts(byzantine_count)

        attack_type_list: List[AttackType] = []
        for attack_type, count in type_counts.items():
            attack_type_list.extend([attack_type] * count)

        self.rng.shuffle(attack_type_list)

        client_to_attack_type = {
            client_id: attack_type
            for client_id, attack_type in zip(
                byzantine_clients,
                attack_type_list,
            )
        }

        client_configs: Dict[int, ClientAttackConfig] = {}

        for client_id in all_clients:
            if client_id not in client_to_attack_type:
                client_configs[client_id] = ClientAttackConfig(
                    client_id=client_id,
                    is_byzantine=False,
                )
                continue

            attack_type = client_to_attack_type[client_id]

            if attack_type == AttackType.LABEL_FLIPPING:
                client_configs[client_id] = ClientAttackConfig(
                    client_id=client_id,
                    is_byzantine=True,
                    attack_type=attack_type,
                    layer_attack_ratio=0.0,
                    layer_selection_seed=None,
                    label_flip_map=self.label_flip_map,
                )

            elif attack_type == AttackType.GAUSSIAN:
                client_configs[client_id] = ClientAttackConfig(
                    client_id=client_id,
                    is_byzantine=True,
                    attack_type=attack_type,
                    layer_attack_ratio=self.layer_attack_ratio,
                    layer_selection_seed=self._new_layer_selection_seed(),
                    gaussian_mean=self.gaussian_mean,
                    gaussian_std=self.gaussian_std,
                )

            elif attack_type == AttackType.SIGN_FLIPPING:
                client_configs[client_id] = ClientAttackConfig(
                    client_id=client_id,
                    is_byzantine=True,
                    attack_type=attack_type,
                    layer_attack_ratio=self.layer_attack_ratio,
                    layer_selection_seed=self._new_layer_selection_seed(),
                    sign_flip_scale=self.sign_flip_scale,
                )

            else:
                raise RuntimeError(f"未处理的攻击类型：{attack_type}")

        return AttackPlan(
            master_seed=self.master_seed,
            num_clients=self.num_clients,
            byzantine_ratio=self.byzantine_ratio,
            client_configs=client_configs,
        )


def resolve_attacked_layers(
    client_config: ClientAttackConfig,
    shared_layers: Sequence[int],
) -> tuple[int, ...]:
    """
    第 2 步得到 shared_layers 后，解析最终 attacked_layers。

    同一个 client_config.layer_selection_seed、
    同一份 shared_layers 和同一个 layer_attack_ratio，
    会得到完全相同的 attacked_layers。
    """

    if not client_config.is_byzantine:
        return ()

    if client_config.attack_type == AttackType.LABEL_FLIPPING:
        return ()

    if client_config.attack_type not in {
        AttackType.GAUSSIAN,
        AttackType.SIGN_FLIPPING,
    }:
        return ()

    if client_config.layer_attack_ratio <= 0:
        return ()

    layers = list(shared_layers)
    if not layers:
        return ()

    if client_config.layer_selection_seed is None:
        raise ValueError(
            f"Client {client_config.client_id} 缺少 layer_selection_seed"
        )

    attack_count = int(
        round(len(layers) * client_config.layer_attack_ratio)
    )

    attack_count = max(1, attack_count)
    attack_count = min(len(layers), attack_count)

    rng = random.Random(client_config.layer_selection_seed)
    attacked_layers = rng.sample(layers, attack_count)

    return tuple(sorted(attacked_layers))


def print_attack_plan(plan: AttackPlan) -> None:
    print("\n" + "=" * 72)
    print("Step 0 - Byzantine Attack Plan")
    print("=" * 72)

    print(f"Master Seed       : {plan.master_seed}")
    print(f"Client Count      : {plan.num_clients}")
    print(f"Byzantine Ratio   : {plan.byzantine_ratio:.2f}")
    print(f"Byzantine Clients : {plan.byzantine_clients}")
    print(f"Benign Clients    : {plan.benign_clients}")

    print("\nClient Details")
    print("-" * 72)

    for client_id in range(plan.num_clients):
        cfg = plan.get(client_id)

        if not cfg.is_byzantine:
            print(f"Client {client_id:02d}: BENIGN")
            continue

        if cfg.attack_type == AttackType.LABEL_FLIPPING:
            print(
                f"Client {client_id:02d}: BYZANTINE | "
                f"{cfg.attack_type.value} | data-layer attack"
            )

        elif cfg.attack_type == AttackType.GAUSSIAN:
            print(
                f"Client {client_id:02d}: BYZANTINE | "
                f"{cfg.attack_type.value} | "
                f"shared-layer ratio={cfg.layer_attack_ratio:.2f} | "
                f"layer_seed={cfg.layer_selection_seed} | "
                f"mean={cfg.gaussian_mean} | std={cfg.gaussian_std}"
            )

        elif cfg.attack_type == AttackType.SIGN_FLIPPING:
            print(
                f"Client {client_id:02d}: BYZANTINE | "
                f"{cfg.attack_type.value} | "
                f"shared-layer ratio={cfg.layer_attack_ratio:.2f} | "
                f"layer_seed={cfg.layer_selection_seed} | "
                f"scale={cfg.sign_flip_scale}"
            )


if __name__ == "__main__":
    generator = ByzantineConfigGenerator(
        num_clients=10,
        byzantine_ratio=0.30,

        attack_type_weights={
            AttackType.LABEL_FLIPPING: 1,
            AttackType.GAUSSIAN: 1,
            AttackType.SIGN_FLIPPING: 1,
        },

        # 这里是“共享层攻击比例”，不是模型总层数比例。
        layer_attack_ratio=0.50,

        gaussian_mean=0.0,
        gaussian_std=1.0,

        sign_flip_scale=-1.0,

        label_flip_map={
            0: 1,
            1: 2,
            2: 3,
            3: 4,
            4: 5,
            5: 6,
            6: 7,
            7: 8,
            8: 9,
            9: 0,
        },

        master_seed=42,
    )

    attack_plan = generator.generate()

    print_attack_plan(attack_plan)

    # --------------------------------------------------------
    # 下面只是裸跑测试：模拟第 2 步已经产生了共享层
    # --------------------------------------------------------
    example_shared_layers = {
        0: [1, 2, 3, 4],
        1: [1, 2],
        2: [1, 2, 3],
        3: [1],
        4: [1, 2, 3, 4, 5],
        5: [1, 2, 3],
        6: [1, 2],
        7: [1, 2, 3, 4],
        8: [1],
        9: [1, 2, 3, 4, 5],
    }

    print("\n" + "=" * 72)
    print("Example - Resolve attacked layers AFTER Step 2")
    print("=" * 72)

    for client_id in attack_plan.byzantine_clients:
        cfg = attack_plan.get(client_id)

        shared_layers = example_shared_layers[client_id]
        attacked_layers = resolve_attacked_layers(
            cfg,
            shared_layers,
        )

        print(
            f"Client {client_id:02d} | "
            f"type={cfg.attack_type.value} | "
            f"shared_layers={shared_layers} | "
            f"attacked_layers={list(attacked_layers)}"
        )
