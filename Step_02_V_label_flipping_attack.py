"""
Step_02_V_label_flipping_attack.py

联邦学习实验第 2 步：Label Flipping 标签翻转攻击模块（可视化相关）。

职责：
1. 读取第 0 步生成的 AttackPlan，识别其中 attack_type == label_flipping 的 Byzantine 客户端；
2. 读取第 1 步 DataDistributionResult，取得每个客户端已经分配好的本地训练集；
3. 对 Label Flipping 客户端“训练集中的每一个样本”执行标签翻转：
   - 原始标签 y 不保留；
   - 从除 y 之外的其他类别中随机选择一个新标签；
   - 每个样本独立随机，因此原本不同标签的样本在翻转后完全可能变成同一个标签；
   - 同一个原始标签的不同样本，也可能被随机翻转成不同的新标签；
4. 客户端自己的本地测试集绝对不修改；
5. 不直接修改 full_dataset 的全局 targets，而是为每个客户端建立“训练标签覆盖层”，
   避免因为第 1 步允许不同客户端复制/共享同一原始样本，而误伤其他客户端的数据；
6. 输出后续 Local Training 可以直接使用的客户端训练 Dataset；
7. 输出 Label Flipping 前后的标签统计、翻转数量和少量样例，供可视化模块展示；
8. 随机过程由第 0 步的 master_seed 派生，保证同一实验配置下结果可复现。

为什么不能直接修改 full_dataset.targets：
- 第 1 步允许不同客户端持有相同的原始样本索引；
- 如果直接把 full_dataset.targets[idx] 改掉，那么某个 Label Flipping 客户端修改一个标签时，
  其他持有同一个 idx 的 benign 客户端也会被一起修改；
- 甚至还可能影响其他客户端的 local test；
- 因此本模块必须采用“每客户端独立的 label override”方式，而不是全局原地修改数据集。

明确不负责：
- 不重新选择 Byzantine 客户端；
- 不重新决定攻击类型；
- 不修改客户端样本归属；
- 不修改 client_train_indices / client_test_indices；
- 不修改客户端本地测试集；
- 不执行 Gaussian Attack；
- 不执行 Sign Flipping；
- 不执行模型训练；
- 不决定共享层数；
- 不执行参数掩码、检测或聚合。

输入：     带 # 的为需要读取作为输入的内容

# - attack_plan：
    （直接读取第 0 步 step_00_V_byzantine_config.py 生成的 AttackPlan。
     本步骤不重新随机 Byzantine 客户端，也不重新随机攻击类型。
     这里只处理其中 attack_type == label_flipping 的客户端。）

# - data_result：
    （直接读取第 1 步 step_01_data_distribution.py 生成的 DataDistributionResult。
     主要使用：
         full_dataset
         client_train_indices
         client_test_indices
         client_label_distribution_train
     本步骤只改变 Label Flipping 客户端训练时“看到的标签”，
     不改变第 1 步已经决定好的样本索引和数据量。）

- master_seed：
    （不单独从前端读取。
     直接使用 attack_plan.master_seed，并为每个 Label Flipping 客户端派生独立随机种子。）

- preview_samples_per_client：
    （仅用于控制给可视化/调试展示多少条“原标签 -> 翻转后标签”的样例。
     属于后端软锁定参数，不影响真实攻击结果。
     默认显示每个受攻击客户端前 10 条样例。）

输出：（带 V，本步骤输出需要提供给可视化模块）

- LabelFlippingResult：
    （本步骤的统一输出对象。）

    - data_result：
        （原始第 1 步 DataDistributionResult。
         样本索引、客户端数据量、global test 等均保持不变。）

    - client_train_datasets：
        （每个客户端后续 Local Training 真正使用的训练 Dataset。
         对 Label Flipping 客户端：返回标签已被覆盖后的训练数据；
         对其他客户端：返回原始训练数据。）

    - client_test_datasets：
        （每个客户端的本地测试 Dataset。
         始终保持第 1 步原始标签，不允许攻击修改。）

    - label_overrides：
        （仅对 Label Flipping 客户端记录：
             client_id -> {原始样本索引: 翻转后的新标签}
         这是实际标签攻击结果。
         不修改 full_dataset.targets。）

    - attacked_clients：
        （实际执行 Label Flipping 的客户端 ID 列表。）

    - client_original_train_label_distribution：
        （各 Label Flipping 客户端攻击前的训练标签分布。）

    - client_attacked_train_label_distribution：
        （各 Label Flipping 客户端攻击后的训练标签分布。
         建议可视化模块重点展示“攻击前 vs 攻击后”。）

    - flipped_sample_counts：
        （每个 Label Flipping 客户端实际翻转的训练样本数量。
         按当前攻击定义，应等于该客户端训练集样本数。）

    - visualization_samples：
        （供可视化展示的少量标签变化样例：
             sample_index
             original_label
             flipped_label
         仅是展示样例，不是全部攻击数据。）

可视化建议：
- 页面展示实际执行 Label Flipping 的客户端；
- 对某个受攻击客户端展示：
      Train Samples: 例如 5000
      Flipped Samples: 5000
      Attack Coverage: 100%
- 用柱状图对比该客户端：
      攻击前标签分布
      攻击后标签分布
- 再展示少量样本变化：
      Sample 125: 3 -> 7
      Sample 814: 3 -> 1
      Sample 996: 8 -> 1
  这样可以直观看出：
      1. 每个训练样本都被翻转；
      2. 同一原标签可以随机变成不同标签；
      3. 不同原标签也可能最终变成同一标签。

后续主要传递关系：

    Step 0 AttackPlan
            +
    Step 1 DataDistributionResult
            |
            v
    Step 2 Label Flipping
            |
            +--> client_train_datasets
            |       -> Local Training
            |
            +--> client_test_datasets
            |       -> Local Evaluation
            |
            +--> 攻击前后标签统计 / visualization_samples
                    -> Visualization

重要：
- Step 1 的 DataDistributionResult 本身不需要因为标签翻转而重新做一遍数据分发；
- Step 2 只是在“客户端训练数据视图”上覆盖 label；
- 所以样本数量、样本索引、Train/Test 划分全部保持不变，
  唯一变化的是指定 Label Flipping 客户端训练样本返回的 label。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import random

from torch.utils.data import Dataset, Subset

from Step_00_V_byzantine_config import (
    AttackPlan,
    AttackType,
    ByzantineConfigGenerator,
)
from Step_01_data_distribution import (
    DataDistributionResult,
    FLDataDistributor,
)


# ============================================================
# 后端软锁定参数
# ============================================================

DEFAULT_PREVIEW_SAMPLES_PER_CLIENT = 10
LABEL_FLIPPING_VALUE = "label_flipping"


class LabelOverrideSubset(Dataset):
    """只覆盖当前客户端训练标签，不修改底层 Dataset。"""

    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        label_overrides: Mapping[int, int],
    ):
        self.dataset = dataset
        self.indices = list(indices)
        self.label_overrides = dict(label_overrides)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        original_index = self.indices[position]
        sample, original_label = self.dataset[original_index]
        new_label = self.label_overrides.get(original_index, int(original_label))
        return sample, new_label


@dataclass
class LabelChangeSample:
    """单条可视化标签变化记录。"""

    sample_index: int
    original_label: int
    flipped_label: int


@dataclass
class LabelFlippingResult:
    """Label Flipping 的统一输出。"""

    data_result: DataDistributionResult

    client_train_datasets: Dict[int, Dataset]
    client_test_datasets: Dict[int, Dataset]

    label_overrides: Dict[int, Dict[int, int]]
    attacked_clients: List[int]

    client_original_train_label_distribution: Dict[int, Dict[int, int]]
    client_attacked_train_label_distribution: Dict[int, Dict[int, int]]

    flipped_sample_counts: Dict[int, int]
    visualization_samples: Dict[int, List[LabelChangeSample]]

    @property
    def total_flipped_samples(self) -> int:
        return sum(self.flipped_sample_counts.values())


class LabelFlippingAttacker:
    """
    根据 Step 0 AttackPlan，对 Step 1 已经分发好的训练集执行 Label Flipping。

    当前攻击定义：
        对指定客户端训练集中的每一个样本，
        将原标签均匀随机替换为“任意一个不同于原标签的其他类别”。

    注意：
        当前工程采用逐样本随机错误标签，因此本模块不使用固定 label_flip_map。
        即使 Step 0 仍保留 label_flip_map 字段作为兼容占位，也不会采用固定映射。
    """

    def __init__(
        self,
        attack_plan: AttackPlan,
        data_result: DataDistributionResult,
        preview_samples_per_client: int = DEFAULT_PREVIEW_SAMPLES_PER_CLIENT,
    ):
        self.attack_plan = attack_plan
        self.data_result = data_result
        self.preview_samples_per_client = preview_samples_per_client

        self._validate_inputs()

        self.num_clients = int(self.attack_plan.num_clients)
        self.num_classes = self._infer_num_classes()

    def _validate_inputs(self) -> None:
        required_attack_plan_attrs = [
            "master_seed",
            "num_clients",
            "client_configs",
            "get",
        ]

        for attr in required_attack_plan_attrs:
            if not hasattr(self.attack_plan, attr):
                raise TypeError(f"attack_plan 缺少必要字段/方法：{attr}")

        required_data_result_attrs = [
            "full_dataset",
            "client_train_indices",
            "client_test_indices",
            "client_sizes",
            "client_train_sizes",
            "client_test_sizes",
        ]

        for attr in required_data_result_attrs:
            if not hasattr(self.data_result, attr):
                raise TypeError(f"data_result 缺少必要字段：{attr}")

        if int(self.attack_plan.num_clients) != len(self.data_result.client_train_indices):
            raise ValueError("Step 0 的 num_clients 与 Step 1 的客户端数量不一致")

        if self.preview_samples_per_client < 0:
            raise ValueError("preview_samples_per_client 必须 >= 0")

    def _get_targets(self):
        dataset = self.data_result.full_dataset
        if not hasattr(dataset, "targets"):
            raise TypeError("当前 full_dataset 不包含 targets 属性，无法直接读取原始标签。")
        return dataset.targets

    def _get_original_label(self, sample_index: int) -> int:
        targets = self._get_targets()
        return int(targets[sample_index])

    def _infer_num_classes(self) -> int:
        dataset = self.data_result.full_dataset

        if hasattr(dataset, "classes"):
            classes = dataset.classes
            if classes is not None and len(classes) > 1:
                return len(classes)

        targets = self._get_targets()
        target_values = targets.tolist() if hasattr(targets, "tolist") else list(targets)
        unique_labels = sorted({int(x) for x in target_values})

        if len(unique_labels) < 2:
            raise ValueError("Label Flipping 至少需要两个类别")

        expected = list(range(max(unique_labels) + 1))
        if unique_labels != expected:
            raise ValueError("当前标签不是连续的 0..C-1，本模块暂不自动推断这种标签编码。")

        return len(unique_labels)

    @staticmethod
    def _attack_type_value(client_config: Any) -> Optional[str]:
        attack_type = getattr(client_config, "attack_type", None)
        if attack_type is None:
            return None
        if hasattr(attack_type, "value"):
            return str(attack_type.value)
        return str(attack_type)

    def _is_label_flipping_client(self, client_id: int) -> bool:
        cfg = self.attack_plan.get(client_id)
        return (
            bool(getattr(cfg, "is_byzantine", False))
            and self._attack_type_value(cfg) == LABEL_FLIPPING_VALUE
        )

    def _derive_client_seed(self, client_id: int) -> int:
        """从 Step 0 master_seed 为每个客户端派生稳定随机种子。"""
        master_seed = int(self.attack_plan.master_seed)
        mixed = (
            (master_seed * 1_000_003)
            ^ (client_id * 97_409)
            ^ 0x4C4142454C
        )
        return mixed & 0xFFFFFFFF

    def _random_wrong_label(
        self,
        original_label: int,
        rng: random.Random,
    ) -> int:
        """从除原标签外的其他类别中均匀随机选择一个新标签。"""
        if not (0 <= original_label < self.num_classes):
            raise ValueError(
                f"原始标签 {original_label} 超出合法范围 [0, {self.num_classes - 1}]"
            )

        candidate = rng.randrange(self.num_classes - 1)
        if candidate >= original_label:
            candidate += 1
        return candidate

    def _attack_one_client(
        self,
        client_id: int,
    ) -> tuple[
        Dict[int, int],
        Dict[int, int],
        Dict[int, int],
        List[LabelChangeSample],
    ]:
        train_indices = self.data_result.client_train_indices[client_id]
        rng = random.Random(self._derive_client_seed(client_id))

        overrides: Dict[int, int] = {}
        original_counter: Counter[int] = Counter()
        attacked_counter: Counter[int] = Counter()
        preview: List[LabelChangeSample] = []

        for sample_index in train_indices:
            original_label = self._get_original_label(sample_index)
            flipped_label = self._random_wrong_label(original_label, rng)

            if flipped_label == original_label:
                raise RuntimeError("Label Flipping 内部错误：翻转后标签与原标签相同")

            overrides[sample_index] = flipped_label
            original_counter[original_label] += 1
            attacked_counter[flipped_label] += 1

            if len(preview) < self.preview_samples_per_client:
                preview.append(
                    LabelChangeSample(
                        sample_index=int(sample_index),
                        original_label=original_label,
                        flipped_label=flipped_label,
                    )
                )

        return (
            overrides,
            dict(sorted(original_counter.items())),
            dict(sorted(attacked_counter.items())),
            preview,
        )

    def apply(self) -> LabelFlippingResult:
        """执行本步骤并构造后续训练可直接使用的数据集。"""

        full_dataset = self.data_result.full_dataset

        client_train_datasets: Dict[int, Dataset] = {}
        client_test_datasets: Dict[int, Dataset] = {}

        label_overrides: Dict[int, Dict[int, int]] = {}
        attacked_clients: List[int] = []

        original_distributions: Dict[int, Dict[int, int]] = {}
        attacked_distributions: Dict[int, Dict[int, int]] = {}

        flipped_sample_counts: Dict[int, int] = {}
        visualization_samples: Dict[int, List[LabelChangeSample]] = {}

        for client_id in range(self.num_clients):
            train_indices = self.data_result.client_train_indices[client_id]
            test_indices = self.data_result.client_test_indices[client_id]

            # 所有客户端的 local test 始终保持原始标签。
            client_test_datasets[client_id] = Subset(full_dataset, test_indices)

            if not self._is_label_flipping_client(client_id):
                client_train_datasets[client_id] = Subset(full_dataset, train_indices)
                continue

            (
                client_overrides,
                original_distribution,
                attacked_distribution,
                preview,
            ) = self._attack_one_client(client_id)

            attacked_clients.append(client_id)
            label_overrides[client_id] = client_overrides
            original_distributions[client_id] = original_distribution
            attacked_distributions[client_id] = attacked_distribution
            flipped_sample_counts[client_id] = len(client_overrides)
            visualization_samples[client_id] = preview

            client_train_datasets[client_id] = LabelOverrideSubset(
                dataset=full_dataset,
                indices=train_indices,
                label_overrides=client_overrides,
            )

        return LabelFlippingResult(
            data_result=self.data_result,
            client_train_datasets=client_train_datasets,
            client_test_datasets=client_test_datasets,
            label_overrides=label_overrides,
            attacked_clients=attacked_clients,
            client_original_train_label_distribution=original_distributions,
            client_attacked_train_label_distribution=attacked_distributions,
            flipped_sample_counts=flipped_sample_counts,
            visualization_samples=visualization_samples,
        )


def apply_label_flipping(
    attack_plan: AttackPlan,
    data_result: DataDistributionResult,
    preview_samples_per_client: int = DEFAULT_PREVIEW_SAMPLES_PER_CLIENT,
) -> LabelFlippingResult:
    """本模块推荐的正式对外入口。"""
    attacker = LabelFlippingAttacker(
        attack_plan=attack_plan,
        data_result=data_result,
        preview_samples_per_client=preview_samples_per_client,
    )
    return attacker.apply()


def print_label_flipping_result(result: LabelFlippingResult) -> None:
    """控制台输出，便于裸跑测试。"""
    print("\n" + "=" * 78)
    print("Step 2 - Label Flipping Attack Result")
    print("=" * 78)

    print(f"Attacked Clients      : {result.attacked_clients}")
    print(f"Total Flipped Samples : {result.total_flipped_samples}")

    if not result.attacked_clients:
        print("当前 AttackPlan 中没有 Label Flipping 客户端。")
        return

    for client_id in result.attacked_clients:
        print("\n" + "-" * 78)
        print(f"Client {client_id:02d}")
        print("-" * 78)

        train_size = result.data_result.client_train_sizes[client_id]
        flipped_count = result.flipped_sample_counts[client_id]
        coverage = flipped_count / train_size if train_size > 0 else 0.0

        print(f"Train Samples   : {train_size}")
        print(f"Flipped Samples : {flipped_count}")
        print(f"Attack Coverage : {coverage:.2%}")
        print(
            "Before Distribution:",
            result.client_original_train_label_distribution[client_id],
        )
        print(
            "After Distribution :",
            result.client_attacked_train_label_distribution[client_id],
        )

        print("Preview:")
        for item in result.visualization_samples[client_id]:
            print(
                f"  sample_index={item.sample_index} | "
                f"{item.original_label} -> {item.flipped_label}"
            )


# ============================================================
# 这个是测试样例
# ============================================================

if __name__ == "__main__":
    # ========================================================
    # 独立运行测试
    # ========================================================
    # 目的：
    # 1. 裸跑 Step 0，生成 AttackPlan；
    # 2. 裸跑 Step 1，生成 DataDistributionResult；
    # 3. 本文件执行真正的 Label Flipping；
    # 4. 检查：
    #       - 只有 Label Flipping 客户端被修改；
    #       - 这些客户端训练集中的每一个样本标签都被替换为其他标签；
    #       - local test 完全不修改；
    #       - full_dataset.targets 完全不修改；
    #       - 相同 AttackPlan 下攻击结果可以复现。
    #
    # 注意：
    # 这只是本文件的独立测试入口，不属于正式完整 FL 流程。
    # 后续正式系统应由 main.py 按
    # step_00 -> step_01 -> step_02 -> ...
    # 的顺序统一调用。
    # ========================================================

    # --------------------------------------------------------
    # Step 0：为了测试，强制全部 Byzantine 都采用 Label Flipping
    # --------------------------------------------------------

    generator = ByzantineConfigGenerator(
        num_clients=5,
        byzantine_ratio=0.40,

        attack_type_weights={
            AttackType.LABEL_FLIPPING: 1,
            AttackType.GAUSSIAN: 0,
            AttackType.SIGN_FLIPPING: 0,
        },

        # Label Flipping 不使用共享层攻击比例，
        # 但 Step 0 构造器仍保留这一统一配置字段。
        layer_attack_ratio=0.50,

        gaussian_mean=0.0,
        gaussian_std=1.0,
        sign_flip_scale=-1.0,

        # 当前 Step 2 的正式攻击逻辑是：
        # “每个训练样本随机替换为任意一个不同于原标签的其他标签”。
        # 因此固定 label_flip_map 不参与本步骤，保持 None。
        label_flip_map=None,

        master_seed=42,
    )

    attack_plan = generator.generate()

    # --------------------------------------------------------
    # Step 1：数据分发
    # --------------------------------------------------------

    distributor = FLDataDistributor(
        dataset_name="cifar10",
        num_clients=attack_plan.num_clients,
        label_heterogeneity=0.5,
        volume_heterogeneity=1.0,
        data_pool_ratio=1.0,
        global_test_ratio=0.1,
        local_test_ratio=0.2,
        min_samples_per_client=30,
        min_labels_per_client=1,
        max_labels_per_client=None,

        # 独立测试中直接沿用 master_seed。
        # 正式 main.py 后续可以统一派生 data_seed。
        seed=attack_plan.master_seed,
    )

    data_result = distributor.distribute()

    # --------------------------------------------------------
    # 记录攻击前 full_dataset.targets
    # 用于检查 Step 2 没有原地修改底层数据集
    # --------------------------------------------------------

    original_targets = [
        int(x)
        for x in data_result.full_dataset.targets
    ]

    # --------------------------------------------------------
    # Step 2：执行 Label Flipping
    # --------------------------------------------------------

    label_result = apply_label_flipping(
        attack_plan=attack_plan,
        data_result=data_result,
        preview_samples_per_client=10,
    )

    print_label_flipping_result(label_result)

    # --------------------------------------------------------
    # 自动检查 1：底层 full_dataset.targets 没有被修改
    # --------------------------------------------------------

    current_targets = [
        int(x)
        for x in data_result.full_dataset.targets
    ]

    print("\n" + "=" * 78)
    print("Automatic Checks")
    print("=" * 78)

    print(
        "full_dataset.targets 未被原地修改 :",
        original_targets == current_targets,
    )

    # --------------------------------------------------------
    # 自动检查 2：所有 Label Flipping 样本都真正变成其他标签
    # --------------------------------------------------------

    all_flipped_are_different = True

    for client_id in label_result.attacked_clients:
        for sample_index, flipped_label in (
            label_result.label_overrides[client_id].items()
        ):
            original_label = int(
                data_result.full_dataset.targets[sample_index]
            )

            if original_label == flipped_label:
                all_flipped_are_different = False
                break

        if not all_flipped_are_different:
            break

    print(
        "所有受攻击训练样本均被翻转为其他标签 :",
        all_flipped_are_different,
    )

    # --------------------------------------------------------
    # 自动检查 3：翻转数量 == 对应客户端训练样本数
    # --------------------------------------------------------

    all_train_samples_flipped = all(
        label_result.flipped_sample_counts[client_id]
        == data_result.client_train_sizes[client_id]
        for client_id in label_result.attacked_clients
    )

    print(
        "Label Flipping 客户端训练样本攻击覆盖率均为 100% :",
        all_train_samples_flipped,
    )

    # --------------------------------------------------------
    # 自动检查 4：local test 始终保持原始标签
    # --------------------------------------------------------

    local_test_untouched = True

    for client_id in range(attack_plan.num_clients):
        test_dataset = label_result.client_test_datasets[client_id]
        test_indices = data_result.client_test_indices[client_id]

        # 独立测试只抽前 20 条检查即可。
        check_count = min(20, len(test_dataset))

        for position in range(check_count):
            _, returned_label = test_dataset[position]
            original_index = test_indices[position]
            original_label = int(
                data_result.full_dataset.targets[original_index]
            )

            if int(returned_label) != original_label:
                local_test_untouched = False
                break

        if not local_test_untouched:
            break

    print(
        "客户端 Local Test 标签保持不变 :",
        local_test_untouched,
    )

    # --------------------------------------------------------
    # 自动检查 5：相同 AttackPlan 重跑时攻击结果保持一致
    # --------------------------------------------------------

    label_result_repeat = apply_label_flipping(
        attack_plan=attack_plan,
        data_result=data_result,
        preview_samples_per_client=10,
    )

    reproducible = (
        label_result.label_overrides
        == label_result_repeat.label_overrides
    )

    print(
        "相同 AttackPlan 下 Label Flipping 可复现 :",
        reproducible,
    )

