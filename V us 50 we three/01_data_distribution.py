"""
01_data_distribution.py

# 虽然说这种不带V的步骤不需要可视化输出，但还是建议在可视化系统之外找个地方展示结果
# 这样子方便验证下代码是不是有问题

# 经过测试，发现了 “学习了没考” 的问题，即训练集中的部分标签，测试集里没有。但是不碍事，工程上是允许的

联邦学习数据分发模块。

职责：
1. 根据 dataset_name 加载指定数据集；
2. 从原始训练集划出全局测试集和客户端数据池；
3. 根据客户端数量、标签异构参数和数据量异构参数，为每个客户端分配本地数据；
4. 保证同一客户端内部 train/test 不重叠；
5. 保证同一客户端的 train/test 使用相同的标签集合，但标签比例可以不同；
6. 允许不同客户端之间持有相同样本（“复制”而不是从公共池中永久拿走）；
7. 通过 seed 控制整套随机分发过程，保证实验可复现；
8. 输出客户端实际持有的数据索引，以及 client_sizes 等后续模块所需元信息。

明确不负责：
- 模型/网络架构；
- 共享层数分配；
- 拜占庭客户端选择与攻击；
- 参数检测；
- 参数聚合；
- 可视化 / demo mode。

输入：     带 # 的为需要读取作为输入的内容

# - dataset_name：数据集名称
    （目前支持 CIFAR-10 和 MNIST。
     如果比赛 Demo 最终统一使用 CIFAR-10，也可以前端默认选 CIFAR-10。）

# - num_clients：客户端总数
    （直接继承第 0 步的 num_clients，不应该让用户在第 1 步重新输入。
     即：第 0 步输入一次，后续全流程复用。）

# - label_heterogeneity：标签异构程度
    （Dirichlet 参数 α。
     数值越小，客户端之间的标签分布差异通常越明显。
     论文主要使用 α=0.5、1.0，建议可视化只提供这两个选项。）

# - volume_heterogeneity：数据量异构程度
    （幂律分布参数 β。
     β 越大，各客户端数据量差异越明显。
     论文实验中使用 β=1.0，可根据最终 Demo 需要决定是否允许修改。）

# - seed：
    （数据分发随机种子。
     完整工程中不建议用户单独输入一个新的 data seed，
     而应由 master_seed 统一派生/传入。
     因此页面实际可以继续只显示第 0 步的 master_seed。）

- data_pool_ratio：客户端数据池使用比例
    （建议作为后端实验参数，不需要在正式可视化页面暴露。）

- global_test_ratio：全局测试集比例
    （建议后端固定，例如 0.1，不需要作为可视化输入。）

- local_test_ratio：客户端本地测试集比例
    （建议后端固定，例如 0.2，不需要作为可视化输入。）

- min_samples_per_client：每个客户端最低样本量约束
    （后端内部约束，不需要可视化输入。）

- min_labels_per_client / max_labels_per_client：
    （客户端可拥有标签种类数约束，属于后端内部数据分发规则，
     不需要可视化输入。）

- data_dir：本地数据集存储路径
    （纯后端参数，不需要可视化。）

"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np
import random
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset

# ============================================================
# 全局默认随机种子
# ============================================================
# 只要输入参数和 seed 相同，数据划分结果就应保持一致。
# 需要生成另一套数据分布时，只需修改 seed。
DEFAULT_SEED = 42


@dataclass
class DataDistributionResult:
    """
    数据分发结果。

    这个对象是 01_data_distribution.py 与后续模块之间的主要接口。

    其中：
    - client_train_indices / client_test_indices:
      供后续 client_training.py 真正构造各客户端本地数据集；
    - client_sizes:
      供后续 architecture_allocation.py 根据数据量决定共享层数；
    - client_label_distribution_*:
      主要用于日志、调试和实验统计，不参与核心训练逻辑。
    """

    dataset_name: str
    seed: int
    full_dataset: Dataset
    global_test_dataset: Dataset

    client_train_indices: Dict[int, List[int]]
    client_test_indices: Dict[int, List[int]]

    # 每个客户端实际拥有的总样本量 = train + local test
    # 列表下标就是 client_id，因此长度必须始终等于 num_clients
    client_sizes: List[int]

    client_train_sizes: List[int]
    client_test_sizes: List[int]

    client_label_distribution_train: Dict[int, Dict[int, int]]
    client_label_distribution_test: Dict[int, Dict[int, int]]

    global_test_indices: List[int]
    data_pool_indices: List[int]


class FLDataDistributor:
    """
    联邦学习数据分发器。

    核心思想：
    1. 先从原始数据集中划出 global test；
    2. 再从剩余数据中选择一定比例构成 client data pool；
    3. 根据标签异构参数，为每个客户端确定其偏好的标签集合/标签权重；
    4. 根据数据量异构参数，确定每个客户端应持有多少样本；
    5. 从公共 data pool 中为各客户端“复制”样本。

    重要：
    - 不同客户端之间允许样本重叠；
    - 同一个客户端自己的训练集和本地测试集不允许重叠。
    """

    def __init__(
        self,
        dataset_name: str = "cifar10",
        num_clients: int = 30,
        label_heterogeneity: float = 0.5,
        volume_heterogeneity: float = 1.0,
        data_pool_ratio: float = 1.0,
        global_test_ratio: float = 0.1,
        local_test_ratio: float = 0.2,
        min_samples_per_client: int = 30,
        min_labels_per_client: int = 1,
        max_labels_per_client: int | None = None,
        data_dir: str = "./data",
        seed: int = DEFAULT_SEED,
    ):
        """
        参数说明
        ----------
        dataset_name:
            数据集名称，目前支持 "cifar10" 和 "mnist"。

        num_clients:
            客户端总数。

        label_heterogeneity:
            标签异构参数。
            数值越小，Dirichlet 分布通常越尖锐，各客户端标签偏好越明显。
            必须 > 0。

        volume_heterogeneity:
            数据量异构参数。
            使用幂律权重 1 / rank^beta。
            beta = 0 时各客户端理论数据量近似相同；
            beta 越大，客户端数据量差距越明显。
            必须 >= 0。

        data_pool_ratio:
            在扣除 global test 后，剩余数据中有多少比例用于客户端数据池。
            取值范围 (0, 1]。
            例如 0.8 表示只使用剩余样本的 80% 进行客户端分发。

        global_test_ratio:
            原始训练数据中预留为全局测试集的比例。
            这部分样本不会再进入客户端数据池。

        local_test_ratio:
            每个客户端拿到的本地数据中，有多少比例用于该客户端本地测试。
            其余用于本地训练。

        min_samples_per_client:
            希望每个客户端至少拥有的理论样本量。
            若数据池规模和客户端数量使该约束根本无法满足，会自动降低到可行范围。

        min_labels_per_client / max_labels_per_client:
            对每个客户端可持有标签种类数的约束。

        seed:
            数据分发随机种子。
            它统一控制本模块中的随机过程，包括：
            - global test / client data pool 的随机划分；
            - 每个客户端的标签偏好生成；
            - 不同标签下具体样本的随机抽取；
            - train/test 的标签比例随机生成。
            相同配置 + 相同 seed => 尽量得到完全相同的数据分发结果；
            修改 seed => 生成另一套随机数据分布。
        """
        self.dataset_name = dataset_name.lower()
        self.num_clients = num_clients
        self.label_heterogeneity = label_heterogeneity
        self.volume_heterogeneity = volume_heterogeneity
        self.data_pool_ratio = data_pool_ratio
        self.global_test_ratio = global_test_ratio
        self.local_test_ratio = local_test_ratio
        self.min_samples_per_client = min_samples_per_client
        self.min_labels_per_client = min_labels_per_client
        self.max_labels_per_client = max_labels_per_client
        self.data_dir = data_dir
        self.seed = seed

        self._validate_config()

        # ============================================================
        # 本模块所有随机行为都从这里读取 seed
        # ============================================================
        # 不直接调用 random.seed()/np.random.seed()，避免影响项目其他模块。
        # Python 随机过程（shuffle/sample 等）使用 self.py_rng；
        # NumPy 随机过程（Dirichlet 等）使用 self.np_rng。
        self.py_rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)

        self.full_dataset: Dataset | None = None
        self.targets: np.ndarray | None = None
        self.num_classes: int | None = None

        self.global_test_indices: List[int] = []
        self.data_pool_indices: List[int] = []

    def _validate_config(self) -> None:
        """检查输入配置，尽早发现明显错误。"""
        if self.num_clients <= 0:
            raise ValueError("num_clients 必须 > 0")

        if self.label_heterogeneity <= 0:
            raise ValueError("label_heterogeneity 必须 > 0")

        if self.volume_heterogeneity < 0:
            raise ValueError("volume_heterogeneity 必须 >= 0")

        if not (0 < self.data_pool_ratio <= 1):
            raise ValueError("data_pool_ratio 必须位于 (0, 1]")

        if not (0 <= self.global_test_ratio < 1):
            raise ValueError("global_test_ratio 必须位于 [0, 1)")

        if not (0 <= self.local_test_ratio < 1):
            raise ValueError("local_test_ratio 必须位于 [0, 1)")

        if self.min_samples_per_client < 1:
            raise ValueError("min_samples_per_client 必须 >= 1")

        if self.min_labels_per_client < 1:
            raise ValueError("min_labels_per_client 必须 >= 1")

    def _load_dataset(self) -> None:
        """
        加载原始数据集。

        注意：
        本模块只关心“样本 + 标签”，不关心后续采用什么模型。
        因此 CIFAR-10 与某个特定 CNN 架构在这里不再绑定。
        """
        if self.dataset_name == "cifar10":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ])

            self.full_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                download=True,
                transform=transform,
            )
            self.num_classes = 10
            self.targets = np.asarray(self.full_dataset.targets)

        elif self.dataset_name == "mnist":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])

            self.full_dataset = torchvision.datasets.MNIST(
                root=self.data_dir,
                train=True,
                download=True,
                transform=transform,
            )
            self.num_classes = 10
            self.targets = np.asarray(self.full_dataset.targets)

        else:
            raise ValueError(
                f"暂不支持数据集 {self.dataset_name!r}，"
                '当前支持 "cifar10" 和 "mnist"'
            )

    def _build_data_pool(self) -> None:
        """
        从原始训练集构造：
        1. global test；
        2. client data pool。

        global test 与 client data pool 完全不重叠。
        data_pool_ratio < 1 时，剩余未被选中的样本暂时不用。
        """
        assert self.full_dataset is not None

        all_indices = list(range(len(self.full_dataset)))
        self.py_rng.shuffle(all_indices)

        global_test_size = int(len(all_indices) * self.global_test_ratio)

        self.global_test_indices = all_indices[:global_test_size]

        remaining_indices = all_indices[global_test_size:]
        pool_size = int(len(remaining_indices) * self.data_pool_ratio)

        if pool_size <= 0:
            raise ValueError("当前比例设置导致客户端数据池为空")

        self.data_pool_indices = remaining_indices[:pool_size]

    def _generate_client_label_preferences(
        self,
    ) -> Dict[int, np.ndarray]:
        """
        为每个客户端生成标签偏好概率。

        原代码的思路是：
        先通过 Dirichlet 得到标签偏好，
        再根据阈值筛出客户端主要拥有的标签。

        这里保留这一思想，但返回完整概率向量，
        后续真正分样本时直接根据该向量决定各标签应分多少数据。
        """
        assert self.num_classes is not None

        max_labels = (
            self.num_classes
            if self.max_labels_per_client is None
            else min(self.max_labels_per_client, self.num_classes)
        )

        if self.min_labels_per_client > max_labels:
            raise ValueError("min_labels_per_client 不能大于 max_labels_per_client")

        preferences: Dict[int, np.ndarray] = {}
        threshold = (1.0 / self.num_classes) * 0.3

        for client_id in range(self.num_clients):
            probs = self.np_rng.dirichlet(
                np.full(self.num_classes, self.label_heterogeneity)
            )

            selected = np.where(probs > threshold)[0].tolist()

            # 标签过少时，按原始 Dirichlet 概率从高到低补足。
            if len(selected) < self.min_labels_per_client:
                ranked = np.argsort(probs)[::-1].tolist()
                for label in ranked:
                    if label not in selected:
                        selected.append(label)
                    if len(selected) >= self.min_labels_per_client:
                        break

            # 标签过多时，保留概率最高的若干标签。
            if len(selected) > max_labels:
                selected = sorted(
                    selected,
                    key=lambda label: probs[label],
                    reverse=True,
                )[:max_labels]

            selected = sorted(selected)

            # 未被选中的标签概率置 0，再重新归一化。
            masked_probs = np.zeros(self.num_classes, dtype=float)
            masked_probs[selected] = probs[selected]
            masked_probs /= masked_probs.sum()

            preferences[client_id] = masked_probs

        return preferences

    def _generate_client_sizes(self) -> List[int]:
        """
        根据幂律分布生成各客户端理论数据量。

        返回长度恒等于 num_clients 的列表：
            client_sizes[client_id] = 该客户端理论总样本数

        这里仍然让所有客户端理论样本量之和约等于 data pool 大小，
        作为实验规模控制。

        注意：
        后续由于不同客户端允许复制同一个公共样本，
        “客户端持有样本总数之和”并不意味着这些样本在全局上互不重复。
        """
        pool_size = len(self.data_pool_indices)

        ranks = np.arange(1, self.num_clients + 1, dtype=float)

        if self.volume_heterogeneity == 0:
            weights = np.ones(self.num_clients, dtype=float)
        else:
            weights = 1.0 / np.power(ranks, self.volume_heterogeneity)

        weights /= weights.sum()

        # 先通过最大余数法得到总和严格等于 pool_size 的整数分配。
        raw_sizes = weights * pool_size
        sizes = np.floor(raw_sizes).astype(int)
        remainder = pool_size - int(sizes.sum())

        if remainder > 0:
            fractional_order = np.argsort(raw_sizes - sizes)[::-1]
            for idx in fractional_order[:remainder]:
                sizes[idx] += 1

        # min_samples_per_client 是一个约束目标。
        # 如果 data pool 太小，不可能让每个客户端都达到该值，则不强行制造非法结果。
        feasible_min = min(
            self.min_samples_per_client,
            pool_size // self.num_clients,
        )

        if feasible_min > 0:
            for client_id in range(self.num_clients):
                deficit = feasible_min - sizes[client_id]
                if deficit <= 0:
                    continue

                # 从当前样本数最多的客户端逐步转移额度。
                while deficit > 0:
                    donor = int(np.argmax(sizes))
                    transferable = sizes[donor] - feasible_min
                    if transferable <= 0:
                        break

                    move = min(deficit, transferable)
                    sizes[donor] -= move
                    sizes[client_id] += move
                    deficit -= move

        return sizes.astype(int).tolist()

    @staticmethod
    def _integer_allocation(total: int, probabilities: np.ndarray) -> np.ndarray:
        """
        按概率将 total 个样本额度分配给不同标签。

        使用最大余数法，保证最终整数数量之和严格等于 total。
        """
        if total <= 0:
            return np.zeros_like(probabilities, dtype=int)

        raw = probabilities * total
        counts = np.floor(raw).astype(int)
        remainder = total - int(counts.sum())

        if remainder > 0:
            order = np.argsort(raw - counts)[::-1]
            for idx in order[:remainder]:
                counts[idx] += 1

        return counts

    @staticmethod
    def _integer_allocation_with_coverage(
        total: int,
        probabilities: np.ndarray,
        active_labels: List[int],
    ) -> np.ndarray:
        """
        在尽量遵循 probabilities 的同时，尽量保证 active_labels 都出现。

        设计目标：
        - 如果 total >= 标签种类数，则每个 active label 至少分到 1 个样本；
        - 剩余样本再按给定概率分配；
        - 如果 total < 标签种类数，则数学上不可能覆盖全部标签，
          此时只能优先覆盖概率较高的标签。

        这个函数用于保证：
            同一客户端 train/test 的“标签集合”尽量完全一致，
            但两者标签比例可以不同。
        """
        counts = np.zeros_like(probabilities, dtype=int)

        if total <= 0 or len(active_labels) == 0:
            return counts

        active_labels = list(active_labels)

        # 样本数足够时：先给每个标签 1 个样本，确保标签覆盖。
        if total >= len(active_labels):
            counts[active_labels] = 1
            remaining = total - len(active_labels)

            if remaining > 0:
                active_probs = probabilities[active_labels].astype(float)
                active_probs = active_probs / active_probs.sum()

                extra = FLDataDistributor._integer_allocation(
                    remaining,
                    active_probs,
                )

                for pos, label in enumerate(active_labels):
                    counts[label] += int(extra[pos])

            return counts

        # 样本数比标签种类数还少时，无法全部覆盖。
        # 工程上优先选择概率最大的 total 个标签。
        ranked_labels = sorted(
            active_labels,
            key=lambda label: probabilities[label],
            reverse=True,
        )
        for label in ranked_labels[:total]:
            counts[label] = 1

        return counts

    def _generate_independent_probs_for_labels(
        self,
        active_labels: List[int],
    ) -> np.ndarray:
        """
        针对已经确定好的标签集合，重新生成一套独立的标签比例。

        用途：
        - train 与 test 共享相同的标签集合 active_labels；
        - 但 test 不直接复用 train 的比例；
        - 因此两者可以有不同的标签占比。

        例如：
            train: label 0/7/4 = 0.2 / 0.5 / 0.3
            test : label 0/7/4 = 0.4 / 0.2 / 0.4

        二者标签集合相同，但比例不同。
        """
        assert self.num_classes is not None

        probs = np.zeros(self.num_classes, dtype=float)

        if len(active_labels) == 0:
            return probs

        local_probs = self.np_rng.dirichlet(
            np.full(len(active_labels), self.label_heterogeneity)
        )

        for pos, label in enumerate(active_labels):
            probs[label] = local_probs[pos]

        return probs

    def _sample_for_one_client(
        self,
        total_size: int,
        train_label_probs: np.ndarray,
        label_to_pool_indices: Dict[int, List[int]],
    ) -> tuple[List[int], List[int]]:
        """
        为单个客户端复制样本。

        核心规则：
        1. 不同客户端之间允许使用同一原始样本；
        2. 同一客户端内部 train 和 local test 严格互斥；
        3. 同一客户端的 train/test 使用相同的标签集合；
        4. train/test 的标签比例不要求相同；
        5. 如果 train 或 test 的样本量小于标签种类数，
           则数学上无法保证全部标签都出现，此时只能尽量覆盖。

        参数
        ----------
        total_size:
            该客户端理论总样本数。

        train_label_probs:
            该客户端训练集所使用的标签偏好概率。
            其中非零位置决定该客户端允许拥有的标签集合。

        label_to_pool_indices:
            公共数据池中的“标签 -> 样本索引列表”映射。
        """
        if total_size <= 0:
            return [], []

        local_test_size = int(total_size * self.local_test_ratio)
        train_size = total_size - local_test_size

        # 如果启用了 local test，且客户端至少有两个样本，则尽量保证 train/test 都非空。
        if self.local_test_ratio > 0 and total_size >= 2:
            local_test_size = max(1, local_test_size)
            train_size = total_size - local_test_size

            if train_size == 0:
                train_size = 1
                local_test_size = total_size - 1

        # train/test 必须共享同一个标签集合。
        active_labels = np.where(train_label_probs > 0)[0].tolist()

        # 测试集重新生成独立比例：
        # 标签集合与 train 一样，但各标签比例可以完全不同。
        test_label_probs = self._generate_independent_probs_for_labels(
            active_labels
        )

        # 分别生成 train / test 的各标签样本额度。
        # 如果样本数足够，则保证 active_labels 中每个标签至少出现一次。
        train_label_counts = self._integer_allocation_with_coverage(
            train_size,
            train_label_probs,
            active_labels,
        )
        test_label_counts = self._integer_allocation_with_coverage(
            local_test_size,
            test_label_probs,
            active_labels,
        )

        selected_train: List[int] = []
        selected_test: List[int] = []

        # 只记录“当前客户端”已经使用过的样本。
        #
        # 因此：
        # - Client A 用过的样本，Client B 依然可以再次使用；
        # - 但 Client A 自己的 train/test 不允许重复使用同一个样本。
        used_by_this_client = set()

        def draw_samples(
            requested_counts: np.ndarray,
            target_list: List[int],
        ) -> None:
            """
            根据每个标签所需的样本数，从公共池中抽取实际样本。

            注意：
            这里不会从全局公共池中删除样本，
            因此天然支持不同客户端之间的数据重复。
            """
            for label, requested in enumerate(requested_counts):
                if requested <= 0:
                    continue

                candidates = [
                    idx
                    for idx in label_to_pool_indices[label]
                    if idx not in used_by_this_client
                ]

                take = min(int(requested), len(candidates))
                if take <= 0:
                    continue

                chosen = self.py_rng.sample(candidates, take)

                target_list.extend(chosen)
                used_by_this_client.update(chosen)

        # 先抽训练集，再抽本地测试集。
        draw_samples(train_label_counts, selected_train)
        draw_samples(test_label_counts, selected_test)

        # ============================================================
        # 样本不足时的工程性补齐
        # ============================================================
        # 某个标签在公共池中的样本数量可能不足，
        # 因此实际拿到的数据量有可能小于理论额度。
        #
        # 此时只从该客户端允许的 active_labels 中继续补齐，
        # 不引入新的标签，同时仍保持当前客户端内部不重复。
        desired_total = train_size + local_test_size
        current_total = len(selected_train) + len(selected_test)
        missing = desired_total - current_total

        if missing > 0:
            fallback_candidates: List[int] = []

            for label in active_labels:
                fallback_candidates.extend(
                    idx
                    for idx in label_to_pool_indices[label]
                    if idx not in used_by_this_client
                )

            self.py_rng.shuffle(fallback_candidates)

            # 优先补足训练集。
            train_missing = max(0, train_size - len(selected_train))
            take_train = min(train_missing, len(fallback_candidates))

            selected_train.extend(fallback_candidates[:take_train])
            used_by_this_client.update(fallback_candidates[:take_train])

            fallback_candidates = fallback_candidates[take_train:]

            # 再补足测试集。
            test_missing = max(0, local_test_size - len(selected_test))
            take_test = min(test_missing, len(fallback_candidates))

            selected_test.extend(fallback_candidates[:take_test])
            used_by_this_client.update(fallback_candidates[:take_test])

        return selected_train, selected_test

    def distribute(self) -> DataDistributionResult:
        """
        执行完整数据分发，并返回标准结果对象。

        这个函数是本模块对外最主要的入口。
        """
        self._load_dataset()
        self._build_data_pool()

        assert self.full_dataset is not None
        assert self.targets is not None
        assert self.num_classes is not None

        label_preferences = self._generate_client_label_preferences()
        theoretical_client_sizes = self._generate_client_sizes()

        # 建立“标签 -> 可用公共样本索引”的映射。
        label_to_pool_indices: Dict[int, List[int]] = defaultdict(list)
        for idx in self.data_pool_indices:
            label = int(self.targets[idx])
            label_to_pool_indices[label].append(idx)

        client_train_indices: Dict[int, List[int]] = {}
        client_test_indices: Dict[int, List[int]] = {}

        client_train_sizes: List[int] = []
        client_test_sizes: List[int] = []
        client_sizes: List[int] = []

        label_dist_train: Dict[int, Dict[int, int]] = {}
        label_dist_test: Dict[int, Dict[int, int]] = {}

        for client_id in range(self.num_clients):
            train_indices, test_indices = self._sample_for_one_client(
                total_size=theoretical_client_sizes[client_id],
                train_label_probs=label_preferences[client_id],
                label_to_pool_indices=label_to_pool_indices,
            )

            client_train_indices[client_id] = train_indices
            client_test_indices[client_id] = test_indices

            train_size = len(train_indices)
            test_size = len(test_indices)

            client_train_sizes.append(train_size)
            client_test_sizes.append(test_size)
            client_sizes.append(train_size + test_size)

            label_dist_train[client_id] = dict(
                Counter(int(self.targets[idx]) for idx in train_indices)
            )
            label_dist_test[client_id] = dict(
                Counter(int(self.targets[idx]) for idx in test_indices)
            )

        global_test_dataset = Subset(
            self.full_dataset,
            self.global_test_indices,
        )

        return DataDistributionResult(
            dataset_name=self.dataset_name,
            seed=self.seed,
            full_dataset=self.full_dataset,
            global_test_dataset=global_test_dataset,
            client_train_indices=client_train_indices,
            client_test_indices=client_test_indices,
            client_sizes=client_sizes,
            client_train_sizes=client_train_sizes,
            client_test_sizes=client_test_sizes,
            client_label_distribution_train=label_dist_train,
            client_label_distribution_test=label_dist_test,
            global_test_indices=self.global_test_indices.copy(),
            data_pool_indices=self.data_pool_indices.copy(),
        )


def get_client_train_dataset(
    result: DataDistributionResult,
    client_id: int,
) -> Dataset:
    """
    根据分发结果取得某个客户端的本地训练集。

    后续 client_training.py 可以直接调用这个辅助函数，
    不需要知道数据分发算法内部如何实现。
    """
    if client_id not in result.client_train_indices:
        raise ValueError(f"客户端 {client_id} 不存在")

    return Subset(
        result.full_dataset,
        result.client_train_indices[client_id],
    )


def get_client_test_dataset(
    result: DataDistributionResult,
    client_id: int,
) -> Dataset:
    """根据分发结果取得某个客户端的本地测试集。"""
    if client_id not in result.client_test_indices:
        raise ValueError(f"客户端 {client_id} 不存在")

    return Subset(
        result.full_dataset,
        result.client_test_indices[client_id],
    )



# 这个是测试样例
if __name__ == "__main__":
    # ============================================================
    # 独立运行测试
    # ============================================================
    # 这里的参数可以直接修改，用于快速检查数据分发效果。
    #
    # 注意：
    # 这只是本文件的独立测试入口，不属于正式联邦学习流程。
    # 后续完整系统运行时，应由 config.py / main.py 统一传入参数。
    # ============================================================

    distributor = FLDataDistributor(
        dataset_name="cifar10",
        num_clients=5,
        label_heterogeneity=0.5,
        volume_heterogeneity=1.0,
        data_pool_ratio=1.0,
        global_test_ratio=0.1,
        local_test_ratio=0.2,
        min_samples_per_client=30,
        min_labels_per_client=1,
        max_labels_per_client=None,
        seed=42,                       # 改这里即可测试不同随机分发
    )

    result = distributor.distribute()

    print("\n========== 数据分发结果 ==========")
    print(f"Dataset: {result.dataset_name}")
    print(f"Seed: {result.seed}")
    print(f"Client 数量: {len(result.client_sizes)}")
    print(f"Global Test Size: {len(result.global_test_indices)}")
    print(f"Client Data Pool Size: {len(result.data_pool_indices)}")

    for client_id in range(len(result.client_sizes)):
        print(f"\n---------- Client {client_id} ----------")

        print(
            f"总数据量: {result.client_sizes[client_id]} "
            f"(Train={result.client_train_sizes[client_id]}, "
            f"Test={result.client_test_sizes[client_id]})"
        )

        print(
            "Train 标签分布:",
            result.client_label_distribution_train[client_id]
        )

        print(
            "Test 标签分布:",
            result.client_label_distribution_test[client_id]
        )

        # 检查同一客户端 train/test 是否存在数据重叠
        train_set = set(result.client_train_indices[client_id])
        test_set = set(result.client_test_indices[client_id])

        overlap = train_set & test_set

        print(f"Train/Test 重叠样本数: {len(overlap)}")

        # 检查 test 中是否出现 train 未见过的新标签
        train_labels = set(
            result.client_label_distribution_train[client_id].keys()
        )
        test_labels = set(
            result.client_label_distribution_test[client_id].keys()
        )

        print(f"Train 标签集合: {sorted(train_labels)}")
        print(f"Test 标签集合 : {sorted(test_labels)}")
        print(
            f"Test 标签均已在 Train 中出现 : "
            f"{test_labels.issubset(train_labels)}"
        )