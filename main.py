"""
main.py

FedCALI / Competition Demo 主流程控制器
===============================================================================

本文件只负责“连接已有 Step 00~09”，不重新实现任何步骤内部算法。

项目文件名必须与 PyCharm 工程保持一致：

    Step_00_V_byzantine_config.py
    Step_01_data_distribution.py
    Step_02_V_label_flipping_attack.py
    Step_03_V_architecture_allocation.py
    Step_03_V_sp_consistent_architecture.py
    Step_04_local_model_trainning.py
    Step_05_V_gaussian_and_sign_flipping_attack.py
    Step_06_V_masking_and_uploading.py
    Step_06_V_sp_uploading.py
    Step_07_V_byzantine_final.py
    Step_07_V_sp_byzantine.py
    Step_08_V_robust_aggregation.py
    Step_09_V_benign_local_accuracy.py

正式流程：

    Step 00 -> Step 01 -> Step 02 -> Step 03 / Step 03-SP
                                      |
                                      v
                 +---------------- T Rounds ----------------+
                 |                                          |
                 | Step 04 -> 05 -> 06 -> 07 -> 08 -> 09   |
                 |    ^                              |       |
                 |    +---- shared/client state -----+       |
                 +------------------------------------------+

注意：
1. Step 00~03 只执行一次。
2. Step 02 与 Step 05 不是“整个实验二选一”：
   - Step 02 只对 Label-Flipping 客户端修改训练标签；
   - Step 05 每轮只真正修改 Gaussian / Sign-Flipping 客户端参数；
   - 二者共同服从同一份 Step 00 AttackPlan。
3. Step 03 与 Step 03-SP 二选一。
4. 安全分支必须成对：
       FULL: Step_06_V_masking_and_uploading
             -> Step_07_V_byzantine_final
       SP:   Step_06_V_sp_uploading
             -> Step_07_V_sp_byzantine
5. Step 08 的 robust_aggregated_parameters 是下一轮 Step 04 的共享参数。
6. Step 04 的 client_state_dicts 是下一轮 Step 04 保留个性化状态的来源。
7. Step 09 每轮执行，只评估 Step 00 中真实 benign clients。
8. 全实验只有一个 MASTER_SEED；各随机阶段由它或由它+round 派生。
===============================================================================
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

from Step_00_V_byzantine_config import (
    AttackType,
    ByzantineConfigGenerator,
)
from Step_01_data_distribution import FLDataDistributor
from Step_02_V_label_flipping_attack import apply_label_flipping

from Step_03_V_architecture_allocation import (
    allocate_architecture_from_data_result as allocate_adaptive_architecture,
)
from Step_03_V_sp_consistent_architecture import (
    allocate_architecture_from_data_result as allocate_consistent_architecture,
)

from Step_04_local_model_trainning import (
    initialize_global_model,
    train_local_models,
)
from Step_05_V_gaussian_and_sign_flipping_attack import (
    apply_parameter_attacks,
)

from Step_06_V_masking_and_uploading import (
    mask_and_upload_from_step05,
)
from Step_06_V_sp_uploading import (
    upload_from_step05,
)

from Step_07_V_byzantine_final import (
    detect_byzantine_from_step06,
)
from Step_07_V_sp_byzantine import (
    detect_from_step06_sp,
)

from Step_08_V_robust_aggregation import (
    robust_aggregate_from_step06_step07,
)
from Step_09_V_benign_local_accuracy import (
    evaluate_benign_local_accuracy,
)


# =============================================================================
# 1. 实验配置
# =============================================================================

@dataclass
class ExperimentConfig:
    # -------------------------------------------------------------------------
    # 全局
    # -------------------------------------------------------------------------
    master_seed: int = 42
    num_rounds: int = 50
    num_clients: int = 30
    dataset_name: str = "cifar10"
    data_dir: str = "./data"
    device: str = "cpu"

    # -------------------------------------------------------------------------
    # Step 00：Byzantine 配置
    # 权重只决定 Byzantine clients 内部三种攻击的分配比例。
    # -------------------------------------------------------------------------
    byzantine_ratio: float = 0.20
    attack_type_weights: Dict[AttackType, float] = field(
        default_factory=lambda: {
            AttackType.LABEL_FLIPPING: 1.0,
            AttackType.GAUSSIAN: 1.0,
            AttackType.SIGN_FLIPPING: 1.0,
        }
    )
    layer_attack_ratio: float = 1.0
    gaussian_mean: float = 0.0
    gaussian_std: float = 1.0
    sign_flip_scale: float = -1.0
    label_flip_map: Optional[Dict[int, int]] = None

    # -------------------------------------------------------------------------
    # Step 01：数据异构
    # -------------------------------------------------------------------------
    label_heterogeneity: float = 0.5
    volume_heterogeneity: float = 1.0
    data_pool_ratio: float = 1.0
    global_test_ratio: float = 0.1
    local_test_ratio: float = 0.2
    min_samples_per_client: int = 30
    min_labels_per_client: int = 1
    max_labels_per_client: Optional[int] = None

    # -------------------------------------------------------------------------
    # Step 03：架构模式
    #
    # architecture_mode:
    #   "adaptive"   -> Step_03_V_architecture_allocation.py
    #   "consistent" -> Step_03_V_sp_consistent_architecture.py
    #
    # consistent_shared_layers 仅在 consistent 模式使用，可取 1~5。
    # -------------------------------------------------------------------------
    architecture_mode: str = "adaptive"
    consistent_shared_layers: int = 3

    # -------------------------------------------------------------------------
    # Step 04：本地训练
    # -------------------------------------------------------------------------
    personalized_epochs: int = 1
    shared_epochs: int = 1
    learning_rate: float = 0.01
    momentum: float = 0.9
    train_batch_size: int = 64

    # -------------------------------------------------------------------------
    # Step 06 / 07：安全模式
    #
    # security_mode:
    #   "full" -> Random Masking + 4 logical servers + CKKS Step07
    #   "sp"   -> Plain upload to TS + plaintext Step07-SP
    # -------------------------------------------------------------------------
    security_mode: str = "full"
    mask_std: float = 1.0
    tau: float = 0.7
    h_min: float = 0.5
    h_max: float = 2.0

    # -------------------------------------------------------------------------
    # Step 09
    # -------------------------------------------------------------------------
    test_batch_size: int = 128
    test_num_workers: int = 0

    # -------------------------------------------------------------------------
    # V / Demo
    # -------------------------------------------------------------------------
    preview_values_per_tensor: int = 5


# =============================================================================
# 2. 可视化统一输出
# =============================================================================

@dataclass
class RoundVisualizationBundle:
    """
    一轮 04~09 的前端接口。

    这里只保存 JSON-friendly / preview / statistics，
    不把每轮完整模型 Tensor 全部长期保存在 history 中。
    """
    round_index: int

    step04_training: Any
    step05_attack: Any
    step06_upload: Any
    step07_detection: Any
    step08_aggregation: Any
    step09_accuracy: Any

    accuracy_plot_point: Dict[str, Any]

    # 各步骤真实 wall-clock 耗时（秒），可直接给前端画耗时柱状图。
    step_timing_seconds: Dict[str, float]
    round_total_seconds: float


@dataclass
class ExperimentVisualizationBundle:
    """
    前端/可视化团队建议优先读取这个对象。

    initialization:
        Step00~03 的一次性展示数据。

    rounds:
        每轮 Step04~09 的展示数据。

    accuracy_history:
        可直接画 Accuracy-vs-Round 曲线。
    """
    initialization: Dict[str, Any]
    rounds: List[RoundVisualizationBundle] = field(default_factory=list)
    accuracy_history: List[Dict[str, Any]] = field(default_factory=list)
    timing_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ExperimentResult:
    """
    main.py 的统一返回值。

    final_global_shared_parameters:
        最后一轮 Step08 的正式鲁棒共享参数。

    final_client_state_dicts:
        最后一轮 Step04 的完整客户端状态。
        其中 personalized layers 用于保持客户端个性化状态。

    visualization:
        前端主要接口。
    """
    config: ExperimentConfig
    attack_plan: Any
    data_result: Any
    label_flipping_result: Any
    architecture_result: Any

    final_global_shared_parameters: Dict[str, List[torch.Tensor]]
    final_client_state_dicts: Dict[int, Dict[str, torch.Tensor]]

    final_accuracy_result: Any
    visualization: ExperimentVisualizationBundle


# =============================================================================
# 3. Seed 管理
# =============================================================================

def _derive_round_seed(master_seed: int, round_index: int) -> int:
    """
    只给“该轮本地训练的通用随机状态”派生 seed。

    Step00 / Step02 / Step05 / Step06 / Step07 内部已有自己的确定性随机逻辑，
    main 不覆盖它们的内部派生规则。
    """
    value = (
        (int(master_seed) & 0x7FFFFFFF) * 1_000_003
        + int(round_index) * 1_299_709
        + 0x524F554E44
    )
    return int(value & 0x7FFFFFFF)


def _seed_process(seed: int) -> None:
    """
    统一设置 Python / NumPy / PyTorch 的进程级随机状态。

    目的：
    - Step01 使用明确传入的 master_seed；
    - Step04 DataLoader shuffle 等通用训练随机性在同一配置下可复现；
    - 每一轮使用不同但确定的 round seed。
    """
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 4. 小型辅助函数
# =============================================================================

def _clone_parameter_structure(
    parameters: Mapping[str, List[torch.Tensor]],
) -> Dict[str, List[torch.Tensor]]:
    return {
        str(layer_name): [
            tensor.detach().cpu().clone()
            for tensor in tensors
        ]
        for layer_name, tensors in parameters.items()
    }


def _clone_client_state_dicts(
    states: Mapping[int, Mapping[str, torch.Tensor]],
) -> Dict[int, Dict[str, torch.Tensor]]:
    return {
        int(client_id): {
            str(name): tensor.detach().cpu().clone()
            for name, tensor in state.items()
        }
        for client_id, state in states.items()
    }


def _safe_visualization_data(result: Any) -> Any:
    """
    各 Step 的 V 接口命名并不完全相同。
    Main 只读取已经存在的展示字段，不重新计算算法结果。
    """
    for attr in (
        "visualization_data",
        "aggregation_visualization_data",
        "accuracy_visualization_data",
    ):
        if hasattr(result, attr):
            return copy.deepcopy(getattr(result, attr))

    return None


def _build_attack_plan_visualization(attack_plan: Any) -> Dict[str, Any]:
    clients = []

    for client_id in range(attack_plan.num_clients):
        cfg = attack_plan.get(client_id)

        clients.append(
            {
                "client_id": int(client_id),
                "is_byzantine": bool(cfg.is_byzantine),
                "attack_type": (
                    cfg.attack_type.value
                    if cfg.attack_type is not None
                    else None
                ),
                "layer_attack_ratio": float(cfg.layer_attack_ratio),
            }
        )

    return {
        "master_seed": int(attack_plan.master_seed),
        "num_clients": int(attack_plan.num_clients),
        "byzantine_ratio": float(attack_plan.byzantine_ratio),
        "byzantine_clients": list(attack_plan.byzantine_clients),
        "benign_clients": list(attack_plan.benign_clients),
        "clients": clients,
    }


def _build_data_visualization(data_result: Any) -> Dict[str, Any]:
    """
    Step01 没必要把 Dataset 对象交给前端。
    Main 只整理其已有统计字段。
    """
    return {
        "dataset_name": str(data_result.dataset_name),
        "seed": int(data_result.seed),
        "client_sizes": list(data_result.client_sizes),
        "client_train_sizes": list(data_result.client_train_sizes),
        "client_test_sizes": list(data_result.client_test_sizes),
        "client_label_distribution_train": copy.deepcopy(
            data_result.client_label_distribution_train
        ),
        "client_label_distribution_test": copy.deepcopy(
            data_result.client_label_distribution_test
        ),
    }


def _build_step02_visualization(result: Any) -> Dict[str, Any]:
    return {
        "attacked_clients": list(result.attacked_clients),
        "total_flipped_samples": int(result.total_flipped_samples),
        "original_train_label_distribution": copy.deepcopy(
            result.client_original_train_label_distribution
        ),
        "attacked_train_label_distribution": copy.deepcopy(
            result.client_attacked_train_label_distribution
        ),
        "flipped_sample_counts": copy.deepcopy(
            result.flipped_sample_counts
        ),
        "visualization_samples": copy.deepcopy(
            result.visualization_samples
        ),
    }


def _build_step04_visualization(result: Any) -> Dict[str, Any]:
    """
    Step04 没有必要把 client_models / state_dict 全塞给前端。
    只暴露训练统计与架构名称。
    """
    return {
        "round_index": int(result.round_index),
        "shared_layer_names": copy.deepcopy(
            result.shared_layer_names
        ),
        "personalized_layer_names": copy.deepcopy(
            result.personalized_layer_names
        ),
        "training_metrics": copy.deepcopy(
            result.training_metrics
        ),
    }


def _validate_config(config: ExperimentConfig) -> None:
    if config.num_rounds <= 0:
        raise ValueError("num_rounds 必须 > 0")

    if config.num_clients <= 0:
        raise ValueError("num_clients 必须 > 0")

    if config.architecture_mode not in {
        "adaptive",
        "consistent",
    }:
        raise ValueError(
            "architecture_mode 必须为 'adaptive' 或 'consistent'"
        )

    if config.security_mode not in {
        "full",
        "sp",
    }:
        raise ValueError(
            "security_mode 必须为 'full' 或 'sp'"
        )

    if (
        config.architecture_mode == "consistent"
        and not (1 <= config.consistent_shared_layers <= 5)
    ):
        raise ValueError(
            "consistent_shared_layers 必须位于 [1, 5]"
        )

    if not (0.0 < config.tau <= 1.0):
        raise ValueError("tau 必须满足 0 < tau <= 1")


# =============================================================================
# 5. 主实验
# =============================================================================

def run_experiment(
    config: ExperimentConfig,
) -> ExperimentResult:
    """
    正式主入口。

    该函数严格负责：
        配置
        -> Step00~03 初始化
        -> Step04~09 循环 T 轮
        -> 状态传递
        -> V 数据收集
        -> 返回最终结果
    """
    _validate_config(config)

    # =========================================================================
    # 全实验初始 Seed
    # =========================================================================
    _seed_process(config.master_seed)

    print("=" * 96)
    print("FedCALI Main Experiment")
    print("=" * 96)
    print(f"Dataset           : {config.dataset_name}")
    print(f"Clients           : {config.num_clients}")
    print(f"Rounds            : {config.num_rounds}")
    print(f"Master Seed       : {config.master_seed}")
    print(f"Architecture Mode : {config.architecture_mode}")
    print(f"Security Mode     : {config.security_mode}")
    print("=" * 96)

    # =========================================================================
    # Step 00：一次性 Byzantine AttackPlan
    # =========================================================================
    print("\n[Step 00] Byzantine configuration")
    _t0 = time.perf_counter()

    attack_generator = ByzantineConfigGenerator(
        num_clients=config.num_clients,
        byzantine_ratio=config.byzantine_ratio,
        attack_type_weights=config.attack_type_weights,
        layer_attack_ratio=config.layer_attack_ratio,
        gaussian_mean=config.gaussian_mean,
        gaussian_std=config.gaussian_std,
        sign_flip_scale=config.sign_flip_scale,
        label_flip_map=config.label_flip_map,
        master_seed=config.master_seed,
    )

    attack_plan = attack_generator.generate()
    initialization_timing = {
        "step00": time.perf_counter() - _t0,
    }
    print(f"          Time: {initialization_timing['step00']:.3f} s")

    # =========================================================================
    # Step 01：一次性数据分发
    # =========================================================================
    print("[Step 01] Data distribution")
    _t0 = time.perf_counter()

    distributor = FLDataDistributor(
        dataset_name=config.dataset_name,
        num_clients=config.num_clients,
        label_heterogeneity=config.label_heterogeneity,
        volume_heterogeneity=config.volume_heterogeneity,
        data_pool_ratio=config.data_pool_ratio,
        global_test_ratio=config.global_test_ratio,
        local_test_ratio=config.local_test_ratio,
        min_samples_per_client=config.min_samples_per_client,
        min_labels_per_client=config.min_labels_per_client,
        max_labels_per_client=config.max_labels_per_client,
        data_dir=config.data_dir,
        seed=config.master_seed,
    )

    data_result = distributor.distribute()
    initialization_timing["step01"] = time.perf_counter() - _t0
    print(f"          Time: {initialization_timing['step01']:.3f} s")

    # =========================================================================
    # Step 02：一次性 Label-Flipping 数据视图
    #
    # 即使本次没有 Label-Flipping client，也允许调用。
    # 此时所有训练数据会原样通过。
    # =========================================================================
    print("[Step 02] Label-flipping preparation")
    _t0 = time.perf_counter()

    label_result = apply_label_flipping(
        attack_plan=attack_plan,
        data_result=data_result,
    )
    initialization_timing["step02"] = time.perf_counter() - _t0
    print(f"          Time: {initialization_timing['step02']:.3f} s")

    client_train_datasets = (
        label_result.client_train_datasets
    )
    client_test_datasets = (
        label_result.client_test_datasets
    )

    # =========================================================================
    # Step 03 / Step 03-SP：一次性架构分配
    # =========================================================================
    _t0 = time.perf_counter()
    if config.architecture_mode == "adaptive":
        print("[Step 03] Adaptive architecture allocation")

        architecture_result = (
            allocate_adaptive_architecture(
                data_result
            )
        )

    else:
        print(
            "[Step 03-SP] Consistent architecture allocation "
            f"(K={config.consistent_shared_layers})"
        )

        architecture_result = (
            allocate_consistent_architecture(
                data_result,
                fixed_shared_layers=(
                    config.consistent_shared_layers
                ),
            )
        )

    initialization_timing["step03"] = time.perf_counter() - _t0
    print(f"          Time: {initialization_timing['step03']:.3f} s")

    client_shared_layer_counts = (
        architecture_result.shared_layers
    )

    # =========================================================================
    # Step 04 第一轮统一初始化模型
    #
    # 注意：
    # initial_global_state_dict 是“第一轮初始模型”。
    # 后续轮真正变化的共享参数通过 global_shared_parameters 传入。
    # =========================================================================
    _, initial_global_state_dict = initialize_global_model(
        dataset_name=config.dataset_name,
        seed=config.master_seed,
        device=config.device,
    )

    # =========================================================================
    # 初始化 V 总接口
    # =========================================================================
    visualization = ExperimentVisualizationBundle(
        initialization={
            "initialization_timing_seconds": copy.deepcopy(initialization_timing),
            "experiment": {
                "master_seed": config.master_seed,
                "num_rounds": config.num_rounds,
                "num_clients": config.num_clients,
                "dataset_name": config.dataset_name,
                "architecture_mode": config.architecture_mode,
                "consistent_shared_layers": (
                    config.consistent_shared_layers
                    if config.architecture_mode == "consistent"
                    else None
                ),
                "security_mode": config.security_mode,
                "tau": config.tau,
            },
            "step00_attack_plan": (
                _build_attack_plan_visualization(
                    attack_plan
                )
            ),
            "step01_data_distribution": (
                _build_data_visualization(
                    data_result
                )
            ),
            "step02_label_flipping": (
                _build_step02_visualization(
                    label_result
                )
            ),
            "step03_architecture": (
                _safe_visualization_data(
                    architecture_result
                )
            ),
        }
    )

    # =========================================================================
    # Round State
    #
    # 第一轮：
    #   previous_client_state_dicts = None
    #   global_shared_parameters    = None
    #
    # 第 t+1 轮：
    #   previous_client_state_dicts <- Step04(t).client_state_dicts
    #   global_shared_parameters    <- Step08(t).robust_aggregated_parameters
    # =========================================================================
    previous_client_state_dicts = None
    global_shared_parameters = None

    final_accuracy_result = None

    # =========================================================================
    # Step 04~09：完整 T 轮循环
    # =========================================================================
    for round_index in range(1, config.num_rounds + 1):

        print("\n" + "=" * 96)
        print(
            f"Round {round_index}/{config.num_rounds}"
        )
        print("=" * 96)

        round_start_time = time.perf_counter()
        step_timing: Dict[str, float] = {}

        # ---------------------------------------------------------------------
        # 该轮通用训练随机状态。
        #
        # 不修改 AttackPlan；
        # 不替代 Step05/06/07 内部自己的 seed 派生。
        # ---------------------------------------------------------------------
        round_seed = _derive_round_seed(
            config.master_seed,
            round_index,
        )
        _seed_process(round_seed)

        # =====================================================================
        # Step 04：Local Training
        # =====================================================================
        print("[Step 04] Local model training")
        _t0 = time.perf_counter()

        local_training_result = train_local_models(
            dataset_name=config.dataset_name,
            client_train_datasets=client_train_datasets,
            client_shared_layer_counts=(
                client_shared_layer_counts
            ),
            initial_global_state_dict=(
                initial_global_state_dict
            ),
            previous_client_state_dicts=(
                previous_client_state_dicts
            ),
            global_shared_parameters=(
                global_shared_parameters
            ),
            round_index=round_index,
            personalized_epochs=(
                config.personalized_epochs
            ),
            shared_epochs=config.shared_epochs,
            learning_rate=config.learning_rate,
            momentum=config.momentum,
            batch_size=config.train_batch_size,
            device=config.device,
        )
        step_timing["step04"] = time.perf_counter() - _t0
        print(f"          Time: {step_timing['step04']:.3f} s")

        # =====================================================================
        # Step 05：Gaussian / Sign-Flipping Parameter Attack
        #
        # 所有 round 都经过 Step05。
        # 是否真正改参数由 Step00 AttackPlan 决定：
        #   benign          -> unchanged
        #   label flipping  -> unchanged here
        #   gaussian/sign   -> attacked here
        # =====================================================================
        print("[Step 05] Parameter attack stage")
        _t0 = time.perf_counter()

        parameter_attack_result = apply_parameter_attacks(
            attack_plan=attack_plan,
            local_training_result=local_training_result,
            preview_values_per_tensor=(
                config.preview_values_per_tensor
            ),
        )
        step_timing["step05"] = time.perf_counter() - _t0
        print(f"          Time: {step_timing['step05']:.3f} s")

        # =====================================================================
        # Step 06 + Step 07：严格配对的安全/SP 分支
        # =====================================================================
        if config.security_mode == "full":

            # -----------------------------------------------------------------
            # Step 06 FULL
            # -----------------------------------------------------------------
            print(
                "[Step 06] Four-server random masking/uploading"
            )
            _t0 = time.perf_counter()

            step06_result = mask_and_upload_from_step05(
                parameter_attack_result=(
                    parameter_attack_result
                ),
                master_seed=attack_plan.master_seed,
                mask_std=config.mask_std,
                preview_values_per_tensor=(
                    config.preview_values_per_tensor
                ),
            )
            step_timing["step06"] = time.perf_counter() - _t0
            print(f"          Time: {step_timing['step06']:.3f} s")

            # -----------------------------------------------------------------
            # Step 07 FULL CKKS
            # -----------------------------------------------------------------
            print(
                "[Step 07] CKKS secure Byzantine detection"
            )
            _t0 = time.perf_counter()

            step07_result = detect_byzantine_from_step06(
                masking_result=step06_result,
                tau=config.tau,
                h_min=config.h_min,
                h_max=config.h_max,
                preview_values_per_tensor=(
                    config.preview_values_per_tensor
                ),
            )
            step_timing["step07"] = time.perf_counter() - _t0
            print(f"          Time: {step_timing['step07']:.3f} s")

        else:

            # -----------------------------------------------------------------
            # Step 06 SP
            # -----------------------------------------------------------------
            print(
                "[Step 06-SP] Plain shared-parameter uploading"
            )
            _t0 = time.perf_counter()

            step06_result = upload_from_step05(
                parameter_attack_result=(
                    parameter_attack_result
                ),
                preview_values_per_tensor=(
                    config.preview_values_per_tensor
                ),
            )
            step_timing["step06"] = time.perf_counter() - _t0
            print(f"          Time: {step_timing['step06']:.3f} s")

            # -----------------------------------------------------------------
            # Step 07 SP
            # -----------------------------------------------------------------
            print(
                "[Step 07-SP] Plaintext Byzantine detection"
            )
            _t0 = time.perf_counter()

            step07_result = detect_from_step06_sp(
                plain_uploading_result=step06_result,
                master_seed=attack_plan.master_seed,
                tau=config.tau,
                preview_values_per_tensor=(
                    config.preview_values_per_tensor
                ),
            )
            step_timing["step07"] = time.perf_counter() - _t0
            print(f"          Time: {step_timing['step07']:.3f} s")

        # =====================================================================
        # Step 08：统一鲁棒聚合
        #
        # Step08 自己识别 Step06 是 FULL 还是 SP。
        # =====================================================================
        print("[Step 08] Robust layer-wise aggregation")
        _t0 = time.perf_counter()

        step08_result = (
            robust_aggregate_from_step06_step07(
                step06_result=step06_result,
                step07_result=step07_result,
                preview_values_per_tensor=(
                    config.preview_values_per_tensor
                ),
            )
        )
        step_timing["step08"] = time.perf_counter() - _t0
        print(f"          Time: {step_timing['step08']:.3f} s")

        # =====================================================================
        # Step 09：良性客户端 local-test accuracy
        #
        # 注意：
        #   真实 Byzantine clients 由 Step00 ground truth 排除。
        #   不使用 Step07 retained set 代替 benign set。
        # =====================================================================
        print("[Step 09] Benign local-test accuracy")
        _t0 = time.perf_counter()

        step09_result = evaluate_benign_local_accuracy(
            local_training_result=(
                local_training_result
            ),
            robust_aggregation_result=(
                step08_result
            ),
            attack_plan=attack_plan,
            client_test_datasets=(
                client_test_datasets
            ),
            batch_size=config.test_batch_size,
            device=config.device,
            num_workers=config.test_num_workers,
        )
        step_timing["step09"] = time.perf_counter() - _t0
        print(f"          Time: {step_timing['step09']:.3f} s")

        round_total_seconds = time.perf_counter() - round_start_time

        final_accuracy_result = step09_result

        # =====================================================================
        # 本轮 V 数据
        # =====================================================================
        round_v = RoundVisualizationBundle(
            round_index=round_index,
            step04_training=(
                _build_step04_visualization(
                    local_training_result
                )
            ),
            step05_attack=(
                _safe_visualization_data(
                    parameter_attack_result
                )
            ),
            step06_upload=(
                _safe_visualization_data(
                    step06_result
                )
            ),
            step07_detection={
                "detection": (
                    _safe_visualization_data(
                        step07_result
                    )
                ),
                "privacy": copy.deepcopy(
                    getattr(
                        step07_result,
                        "privacy_visualization_data",
                        None,
                    )
                ),
                "ckks_chain": copy.deepcopy(
                    getattr(
                        step07_result,
                        "ckks_chain_visualization_data",
                        None,
                    )
                ),
                "distance": copy.deepcopy(
                    getattr(
                        step07_result,
                        "distance_visualization_data",
                        None,
                    )
                ),
            },
            step08_aggregation={
                "components": copy.deepcopy(
                    getattr(
                        step08_result,
                        "aggregation_visualization_data",
                        None,
                    )
                ),
                "summary": copy.deepcopy(
                    getattr(
                        step08_result,
                        "aggregation_visualization_summary",
                        None,
                    )
                ),
            },
            step09_accuracy=copy.deepcopy(
                step09_result.accuracy_visualization_data
            ),
            accuracy_plot_point=copy.deepcopy(
                step09_result.round_accuracy_plot_point
            ),
            step_timing_seconds=copy.deepcopy(step_timing),
            round_total_seconds=float(round_total_seconds),
        )

        visualization.rounds.append(round_v)
        visualization.accuracy_history.append(
            copy.deepcopy(
                step09_result.round_accuracy_plot_point
            )
        )
        visualization.timing_history.append(
            {
                "round_index": round_index,
                **copy.deepcopy(step_timing),
                "round_total": float(round_total_seconds),
            }
        )

        print(
            f"Round {round_index} Accuracy | "
            f"mean-client="
            f"{step09_result.mean_client_accuracy * 100.0:.4f}% | "
            f"sample-weighted="
            f"{step09_result.sample_weighted_accuracy * 100.0:.4f}%"
        )

        print("Round Timing Summary")
        for _step_name in (
            "step04", "step05", "step06",
            "step07", "step08", "step09",
        ):
            print(
                f"  {_step_name.upper():<8}: "
                f"{step_timing[_step_name]:10.3f} s"
            )
        print(
            f"  {'TOTAL':<8}: "
            f"{round_total_seconds:10.3f} s"
        )

        # =====================================================================
        # 最关键：状态传给下一轮
        #
        # 1. 个性化状态：
        #       Step04(t).client_state_dicts
        #       -> Step04(t+1).previous_client_state_dicts
        #
        # 2. 共享状态：
        #       Step08(t).robust_aggregated_parameters
        #       -> Step04(t+1).global_shared_parameters
        #
        # Step09 只评估，不修改这两个训练状态。
        # =====================================================================
        previous_client_state_dicts = (
            _clone_client_state_dicts(
                local_training_result.client_state_dicts
            )
        )

        global_shared_parameters = (
            _clone_parameter_structure(
                step08_result.robust_aggregated_parameters
            )
        )

        # 主循环不需要继续持有 Step06/07 的大对象；
        # round_v 中已经只保存前端需要的轻量展示数据。
        del step06_result
        del step07_result

    # =========================================================================
    # 实验结束
    # =========================================================================
    assert global_shared_parameters is not None
    assert previous_client_state_dicts is not None
    assert final_accuracy_result is not None

    print("\n" + "=" * 96)
    print("Experiment Finished")
    print("=" * 96)
    print(
        "Final Mean Client Accuracy      : "
        f"{final_accuracy_result.mean_client_accuracy * 100.0:.4f}%"
    )
    print(
        "Final Sample-weighted Accuracy  : "
        f"{final_accuracy_result.sample_weighted_accuracy * 100.0:.4f}%"
    )

    print("\nAverage Step Timing Across Rounds")
    for _step_name in (
        "step04", "step05", "step06",
        "step07", "step08", "step09",
    ):
        _avg = sum(
            item[_step_name]
            for item in visualization.timing_history
        ) / len(visualization.timing_history)
        print(f"  {_step_name.upper():<8}: {_avg:10.3f} s")

    return ExperimentResult(
        config=config,
        attack_plan=attack_plan,
        data_result=data_result,
        label_flipping_result=label_result,
        architecture_result=architecture_result,
        final_global_shared_parameters=(
            global_shared_parameters
        ),
        final_client_state_dicts=(
            previous_client_state_dicts
        ),
        final_accuracy_result=(
            final_accuracy_result
        ),
        visualization=visualization,
    )


# =============================================================================
# 6. PyCharm 直接运行入口
# =============================================================================

if __name__ == "__main__":
    # =========================================================================
    # 这里是当前工程唯一建议直接修改的实验入口。
    #
    # 比赛前端接入后，可以把这些参数改为由 UI / API 传入，
    # 但 run_experiment(...) 本身无需改变。
    # =========================================================================

    config = ExperimentConfig(
        # 全局
        master_seed=42,
        num_rounds=10,
        num_clients=20,
        dataset_name="cifar10",
        data_dir="./data",
        device=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),

        # Byzantine
        byzantine_ratio=0.20,
        attack_type_weights={
            AttackType.LABEL_FLIPPING: 1.0,
            AttackType.GAUSSIAN: 1.0,
            AttackType.SIGN_FLIPPING: 1.0,
        },
        layer_attack_ratio=1.0,
        gaussian_mean=0.0,
        gaussian_std=1.0,
        sign_flip_scale=-1.0,

        # Data heterogeneity
        label_heterogeneity=0.5,
        volume_heterogeneity=1.0,

        # Architecture
        # "adaptive"   -> Step_03_V_architecture_allocation.py
        # "consistent" -> Step_03_V_sp_consistent_architecture.py
        architecture_mode="adaptive",
        consistent_shared_layers=3,

        # Local training
        personalized_epochs=2,
        shared_epochs=2,
        learning_rate=0.01,
        momentum=0.9,
        train_batch_size=64,

        # Security
        # "full" -> masking + CKKS
        # "sp"   -> plaintext comparison branch
        security_mode="full",
        mask_std=1.0,
        tau=0.7,

        # Evaluation
        test_batch_size=128,
    )

    experiment_result = run_experiment(
        config
    )

    # -------------------------------------------------------------------------
    # 前端最重要的统一变量：
    #
    # experiment_result.visualization.initialization
    # experiment_result.visualization.rounds
    # experiment_result.visualization.accuracy_history
    # experiment_result.visualization.timing_history
    #
    # 例如：
    # accuracy_curve = (
    #     experiment_result
    #     .visualization
    #     .accuracy_history
    # )
    # -------------------------------------------------------------------------
