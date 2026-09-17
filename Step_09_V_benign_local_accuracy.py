"""
Step_09_V_benign_local_accuracy.py

===============================================================================
Step 09：良性客户端 Local Test Accuracy（可视化相关）
===============================================================================

【职责】
Step 03 只在训练前确定一次架构；Step 04~09 构成完整 round loop：

    04 Local Training
      -> 05 Byzantine Attack
      -> 06 Upload
      -> 07 Byzantine Detection
      -> 08 Robust Aggregation
      -> 09 Benign Local-Test Evaluation
      -> next Round 04

Step 09 必须在 Step 08 后评估。对于真实良性客户端 k：

    M_k^(t) = G_shared,k^(t) + P_k^(t)

其中：
    G_shared,k^(t) 来自 Step 08 robust_aggregated_parameters；
    P_k^(t) 保留 Step 04 本轮训练后的客户端个性化层。

随后在该客户端自己的 local test set 上计算：

    Acc_k^(t) = correct_k / |D_k,test|

【Byzantine 客户端】
Step 09 是实验 evaluation，不是检测算法，因此使用 Step 00 的真实
attack_plan.benign_clients 决定评估集合。真实 Byzantine 客户端：
    - 不构造 evaluation model；
    - 不遍历 local test set；
    - 不进入 per-client accuracy；
    - 不进入任何总体 accuracy。

不能使用 Step 07 retained_clients 冒充真实 benign client set。

【总体指标】
同时输出：
    mean_client_accuracy
        = 所有良性客户端 accuracy 的等权平均；
    sample_weighted_accuracy
        = 所有良性客户端 correct 总数 / test sample 总数。

【V 接口】
本文件不画图，只准备可直接绘图的数据：
    result.accuracy_visualization_data
    result.round_accuracy_plot_point

前者包含每个良性客户端准确率、样本数、共享/个性化层、总体 mean/min/max/std；
后者可被多轮主程序直接 append，用于 accuracy-vs-round 曲线。

【下一轮】
Step 09 不修改模型。
下一轮 Step 04：
    global_shared_parameters
        <- Step 08 robust_aggregated_parameters
    previous_client_state_dicts
        <- 本轮 Step 04 client_state_dicts
===============================================================================
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class ClientAccuracyMetric:
    client_id: int
    correct: int
    total: int
    accuracy: float
    shared_layers: List[str]
    personalized_layers: List[str]

    @property
    def visualization_dict(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "correct": self.correct,
            "total": self.total,
            "accuracy": self.accuracy,
            "accuracy_percent": self.accuracy * 100.0,
            "shared_layers": self.shared_layers.copy(),
            "personalized_layers": self.personalized_layers.copy(),
            "shared_layer_count": len(self.shared_layers),
            "personalized_layer_count": len(self.personalized_layers),
        }


@dataclass
class BenignLocalAccuracyResult:
    dataset_name: str
    round_index: int
    evaluated_client_ids: List[int]
    excluded_byzantine_client_ids: List[int]
    client_metrics: Dict[int, ClientAccuracyMetric]

    mean_client_accuracy: float
    sample_weighted_accuracy: float
    min_client_accuracy: float
    max_client_accuracy: float
    std_client_accuracy: float
    total_correct: int
    total_test_samples: int

    # 仅供核对/展示；下一轮个性化状态仍来自 Step 04 client_state_dicts。
    evaluation_client_state_dicts: Dict[int, Dict[str, torch.Tensor]]

    @property
    def accuracy_visualization_data(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "round_index": self.round_index,
            "evaluated_client_ids": self.evaluated_client_ids.copy(),
            "excluded_byzantine_client_ids":
                self.excluded_byzantine_client_ids.copy(),
            "benign_client_count": len(self.evaluated_client_ids),
            "excluded_byzantine_client_count":
                len(self.excluded_byzantine_client_ids),

            "mean_client_accuracy": self.mean_client_accuracy,
            "mean_client_accuracy_percent":
                self.mean_client_accuracy * 100.0,
            "sample_weighted_accuracy": self.sample_weighted_accuracy,
            "sample_weighted_accuracy_percent":
                self.sample_weighted_accuracy * 100.0,

            "min_client_accuracy": self.min_client_accuracy,
            "min_client_accuracy_percent":
                self.min_client_accuracy * 100.0,
            "max_client_accuracy": self.max_client_accuracy,
            "max_client_accuracy_percent":
                self.max_client_accuracy * 100.0,
            "std_client_accuracy": self.std_client_accuracy,
            "std_client_accuracy_percent":
                self.std_client_accuracy * 100.0,

            "total_correct": self.total_correct,
            "total_test_samples": self.total_test_samples,

            "per_client": [
                self.client_metrics[cid].visualization_dict
                for cid in self.evaluated_client_ids
            ],
        }

    @property
    def visualization_data(self) -> Dict[str, Any]:
        return self.accuracy_visualization_data

    @property
    def round_accuracy_plot_point(self) -> Dict[str, Any]:
        return {
            "round_index": self.round_index,
            "accuracy": self.mean_client_accuracy,
            "accuracy_percent": self.mean_client_accuracy * 100.0,
            "sample_weighted_accuracy": self.sample_weighted_accuracy,
            "sample_weighted_accuracy_percent":
                self.sample_weighted_accuracy * 100.0,
        }


def _clone_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
    }


def _resolve_benign_clients(attack_plan: Any) -> List[int]:
    if hasattr(attack_plan, "benign_clients"):
        values = attack_plan.benign_clients
    elif hasattr(attack_plan, "benign_client_ids"):
        values = attack_plan.benign_client_ids
    else:
        raise TypeError(
            "attack_plan 缺少 benign_clients/benign_client_ids；"
            "Step 09 不允许用 Step 07 retained_clients 代替真实良性集合。"
        )

    ids = sorted({int(x) for x in values})
    if not ids:
        raise ValueError("没有良性客户端，无法计算测试准确率")
    return ids


def _resolve_byzantine_clients(
    attack_plan: Any,
    all_client_ids: Sequence[int],
    benign_ids: Sequence[int],
) -> List[int]:
    for name in ("byzantine_clients", "byzantine_client_ids"):
        if hasattr(attack_plan, name):
            return sorted({
                int(x) for x in getattr(attack_plan, name)
            })

    benign = set(benign_ids)
    return sorted(
        int(x) for x in all_client_ids
        if int(x) not in benign
    )


def _apply_robust_shared_parameters(
    model: nn.Module,
    shared_layers: Sequence[str],
    robust_parameters: Mapping[str, Sequence[torch.Tensor]],
) -> None:
    if not hasattr(model, "set_layer_params"):
        raise TypeError(
            "Step 04 model 缺少 set_layer_params(...)"
        )

    for layer_name in shared_layers:
        if layer_name not in robust_parameters:
            raise KeyError(
                f"Step 08 聚合结果缺少 {layer_name}"
            )
        model.set_layer_params(
            layer_name,
            list(robust_parameters[layer_name]),
        )


def _evaluate(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    num_workers: int,
) -> tuple[int, int, float]:
    if len(dataset) <= 0:
        raise ValueError("local test set 不能为空")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            prediction = output.argmax(dim=1)
            correct += int((prediction == target).sum().item())
            total += int(target.numel())

    if total == 0:
        raise RuntimeError("没有有效测试样本")

    return correct, total, correct / total


def evaluate_benign_local_accuracy(
    local_training_result: Any,
    robust_aggregation_result: Any,
    attack_plan: Any,
    client_test_datasets: Mapping[int, Dataset],
    *,
    batch_size: int = 128,
    device: str = "cpu",
    num_workers: int = 0,
) -> BenignLocalAccuracyResult:
    """
    Step 09 正式入口。

    local_training_result      <- 当前轮 Step 04
    robust_aggregation_result  <- 当前轮 Step 08
    attack_plan                <- Step 00
    client_test_datasets       <- Step 01 local test datasets

    Byzantine 客户端不会进入下面的 evaluation loop。
    """
    if batch_size <= 0:
        raise ValueError("batch_size 必须 > 0")
    if num_workers < 0:
        raise ValueError("num_workers 必须 >= 0")

    for name in (
        "dataset_name",
        "round_index",
        "client_models",
        "client_state_dicts",
        "shared_layer_names",
        "personalized_layer_names",
    ):
        if not hasattr(local_training_result, name):
            raise TypeError(f"Step 04 输出缺少 {name}")

    if not hasattr(
        robust_aggregation_result,
        "robust_aggregated_parameters",
    ):
        raise TypeError(
            "Step 08 输出缺少 robust_aggregated_parameters"
        )

    round_index = int(local_training_result.round_index)

    if hasattr(robust_aggregation_result, "round_index"):
        if int(robust_aggregation_result.round_index) != round_index:
            raise ValueError("Step 04 与 Step 08 round_index 不一致")

    all_ids = sorted(
        int(x) for x in local_training_result.client_models
    )
    benign_ids = _resolve_benign_clients(attack_plan)
    byzantine_ids = _resolve_byzantine_clients(
        attack_plan,
        all_ids,
        benign_ids,
    )

    if set(benign_ids) & set(byzantine_ids):
        raise ValueError("benign 与 Byzantine client 集合存在重叠")

    missing_models = sorted(set(benign_ids) - set(all_ids))
    if missing_models:
        raise KeyError(
            f"Step 04 缺少良性客户端模型: {missing_models}"
        )

    missing_tests = [
        cid for cid in benign_ids
        if cid not in client_test_datasets
    ]
    if missing_tests:
        raise KeyError(
            f"良性客户端缺少 local test set: {missing_tests}"
        )

    device_obj = torch.device(device)
    robust_parameters = (
        robust_aggregation_result.robust_aggregated_parameters
    )

    metrics: Dict[int, ClientAccuracyMetric] = {}
    eval_states: Dict[int, Dict[str, torch.Tensor]] = {}

    # 关键：循环集合仅为真实 benign_ids。
    for client_id in benign_ids:
        model = copy.deepcopy(
            local_training_result.client_models[client_id]
        ).to(device_obj)

        shared_layers = list(
            local_training_result.shared_layer_names[client_id]
        )
        personalized_layers = list(
            local_training_result.personalized_layer_names[client_id]
        )

        # 保留 Step 04 personalized layers，只覆盖 shared layers。
        _apply_robust_shared_parameters(
            model,
            shared_layers,
            robust_parameters,
        )

        correct, total, accuracy = _evaluate(
            model,
            client_test_datasets[client_id],
            batch_size,
            device_obj,
            num_workers,
        )

        metrics[client_id] = ClientAccuracyMetric(
            client_id=client_id,
            correct=correct,
            total=total,
            accuracy=float(accuracy),
            shared_layers=shared_layers,
            personalized_layers=personalized_layers,
        )

        eval_states[client_id] = _clone_state_dict(
            model.state_dict()
        )

    values = torch.tensor(
        [metrics[cid].accuracy for cid in benign_ids],
        dtype=torch.float64,
    )

    total_correct = sum(x.correct for x in metrics.values())
    total_samples = sum(x.total for x in metrics.values())

    return BenignLocalAccuracyResult(
        dataset_name=str(local_training_result.dataset_name),
        round_index=round_index,
        evaluated_client_ids=benign_ids.copy(),
        excluded_byzantine_client_ids=byzantine_ids.copy(),
        client_metrics=metrics,

        mean_client_accuracy=float(values.mean().item()),
        sample_weighted_accuracy=float(
            total_correct / total_samples
        ),
        min_client_accuracy=float(values.min().item()),
        max_client_accuracy=float(values.max().item()),
        std_client_accuracy=float(
            values.std(unbiased=False).item()
        ),

        total_correct=total_correct,
        total_test_samples=total_samples,
        evaluation_client_state_dicts=eval_states,
    )


def print_benign_local_accuracy_result(
    result: BenignLocalAccuracyResult,
) -> None:
    print("\n" + "=" * 88)
    print("Step 09 - Benign Client Local Test Accuracy")
    print("=" * 88)
    print(f"Dataset                    : {result.dataset_name}")
    print(f"Round Index                : {result.round_index}")
    print(f"Evaluated Benign Clients   : {result.evaluated_client_ids}")
    print(
        "Excluded Byzantine Clients : "
        f"{result.excluded_byzantine_client_ids}"
    )

    for cid in result.evaluated_client_ids:
        m = result.client_metrics[cid]
        print(
            f"Client {cid:02d} | "
            f"accuracy={m.accuracy * 100.0:8.4f}% | "
            f"{m.correct}/{m.total}"
        )

    print(
        f"Mean Client Accuracy       : "
        f"{result.mean_client_accuracy * 100.0:.4f}%"
    )
    print(
        f"Sample-weighted Accuracy   : "
        f"{result.sample_weighted_accuracy * 100.0:.4f}%"
    )


if __name__ == "__main__":
    # 独立裸跑测试，不重跑真实 Step 00~08。
    from dataclasses import dataclass

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(2, 2, bias=False)  # shared
            self.layer2 = nn.Linear(2, 2, bias=False)  # personalized

        def forward(self, x):
            return self.layer2(self.layer1(x))

        def set_layer_params(self, layer_name, params):
            targets = list(getattr(self, layer_name).parameters())
            if len(targets) != len(params):
                raise ValueError("参数数量不匹配")
            with torch.no_grad():
                for target, source in zip(targets, params):
                    target.copy_(source.to(target.device))

    class TinyDataset(Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, i):
            x, y = self.items[i]
            return (
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long),
            )

    @dataclass
    class Fake04:
        dataset_name: str
        round_index: int
        client_models: Dict[int, nn.Module]
        client_state_dicts: Dict[int, Dict[str, torch.Tensor]]
        shared_layer_names: Dict[int, List[str]]
        personalized_layer_names: Dict[int, List[str]]

    @dataclass
    class Fake08:
        round_index: int
        robust_aggregated_parameters: Dict[str, List[torch.Tensor]]

    @dataclass
    class FakePlan:
        benign_clients: List[int]
        byzantine_clients: List[int]

    eye = torch.eye(2)
    swap = torch.tensor([[0., 1.], [1., 0.]])

    def make_model(personalized):
        m = TinyModel()
        with torch.no_grad():
            m.layer1.weight.copy_(swap)  # 故意错误，必须被 Step08 覆盖
            m.layer2.weight.copy_(personalized)
        return m

    models = {
        0: make_model(eye),
        1: make_model(swap),
        2: make_model(eye),  # Byzantine
    }

    step04 = Fake04(
        "synthetic",
        5,
        models,
        {cid: _clone_state_dict(m.state_dict()) for cid, m in models.items()},
        {0: ["layer1"], 1: ["layer1"], 2: ["layer1"]},
        {0: ["layer2"], 1: ["layer2"], 2: ["layer2"]},
    )
    step08 = Fake08(5, {"layer1": [eye.clone()]})
    plan = FakePlan([0, 1], [2])

    tests = {
        0: TinyDataset([
            ([2., 0.], 0), ([0., 2.], 1),
        ]),
        1: TinyDataset([
            ([2., 0.], 1), ([0., 2.], 0),
        ]),
        # 故意给 Byzantine client 测试集，但绝不能使用。
        2: TinyDataset([
            ([2., 0.], 1),
        ]),
    }

    result = evaluate_benign_local_accuracy(
        step04,
        step08,
        plan,
        tests,
        batch_size=2,
    )

    print_benign_local_accuracy_result(result)

    print("\nAutomatic Checks")
    print(
        "Byzantine client 未参与:",
        result.evaluated_client_ids == [0, 1]
        and 2 not in result.client_metrics
        and 2 not in result.evaluation_client_state_dicts,
    )
    print(
        "两个良性客户端均为 100%:",
        all(
            abs(result.client_metrics[cid].accuracy - 1.0) < 1e-12
            for cid in [0, 1]
        ),
    )
    print(
        "Step08 shared 参数已覆盖:",
        torch.allclose(
            result.evaluation_client_state_dicts[0]["layer1.weight"],
            eye,
        ),
    )
    print(
        "Client1 personalized 参数保持:",
        torch.allclose(
            result.evaluation_client_state_dicts[1]["layer2.weight"],
            swap,
        ),
    )
    print(
        "V 数据已生成:",
        bool(result.accuracy_visualization_data)
        and bool(result.round_accuracy_plot_point),
    )
