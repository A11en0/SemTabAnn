import os
import math
import copy
import json
import pickle
from typing import Dict, Any, Optional, List, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalLoopOutput, has_length
from transformers.trainer_callback import TrainerCallback

from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


# ------------------ utils ------------------ #

def linear_rampup(current, rampup_length):
    """Linear rampup (FreeAL 原版)"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


# ------------------ Trainer 本体 ------------------ #

class FreeALHFTrainer(Trainer):
    """
    FreeAL-style Trainer，改到我们的场景：

    - 数据：列级 MultiPLL，batch 里是“整表拼在一起”的 token
    - 每列有一个全局列 ID：global_col_indices ∈ [0, num_columns)
    - 训练标签用 LLM 给的 pseudo label：candidate_labels 的第一个元素
    - Eval 用真 labels（inputs["labels"]）

    算法核心（完全对齐 FreeAL）：
    - 每个 epoch 开头，用上一轮模型做一遍 teacher pass 扫全训练集：
        * 对每一列算 CE loss（用 pseudo label）
        * GMM 选低损失的为 clean
        * 在 clean 且“预测==pseudo label”的样本中，按类内 confidence 排序，取 top-ρ
        * 对这些 samples 做 KMeans，选 medoids 作为 demo seed
        * 对所有样本做相似度检索，得到 demo index
        * 在 epoch == 1 时，把
              - clean index
              - demo index
              - pseudo label
          分别存成 pkl
    - 训练时：
        * self.chosen_list[g_id] == 1 的样本视为 clean，用 CE + embedding-level mixup
        * 额外对所有样本做 consistency regularization（CR）
    """

    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        data_collator: Optional[Any] = None,
        compute_metrics: Optional[callable] = None,
        callbacks: Optional[List[Any]] = None,
        optimizers: Optional[tuple] = None,
        preprocess_logits_for_metrics: Optional[callable] = None,
        type_ontology: Optional[List[str]] = None,
        **kwargs
    ):
        init_kwargs = {
            "model": model,
            "args": args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "tokenizer": tokenizer,
            "data_collator": data_collator,
            "compute_metrics": compute_metrics,
            "callbacks": callbacks,
            "preprocess_logits_for_metrics": preprocess_logits_for_metrics,
            **kwargs,
        }
        if optimizers is not None:
            init_kwargs["optimizers"] = optimizers

        super().__init__(**init_kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.type_ontology = type_ontology
        self.num_labels: int = getattr(model.config, "num_labels", None)
        if self.num_labels is None:
            raise ValueError("model.config.num_labels 必须存在。")

        # ---- 训练集列总数，用 global_col_indices 推导 ----
        self.num_train_columns = self._compute_num_train_columns(train_dataset)

        # chosen_list[g_id] == 1 表示该列在当前 epoch 被 teacher 选为 clean
        self.chosen_list = torch.ones(self.num_train_columns, device=self.device)

        # 一些 FreeAL 超参（可以从外部 config 传入，也可以用默认值）
        self.temp_u = getattr(self.args, "temp_u", 1.0)
        self.warmup_epochs = getattr(self.args, "warmup_epochs", 1)      # 对应 FreeAL 的 args.warmup
        self.embedding_dim = getattr(self.args, "embedding_dim", None)    # 若 None，会在第一次 teacher pass 里自动推断
        self.select_demo_num = getattr(self.args, "select_demo_num", 100) # 总 demo 数
        self.shot_num = getattr(self.args, "shot_num", 10)                # 每个样本最多检索 demo 数
        self.learning_setting = getattr(self.args, "learning_setting", "transductive")

    # ---- 用 train_dataset.table_df 计算列数 ----
    def _compute_num_train_columns(self, train_dataset) -> int:
        if train_dataset is None:
            return 0
        # MultiPLLDataset 里每行是一个表，global_col_indices 是 [num_cols]
        if hasattr(train_dataset, "table_df"):
            max_gid = -1
            for _, row in train_dataset.table_df.iterrows():
                g = row["global_col_indices"]
                g_max = int(g.max().item())
                if g_max > max_gid:
                    max_gid = g_max
            return max_gid + 1
        # 兜底：普通分类数据集时，用 len(dataset)
        return len(train_dataset)

    # ==================== CLS 抽取：logits / embeddings ==================== #

    def _extract_cls_logits(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        从 [B, T, C] 里抽取所有 CLS 位置的 logits，得到 [N_cols, C]。
        - 使用 tokenizer.cls_token_id 自动寻找 CLS 位置；
        - 结果顺序与你 collator 输出的 labels / global_col_indices 展平顺序一致：
            先表 0 的所有列（按 CLS 出现顺序），再表 1 的所有列 ...
        """
        # 和你之前 ProDEN 版一致：从 processing_class 里拿 cls_token_id
        cls_token_id = getattr(self.processing_class, "cls_token_id", None)
        if cls_token_id is None:
            raise ValueError("processing_class.cls_token_id is required to auto-detect CLS positions.")
        cls_positions = torch.nonzero(input_ids == cls_token_id, as_tuple=False)  # [N_cols, 2]
        batch_idx, pos_idx = cls_positions[:, 0], cls_positions[:, 1]
        return logits[batch_idx, pos_idx, :]

    def _extract_cls_embeddings(
        self,
        hidden_states: torch.Tensor,   # [B, T, H]
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        和 _extract_cls_logits 类似，从 [B, T, H] 抽取列级 [N_cols, H] CLS embedding。
        """
        cls_token_id = getattr(self.processing_class, "cls_token_id", None)
        if cls_token_id is None:
            raise ValueError("processing_class.cls_token_id is required to auto-detect CLS positions.")
        cls_positions = torch.nonzero(input_ids == cls_token_id, as_tuple=False)
        batch_idx, pos_idx = cls_positions[:, 0], cls_positions[:, 1]
        return hidden_states[batch_idx, pos_idx, :]

    def _forward_with_features(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        return_features: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        统一封装一版 forward，兼容你的 model 接口：

        期望：
        - model(input_ids=..., return_features=True) 返回:
            (logits, hidden_states) 或 (logits, features)
        - 若 return_features=False，则可只返回 logits
        """
        outputs = model(input_ids=input_ids, return_features=return_features)

        if isinstance(outputs, tuple):
            logits = outputs[0]
            feats = outputs[1] if return_features and len(outputs) > 1 else None
        else:
            # huggingface style
            logits = outputs.logits
            feats = None
            if return_features:
                if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                    feats = outputs.hidden_states[-1]
                elif hasattr(outputs, "last_hidden_state"):
                    feats = outputs.last_hidden_state

        # 多列：logits [B, T, C] -> [N_cols, C]；embedding [B, T, H] -> [N_cols, H]
        if logits.dim() == 3:
            logits = self._extract_cls_logits(logits, input_ids)
        if feats is not None and feats.dim() == 3:
            feats = self._extract_cls_embeddings(feats, input_ids)

        return logits, feats

    # ==================== teacher pass：每个 epoch 开头扫全数据 ==================== #

    def build_chosen_list_for_epoch(self, epoch: int):
        """
        完整复现 FreeAL 的 epoch 级 teacher pass，
        使用 global_col_indices 作为“全局样本索引”。
        """
        args = self.args
        device = args.device
        length = self.num_train_columns
        if length <= 0 or self.train_dataset is None:
            # 没有训练数据或列数未知，全部视为 clean
            self.chosen_list = torch.ones(length, device=device)
            return

        # warm-up：前 warmup_epochs 直接全 1
        if epoch <= self.warmup_epochs:
            self.chosen_list = torch.ones(length, device=device)
            return

        print(f"[FreeAL] Teacher pass at epoch {epoch} ...")
        train_dataloader = self.get_train_dataloader()

        # FreeAL 原代码：每个 epoch deep copy 一份 teacher model
        t_model = copy.deepcopy(self.model).to(device)
        t_model.eval()

        num_labels = self.num_labels
        self.temp_u = getattr(args, "temp_u", 1.0)

        # 全数据缓存：按“全局列 id”存
        targets_all = torch.zeros((length,), dtype=torch.long, device=device)
        outputs_all = torch.zeros((length, num_labels), device=device)
        loss_all = torch.zeros((length,), device=device)
        # embedding_dim 不知道的话第一次自动推断
        if self.embedding_dim is None:
            with torch.no_grad():
                for batch in train_dataloader:
                    batch = self._prepare_inputs(batch)
                    input_ids = batch["input_ids"].to(device)
                    _, feats = self._forward_with_features(t_model, input_ids, return_features=True)
                    if feats is None:
                        raise ValueError("模型在 return_features=True 时必须返回 embedding / hidden_states。")
                    self.embedding_dim = feats.size(-1)
                    break
        embeddings_all = torch.zeros((length, self.embedding_dim), dtype=torch.float32)

        loss_fct = nn.CrossEntropyLoss(reduction="none")

        # ---------- 第一次 pass：收集 logits / loss / embedding ---------- #
        for step, batch in enumerate(train_dataloader):
            batch = self._prepare_inputs(batch)
            input_ids = batch["input_ids"].to(device)

            # pseudo label：candidate_labels 的第一个元素
            cand_list = batch["candidate_labels"]      # List[List[int]] 展平后的 list，长度 = N_cols_batch
            pseudo_labels = torch.tensor(
                [(c[0] if isinstance(c, (list, tuple)) and len(c) > 0 else 0) for c in cand_list],
                dtype=torch.long,
                device=device,
            )

            gids = batch["global_col_indices"].view(-1).to(device)  # [N_cols_batch]

            with torch.no_grad():
                logits, feats = self._forward_with_features(t_model, input_ids, return_features=True)
                if feats is None:
                    raise ValueError("模型在 return_features=True 时必须返回 embedding / hidden_states。")

                # 写入全局缓存
                outputs_all[gids] = logits
                targets_all[gids] = pseudo_labels
                batch_loss = loss_fct(logits, pseudo_labels)
                loss_all[gids] = batch_loss
                embeddings_all[gids.cpu()] = feats.detach().cpu()

        # ---------- GMM：在 loss 上做两簇划分，选低损失簇 ---------- #
        valid_idx_all = targets_all >= 0  # 这里其实全 True，因为没有 ambiguous，保持接口一致
        loss_all_cpu = loss_all.detach().cpu()
        valid_indices = torch.where(valid_idx_all)[0].cpu()

        if valid_indices.numel() == 0:
            # 极端情况：没有有效样本，全 clean
            self.chosen_list = torch.ones(length, device=device)
            return

        # min-max 归一化（FreeAL 原版）
        v_loss = loss_all_cpu[valid_indices]
        loss_min, loss_max = v_loss.min(), v_loss.max()
        loss_norm = (loss_all_cpu - loss_min) / (loss_max - loss_min + 1e-8)
        loss_norm = loss_norm.view(-1, 1)

        loss_tmp = loss_norm[valid_indices]
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(loss_tmp.numpy())
        prob = gmm.predict_proba(loss_tmp.numpy())
        prob_clean = prob[:, gmm.means_.argmin()]  # 均值小的簇是 clean 簇
        chosen_idx_all_gmm = np.where(prob_clean > 0.7)[0]          # clean 阈值 0.7（FreeAL）
        # 这些是“全局列 id”
        chosen_gids_gmm = valid_indices[chosen_idx_all_gmm]

        chosen_list = torch.zeros((length,), device=device)
        chosen_list[chosen_gids_gmm] = 1.0

        # ---------- 按类构建 demo pool（完全照 FreeAL） ---------- #
        with torch.no_grad():
            high_conf_all = outputs_all.max(dim=-1)[0]  # 每列最高 logit（不再转 softmax）
            pred_idx_all = outputs_all.max(dim=-1)[1]
        matched_idx_all = (pred_idx_all == targets_all) & valid_idx_all  # 预测 == pseudo label

        rho_sel = 0.2   # FreeAL 固定 0.2
        chosen_list_sel = torch.zeros((length,), device=device)
        chosen_top_indices_list = []

        embeddings_all_t = torch.from_numpy(embeddings_all.numpy())  # torch.Tensor on CPU

        for j in range(num_labels):
            # 同一类别且预测正确的样本
            index_j_matched = torch.where((pred_idx_all == j) & matched_idx_all)[0].cpu()
            if index_j_matched.numel() == 0:
                continue

            max_score_j = high_conf_all[index_j_matched]
            sort_index_j = (-max_score_j).sort()[1].cpu().numpy()  # 按 confidence 降序排序
            partition_j_sel = int(index_j_matched.shape[0] * rho_sel)
            if partition_j_sel == 0:
                continue

            index_j_sel = index_j_matched[sort_index_j[:partition_j_sel]]
            chosen_list_sel[index_j_sel] = 1.0

            # k-medoids（用 KMeans 近似）在这些 clean 样本上选 demo seed
            embeddings_j = embeddings_all_t[index_j_sel]  # [K, H]
            num_clusters = max(1, self.select_demo_num // max(1, num_labels))
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings_j)
            kmeans_labels = kmeans.labels_

            idx_all_representative = []
            embedding_all_representative = []
            for k in range(num_clusters):
                cluster_mask = (kmeans_labels == k)
                if cluster_mask.sum() == 0:
                    continue
                vectors_in_cluster = embeddings_j[cluster_mask]
                idx_in_cluster = index_j_sel[cluster_mask]
                centroid = vectors_in_cluster.mean(dim=0)
                distances = torch.norm(vectors_in_cluster - centroid, dim=1)
                index_of_rep = torch.argmin(distances)
                embedding_all_representative.append(vectors_in_cluster[index_of_rep])
                idx_all_representative.append(idx_in_cluster[index_of_rep].view(1))

            if len(embedding_all_representative) == 0:
                continue

            embedding_all_representative = torch.cat(
                [emb.view(1, -1) for emb in embedding_all_representative], dim=0
            )  # [R, H]
            idx_all_representative = torch.cat(idx_all_representative, dim=0)  # [R]

            # 对所有样本做相似度检索，选 top-K demo（FreeAL 原版做法）
            cos_similarities = cosine_similarity(
                embeddings_all_t.numpy(), embedding_all_representative.numpy()
            )  # [length, R]
            sort_result = torch.sort(torch.from_numpy(cos_similarities), dim=1, descending=True)
            top_indices = sort_result[1][:, :(self.shot_num // max(1, num_labels))]
            # 把代表点 index 映射回全局样本 index
            for i in range(top_indices.shape[0]):
                top_indices[i, :] = idx_all_representative[top_indices[i, :]]
            chosen_top_indices_list.append(top_indices)

        # 拼接所有类别的 demo 检索结果 [length, #demo_total]
        if len(chosen_top_indices_list) > 0:
            chosen_top_indices = torch.cat(chosen_top_indices_list, dim=1)  # [length, total_demo]
        else:
            chosen_top_indices = torch.zeros((length, 0), dtype=torch.long)

        # epoch == 1 时，按照 FreeAL 保存 feedback
        if self.learning_setting == "transductive" and epoch == 10:
            os.makedirs("./feedback", exist_ok=True)
            # clean 样本（类内 top-ρ & 预测正确）
            chosen_idx_sel = torch.where(chosen_list_sel > 0)[0].cpu()
            with open("./feedback/right_list_mr.pkl", "wb") as f:
                pickle.dump(chosen_idx_sel.tolist(), f)
            # demo 检索 index
            with open("./feedback/demo_index_mr.pkl", "wb") as f:
                pickle.dump(chosen_top_indices.tolist(), f)
            # teacher pseudo labels（用 pred_idx_all）
            with open("./feedback/pred_label_mr.pkl", "wb") as f:
                pickle.dump(pred_idx_all.cpu().tolist(), f)
            print("[FreeAL] Saved feedback to ./feedback/right_list_mr.pkl / demo_index_mr.pkl / pred_label_mr.pkl")

        # 更新当前 epoch 的 chosen_list（用于训练阶段的 clean / unclean 划分）
        self.chosen_list = chosen_list.to(device)

    # ==================== 训练时的 compute_loss（完全按 FreeAL 的思路） ==================== #

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        FreeAL-style loss：

        - pseudo label：candidate_labels 第一项
        - clean subset：由 self.chosen_list[global_col_indices] 决定
        - 在 clean subset 上：
            * CE
            * embedding-level mixup
        - 可选：Consistency Regularization（CR）：
            * 对 back-translated batch（input_ids_bt）做 CE
            * 对 unclean 部分做 symmetrical KL
        """
        # 计算 ramp-up 权重
        current_epoch = getattr(self.state, "epoch", 0.0) or 0.0
        w = linear_rampup(current_epoch, 10)

        # 是否处于训练（有 BT 样本）
        is_in_train = ("input_ids_bt" in inputs)

        # pseudo labels from candidate_labels
        cand_list = inputs["candidate_labels"]
        pseudo_labels = torch.tensor(
            [(c[0] if isinstance(c, (list, tuple)) and len(c) > 0 else 0) for c in cand_list],
            dtype=torch.long,
            device=self.args.device,
        )

        if is_in_train:
            batch_gids = inputs["global_col_indices"].to(self.args.device)
            # clean 样本 mask
            batch_mask = self.chosen_list[batch_gids.long()].bool()
            # unclean / 未选择的样本
            batch_mask_un = (~batch_mask)
        else:
            # eval 或无 BT：所有都视为 clean
            batch_mask = torch.ones_like(pseudo_labels, dtype=torch.bool, device=self.args.device)
            batch_mask_un = torch.zeros_like(pseudo_labels, dtype=torch.bool, device=self.args.device)

        index_x = torch.where(batch_mask)[0]   # clean subset
        index_u = torch.where(batch_mask_un)[0]  # unclean subset

        # -------- forward（取 CLS logits + CLS embedding） -------- #
        input_ids = inputs["input_ids"].to(self.args.device)
        logits, embeddings = self._forward_with_features(model, input_ids, return_features=True)
        if embeddings is None:
            raise ValueError("模型在 return_features=True 时必须返回 embedding / hidden_states。")

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss_fct_un = nn.KLDivLoss(reduction="sum")

        # -------- 1) clean subset CE -------- #
        if index_x.numel() > 0:
            loss_l = loss_fct(logits[index_x], pseudo_labels[index_x]).mean()
        else:
            loss_l = torch.tensor(0.0, device=self.args.device)

        # -------- 2) embedding-level mixup on clean subset（FreeAL） -------- #
        if index_x.numel() > 1:
            valid_embeddings = embeddings[index_x]
            with torch.no_grad():
                targets_l = F.one_hot(pseudo_labels[index_x].detach(), num_classes=self.num_labels).float()
                all_targets = targets_l.detach()
            rand_idx = torch.randperm(valid_embeddings.size(0))
            lam = np.random.beta(4, 4)
            lam = max(lam, 1 - lam)
            mixed_embeddings = lam * valid_embeddings + (1 - lam) * valid_embeddings[rand_idx]
            mixed_targets = lam * all_targets + (1 - lam) * all_targets[rand_idx]
            mixed_logits = model.classifier(mixed_embeddings) if hasattr(model, "classifier") else logits[index_x]
            loss_mix = -torch.mean(torch.sum(F.log_softmax(mixed_logits, dim=-1) * mixed_targets, dim=-1))
        else:
            loss_mix = torch.tensor(0.0, device=self.args.device)

        final_loss = loss_l + w * loss_mix

        # -------- 3) Consistency Regularization（可选，完全对齐 FreeAL） -------- #
        if is_in_train and index_x.numel() > 0:
            input_ids_bt = inputs["input_ids_bt"].to(self.args.device)
            logits_bt, _ = self._forward_with_features(model, input_ids_bt, return_features=False)

            # 有标签（clean subset）上的 CE
            loss_cr = loss_fct(logits_bt[index_x], pseudo_labels[index_x]).mean()

            # 对 unclean 部分做 symmetrical KL（两条增强之间的对称 KL）
            if index_u.numel() > 0:
                temp_u = getattr(self.args, "temp_u", 1.0)
                p1 = F.log_softmax(logits[index_u] / temp_u, dim=-1)
                q1 = F.softmax(logits_bt[index_u].detach().clone(), dim=-1)
                p2 = F.log_softmax(logits_bt[index_u] / temp_u, dim=-1)
                q2 = F.softmax(logits[index_u].detach().clone(), dim=-1)
                loss_cr_un = 0.5 * loss_fct_un(p1, q1) + 0.5 * loss_fct_un(p2, q2)
                loss_cr_un = loss_cr_un / index_u.numel()
            else:
                loss_cr_un = torch.tensor(0.0, device=self.args.device)

            final_loss = final_loss + w * (loss_cr + loss_cr_un)

        if return_outputs:
            return final_loss, logits
        return final_loss

    # ==================== Eval loop：用真标签做 CE + F1 ==================== #

    def evaluation_loop(
        self,
        dataloader: Any,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Any:
        self.model.eval()

        total_loss = 0.0
        all_predictions, all_true_labels = [], []
        all_logits = []

        if prediction_loss_only is None:
            prediction_loss_only = self.args.prediction_loss_only

        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                batch = self._prepare_inputs(batch)

                input_ids = batch["input_ids"].to(self.args.device)
                logits, _ = self._forward_with_features(self.model, input_ids, return_features=False)

                if not prediction_loss_only:
                    if "labels" not in batch:
                        raise ValueError("Evaluation requires 'labels' field for loss computation.")
                    loss = torch.nn.functional.cross_entropy(logits, batch["labels"].to(self.args.device))
                    total_loss += loss.item()

                predictions = logits.argmax(dim=-1)
                all_predictions.extend(predictions.cpu().detach().numpy().tolist())
                all_true_labels.extend(batch["labels"].cpu().detach().numpy().tolist())
                all_logits.extend(logits.cpu().detach().numpy().tolist())

        if not prediction_loss_only:
            avg_loss = total_loss / max(len(dataloader), 1)
        else:
            avg_loss = 0.0

        eval_pred = (np.array(all_logits), np.array(all_true_labels))

        metrics = {}
        if self.compute_metrics is not None:
            raw_metrics = self.compute_metrics(eval_pred)
            for k, v in raw_metrics.items():
                metrics[f"{metric_key_prefix}_{k}"] = v

        metrics[f"{metric_key_prefix}_loss"] = avg_loss

        return EvalLoopOutput(
            predictions=all_predictions,
            label_ids=all_true_labels,
            metrics=metrics,
            num_samples=len(all_predictions),
        )


# ==================== Callback：每个 epoch 开头调用 teacher pass ==================== #

class FreeALCallback(TrainerCallback):
    """
    在每个 epoch 开头调用 FreeALHFTrainer.build_chosen_list_for_epoch(epoch)，
    不用重写 _inner_training_loop。
    """

    def __init__(self, trainer: FreeALHFTrainer):
        self.trainer = trainer

    def on_epoch_begin(self, args, state, control, **kwargs):
        # state.epoch 是 float（当前 epoch + 进度），取 floor 即可
        epoch = int(state.epoch) if state.epoch is not None else 0
        self.trainer.build_chosen_list_for_epoch(epoch)


# ==================== metrics & factory ==================== #

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    micro_f1 = f1_score(labels, preds, average='micro')
    macro_f1 = f1_score(labels, preds, average='macro')

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }


def create_hf_trainer(
    model: nn.Module,
    config,
    train_dataset: Optional[Any] = None,
    eval_dataset: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    data_collator: Optional[Any] = None,
    type_ontology: Optional[List[str]] = None,
    **kwargs
) -> FreeALHFTrainer:
    """
    创建 FreeAL 版 Trainer，尽量保持你原来的 TrainingArguments 设定不变，
    只是额外塞入一些 FreeAL 需要的超参。
    """
    import swanlab
    swanlab.init(
        project=f"{config.project_name}",
        experiment_name=f"{config.experiment_name}"
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_num_workers=0,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        warmup_steps=getattr(config, 'warmup_steps', 0),
        fp16=False,
        bf16=True,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=1,
        logging_dir=config.output_dir,
        report_to=["swanlab"],
        logging_first_step=True,
        max_grad_norm=1.0,
    )

    # 补充 FreeAL 相关超参（可从外部 config 读取，也可以直接在 config 里加字段）
    training_args.temp_u = getattr(config, "temp_u", 1.0)
    training_args.warmup_epochs = getattr(config, "warmup_epochs", 1)
    training_args.embedding_dim = getattr(config, "embedding_dim", getattr(model.config, "hidden_size", 768))
    training_args.select_demo_num = getattr(config, "select_demo_num", 100)
    training_args.shot_num = getattr(config, "shot_num", 10)
    training_args.learning_setting = getattr(config, "learning_setting", "transductive")

    trainer = FreeALHFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        type_ontology=type_ontology,
        **kwargs,
    )

    # 注册 FreeAL callback：每个 epoch 开头跑 teacher pass
    trainer.add_callback(FreeALCallback(trainer))

    return trainer
