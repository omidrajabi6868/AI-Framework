from abc import ABC, abstractmethod
import math
import time
import torch
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from Validation.DetectionSegmentation import ValDetectionSegmentation
from Network.YoloIPC8 import YOLOIPC8Loss, YOLOIPC8InstanceSegLoss, IPC8Predictor
from Network.utils import load_state_dict_compat
from pathlib import Path
from Train.utils import load_pretrained_yolo


class ITrain(ABC):
    @abstractmethod
    def start(self, epoch_num, device):
        pass
    @abstractmethod
    def optimization_algorithm(self, name, params):
        pass 
    @abstractmethod
    def make_scheduler(self, name):
        pass
    @abstractmethod
    def make_cost_function(self, name):
        pass


class TBase(ITrain):
    def __init__(self, structure, 
                 dataset, 
                 optimizer_name='Adam',
                 learning_rate=1e-3,
                 isScheduled=False, 
                 scheduler_name="CosineAnnealingLR",
                 cost_function_name='object_detection_cost', 
                 **extras):
        
        self.structure = structure
        self.dataset = dataset
        self.lr = learning_rate
        self.isScheduled = isScheduled
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.optimizer = None
        self.cost_function_name = cost_function_name 
        self.extras = extras
        pass

    def start(self, saving_weights_file, epoch_num=20, report_epoch=10, device='cuda'):
        pass

    @staticmethod
    def _format_loss_components(components):
        ordered_keys = ["box", "cls", "dfl", "seg"]
        parts = []
        for key in ordered_keys:
            if key in components:
                parts.append(f"{key}: {components[key]:.4f}")
        for key, value in components.items():
            if key not in ordered_keys:
                parts.append(f"{key}: {value:.4f}")
        return " ".join(parts)

    def _print_epoch_loss_summary(self, epoch, epoch_num, stage, loss_value, components):
        details = self._format_loss_components(components)
        print(f"Epoch {epoch+1}/{epoch_num} - {stage} loss: {loss_value:.4f} {details}".rstrip())

    @staticmethod
    def _history_csv_header():
        return [
            "epoch",
            "train_loss",
            "train_box",
            "train_cls",
            "train_dfl",
            "train_seg",
            "val_loss",
            "val_box",
            "val_cls",
            "val_dfl",
            "val_seg",
        ]

    @staticmethod
    def _history_row(epoch, train_stats, val_stats):
        return [
            epoch,
            f"{train_stats.get('loss', 0.0):.8f}",
            f"{train_stats.get('box', 0.0):.8f}",
            f"{train_stats.get('cls', 0.0):.8f}",
            f"{train_stats.get('dfl', 0.0):.8f}",
            f"{train_stats.get('seg', 0.0):.8f}",
            f"{val_stats.get('loss', 0.0):.8f}",
            f"{val_stats.get('box', 0.0):.8f}",
            f"{val_stats.get('cls', 0.0):.8f}",
            f"{val_stats.get('dfl', 0.0):.8f}",
            f"{val_stats.get('seg', 0.0):.8f}",
        ]

    def _init_loss_tracking(self, saving_weights_file, resume):
        train_losses = []
        val_losses = []
        backup_dir = Path(saving_weights_file).parent
        backup_dir.mkdir(parents=True, exist_ok=True)
        loss_log_file = backup_dir / "loss_history.csv"
        loss_chart_file = backup_dir / "loss_curve.png"

        if resume and loss_log_file.exists():
            with loss_log_file.open("r", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    try:
                        train_losses.append(float(row["train_loss"]))
                        val_losses.append(float(row["val_loss"]))
                    except (KeyError, ValueError, TypeError):
                        continue
            if train_losses and val_losses:
                print(f"Loaded {len(train_losses)} historical loss points from {loss_log_file}")
        else:
            with loss_log_file.open("w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(self._history_csv_header())

        return train_losses, val_losses, loss_log_file, loss_chart_file

    def _append_loss_history(self, loss_log_file, epoch, train_stats, val_stats):
        with loss_log_file.open("a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(self._history_row(epoch, train_stats, val_stats))

    def _resume_from_checkpoint(self, saving_weights_file, device):
        best_val_loss = float("inf")
        if not Path(f"{saving_weights_file}_best.pt").exists():
            return best_val_loss

        ckpt = torch.load(f"{saving_weights_file}_last.pt", map_location=device)
        direction, remapped = load_state_dict_compat(self.structure.model, ckpt["model_state_dict"], strict=True)
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if not self.isScheduled:
                for group in self.optimizer.param_groups:
                    group["lr"] = float(self.lr)
                    if "initial_lr" in group:
                        group["initial_lr"] = float(self.lr)
                print(f"Scheduler disabled: reset resumed optimizer LR to {self.lr:.8f}")
        print(f"Loaded checkpoint from {saving_weights_file}_last.pt")
        if remapped > 0:
            print(f"Checkpoint key remap applied: {direction} ({remapped} tensors)")
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        print(f"best_val_loss:{best_val_loss}")
        return best_val_loss
    
    def optimization_algorithm(self, name, params):
        if name == 'Adam':
            self.optimizer = torch.optim.Adam(params=params,
                                              lr=self.lr, 
                                              weight_decay=self.extras.get("weight_decay", 0),
                                              betas=self.extras.get('betas', (0.9, 0.999)),
                                              eps=self.extras.get('eps', 1e-8))
   
    def make_scheduler(self, name):
        if name == 'CosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, 
                                                            T_max=self.extras.get('T_max', 100),
                                                            eta_min=self.extras.get('eta_min', 0.00001),
                                                            last_epoch=self.extras.get('last_epoch', -1)
                                                            )
        elif name == 'OneCycleLR':
            return torch.optim.lr_scheduler.OneCycleLR(optimizer = self.optimizer, 
                                                    max_lr=self.extras.get('max_lr', self.lr),                 # Default to base LR if not provided
                                                    total_steps=self.extras.get('total_steps', 100),           # Total epochs
                                                    pct_start=self.extras.get('pct_start', 0.1),               # Spend 10% of time warming up (10 epochs)
                                                    anneal_strategy=self.extras.get('anneal_strategy', 'cos'), # Use cosine shape
                                                    final_div_factor=self.extras.get('final_div_factor', 1000) # Final LR will be max_lr / 1000
                                                    )
    
    def make_cost_function(self, name):
        if name == 'yolo_cost_function':
            task_name = str(getattr(self.structure, "task", "detection")).lower()
            if task_name == "instance_segmentation":
                return YOLOIPC8InstanceSegLoss(
                    nc=self.dataset.class_num,
                    reg_max=self.structure.reg_max,
                    class_balance=self.extras.get("class_balance", True),
                    class_balance_power=self.extras.get("class_balance_power", 1.0),
                    class_balance_min=self.extras.get("class_balance_min", 0.25),
                    class_balance_max=self.extras.get("class_balance_max", 4.0),
                    class_balance_eps=self.extras.get("class_balance_eps", 1e-6),
                    seg_weight=self.extras.get("seg_weight", 1.0),
                    mask_bce_weight=self.extras.get("mask_bce_weight", 1.0),
                    mask_dice_weight=self.extras.get("mask_dice_weight", 1.0),
                )
            return YOLOIPC8Loss(
                nc=self.dataset.class_num,
                reg_max=self.structure.reg_max,
                class_balance=self.extras.get("class_balance", True),
                class_balance_power=self.extras.get("class_balance_power", 1.0),
                class_balance_min=self.extras.get("class_balance_min", 0.25),
                class_balance_max=self.extras.get("class_balance_max", 4.0),
                class_balance_eps=self.extras.get("class_balance_eps", 1e-6)
            )
        
    @staticmethod
    def get_gpu_status(device_id=0):
        # 1. Current memory used by your tensors
        allocated = torch.cuda.memory_allocated(device_id) / 1024**2
        
        # 2. Total memory PyTorch has cached/reserved from the GPU
        reserved = torch.cuda.memory_reserved(device_id) / 1024**2
        
        # 3. The absolute peak memory reached during the session
        max_mem = torch.cuda.max_memory_allocated(device_id) / 1024**2

        print(f"Allocated: {allocated:.2f} MB")
        print(f"Reserved:  {reserved:.2f} MB")
        print(f"Peak:      {max_mem:.2f} MB")
    
    @staticmethod
    def _save_loss_chart(train_losses, val_losses, output_file):
        if not train_losses or not val_losses:
            return
        epochs = list(range(1, len(train_losses) + 1))
        plt.style.use("seaborn-v0_8-whitegrid")
        width = max(9.0, min(20.0, 9.0 + (len(epochs) / 30.0)))
        fig, ax = plt.subplots(figsize=(width, 6.0), dpi=120)

        marker_size = 3 if len(epochs) > 100 else 4
        ax.plot(epochs, train_losses, color="#1f77b4", marker="o", linewidth=2.2, markersize=marker_size, label="Train Loss")
        ax.plot(epochs, val_losses, color="#d62728", marker="s", linewidth=2.2, markersize=marker_size, label="Validation Loss")

        best_idx = min(range(len(val_losses)), key=lambda i: val_losses[i])
        best_epoch = best_idx + 1
        best_loss = val_losses[best_idx]
        ax.scatter([best_epoch], [best_loss], color="#2ca02c", s=60, zorder=4, label="Best Val Loss")
        ax.annotate(
            f"Best: {best_loss:.4f} (E{best_epoch})",
            xy=(best_epoch, best_loss),
            xytext=(10, -16),
            textcoords="offset points",
            fontsize=10,
            color="#2ca02c",
            fontweight="bold",
        )

        ax.set_title("Training and Validation Loss", fontsize=15, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.legend(frameon=True, fancybox=True)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=min(12, max(5, len(epochs) // 10 + 1)), integer=True))
        ax.grid(True, which="major", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(output_file)
        plt.close(fig)

            
class TYOLO(TBase):
    def __init__(self, structure, dataset, optimizer_name='Adam', learning_rate=0.001, isScheduled=False, scheduler_name="CosineAnnealingLR", cost_function_name='object_detection_cost', **extras):
        super().__init__(structure, dataset, optimizer_name, learning_rate, isScheduled, scheduler_name, cost_function_name, **extras)
        self.task = str(getattr(structure, "task", "detection")).lower()

    def start(self, saving_weights_file, epoch_num=20, report_epoch=10, resume=True, device='cuda'):
        self.optimization_algorithm(name="Adam", params = self.structure.model.parameters())
        grad_accum_steps = max(1, int(self.extras.get("subdivisions", self.extras.get("grad_accum_steps", 1))))
        use_amp = bool(self.extras.get("use_amp", True)) and str(device).startswith("cuda") and torch.cuda.is_available()
        grad_clip_norm = float(self.extras.get("grad_clip_norm", 10.0))
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        total_trainable = sum(p.numel() for p in self.structure.model.parameters() if p.requires_grad)
        print("Trainable parameters:", total_trainable)
        print(f"Gradient accumulation (subdivisions): {grad_accum_steps}")
        print(f"AMP enabled: {use_amp}")
        print(f"Gradient clip norm: {grad_clip_norm:.2f}")

        total = sum(len(g["params"]) for g in self.optimizer.param_groups)
        print("total tensors in optimizer:", total)

        opt_trainable = sum(p.numel() for g in self.optimizer.param_groups for p in g["params"] if p.requires_grad)
        print("Parameters in optimizer:", opt_trainable)
        
        cost_function = self.make_cost_function(self.cost_function_name)

        self.structure.model.to(device)


        train_loader, val_loader = self.dataset.get_train_val_loaders()
        print("Number of training batches:", len(train_loader))
        print("Number of validation batches:", len(val_loader))
        print(f"Training started on device={device} for {epoch_num} epochs")

        scheduler = None
        if self.isScheduled:
            if self.scheduler_name == "OneCycleLR":
                steps_per_epoch = max(1, len(train_loader))
                total_steps = int(math.ceil(epoch_num * steps_per_epoch / max(1, grad_accum_steps)))
                self.extras["total_steps"] = total_steps
                print(f"OneCycleLR total_steps={total_steps} (steps_per_epoch={steps_per_epoch}, grad_accum={grad_accum_steps})")
            scheduler = self.make_scheduler(name=self.scheduler_name)

        pretrained_weights = self.extras.get("pretrained_weights")
        if pretrained_weights:
            pretrained_path = Path(pretrained_weights)
            if pretrained_path.exists():
                self.structure.model = load_pretrained_yolo(self.structure.model, str(pretrained_path))
                print(f"Loaded pretrained bootstrap weights from {pretrained_path}")
            else:
                print(f"Warning: pretrained_weights not found, skipping bootstrap load: {pretrained_path}")

        best_val_loss = float("inf")
        if resume:
            best_val_loss = self._resume_from_checkpoint(saving_weights_file, device)
        
        last_reported_size = None
        last_reported_lr = self.optimizer.param_groups[0]["lr"]
        print(f"Initial learning rate: {last_reported_lr:.8f}")
        train_losses, val_losses, loss_log_file, loss_chart_file = self._init_loss_tracking(
            saving_weights_file=saving_weights_file,
            resume=resume,
        )

        for epoch in range(epoch_num):
            self.structure.model.train()
            running_train_loss = 0.0
            running_train_box = 0.0
            running_train_cls = 0.0
            running_train_dfl = 0.0
            running_train_seg = 0.0
            epoch_start = time.time()
            log_every = max(1, len(train_loader) // 10)
            print(f"\n[Epoch {epoch+1}/{epoch_num}] starting...")
            
            for batch_idx, (images, targets, _) in enumerate(train_loader, start=1):

                current_size = int(images.shape[-1])
                if current_size != last_reported_size:
                    print(
                        f"[Epoch {epoch+1}/{epoch_num}][Batch {batch_idx}/{len(train_loader)}] "
                        f"multiscale size changed to {current_size}x{current_size}"
                    )
                    last_reported_size = current_size

                # Darknet-like subdivisions: split one dataloader batch into micro-batches.
                batch_size = int(images.shape[0])
                num_chunks = max(1, min(grad_accum_steps, batch_size))
                chunk_size = max(1, (batch_size + num_chunks - 1) // num_chunks)

                self.optimizer.zero_grad(set_to_none=True)
                batch_loss = 0.0
                batch_box = 0.0
                batch_cls = 0.0
                batch_dfl = 0.0
                batch_seg = 0.0
                skip_batch = False

                for start in range(0, batch_size, chunk_size):
                    end = min(start + chunk_size, batch_size)
                    micro_images = images[start:end].to(device)
                    micro_targets = self.format_targets_for_ipc8(
                        targets[start:end],
                        micro_images,
                        device,
                        include_masks=(self.task == "instance_segmentation"),
                    )
                    micro_weight = float(end - start) / float(batch_size)

                    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                        outputs = self.structure.model(micro_images)

                    # Keep loss in FP32 only when AMP is enabled to avoid extra memory duplication.
                    outputs_for_loss = self._cast_outputs_for_loss(outputs, use_amp=use_amp)
                    loss, loss_items = cost_function(outputs_for_loss, micro_targets, return_items=True)

                    if not torch.isfinite(loss):
                        print(
                            f"[Epoch {epoch+1}/{epoch_num}][Batch {batch_idx}/{len(train_loader)}] "
                            "non-finite loss encountered; skipping optimizer update for this batch."
                        )
                        self.optimizer.zero_grad(set_to_none=True)
                        skip_batch = True
                        break

                    scaler.scale(loss / num_chunks).backward()
                    batch_loss += loss.item() * micro_weight
                    batch_box += loss_items["box"] * micro_weight
                    batch_cls += loss_items["cls"] * micro_weight
                    batch_dfl += loss_items["dfl"] * micro_weight
                    batch_seg += float(loss_items.get("seg", 0.0)) * micro_weight

                if skip_batch:
                    continue

                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.structure.model.parameters(), grad_clip_norm)
                scaler.step(self.optimizer)
                scaler.update()

                running_train_loss += batch_loss
                running_train_box += batch_box
                running_train_cls += batch_cls
                running_train_dfl += batch_dfl
                running_train_seg += batch_seg
                if batch_idx % log_every == 0 or batch_idx == len(train_loader):
                    elapsed = time.time() - epoch_start
                    if self.task == "instance_segmentation":
                        print(
                            f"[Epoch {epoch+1}/{epoch_num}][Batch {batch_idx}/{len(train_loader)}] "
                            f"loss={batch_loss:.4f} avg_loss={running_train_loss / batch_idx:.4f} "
                            f"box={batch_box:.4f} cls={batch_cls:.4f} dfl={batch_dfl:.4f} seg={batch_seg:.4f} "
                            f"lr={self.optimizer.param_groups[0]['lr']:.8f} elapsed={elapsed:.1f}s"
                        )
                    else:
                        print(
                            f"[Epoch {epoch+1}/{epoch_num}][Batch {batch_idx}/{len(train_loader)}] "
                            f"loss={batch_loss:.4f} avg_loss={running_train_loss / batch_idx:.4f} "
                            f"box={batch_box:.4f} cls={batch_cls:.4f} dfl={batch_dfl:.4f} "
                            f"lr={self.optimizer.param_groups[0]['lr']:.8f} elapsed={elapsed:.1f}s"
                        )
            
            train_loss = running_train_loss / max(1, len(train_loader))
            train_box = running_train_box / max(1, len(train_loader))
            train_cls = running_train_cls / max(1, len(train_loader))
            train_dfl = running_train_dfl / max(1, len(train_loader))
            train_seg = running_train_seg / max(1, len(train_loader))
            train_losses.append(train_loss)
            train_components = {
                "box": train_box,
                "cls": train_cls,
                "dfl": train_dfl,
            }
            if self.task == "instance_segmentation":
                train_components["seg"] = train_seg
            self._print_epoch_loss_summary(epoch, epoch_num, "train", train_loss, train_components)

            if scheduler is not None:
                scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                if current_lr != last_reported_lr:
                    print(
                        f"[Epoch {epoch+1}/{epoch_num}] learning rate changed: "
                        f"{last_reported_lr:.8f} -> {current_lr:.8f}"
                    )
                    last_reported_lr = current_lr

            with torch.no_grad():
                self.structure.model.eval()
                running_val_loss = 0.0
                running_val_box = 0.0
                running_val_cls = 0.0
                running_val_dfl = 0.0
                running_val_seg = 0.0

                for images, targets, _ in val_loader:
                    images = images.to(device)
                    targets = self.format_targets_for_ipc8(
                        targets,
                        images,
                        device,
                        include_masks=(self.task == "instance_segmentation"),
                    )

                    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                        outputs = self.structure.model(images)
                    outputs_for_loss = self._cast_outputs_for_loss(outputs, use_amp=use_amp)
                    loss, loss_items = cost_function(outputs_for_loss, targets, return_items=True)

                    if not torch.isfinite(loss):
                        print(
                            f"[Epoch {epoch+1}/{epoch_num}] "
                            "non-finite validation loss encountered; skipping this val batch."
                        )
                        continue

                    running_val_loss += loss.item()
                    running_val_box += loss_items["box"]
                    running_val_cls += loss_items["cls"]
                    running_val_dfl += loss_items["dfl"]
                    running_val_seg += float(loss_items.get("seg", 0.0))

                val_loss = running_val_loss / max(1, len(val_loader))
                val_box = running_val_box / max(1, len(val_loader))
                val_cls = running_val_cls / max(1, len(val_loader))
                val_dfl = running_val_dfl / max(1, len(val_loader))
                val_seg = running_val_seg / max(1, len(val_loader))
                val_losses.append(val_loss)
                val_components = {
                    "box": val_box,
                    "cls": val_cls,
                    "dfl": val_dfl,
                }
                if self.task == "instance_segmentation":
                    val_components["seg"] = val_seg
                self._print_epoch_loss_summary(epoch, epoch_num, "val", val_loss, val_components)

                train_stats = {
                    "loss": train_loss,
                    "box": train_box,
                    "cls": train_cls,
                    "dfl": train_dfl,
                    "seg": train_seg,
                }
                val_stats = {
                    "loss": val_loss,
                    "box": val_box,
                    "cls": val_cls,
                    "dfl": val_dfl,
                    "seg": val_seg,
                }
                self._append_loss_history(
                    loss_log_file=loss_log_file,
                    epoch=len(train_losses),
                    train_stats=train_stats,
                    val_stats=val_stats,
                )
                self._save_loss_chart(train_losses, val_losses, loss_chart_file)


                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({"epoch": epoch,
                                "model_state_dict": self.structure.model.state_dict(),
                                "optimizer_state_dict": self.optimizer.state_dict(),
                                "best_val_loss": best_val_loss}, f"{saving_weights_file}_best.pt")
                    print(f'The best checkpoint is saved in {saving_weights_file}_best.pt')
                

                torch.save({"epoch": epoch,
                                "model_state_dict": self.structure.model.state_dict(),
                                "optimizer_state_dict": self.optimizer.state_dict(),
                                "best_val_loss": best_val_loss}, f"{saving_weights_file}_last.pt")
                print(f'The last checkpoint is saved in {saving_weights_file}_last.pt')

                if (epoch + 1) % report_epoch == 0:
                    self.structure.build_predictor(self.dataset.target_classes)
                    val = ValDetectionSegmentation(self.structure.predictor, self.dataset)
                    val.start()

    @staticmethod
    def _cast_outputs_for_loss(outputs, use_amp: bool):
        if not use_amp:
            return outputs
        if isinstance(outputs, list):
            return [o.float() for o in outputs]
        if isinstance(outputs, dict):
            out = {}
            for k, v in outputs.items():
                if torch.is_tensor(v):
                    out[k] = v.float()
                elif isinstance(v, list):
                    out[k] = [vv.float() if torch.is_tensor(vv) else vv for vv in v]
                else:
                    out[k] = v
            return out
        return outputs

    def format_targets_for_ipc8(self, targets, images, device, include_masks: bool = False):
        """Convert normalized xywh targets to absolute xyxy expected by YOLOIPC8Loss."""
        _, _, h, w = images.shape
        formatted_targets = []

        for target in targets:
            boxes = target["boxes"].to(device)
            labels = target["labels"].to(device).long()
            masks = target.get("masks")
            if include_masks and masks is not None and torch.is_tensor(masks):
                masks = (masks.to(device).to(torch.float32) > 0.5).to(torch.float32)
            elif include_masks:
                masks = torch.zeros((0, h, w), dtype=torch.float32, device=device)

            if boxes.numel() == 0:
                item = {"boxes": boxes.reshape(0, 4), "labels": labels}
                if include_masks:
                    item["masks"] = masks if masks is not None else torch.zeros((0, h, w), dtype=torch.float32, device=device)
                formatted_targets.append(item)
                continue

            x_center = boxes[:, 0]
            y_center = boxes[:, 1]
            bw = boxes[:, 2]
            bh = boxes[:, 3]

            x1 = (x_center - bw / 2.0) * w
            y1 = (y_center - bh / 2.0) * h
            x2 = (x_center + bw / 2.0) * w
            y2 = (y_center + bh / 2.0) * h

            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
            boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clamp(0, w - 1)
            boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clamp(0, h - 1)

            item = {
                "boxes": boxes_xyxy,
                "labels": labels,
            }
            if include_masks:
                n = min(int(boxes_xyxy.shape[0]), int(masks.shape[0]) if masks is not None else 0)
                item["boxes"] = boxes_xyxy[:n]
                item["labels"] = labels[:n]
                item["masks"] = masks[:n] if masks is not None else torch.zeros((0, h, w), dtype=torch.float32, device=device)
            formatted_targets.append(item)

        return formatted_targets
    
    def optimization_algorithm(self, name, params):
        return super().optimization_algorithm(name, params)
    
    def make_scheduler(self, name):
        return super().make_scheduler(name)
