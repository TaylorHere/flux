import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from flux.model import Flux, FluxParams

from src.dataset import create_dataloader

logger = get_logger(__name__)


@dataclass
class TrainingArguments:
    output_dir: str = "outputs"
    num_train_epochs: int = 100
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    save_steps: int = 1000
    mixed_precision: Optional[str] = "fp16"
    gradient_checkpointing: bool = False
    resume_from_checkpoint: Optional[str] = None
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_total_limit: int = 5


def get_deepspeed_config():
    """配置DeepSpeed以实现混合并行策略"""
    return {
        "train_batch_size": 32,  # 全局批次大小
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "zero_optimization": {
            "stage": 2,  # 使用ZeRO-2优化
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
        },
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
        # 配置模型并行度
        "tensor_parallel": {
            "enabled": True,
            "size": 2,  # 每2张卡进行模型并行
        },
        # 配置流水线并行度
        "pipeline_parallel": {
            "enabled": False,  # 不使用流水线并行
        },
    }


def setup_accelerator(args: TrainingArguments):
    """设置Accelerator以支持混合并行训练"""
    deepspeed_plugin = DeepSpeedPlugin(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clipping=args.max_grad_norm,
        zero_stage=2,
        offload_optimizer_device="none",
        offload_param_device="none",
        zero3_init_flag=False,
        zero3_save_16bit_model=False,
        deepspeed_config=get_deepspeed_config(),
    )

    project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs")
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=project_config,
        deepspeed_plugin=deepspeed_plugin,
        split_batches=True,  # 启用批次拆分
        dispatch_batches=True,  # 启用批次分发
    )

    return accelerator


def setup_logging(args: TrainingArguments):
    """设置日志记录"""
    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志文件
    log_file = log_dir / f"training_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logger.add_file_handler(log_file)

    # 记录训练参数
    logger.info(f"Training arguments: {args}")


def cleanup_checkpoints(checkpoint_dir: Path, save_total_limit: int):
    """清理旧的检查点，只保留最新的几个"""
    checkpoints = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda x: int(x.name.split("-")[1]),
    )

    if len(checkpoints) > save_total_limit:
        for checkpoint in checkpoints[:-save_total_limit]:
            try:
                logger.info(f"Deleting old checkpoint: {checkpoint}")
                import shutil

                shutil.rmtree(checkpoint)
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint {checkpoint}: {e}")


def create_lora_model(model: Flux):
    """配置和创建LoRA模型"""
    lora_config = LoraConfig(
        r=16,  # LoRA秩
        lora_alpha=32,  # LoRA alpha参数
        target_modules=["img_in", "txt_in", "qkv", "out_proj"],  # 需要应用LoRA的模块
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    return model


def train(
    model: Flux,
    train_dataloader: DataLoader,
    args: TrainingArguments,
):
    try:
        setup_logging(args)

        # 初始化带有混合并行配置的accelerator
        accelerator = setup_accelerator(args)

        # 准备优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        # 将模型、优化器和数据加载器包装到accelerator中
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

        # 配置模型并行组
        if accelerator.state.deepspeed_plugin is not None:
            model.set_tensor_parallel_group(accelerator.state.deepspeed_plugin.tensor_parallel_group)

        # 总训练步数
        num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

        # 从检查点恢复
        starting_epoch = 0
        global_step = 0
        if args.resume_from_checkpoint:
            try:
                checkpoint_path = Path(args.resume_from_checkpoint)
                accelerator.load_state(checkpoint_path)
                step = int(checkpoint_path.name.split("-")[1])
                global_step = step
                starting_epoch = global_step // num_update_steps_per_epoch
                logger.info(f"Resumed from checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                logger.info("Starting training from scratch")

        # 进度条
        progress_bar = tqdm(
            total=max_train_steps,
            initial=global_step,
            disable=not accelerator.is_local_main_process,
        )

        # 训练循环
        for epoch in range(starting_epoch, args.num_train_epochs):
            model.train()
            total_loss = 0

            for step, batch in enumerate(train_dataloader):
                try:
                    with accelerator.accumulate(model):
                        # 前向传播
                        outputs = model(
                            img=batch["img"],
                            img_ids=batch["img_ids"],
                            txt=batch["txt"],
                            txt_ids=batch["txt_ids"],
                            timesteps=batch["timesteps"],
                            y=batch["y"],
                            guidance=batch.get("guidance", None),
                        )

                        loss = torch.nn.functional.mse_loss(outputs, batch["target"])

                        # 反向传播
                        accelerator.backward(loss)

                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                        optimizer.step()
                        optimizer.zero_grad()

                        total_loss += loss.detach().float()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if hasattr(torch.cuda, "empty_cache"):
                            torch.cuda.empty_cache()
                        logger.warning(f"OOM error in batch {step}, skipping...")
                        if "optimizer" in locals():
                            optimizer.zero_grad()
                        continue
                    raise e

                # 更新进度条和日志
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % args.logging_steps == 0:
                        avg_loss = total_loss / args.logging_steps
                        logger.info(f"Step {global_step}: loss = {avg_loss:.4f}")
                        total_loss = 0

                    # 保存检查点
                    if global_step % args.save_steps == 0:
                        if accelerator.is_main_process:
                            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                            accelerator.save_state(checkpoint_dir)
                            logger.info(f"Saved checkpoint: {checkpoint_dir}")

                            # 清理旧检查点
                            cleanup_checkpoints(Path(args.output_dir), args.save_total_limit)

            # 每个epoch结束时的日志
            avg_epoch_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch}: Average loss = {avg_epoch_loss:.4f}")

            # 保存最后一个epoch的检查点
            if accelerator.is_main_process:
                checkpoint_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                accelerator.save_state(checkpoint_dir)
                logger.info(f"Saved final checkpoint: {checkpoint_dir}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if accelerator.is_main_process:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-interrupted-{global_step}"
            accelerator.save_state(checkpoint_dir)
            logger.info(f"Saved interrupt checkpoint: {checkpoint_dir}")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

    finally:
        accelerator.free_memory()
        if "progress_bar" in locals():
            progress_bar.close()


def main():
    try:
        args = TrainingArguments()
        os.makedirs(args.output_dir, exist_ok=True)

        model_params = FluxParams(
            in_channels=3,
            vec_in_dim=512,
            context_in_dim=768,
            hidden_size=1024,
            mlp_ratio=4.0,
            num_heads=16,
            depth=12,
            depth_single_blocks=4,
            axes_dim=[16, 16, 16, 16],
            theta=10000,
            qkv_bias=True,
            guidance_embed=True,
        )

        model = Flux(model_params)
        model = create_lora_model(model)

        train_dataloader = create_dataloader(
            database_url="sqlite:///pics.db",
            batch_size=args.per_device_train_batch_size,
            num_workers=4,
            shuffle=True,
        )

        train(model, train_dataloader, args)

    except Exception as e:
        logger.error(f"Program failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
