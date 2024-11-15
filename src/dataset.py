import os
from dataclasses import dataclass
from typing import Dict

import torch
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer

from src.data_model import Pic, TrainingData


@dataclass
class DataConfig:
    """数据配置类"""

    database_url: str
    gcs_mount_path: str  # GCS挂载的本地路径
    image_size: int = 256
    max_txt_length: int = 77

    def convert_url_to_path(self, url: str) -> str:
        """将原始URL转换为本地路径"""
        # 假设url格式为: gs://bucket_name/path/to/image.jpg
        # 将转换为: /mount_path/path/to/image.jpg
        if url.startswith("gs://"):
            relative_path = "/".join(url.split("/")[3:])  # 移除 "gs://bucket_name/"
            return os.path.join(self.gcs_mount_path, relative_path)
        return url


class FluxDataset(Dataset):
    """用于Flux模型训练的数据集类"""

    def __init__(
        self,
        config: DataConfig,
        batch_size: int,
        tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("bert-base-chinese"),
    ):
        self.config = config
        self.batch_size = batch_size
        self.tokenizer = tokenizer

        # 使用SQLAlchemy连接数据库
        self.engine = create_engine(config.database_url)
        self.session = Session(self.engine)

        # 获取训练数据总数
        self.length = (
            self.session.query(TrainingData).join(TrainingData.pic).filter(Pic.saved.is_(True)).count()
        )

        # 图像预处理
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return self.length

    def _load_image(self, url: str) -> torch.Tensor:
        """加载并预处理图像"""
        image_path = self.config.convert_url_to_path(url)
        try:
            image = Image.open(image_path).convert("RGB")
            return self.image_transforms(image)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {str(e)}")

    def _generate_position_ids(self, sequence_length: int) -> torch.Tensor:
        """生成位置编码ID"""
        return torch.arange(sequence_length, dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个数据样本"""
        # 从数据库获取训练数据
        training_data = (
            self.session.query(TrainingData)
            .join(TrainingData.pic)
            .filter(Pic.saved.is_(True))
            .offset(idx)
            .first()
        )

        if not training_data:
            raise IndexError(f"No training data found at index {idx}")

        # 加载图像
        image = self._load_image(training_data.pic.url)

        # 处理文本描述
        tokens = self.tokenizer(
            training_data.text,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_txt_length,
            return_tensors="pt",
        )
        text_tokens = tokens.input_ids[0]

        # 生成位置ID
        img_length = image.shape[-1] // self.batch_size
        img_ids = self._generate_position_ids(img_length * img_length)
        txt_ids = self._generate_position_ids(len(text_tokens))

        # 生成时间步和条件向量
        timesteps = torch.rand(1)
        y = torch.randn(512)
        guidance = torch.tensor([7.5])

        # 构建训练目标
        target = torch.randn_like(image)

        return {
            "img": image.reshape(-1, 3),
            "img_ids": img_ids,
            "txt": text_tokens.unsqueeze(0),
            "txt_ids": txt_ids,
            "timesteps": timesteps,
            "y": y,
            "guidance": guidance,
            "target": target.reshape(-1, 3),
        }

    def __del__(self):
        """关闭数据库连接"""
        self.session.close()


def create_dataloader(
    database_url: str,
    gcs_mount_path: str,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    image_size: int = 256,
    max_txt_length: int = 77,
) -> torch.utils.data.DataLoader:
    """创建数据加载器"""
    config = DataConfig(
        database_url=database_url,
        gcs_mount_path=gcs_mount_path,
        image_size=image_size,
        max_txt_length=max_txt_length,
    )

    dataset = FluxDataset(
        config=config,
        batch_size=batch_size,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
