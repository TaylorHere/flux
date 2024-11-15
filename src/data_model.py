import asyncio
import time
from collections import deque

from sqlalchemy import (
    Boolean,
    Column,
    ForeignKey,
    Integer,
    String,
    Table,
    select,
    text,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

DB_LOCK = asyncio.Lock()  # 添加数据库锁
DB_QUEUE = deque()  # 添加数据库队列

MAPPING = {
    "Gender:": "gender",
    "Country:": "country",
    "Ethnicity:": "ethnicity",
    "Nationality:": "nationality",
    "Birthday:": "birthday",
    "Height:": "height",
    "Weight:": "weight",
    "Hair color:": "hair_color",
    # "Tattoos:": "tattoos",  # 不在对象属性中，需要决定如何处理
    "Breast size:": "breast_size",
    "Breast type:": "breast_type",
    "Categories:": "categories",
}

Base = declarative_base()

# 关系表定义
gallery_pics = Table(
    "gallery_pics",
    Base.metadata,
    Column("gallery_id", String, ForeignKey("galleries.id")),
    Column("pic_id", String, ForeignKey("pics.id")),
)

gallery_channels = Table(
    "gallery_channels",
    Base.metadata,
    Column("gallery_id", String, ForeignKey("galleries.id")),
    Column("channel_url", String, ForeignKey("channels.url")),
)

gallery_stars = Table(
    "gallery_stars",
    Base.metadata,
    Column("gallery_id", String, ForeignKey("galleries.id")),
    Column("star_url", String, ForeignKey("stars.url")),
)

gallery_categories = Table(
    "gallery_categories",
    Base.metadata,
    Column("gallery_id", String, ForeignKey("galleries.id")),
    Column("category_id", String, ForeignKey("categories.id")),
)

gallery_tags = Table(
    "gallery_tags",
    Base.metadata,
    Column("gallery_id", String, ForeignKey("galleries.id")),
    Column("tag_id", String, ForeignKey("tags.id")),
)


class Category(Base):
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)

    galleries = relationship(
        "Gallery", secondary=gallery_categories, back_populates="categories"
    )  # 修复了Category类中缺少的galleries关系


class Tag(Base):
    __tablename__ = "tags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)


class Star(Base):
    __tablename__ = "stars"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String)
    name = Column(String)
    gender = Column(String)
    country = Column(String)
    ethnicity = Column(String)
    nationality = Column(String)
    birthday = Column(String)
    height = Column(String)
    weight = Column(String)
    hair_color = Column(String)
    breast_size = Column(String)
    breast_type = Column(String)
    categories = Column(String)
    meta_saved = Column(Boolean, default=False)

    galleries = relationship("Gallery", secondary=gallery_stars, back_populates="stars")


class Pic(Base):
    __tablename__ = "pics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String)
    saved = Column(Boolean, default=False)

    galleries = relationship("Gallery", secondary=gallery_pics, back_populates="pics")


class Gallery(Base):
    __tablename__ = "galleries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String)
    rating = Column(String)
    views = Column(String)

    pics = relationship("Pic", secondary=gallery_pics, back_populates="galleries")
    channels = relationship("Channel", secondary=gallery_channels, back_populates="galleries")
    stars = relationship("Star", secondary=gallery_stars, back_populates="galleries")
    categories = relationship(
        "Category", secondary=gallery_categories, back_populates="galleries"
    )  # 修复了Gallery类中缺少的categories关系
    tags = relationship("Tag", secondary=gallery_tags)


class Channel(Base):
    __tablename__ = "channels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String)
    name = Column(String)
    description = Column(String, default="")
    official_link = Column(String, default="")

    galleries = relationship("Gallery", secondary=gallery_channels, back_populates="channels")


class TrainingData(Base):
    __tablename__ = "training_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_path = Column(String)  # 图片文件路径
    text = Column(String)  # 对应的文本描述
    pic_id = Column(Integer, ForeignKey("pics.id"))  # 关联的图片ID

    pic = relationship("Pic", backref="training_data")


class PicsDataset:
    def __init__(self, dataset="sqlite+aiosqlite:///pics.db"):
        self.engine = create_async_engine(dataset)
        self.Session = sessionmaker(bind=self.engine, class_=AsyncSession)

    async def create_all(self):
        async with self.engine.begin() as conn:
            # 检查数据库是否已存在
            result = await conn.run_sync(
                lambda sync_conn: sync_conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                ).fetchall()
            )
            if not result:  # 如果没有表存在
                await conn.run_sync(Base.metadata.create_all)

    async def save(self, item):
        try:
            # 使用队列存储待保存的item
            DB_QUEUE.append(item)

            # 获取数据库锁
            async with DB_LOCK:
                # 如果队列不为空且距离上次提交超过30秒,则批量提交
                current_time = time.time()
                if DB_QUEUE and (
                    not hasattr(self, "_last_commit_time") or current_time - self._last_commit_time >= 30
                ):
                    async with self.Session() as session:
                        while DB_QUEUE:
                            session.add(DB_QUEUE.popleft())
                        await session.commit()
                        print(f"\033[32mSaved {len(DB_QUEUE)} items to database\033[0m")
                        self._last_commit_time = current_time

        except Exception as e:
            print(f"\033[31mError saving to database: {e}\033[0m")
            if "session" in locals():
                await session.rollback()

    async def get_gallery(self, url: str) -> Gallery | None:
        async with AsyncSession(self.engine) as session:
            async with session.begin():
                gallery = await session.execute(select(Gallery).filter(Gallery.url == url))
                return gallery.scalars().first()

    async def get_star(self, url):
        async with AsyncSession(self.engine) as session:  # 确保使用 AsyncSession
            async with session.begin():  # 开始一个事务
                star = await session.execute(select(Star).filter(Star.url == url))  # 使用 execute 方法
                return star.scalars().first()  # 获取第一个结果
