import os
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm
from collections import defaultdict
from torchvision import transforms
from albumentations import (
    Compose,
    OneOf,
    RandomBrightnessContrast,
    RandomGamma,
    ShiftScaleRotate,
)


ImageFile.LOAD_TRUNCATED_IMAGES = True

TRAIN_PATH = "../../../datasets/rsna-pneumonia-detection-challenge/train_png/"


class SIIMDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            image_ids,
            transform=True,
            preprocessing_fn=None,
    ):
        """
        セグメンテーション用のデータセットのクラス
        :param image_ids: 画像インデックスのリスト
        :param transform: データ拡張するか否かの真偽値、検証時はデータ拡張しない
        :param preprocessing_fn: 画像の前処理の関数
        """

        # 元画像とマスクのパスを格納する辞書
        self.data = defaultdict(dict)

        # データ拡張
        self.transform = transform

        # 画像を正規化する前処理の関数
        self.preprocessing_fn = preprocessing_fn

        # albumentations によるデータ拡張
        # 移動・拡大縮小・回転を 80% の確率で適用
        self.aug = Compose(
            [
                ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=10,
                    p=0.8
                ),
                OneOf(
                    [
                        RandomGamma(gamma_limit=(90, 110)),
                        RandomBrightnessContrast(
                            brightness_limit=0.1,
                            contrast_limit=0.1,
                        ),
                    ],
                    p=0.5,
                ),
            ]
        )

        # すべての画像インデックスについてのループ
        for index, imgid in enumerate(image_ids):
            # files = glob.glob(os.path.join(TRAIN_PATH, imgid, "*.png"))
            self.data[index] = {
                "img_path": os.path.join(TRAIN_PATH, imgid + ".png"),
                "mask_path": os.path.join(TRAIN_PATH, imgid + "_mask.png"),
            }

    def __len__(self):
        # データセットの大きさを返す
        return len(self.data)

    def __getitem__(self, item):
        # 指定されたインデックスに対して、元画像とマスクを読み込んで返す
        img_path = self.data[item]["img_path"]
        mask_path = self.data[item]["mask_path"]

        # 画像を読み込んで RGB に変換
        img = Image.open(img_path)
        img = img.convert("RGB")

        # PIL 形式の画像を numpy 配列に変換
        img = np.array(img)

        # マスク画像の読み込み
        mask = Image.open(mask_path)

        # float32 型の二値行列に変換
        mask = (mask >= 1).astype("float32")

        # 学習用データセットに対してはデータ拡張を適用
        if self.transform is True:
            augmented = self.aug(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # 画像の前処理
        # ここでは正規化
        if self.preprocessing_fn is not None:
            img = self.preprocessing_fn(img)

        # 画像とマスクのテンソルを返す
        return {
            "image": transforms.ToTensor()(img),
            "mask": transforms.ToTensor()(mask).float(),
        }
