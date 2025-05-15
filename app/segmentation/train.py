import os
import sys
import torch

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim

# from apex import amp
from torch.cuda import amp
from collections import OrderedDict
from sklearn import model_selection
# from tqdm import tqdm
from torch.optim import lr_scheduler

from dataset import SIIMDataset


# 学習用の CSV ファイルの読み込み
TRAINING_CSV = "../../../datasets/rsna-pneumonia-detection-challenge/train.csv"

# 学習用と評価用のバッチサイズ
TRAINING_BATCH_SIZE = 16
TEST_BATCH_SIZE = 4

# エポック数
EPOCHS = 10

# U-Net のエンコーダの定義
# 対応しているエンコーダについては以下を参照
# https://github.com/qubvel/segmentation_models.pytorch
ENCODER = "resnet18"
# ImageNet で事前学習済みの重みをエンコーダで利用
ENCODER_WEIGHTS = "imagenet"

# GPU で学習
DEVICE = "cuda"

def train(dataset, data_loader, model, criterion, optimizer):
    """
    1 エポック学習する関数
    :param dataset: データセットのクラス (SIIMDataset)
    :param data_loader: データローダ
    :param model: モデル
    :param criterion: 損失関数
    :param optimizer: オプティマイザ
    """
    # モデルを学習モードに
    model.train()

    # バッチ数を計算
    num_batches = int(len(dataset) / data_loader.batch_size)

    # 進捗を可視化する tqdm の初期化
    tk0 = tqdm(data_loader, total=num_batches)

    # すべてのバッチについてのループ
    for d in tk0:
        # バッチから元画像とマスクを取り出す
        inputs = d["image"]
        targets = d["mask"]

        # 元画像とマスクをデバイスに転送
        inputs = inputs.to(DEVICE, dtype=torch.float)
        targets = targets.to(DEVICE, dtype=torch.float)

        # オプティマイザの勾配を 0 で初期化
        optimizer.zero_grad()

        # モデルの学習
        outputs = model(inputs)

        # 損失の計算
        loss = criterion(outputs, targets)

        # 混合精度学習における損失スケーリング
        # 混合精度学習を使わない場合は、この2行を削除して loss.backward() を利用
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        # パラメータ更新
        optimizer.step()

    # tqdm の終了
    tk0.close()

def evaluate(dataset, data_loader, model):
    """
    1 エポック評価する関数
    :param dataset: データセットのクラス (SIIMDataset)
    :param data_loader: データローダ
    :param model: モデル
    """
    # モデルを評価モードに
    model.eval()
    # 損失を 0 で初期化
    final_loss = 0
    # バッチ数を計算
    num_batches = int(len(dataset) / data_loader.batch_size)
    # 進捗を可視化する tqdm の初期化
    tk0 = tqdm(data_loader, total=num_batches)

    # メモリ節約のために勾配を計算しない
    with torch.no_grad():
        for d in tk0:
            inputs = d["image"]
            targets = d["mask"]

            inputs = inputs.to(DEVICE, dtype=torch.float)
            targets = targets.to(DEVICE, dtype=torch.float)

            output = model(inputs)
            loss = criterion(output, targets)
            # 計算した損失の追加
            final_loss += loss

    # tqdm の終了
    tk0.close()
    # バッチ平均の損失を返す
    return final_loss / num_batches

if __name__ == "__main__":
    # 学習用の CSV ファイルの読み込み
    df = pd.read_csv(TRAINING_CSV)

    # データセットを学習用と検証用に分割
    df_train, df_valid = model_selection.train_test_split(
        df, random_state=42, test_size=0.1
    )

    # 学習用と検証用の画像 ID
    training_images = df_train.patientId.values
    validation_images = df_valid.patientId.values

    # エンコーダ構造を指定して U-Net モデルを Segmentation Models Pytorch から取得
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=1,
        activation=None,
    )

    # Segmentation Models Pytorch では、正規化などの前処理の関数を指定できる
    # 正規化は元画像のみに適用し、マスクには適用しない
    prep_fn = smp.encoders.get_preprocessing_fn(
        ENCODER,
        ENCODER_WEIGHTS
    )

    # モデルをデバイスに転送
    model.to(DEVICE)

    # 学習用データセットの準備
    # データ拡張を適用
    train_dataset = SIIMDataset(
        training_images,
        transform=True,
        preprocessing_fn=prep_fn,
    )

    # データローダの準備
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAINING_BATCH_SIZE,
        shuffle=True,
        num_workers=12
    )

    # 検証用データセットの準備
    # データ拡張は適用しない
    valid_dataset = SIIMDataset(
        validation_images,
        transform=False,
        preprocessing_fn=prep_fn,
    )

    # 検証用データローダの準備
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    # 損失関数の定義
    criterion = nn.BCEWithLogitsLoss()

    # オプティマイザの定義
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 学習率スケジューラの定義
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, mode="min")

    # 混合精度学習の準備
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    # 学習ループ
    for epoch in range(EPOCHS):
        print(f"Training Epoch: {epoch}")
        # 1 エポック学習
        train(train_dataset, train_loader, model, criterion, optimizer)
        print(f"Validation Epoch: {epoch}")
        # 検証用データセットに対する損失の計算
        val_log = evaluate(
            valid_dataset,
            valid_loader,
            model
        )
        # スケジューラの更新
        scheduler.step(val_log["loss"])
        print(f"\n")
