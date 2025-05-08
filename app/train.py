import os
import pandas as pd
import numpy as np
import albumentations
import torch
from sklearn import metrics
from sklearn.model_selection import train_test_split

import dataset
import engine
from model import get_model


def main():
    data_path = '../../datasets/rsna-pneumonia-detection-challenge/'
    device = "mps"
    epochs = 10

    df = pd.read_csv(os.path.join(data_path, "train.csv"))

    images = df.patientId.values.tolist()
    images = [
        os.path.join(data_path, "train_png", i+".png") for i in images
    ]

    targets = df.Target.values

    model = get_model(pretrained=True)
    model.to(device)

    # TODO: 何をしているのか調べる -> https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    aug = albumentations.Compose(
        [
            albumentations.Normalize(
                mean, std, max_pixel_value=255.0
            )
        ]
    )

    # TODO: k-fold交差検証の代わりにホールドアウト検証を使うのはなぜ？
    train_images, valid_images, train_targets, valid_targets = train_test_split(
        images, targets, stratify=targets, random_state=42
    )

    train_dataset = dataset.ClassificationDataset(
        image_paths=train_images,
        targets=train_targets,
        resize=(227, 227),
        augmentations=aug
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=4
    )

    valid_dataset = dataset.ClassificationDataset(
        image_paths=valid_images,
        targets=valid_targets,
        resize=(227, 227),
        augmentations=aug
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=16, shuffle=True, num_workers=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    for epoch in range(epochs):
        engine.train(train_loader, model, optimizer, device=device)
        predictions, valid_targets = engine.evaluate(valid_loader, model, device=device)
        roc_auc= metrics.roc_auc_score(valid_targets, predictions)
        print(
            f"Epoch={epoch}, Valid ROC AUC={roc_auc}"
        )


if __name__ == '__main__':
    main()
