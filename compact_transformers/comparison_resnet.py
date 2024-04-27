import os
from argparse import Namespace

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from timm.data import create_dataset, create_loader
from timm.utils import CheckpointSaver, AverageMeter, accuracy
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights


def create_resnet(num_classes, resnet, weights=None):
    model = resnet(weights=weights)
    model.fc = nn.Linear(512, num_classes)  # Modify output layer to produce batch size x num_classes
    return model


def train(args):
    # Load the pretrained ResNet on ImageNet
    model = create_resnet(args.num_classes, resnet34, ResNet34_Weights.IMAGENET1K_V1)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(param_count, "parameters")
    model.train()
    model.cuda()

    # Load the datasets for the train and validation sets from the coursework data folders
    train_dataset = create_dataset(
        args.dataset, root=args.data_dir, split=args.train_split, is_training=False, batch_size=args.batch_size)
    val_dataset = create_dataset(
        args.dataset, root=args.data_dir, split=args.test_split, is_training=False, batch_size=args.batch_size)

    # Create the train and validation loader
    train_loader = create_loader(
        train_dataset,
        input_size=args.img_size,
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        interpolation=args.interpolation,
        num_workers=args.workers,
        distributed=False,
        pin_memory=args.pin_mem,
    )
    val_loader = create_loader(
        val_dataset,
        input_size=args.img_size,
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=args.interpolation,
        num_workers=args.workers,
        distributed=False,
        pin_memory=args.pin_mem,
    )

    # Initialise the loss function, optimiser, and LR scheduler
    loss_fn = nn.CrossEntropyLoss().cuda()
    # optimiser = torch.optim.SGD(model.parameters(), args.lr_warmup,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    # scheduler = None
    optimiser = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = StepLR(optimiser, step_size=30, gamma=0.1)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=False)
    saver = CheckpointSaver(
        model=model, optimizer=optimiser, model_ema=None, checkpoint_dir=output_dir, recovery_dir=output_dir,
        decreasing=False, max_history=args.checkpoint_hist)

    for epoch in range(args.epochs):
        # top1_train = AverageMeter()

        for (batch, targets) in train_loader:
            if not args.prefetcher:
                batch = batch.cuda()
                targets = targets.cuda()

            outputs = model(batch)
            loss = loss_fn(outputs, targets)

            # Track training accuracy
            # acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            # torch.cuda.synchronize()
            # top1_train.update(acc1.item(), outputs.size(0))

            # Gradient descent
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        # Validate and save best weights
        top1_avg = eval(model, val_loader, args)
        best_metric, best_epoch = saver.save_checkpoint(epoch, metric=top1_avg)
        model.train()

        print(epoch, best_metric, best_epoch)

        # Increment the LR scheduler
        if scheduler is not None:
            scheduler.step()

        # Training error percentage is below 80%, so warmup is complete
        # if scheduler is None and top1_train.avg > 0.2:
        #     for group in optimiser.param_groups:
        #         group["lr"] = args.lr
        #
        #     scheduler = StepLR(optimiser, step_size=30, gamma=0.1)


def eval(model, data_loader, args):
    top1 = AverageMeter()

    model.eval()

    with torch.no_grad():
        for (batch, targets) in data_loader:
            if not args.prefetcher:
                batch = batch.cuda()
                targets = targets.cuda()

            outputs = model(batch)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

            torch.cuda.synchronize()
            top1.update(acc1.item(), outputs.size(0))

    return top1.avg


def test(args):
    model = create_resnet(args.num_classes, resnet18)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])

    model.cuda()

    test_dataset = create_dataset(
        args.dataset, root=args.data_dir, split=args.test_split, is_training=False, batch_size=args.batch_size)

    test_loader = create_loader(
        test_dataset,
        input_size=args.img_size,
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=args.interpolation,
        num_workers=args.workers,
        distributed=False,
        pin_memory=args.pin_mem,
    )

    top1 = eval(model, test_loader, args)
    print("Accuracy:", top1)


if __name__ == "__main__":
    task = "train"

    if task == "train":
        args = Namespace(data_dir="../data", epochs=300, output_dir="output/train/resnet34",
                         dataset="ImageFolder", train_split="train_split", test_split="val", prefetcher=True,
                         num_classes=15, gp=None, img_size=(3, 224, 224), input_size=None,
                         crop_pct=0.9, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                         interpolation="bicubic", batch_size=16, workers=4,
                         pin_mem=False, lr_warmup=0.01, lr=0.01, momentum=0.9, weight_decay=1e-4, checkpoint_hist=5)
        train(args)
    else:
        args = Namespace(checkpoint_path="output/train/resnet18/model_best.pth.tar", data_dir="../data",
                         dataset="ImageFolder", test_split="test", prefetcher=True,
                         num_classes=15, img_size=(3, 224, 224), input_size=None,
                         crop_pct=0.9, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                         interpolation="bicubic", batch_size=16, workers=4,
                         pin_mem=False)
        test(args)
