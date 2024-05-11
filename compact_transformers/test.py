import time

import torch
import torch.nn.functional as F
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.models.layers import convert_splitbn_model
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler
from contextlib import suppress
from argparse import Namespace
from src import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def test(args, amp_autocast=suppress):
    abbr_labels = ['Kit', 'Sto', 'Bed', 'Liv', 'Hou', 'Ind', 'Sta',
                   'Und', 'Bld', 'Str', 'HW', 'Fld', 'Cst', 'Mnt', 'For']

    dataset_eval = create_dataset(
        args.dataset, root=args.data_dir, split=args.test_split, is_training=False, batch_size=args.batch_size)

    data_loader = create_loader(
        dataset_eval,
        input_size=args.img_size,
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=args.interpolation,
        num_workers=args.workers,
        distributed=False,
        pin_memory=args.pin_mem,
        mean=args.mean,
        std=args.std,
        crop_pct=args.crop_pct
    )

    model = create_model(
        args.model,
        pretrained=False,
        progress=False,
        checkpoint_path=args.checkpoint_path,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript)

    model.eval()
    model.cuda()

    y_pred = []
    y_true = []
    inference_times = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            if not args.prefetcher:
                inputs = inputs.cuda()
                targets = targets.cuda()

            with amp_autocast():
                start = time.perf_counter()
                outputs = model(inputs)
                end = time.perf_counter()
                inference_times.append(end - start)

            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            # Calculate class confidences, and convert the max to the class label
            logits = F.softmax(outputs, dim=1)
            class_max = torch.argmax(logits, dim=1)

            y_pred.extend(class_max.detach().cpu().tolist())
            y_true.extend(targets.detach().cpu().tolist())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    mean_time = sum(inference_times) / len(inference_times)

    accuracy = np.sum(y_pred == y_true) / len(y_true)
    print(f"Accuracy: {accuracy}")
    print(f"Mean inference time: {mean_time}")

    c_mat = confusion_matrix(y_true, y_pred)
    c_mat_display = ConfusionMatrixDisplay(confusion_matrix=c_mat, display_labels=abbr_labels)
    c_mat_display.plot()
    plt.show()


if __name__ == "__main__":
    args = Namespace(data_dir='../data',
                     checkpoint_path='output/train/cct_224_14_trial_re_6/model_best.pth.tar',
                     dataset='ImageFolder', test_split='test', model='cct_sun_224_14', prefetcher=True,
                     num_classes=15, gp=None, img_size=(3, 224, 224), input_size=224,
                     crop_pct=0.9, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                     interpolation='bicubic', batch_size=16, drop=0.0, drop_connect=None, drop_path=None,
                     drop_block=None, bn_tf=False, bn_momentum=None, bn_eps=None, workers=8,
                     pin_mem=False, tta=0, torchscript=False)
    test(args)
