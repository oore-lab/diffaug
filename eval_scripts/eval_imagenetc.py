import argparse
import os

import logging

logging.level = logging.INFO

import diffaug

import torch
from torchvision import datasets, transforms

from collections import defaultdict

parser = argparse.ArgumentParser()

parser.add_argument('--clean_data',
                    type=str,
                    default="/scratch/ssd004/datasets/imagenet")
parser.add_argument('--corrupted_data',
                    type=str,
                    default="/scratch/ssd002/datasets/imagenet-c",
                    help='path to ImageNet-C dataset')

parser.add_argument("--worker_id", type=int)
parser.add_argument("--num_workers", type=int)
parser.add_argument("--eval_batch_size", default=64, type=int)

import sys

args = parser.parse_args(sys.argv[1:])

TIMES = list(range(0, 451, 50))


CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]


def chunk(l, num_chunks):
    per_chunk = len(l) // num_chunks
    ret = []
    i = 0
    while len(ret) != num_chunks:
        if len(ret) == num_chunks - 1:
            ret.append(l[i:])
        else:
            ret.append(l[i:i + per_chunk])
        i += per_chunk
    return ret


CORRUPTIONS = chunk(CORRUPTIONS, args.num_workers)[args.worker_id]
RANK = 0
print(CORRUPTIONS)


def test(net, test_loader):
    """Evaluate network on given dataset."""
    net.eval()

    all_logits = defaultdict(list)
    all_orig_logits = defaultdict(list)
    all_targets = None
    nums = 0
    num_lines = 0
    headers = list(net.nets.keys())
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(RANK), targets
            nums += images.shape[0]
            if all_targets is None:
                all_targets = targets
            else:
                all_targets = torch.cat([all_targets, targets], dim=0)
            for _ in range(num_lines):
                print("\033[F", end="")
            num_lines = 1
            print(f"{str(nums):^6s}" +
                  "".join(list(map(lambda x: f"{x:^16s}", headers))))
            for idx, t in enumerate(TIMES):
                if t == 0:
                    logits = net(preprocess(images))
                    orig_logits = orig_net(preprocess(images))
                else:
                    da_images = diffaugmenter.test_time_augment(
                        2 * images - 1, t)
                    da_images = preprocess_augmented(da_images)

                    logits = net(da_images)
                    orig_logits = orig_net(da_images)
                for k in logits:
                    if len(all_logits[k]) == idx:
                        all_logits[k].append(logits[k])
                        all_orig_logits[k].append(orig_logits[k])
                    else:
                        all_logits[k][idx] = torch.cat(
                            [all_logits[k][idx], logits[k]], dim=0)
                        all_orig_logits[k][idx] = torch.cat(
                            [all_orig_logits[k][idx], orig_logits[k]], dim=0)

                print_str = f"{str(t):^6s}"
                for k in headers:
                    print_str += f"{(all_orig_logits[k][idx].max(dim=1)[1]==all_targets.data).float().mean():^8.4f}{(all_logits[k][idx].max(dim=1)[1]==all_targets.data).float().mean():^8.4f}"
                print(print_str)
                num_lines += 1
            print_str = f"{'MN':^6s}"

            for k in headers:
                print_str += f"{(torch.softmax(torch.stack(all_orig_logits[k]),dim=-1).mean(dim=0).max(dim=1)[1]==all_targets.data).float().mean():^8.4f}{(torch.softmax(torch.stack(all_logits[k]),dim=-1).mean(dim=0).max(dim=1)[1]==all_targets.data).float().mean():^8.4f}"
            print(print_str)
            num_lines += 1

            total_correct = get_cumm_accs(all_logits, all_targets)
            total_correct_orig = get_cumm_accs(all_orig_logits, all_targets)
            for i in range(len(all_logits[k])):
                print_str = f"{f'MN{i}':^6s}"
                for k in headers:
                    print_str += f"{total_correct_orig[k][0][i]:^8.4f}{total_correct[k][0][i]:^8.4f}"
                print(print_str)
                num_lines += 1

    print()

    total_correct = get_cumm_accs(all_logits, all_targets)
    total_correct_orig = get_cumm_accs(all_orig_logits, all_targets)
    return total_correct, total_correct_orig, {
        "all_logits": all_logits,
        "all_orig_logits": all_orig_logits,
        "targets": all_targets
    }


def get_cumm_accs(all_logits, all_targets):
    result = {}
    for k in all_logits:
        probs = 0
        cum_accs = []
        instant_accs = []
        for l in all_logits[k]:
            preds = torch.softmax(l, dim=1)
            probs += preds
            cum_accs.append(
                (probs.max(dim=1)[1] == all_targets).float().mean())
            instant_accs.append(
                (preds.max(dim=1)[1] == all_targets).float().mean())
        result[k] = (cum_accs, instant_accs)
    return result


def test_c(net, test_transform):
    """Evaluate network on given corrupted dataset."""
    corruption_accs = {}

    for c in CORRUPTIONS:
        ckpt = f"workdirs/corr_{c}.pt"
        if ckpt is not None and os.path.exists(ckpt):
            corruption_accs = torch.load(ckpt)
        print(c)
        if c in corruption_accs:
            s = 5 - len(corruption_accs[c])
        else:
            s = 5
            corruption_accs[c] = []
        while s >= 1:
            valdir = os.path.join(args.corrupted_data, c, str(s))
            val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, test_transform),
                batch_size=args.eval_batch_size,
                shuffle=True,
                num_workers=1,
                pin_memory=True)

            acc1, acc1_orig, results = test(net, val_loader)
            corruption_accs[c].append({'ft': acc1, 'orig': acc1_orig})

            print('\ts={}: {} {}'.format(s, acc1_orig, acc1))
            s -= 1
            if ckpt is not None:
                torch.save(corruption_accs, ckpt)


def main():
    global orig_net, diffaugmenter, preprocess, preprocess_augmented, mean_t, std_t
    from eval_scripts.nets_inference import get_nets
    import os
    print(os.getcwd())
    orig_net, net = get_nets()

    diffaugmenter = diffaug.DiffAug()

    # Load datasets
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean_t = torch.FloatTensor(mean).reshape(1, 3, 1, 1)
    std_t = torch.FloatTensor(std).reshape(1, 3, 1, 1)
    preprocess = transforms.Normalize(mean, std)
    preprocess_augmented = transforms.Compose([
        transforms.Normalize([-1, -1, -1], [2, 2, 2]),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    test_c(net, test_transform)


if __name__ == "__main__":
    main()
