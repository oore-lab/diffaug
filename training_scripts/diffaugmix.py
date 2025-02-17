import argparse
import os
import time

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

import logging
logging.level = logging.INFO


import random
import torch
import numpy as np
from torchvision import datasets, transforms, models
import training_scripts.augmentations as augmentations
from torch.nn.parallel import DistributedDataParallel as DDP

augmentations.IMAGE_SIZE = 224
torch.backends.cudnn.benchmark=True

import diffaug


parser = argparse.ArgumentParser()
parser.add_argument('--clean_data', type=str, default="/scratch/ssd004/datasets/imagenet")

# AugMix options
parser.add_argument(
    '--mixture-width',
    default=3,
    type=int,
    help='Number of augmentation chains to mix per augmented example')
parser.add_argument(
    '--mixture-depth',
    default=-1,
    type=int,
    help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument(
    '--aug-severity',
    default=1,
    type=int,
    help='Severity of base augmentation operators')
parser.add_argument(
    '--aug-prob-coeff',
    default=1.,
    type=float,
    help='Probability distribution coefficients')
parser.add_argument(
    '--no-jsd',
    '-nj',
    action='store_true',
    help='Turn off JSD consistency loss.')
parser.add_argument(
    '--all-ops',
    '-all',
    action='store_true',
    help='Turn on all operations (+brightness,contrast,color,sharpness).')

# Checkpointing options
parser.add_argument(
    '--resume',
    '-r',
    type=str,
    default='workdirs/AM_diffaug.ckpt',
    help='Checkpoint path for resume / test.')
parser.add_argument(
    '--print-freq',
    type=int,
    default=10,
    help='Training loss print frequency (batches).')
parser.add_argument(
    '--batch-size', '-b', type=int, default=256, help='Batch size.')

# Acceleration
parser.add_argument(
    '--num-workers',
    type=int,
    default=4,
    help='Number of pre-fetching threads.')


import sys
args = parser.parse_args(sys.argv[1:])

USE_PYTORCH_DDP = 'LOCAL_RANK' in os.environ
RANK = int(os.environ['LOCAL_RANK']) if USE_PYTORCH_DDP else 0
WORLD_SIZE = int(os.environ['WORLD_SIZE']) if USE_PYTORCH_DDP else 1
args.batch_size = args.batch_size//WORLD_SIZE 

if USE_PYTORCH_DDP:
    dist.init_process_group('nccl')
    torch.cuda.set_device(RANK)
    print_fn = print 
    def print(*args,**kwargs):
       if RANK==0:
        print_fn(RANK,*args,**kwargs)


def aug(image, preprocess):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

  Returns:
    mixed: Augmented and mixed image.
  """
  aug_list = augmentations.augmentations
  if args.all_ops:
    aug_list = augmentations.augmentations_all

  ws = np.float32(
      np.random.dirichlet([args.aug_prob_coeff] * args.mixture_width))
  m = np.float32(np.random.beta(args.aug_prob_coeff, args.aug_prob_coeff))

  mix = torch.zeros_like(preprocess(image))
  for i in range(args.mixture_width):
    image_aug = image.copy()
    depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, args.aug_severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug)

  mixed = (1 - m) * preprocess(image) + m * mix
  return mixed


class AugMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess, no_jsd=False):
    self.dataset = dataset
    self.preprocess = preprocess
    self.no_jsd = no_jsd
    self.to_tensor = transforms.ToTensor()
    self.all_idxs = list(range(len(self.dataset)))
    if USE_PYTORCH_DDP:
       random.Random(6158).shuffle(self.all_idxs)
       per_device = len(self.all_idxs)//WORLD_SIZE
       self.all_idxs = self.all_idxs[RANK*per_device:(RANK+1)*per_device]
       print(RANK, WORLD_SIZE, len(self.all_idxs))
    
  def __getitem__(self, i):
    i = self.all_idxs[i]
    x, x_da, y = self.dataset[i]
    if self.no_jsd:
      return 2*self.to_tensor(x_da)-1, aug(x, self.preprocess), y
    else:
      im_tuple = (2*self.to_tensor(x_da)-1, self.preprocess(x), aug(x, self.preprocess),
                  aug(x, self.preprocess))
      return im_tuple, y

  def __len__(self):
    return len(self.all_idxs)

class CustomImageFolder(datasets.ImageFolder):
   def __init__(self, root, train_transform=None, da_transform=None):
      super().__init__(root)
      self.train_transform = train_transform
      self.da_transform = da_transform
    
   def __getitem__(self, index: int):
       im, label = super().__getitem__(index)
       return self.train_transform(im), self.da_transform(im), label

def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k."""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res

def main():
    diffaugmenter = diffaug.DiffAug(gpu_id=RANK)
    
    # Load datasets
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose(
        [transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip()])
    da_transform = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224)])
    
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    preprocess_augmented = transforms.Compose([transforms.Normalize([-1,-1,-1],[2,2,2]), 
                                              transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        preprocess,
    ])

    traindir = os.path.join(args.clean_data, 'train')
    valdir = os.path.join(args.clean_data, 'val')
    train_dataset = CustomImageFolder(traindir, train_transform=train_transform, da_transform=da_transform)
    train_dataset = AugMixDataset(train_dataset, preprocess)
    print(args.batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,pin_memory=True,
        shuffle=True,
        num_workers=1)

    def train_1epoch(net, train_loader, optimizer, num_steps):
        """Train for one epoch."""
        
        data_ema = 0.
        batch_ema = 0.
        loss_ema = 0.
        acc1_ema = 0.
        acc5_ema = 0.

        end = time.time()
        for i, (images, targets) in enumerate(train_loader):
            net.train()
            
            # Compute data loading time
            data_time = time.time() - end
            optimizer.zero_grad()

            images = [im.cuda(RANK) for im in images]
            da_im = images[0]
            images = images[1:]
            with torch.no_grad():
                da_im = diffaugmenter.augment(da_im)
                da_im = preprocess_augmented(da_im)
            
            images_all = torch.cat(images, 0)
            images_all = torch.cat([images_all,da_im],0)

            targets = targets.cuda(RANK)
            logits_all = net(images_all)
            
            logits_clean, logits_aug1, logits_aug2, logits_da = torch.split(
                logits_all, images[0].size(0))

            # Cross-entropy is only computed on clean/denoised images
            loss = F.cross_entropy(logits_clean, targets)+F.cross_entropy(logits_da, targets)

            p_clean, p_aug1, p_aug2 = F.softmax(
                logits_clean, dim=1), F.softmax(
                    logits_aug1, dim=1), F.softmax(
                        logits_aug2, dim=1)

            # Clamp mixture distribution to avoid exploding KL divergence
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
            acc1, acc5 = accuracy(logits_clean, targets, topk=(1, 5))  # pylint: disable=unbalanced-tuple-unpacking
            
            loss.backward()
            optimizer.step()

            # Compute batch computation time and update moving averages.
            batch_time = time.time() - end
            end = time.time()

            data_ema = data_ema * 0.1 + float(data_time) * 0.9
            batch_ema = batch_ema * 0.1 + float(batch_time) * 0.9
            loss_ema = loss_ema * 0.1 + float(loss) * 0.9
            acc1_ema = acc1_ema * 0.1 + float(acc1) * 0.9
            acc5_ema = acc5_ema * 0.1 + float(acc5) * 0.9

            if i % args.print_freq == 0:
                print(
                    'Batch {}: Data Time {:.3f} | Batch Time {:.3f} | Train Loss {:.3f} | Train Acc1 '
                    '{:.3f} | Train Acc5 {:.3f}'.format(num_steps, data_ema,
                                                        batch_ema, loss_ema, acc1_ema,
                                                        acc5_ema))
            num_steps += 1
            if RANK==0 and num_steps%100 == 0:
                checkpoint = {
                    "steps":num_steps,
                    "state_dict":net.state_dict(),
                    "optimizer":optimizer.state_dict(),
                }
                torch.save(checkpoint,args.resume)
            
        checkpoint = {
                    "steps":num_steps,
                    "state_dict":net.state_dict(),
                    "optimizer":optimizer.state_dict(),
                }
        return loss_ema, acc1_ema, batch_ema, checkpoint
    
    net = models.resnet50(weights=None).cuda(RANK)
    if USE_PYTORCH_DDP:
      net = DDP(net, device_ids=[RANK], output_device=RANK)
    else:
      net = torch.nn.DataParallel(net)
    net.load_state_dict(
            torch.load("workdirs/checkpoint.pth.tar",map_location=f"cpu")['state_dict'])
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    num_steps = 0
    optimizer = torch.optim.SGD(
      net.parameters(),
      0.1,
      momentum=0.9,
      weight_decay=0.0001)
    
    optimizer.load_state_dict(
            torch.load("workdirs/checkpoint.pth.tar",map_location='cpu')['optimizer'])
    
    print(optimizer.param_groups[0]['lr'])
    if os.path.exists(args.resume):
       checkpoint = torch.load(args.resume)
       net.load_state_dict(checkpoint['state_dict'])
       optimizer.load_state_dict(checkpoint['optimizer'])
       num_steps = checkpoint.get("steps",0)
       print(f"Loaded model finetuned for {num_steps} steps")
    
    while True:
       if num_steps>50000:
          break
       train_loss_ema, train_acc1_ema, batch_ema, checkpoint = train_1epoch(net, train_loader,
                                                      optimizer, num_steps)
       num_steps = checkpoint["steps"]

if __name__ == "__main__":
   main()
    
