import argparse
import os
import time

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

import logging
logging.level = logging.INFO

import diffaug

import random

import torch
from torchvision import datasets, transforms, models

from torch.nn.parallel import DistributedDataParallel as DDP

torch.backends.cudnn.benchmark=True


parser = argparse.ArgumentParser()

parser.add_argument('--clean_data', type=str, default="/scratch/ssd004/datasets/imagenet", help="ImageNet data.")

# Optimization options
parser.add_argument(
    '--batch-size', '-b', type=int, default=256, help='Batch size.')
parser.add_argument('--eval-batch-size', type=int, default=1000)
parser.add_argument(
    '--diffaug', action='store_true', help='Specify to enable diffaug. Else, just train with synthetic data.')

parser.add_argument(
    '--print-freq',
    type=int,
    default=10,
    help='Training loss print frequency (batches).')

# Acceleration
parser.add_argument(
    '--num-workers',
    type=int,
    default=4,
    help='Number of pre-fetching threads.')

import sys
args = parser.parse_args(sys.argv[1:])
if args.diffaug:
    args.resume = "workdirs/Synth_diffaug.ckpt"
else:
    args.resume = "workdirs/Synth_base.ckpt"

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

class ConcatDataset(torch.utils.data.Dataset):
   def __init__(self, datasets, train_transform, da_transform):
      super().__init__()
      lens = [len(d) for d in datasets]
      assert len(set(lens)) == 1
      self.datasets = datasets 
      self.all_idxs = list(range(lens[0]))
      self.train_transform = train_transform
      self.da_transform = da_transform
      if USE_PYTORCH_DDP:
        random.Random(6158).shuffle(self.all_idxs)
        per_device = len(self.all_idxs)//WORLD_SIZE
        self.all_idxs = self.all_idxs[RANK*per_device:(RANK+1)*per_device]
        print(RANK, WORLD_SIZE, len(self.all_idxs))
   
   def __getitem__(self, i):
      i = self.all_idxs[i]
      xys = [d[i] for d in self.datasets]
      return [self.train_transform(xy[0]) for xy in xys],[self.da_transform(xy[0]) for xy in xys],xys[0][1] 
   
   def __len__(self):
      return len(self.all_idxs)


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
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),])
    da_transform = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
        transforms.ToTensor(),])
    normalize = transforms.Normalize(mean, std)
    
    normalize_augmented = transforms.Compose([transforms.Normalize([-1,-1,-1],[2,2,2]), 
                                              transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    traindir = os.path.join(args.clean_data, 'train')
    valdir = os.path.join(args.clean_data, 'val')
    synth_data = "/datasets/generative-robustness/Base-Generated-ImageNet-1K/train/train_data/train"
    
    train_dataset = torch.utils.data.ConcatDataset([datasets.ImageFolder(traindir),datasets.ImageFolder(synth_data)])
    concat_dataset = ConcatDataset([train_dataset],
                                   train_transform=train_transform, 
                                   da_transform=da_transform)
    print(args.batch_size)
    train_loader = torch.utils.data.DataLoader(
        concat_dataset,
        batch_size=args.batch_size,pin_memory=True,
        shuffle=True,
        num_workers=1)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, test_transform),
        batch_size=args.eval_batch_size,
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
        
        for i, (images, da_images, targets) in enumerate(train_loader):
            net.train()
               
            # Compute data loading time
            data_time = time.time() - end
            optimizer.zero_grad()
            
            clean, = images 
            if args.diffaug:
                da_clean, = da_images
                with torch.no_grad():
                    da_clean = diffaugmenter.augment(2*da_clean.cuda(RANK)-1)
                    da_clean = normalize_augmented(da_clean)

                images_all = torch.cat([normalize(clean).cuda(RANK),
                                        da_clean,
                                    ], dim=0)
            else:
                images_all = normalize(clean).cuda(RANK)
            
            targets = targets.cuda(RANK)
            
            logits_all = net(images_all)
            loss = 0
            for idx,logits in enumerate(torch.split(logits_all, clean.size(0))):
                loss = loss + F.cross_entropy(logits, targets) 
                if idx==0:
                    logits_clean = logits
            
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
    
    net = models.resnet50(weights='IMAGENET1K_V1').cuda(RANK)
    if USE_PYTORCH_DDP:
      net = DDP(net, device_ids=[RANK], output_device=RANK)
    else:
      net = torch.nn.DataParallel(net)
    
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    num_steps = 0
    optimizer = torch.optim.AdamW(
      net.parameters(),
      1e-5,
      weight_decay=0.0001)
    
    print(optimizer.param_groups[0]['lr'])
    print(optimizer.param_groups[0]['weight_decay'])

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
    
