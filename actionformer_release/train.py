# python imports
import argparse
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
import datetime
from pprint import pprint
import matplotlib.pyplot as plt

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
# for visualization
from torch.utils.tensorboard import SummaryWriter

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)


################################################################################
# グラフを描画する関数

def plot_losses(train_losses, val_losses, save_dir='./loss/', filename_prefix='plot', lr=None):

    if isinstance(train_losses, torch.Tensor):
        train_losses = train_losses.cpu().numpy()  # CUDA -> CPU -> NumPy変換
    if isinstance(val_losses, torch.Tensor):
        val_losses = val_losses.cpu().numpy()

    num_epoch = 50
    x1 = range(1, num_epoch+1)
    x2 = range(10, num_epoch+1, 10)
    LR = lr

    plt.figure(figsize=(10, 6))
    plt.plot(x1, train_losses, label='Training Loss')
    plt.plot(x2, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    save_path = f'{save_dir}/{filename_prefix}_lr{LR}_buffer17.png'
    plt.savefig(save_path)  # 例: 'loss_plot.png'
    plt.show()

# def plot_losses(train_losses, save_dir='./loss/', filename_prefix='plot', lr=None):

#     if isinstance(train_losses, torch.Tensor):
#         train_losses = train_losses.cpu().numpy()  # CUDA -> CPU -> NumPy変換

#     num_epoch = 50
#     x1 = range(1, num_epoch+1)
#     x2 = range(10, num_epoch+1, 10)
#     LR = lr

#     plt.figure(figsize=(10, 6))
#     plt.plot(train_losses, label='Training Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss Over Epochs')
#     plt.legend()
#     plt.grid(True)
#     save_path = f'{save_dir}/{filename_prefix}_lr{LR}.png'
#     plt.savefig(save_path)  # 例: 'loss_plot.png'
#     plt.show()

def main(args):
    """main function that handles training / inference"""


    """1. setup parameters / folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    pprint(cfg)

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """2. create dataset / dataloader"""
    # breakpoint()
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])

    train_subset = torch.utils.data.Subset(train_dataset, range(400))

    train_val_loader = make_data_loader(
        train_dataset, False, rng_generator, 1, num_workers=4)

    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['train_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume,
                map_location = lambda storage, loc: storage.cuda(
                    cfg['devices'][0]))
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # 訓練データと検証データのロスを保存するリスト
    train_losses = []  # 訓練時のロスを格納
    val_losses = []
    
    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )

    epoch = 0
    # det_eval, output_file = None, None
    # if not args.saveonly:
    #     print("yes")
    #     print(train_dataset.split[0])
    #     val_db_vars = train_dataset.get_attributes()
    #     det_eval = ANETdetection(
    #         train_dataset.json_file,
    #         train_dataset.split[0],
    #         tiou_thresholds = val_db_vars['tiou_thresholds'],
    #         num_epoch = epoch
    #     )
    # else:
    #     output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')

    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        x = train_one_epoch(
                train_loader,
                model,
                optimizer,
                scheduler,
                epoch,
                model_ema = model_ema,
                clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
                tb_writer=tb_writer,
                print_freq=args.print_freq
            )

        train_losses.append(x)

        # save ckpt once in a while
        if (
            ((epoch + 1) == max_epochs) or
            ((args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0))
        ):
            save_states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            save_states['state_dict_ema'] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_folder,
                file_name='epoch_{:03d}.pth.tar'.format(epoch + 1)
            )

        if epoch % 10 == 9:
        # if epoch % 1 == 0:

            det_eval, output_file = None, None
            if not args.saveonly:
                # print("yes")
                # print(val_dataset.split[0])
                val_db_vars = train_dataset.get_attributes()
                det_eval = ANETdetection(
                    train_dataset.json_file,
                    train_dataset.split[0],
                    tiou_thresholds = val_db_vars['tiou_thresholds'],
                    num_epoch = epoch,
                    train_val = "train"
                )
            else:
                output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')

            print("\nStart testing model {:s} ...".format(cfg['model_name']))
            start = time.time()
            mAP, z = valid_one_epoch(
                train_val_loader,
                model,
                -1,
                evaluator=det_eval,
                output_file=output_file,
                ext_score_file=cfg['test_cfg']['ext_score_file'],
                tb_writer=None,
                print_freq=args.print_freq
            )
            end = time.time()
            print("All done! Total time: {:0.2f} sec".format(end - start))

            # print("=> loading checkpoint '{}'".format(ckpt_file))
            # # load ckpt, reset epoch / best rmse
            # checkpoint = torch.load(
            #     ckpt_file,
            #     map_location = lambda storage, loc: storage.cuda(cfg['devices'][0])
            # )
            # # load ema model instead
            # print("Loading from EMA model ...")
            # model.load_state_dict(checkpoint['state_dict_ema'])
            # del checkpoint

            # set up evaluator
            det_eval, output_file = None, None
            if not args.saveonly:
                val_db_vars = val_dataset.get_attributes()
                det_eval = ANETdetection(
                    val_dataset.json_file,
                    val_dataset.split[0],
                    tiou_thresholds = val_db_vars['tiou_thresholds'],
                    num_epoch = epoch,
                    train_val = "train"
                )
            else:
                output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')

            #breakpoint()
            print("\nStart testing model {:s} ...".format(cfg['model_name']))
            start = time.time()
            mAP, y = valid_one_epoch(
                val_loader,
                model,
                -1,
                evaluator=det_eval,
                output_file=output_file,
                ext_score_file=cfg['test_cfg']['ext_score_file'],
                tb_writer=None,
                print_freq=args.print_freq
            )
            end = time.time()
            print("All done! Total time: {:0.2f} sec".format(end - start))

            value = y.cpu().item()
            #print(value)
            val_losses.append(value)


    # wrap up
    tb_writer.close()
    # 可視化
    plot_losses(train_losses, val_losses, lr=cfg['opt']["learning_rate"])

    print("All done!")
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)'),
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    args = parser.parse_args()
    main(args)