# -*- coding: UTF-8 -*-
import os
import sys
import torch
import shutil
import argparse
import numpy as np
import tensorboardX
import torch.nn as nn
from PIL import Image
from trainer import SCS_UIT
from scipy.stats import entropy
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from utils import get_all_data_loaders, prepare_sub_folder, \
    write_html, write_loss, get_config, write_2images, Timer, visualize_network_params, load_inception, get_data_loader_folder


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='configs/cat2human.yaml',
                    help='Path to the config file.')
parser.add_argument('-o', '--output_path', type=str, default='/home/lyf/repo/checkpoints/SCS-IT/',
                    help="outputs path")
parser.add_argument('-r', "--resume", action="store_true")
parser.add_argument("--lr", type=float,default=None)
parser.add_argument('--in_incubator', action='store_true', default=False,
                    help='use little training set and less epochs to fast evaluate ideas.')
parser.add_argument('-g', '--gpu_id', type=int, default=2, help="gpu id")
parser.add_argument('-ckp', '--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('-si', '--style', type=str, default='', help="style image path")
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and 0 for b2a")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('-ns', '--num_style',type=int, default=10, help="number of styles to sample")
parser.add_argument('-fix', '--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('-oo', '--output_only', action='store_true', help="whether only save the output images or also save the input images")
parser.add_argument('--compute_IS', action='store_true', help="whether to compute Inception Score or not")
parser.add_argument('--compute_CIS', action='store_true', help="whether to compute Conditional Inception Score or not")
parser.add_argument('--inception_a', type=str, default='.', help="path to the pretrained inception network for domain A")
parser.add_argument('--inception_b', type=str, default='.', help="path to the pretrained inception network for domain B")

opts = parser.parse_args()

print(' ##################### Arguments: ########################')
print(' ┌────────────────────────────────────────────────────────────────')
for arg in vars(opts):
    print(' │ {:<12} : {}'.format(arg, getattr(opts, arg)))
print(' └────────────────────────────────────────────────────────────────')

cudnn.benchmark = True
torch.cuda.set_device(opts.gpu_id)

# Load experiment setting
config                = get_config(opts.config)
display_size          = config.display_size
config.vgg_model_path = opts.output_path

# Setup model and data loader
trainer = SCS_UIT(config)
trainer.cuda()
# visualize_network_params(trainer, (torch.randn(1, 3, 128, 128).cuda(), torch.randn(1, 3, 128, 128).cuda()))


def train():
    if opts.in_incubator:
        train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config, len_limit=500)
        max_iter = config.max_itermax_iter // 100
    else:
        train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)

    # Setup logger and output folders
    model_name       = os.path.splitext(os.path.basename(opts.config))[0]
    train_writer     = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)

    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

    # set lr
    if opts.lr is not None:
        trainer.set_lr(opts.lr)
    
    tr_im_a, tr_im_b, tr_se_a, tr_se_b, te_im_a, te_im_b, te_se_a, te_se_b = [], [], [], [], [], [], [], []
    for i in range(display_size):
        sample = train_loader_a.dataset[i][:2]
        tr_im_a.append(sample[0])
        tr_se_a.append(sample[1])
        sample = test_loader_a.dataset[i][:2]
        te_im_a.append(sample[0])
        te_se_a.append(sample[1])
        sample = train_loader_b.dataset[i][:2]
        tr_im_b.append(sample[0])
        tr_se_b.append(sample[1])
        sample = test_loader_b.dataset[i][:2]
        te_im_b.append(sample[0])
        te_se_b.append(sample[1])

    train_display_images_a  , train_display_images_b   = torch.stack(tr_im_a).cuda(), torch.stack(tr_im_b).cuda()
    test_display_images_a   , test_display_images_b    = torch.stack(te_im_a).cuda(), torch.stack(te_im_b).cuda()
    train_display_semantic_a, train_display_semantic_b = torch.stack(tr_se_a).cuda(), torch.stack(tr_se_b).cuda()
    test_display_semantic_a , test_display_semantic_b  = torch.stack(te_se_a).cuda(), torch.stack(te_se_b).cuda()

    # Start training
    iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0

    while True:
        for _, (samples_a, samples_b) in enumerate(zip(train_loader_a, train_loader_b)):
            images_a  , images_b   = samples_a[0].cuda().detach(), samples_b[0].cuda().detach()
            semantic_a, semantic_b = samples_a[1].cuda().detach(), samples_b[1].cuda().detach()

            with Timer("Elapsed time in update: %f"):
                trainer.update_learning_rate()
                # Main training code
                trainer.dis_update(images_a, images_b, config)
                trainer.gen_update(images_a, images_b, semantic_a, semantic_b, config)
                torch.cuda.synchronize()

            # Dump training stats in log file
            if (iterations + 1) % config['log_iter'] == 0:
                print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
                trainer.print_losses()
                write_loss(iterations, trainer, train_writer)

            # Write images
            if (iterations + 1) % config['image_save_iter'] == 0:
                with torch.no_grad():
                    test_image_outputs = trainer.sample(test_display_images_a,   test_display_images_b,
                                                        test_display_semantic_a, test_display_semantic_b)
                    train_image_outputs = trainer.sample(train_display_images_a,   train_display_images_b,
                                                         train_display_semantic_a, train_display_semantic_b)

                write_2images(test_image_outputs,  display_size, image_directory, 'test_%08d'  % (iterations + 1))
                write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
                # HTML
                write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

            if (iterations + 1) % config['image_display_iter'] == 0:
                with torch.no_grad():
                    image_outputs = trainer.sample(train_display_images_a, train_display_images_b,
                                                train_display_semantic_a, train_display_semantic_b)
                write_2images(image_outputs, display_size, image_directory, 'train_current')

            # Save network weights
            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.save(checkpoint_directory, iterations)

            iterations += 1
            if iterations >= max_iter:
                sys.exit('Finish training')


def test():
    trainer.eval()
    style_dim  = config.gen.style_dim
    state_dict = torch.load(opts.checkpoint)

    # Load the inception networks if we need to compute IS or CIIS
    if opts.compute_IS or opts.compute_IS:
        inception = load_inception(opts.inception_b) if opts.a2b else load_inception(opts.inception_a)
        # freeze the inception models and set eval mode
        inception.eval()
        for param in inception.parameters():
            param.requires_grad = False
        inception_up = nn.Upsample(size=(299, 299), mode='bilinear')

    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])

    if opts.compute_IS:
        IS = []
        all_preds = []
    if opts.compute_CIS:
        CIS = []

    if 'new_size' in config:
        new_size = config.new_size
    else:
        if opts.a2b == 1:
            new_size = config.new_size_a
        else:
            new_size = config.new_size_b

    trainer.cuda()
    trainer.eval()
    
    if os.path.exists(opts.style):
        style_encode = trainer.gen_b.encode if opts.a2b else trainer.gen_a.encode # encode function
        with torch.no_grad():
            transform = transforms.Compose([transforms.Resize(new_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            style_image = Variable(transform(Image.open(opts.style).convert('RGB')).unsqueeze(0).cuda()) if opts.style != '' else None
            _, style = style_encode(style_image)
    else:
        style = None
    
    style_fixed = Variable(torch.randn(opts.num_style, style_dim).cuda(), volatile=True)

    data_loader = get_data_loader_folder(opts.input_folder, None, 1, return_paths=True, new_size=config.new_size_a, crop=False, train=False)

    for _, (images, _, names) in enumerate(data_loader):
        if opts.compute_CIS:
            cur_preds = []
        print(names[0])
        images = Variable(images.cuda(), volatile=True)
        if opts.a2b:
            content, _      = trainer.gen_a.encode(images)
            encode_features = trainer.gen_a.encode_features['end_feature']
        else:
            content, _      = trainer.gen_b.encode(images)
            encode_features = trainer.gen_b.encode_features['end_feature']

        style = style_fixed if opts.synchronized else Variable(torch.randn(opts.num_style, style_dim).cuda(), volatile=True)
        for j in range(opts.num_style):
            s = style[j].unsqueeze(0)
            if opts.a2b:
                _, semantics = trainer.gen_a.decode(content, s, encode_features)
                outputs, _   = trainer.gen_b.decode(content, s, encode_features)
            else:
                _, semantics = trainer.gen_b.decode(content, s, encode_features)
                outputs, _   = trainer.gen_a.decode(content, s, encode_features)

            outputs = (outputs + 1) / 2.
            semantics = trainer.get_semantic_color_map(semantics, is_pred=True, mode='human' if opts.a2b else 'cat')
            if opts.compute_IS or opts.compute_CIS:
                pred = F.softmax(inception(inception_up(outputs)),
                                    dim=1).cpu().data.numpy()  # get the predicted class distribution
            if opts.compute_IS:
                all_preds.append(pred)
            if opts.compute_CIS:
                cur_preds.append(pred)
            # path = os.path.join(opts.output_folder, 'input{:03d}_output{:03d}.jpg'.format(i, j))
            basename = os.path.basename(names[0])
            path = os.path.join(opts.output_folder, "rlt_%02d" % j)
            if not os.path.exists(path):
                os.makedirs(path)
                os.makedirs(path + '-sem')
            vutils.save_image(outputs.data, os.path.join(path, basename), padding=0, normalize=True)
            vutils.save_image(semantics, os.path.join(path + '-sem', basename), padding=0, normalize=False)
        if opts.compute_CIS:
            cur_preds = np.concatenate(cur_preds, 0)
            py = np.sum(cur_preds, axis=0)  # prior is computed from outputs given a specific input
            for j in range(cur_preds.shape[0]):
                pyx = cur_preds[j, :]
                CIS.append(entropy(pyx, py))
        if not opts.output_only:
            # also save input images
            vutils.save_image(images.data, os.path.join(opts.output_folder, basename), padding=0,
                                normalize=True)
    if opts.compute_IS:
        all_preds = np.concatenate(all_preds, 0)
        py = np.sum(all_preds, axis=0)  # prior is computed from all outputs
        for j in range(all_preds.shape[0]):
            pyx = all_preds[j, :]
            IS.append(entropy(pyx, py))

    if opts.compute_IS:
        print("Inception Score: {}".format(np.exp(np.mean(IS))))
    if opts.compute_CIS:
        print("conditional Inception Score: {}".format(np.exp(np.mean(CIS))))


if __name__ == '__main__':
    train()
    # test()
