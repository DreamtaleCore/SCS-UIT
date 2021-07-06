"""
Based on the MUNIT repo, `utils.py`
"""
import os
import yaml
import time
import math
import torch
import easydict
import numpy as np
import torch.nn as nn
from thop import profile
import torch.nn.init as init
from data import ImageFolder
from thop import clever_format
import torchvision.utils as vutils
from torch.autograd import Variable
from torchvision.models import vgg16
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.models import inception_v3


def visualize_network_params(model, inputs):
    macs, params = profile(model, inputs=inputs)
    print('========================')
    print('MACs:   ', macs)
    print('PARAMs: ', params)
    print('------------------------')
    macs, params = clever_format([macs, params], "%.3f")
    print('Clever MACs:   ', macs)
    print('Clever PARAMs: ', params)
    print('========================')


def get_all_data_loaders(conf, len_limit=None):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    if 'new_size' in conf:
        new_size_a = new_size_b = conf['new_size']
    else:
        new_size_a = conf['new_size_a']
        new_size_b = conf['new_size_b']
    height = conf['crop_image_height']
    width = conf['crop_image_width']

    train_loader_a = get_data_loader_folder(os.path.join(conf['data_root'], 'trainA'),
                                            os.path.join(conf['semantic_root'], 'trainA'),
                                            batch_size, True, False, new_size_a, height, width,
                                            num_workers, True, len_limit=len_limit)
    test_loader_a = get_data_loader_folder(os.path.join(conf['data_root'], 'testA'),
                                           os.path.join(conf['semantic_root'], 'testA'),
                                           batch_size, False, True, new_size_a, new_size_a, new_size_a,
                                           num_workers, True, len_limit=len_limit)
    train_loader_b = get_data_loader_folder(os.path.join(conf['data_root'], 'trainB'),
                                            os.path.join(conf['semantic_root'], 'trainB'),
                                            batch_size, True, False, new_size_b, height, width,
                                            num_workers, True, len_limit=len_limit)
    test_loader_b = get_data_loader_folder(os.path.join(conf['data_root'], 'testB'),
                                           os.path.join(conf['semantic_root'], 'testB'),
                                           batch_size, False, True, new_size_b, new_size_b, new_size_b,
                                           num_workers, True, len_limit=len_limit)

    return train_loader_a, train_loader_b, test_loader_a, test_loader_b


def get_data_loader_folder(input_folder, semantic_folder,  batch_size, train, return_paths=False, new_size=None,
                           height=256, width=256, num_workers=4, crop=True, len_limit=None):

    dataset = ImageFolder(input_folder, semantic_folder, return_paths=return_paths, use_da=train, len_limit=len_limit,
                          height=height, width=width)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader


def get_config(config):
    with open(config, 'r') as stream:
        return easydict.EasyDict(yaml.load(stream))


def eformat(f, prec):
    s = "%.*e"%(prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%d"%(mantissa, int(exp))


def __write_images(image_outputs, display_image_num, file_name):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs] # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1)


def write_2images(image_outputs, display_image_num, image_directory, postfix):
    n = len(image_outputs)
    __write_images(image_outputs[0:n//2], display_image_num, '%s/gen_a2b_%s.jpg' % (image_directory, postfix))
    __write_images(image_outputs[n//2:n], display_image_num, '%s/gen_b2a_%s.jpg' % (image_directory, postfix))


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def write_one_row_html(html_file, iterations, img_filename, all_size):
    html_file.write("<h3>iteration [%d] (%s)</h3>" % (iterations,img_filename.split('/')[-1]))
    html_file.write("""
        <p><a href="%s">
          <img src="%s" style="width:%dpx">
        </a><br>
        <p>
        """ % (img_filename, img_filename, all_size))
    return


def write_html(filename, iterations, image_save_iterations, image_directory, all_size=1536):
    html_file = open(filename, "w")
    html_file.write('''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Experiment name = %s</title>
      <meta http-equiv="refresh" content="30">
    </head>
    <body>
    ''' % os.path.basename(filename))
    html_file.write("<h3>current</h3>")
    write_one_row_html(html_file, iterations, '%s/gen_a2b_train_current.jpg' % (image_directory), all_size)
    write_one_row_html(html_file, iterations, '%s/gen_b2a_train_current.jpg' % (image_directory), all_size)
    for j in range(iterations, image_save_iterations-1, -1):
        if j % image_save_iterations == 0:
            write_one_row_html(html_file, j, '%s/gen_a2b_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_a2b_train_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_train_%08d.jpg' % (image_directory, j), all_size)
    html_file.write("</body></html>")
    html_file.close()


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


def slerp(val, low, high):
    """
    original: Animating Rotation with Quaternion Curves, Ken Shoemake
    https://arxiv.org/abs/1609.04468
    Code: https://github.com/soumith/dcgan.torch/issues/14, Tom White
    """
    omega = np.arccos(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def get_slerp_interp(nb_latents, nb_interp, z_dim):
    """
    modified from: PyTorch inference for "Progressive Growing of GANs" with CelebA snapshot
    https://github.com/ptrblck/prog_gans_pytorch_inference
    """

    latent_interps = np.empty(shape=(0, z_dim), dtype=np.float32)
    for _ in range(nb_latents):
        low = np.random.randn(z_dim)
        high = np.random.randn(z_dim)  # low + np.random.randn(512) * 0.7
        interp_vals = np.linspace(0, 1, num=nb_interp)
        latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                 dtype=np.float32)
        latent_interps = np.vstack((latent_interps, latent_interp))

    return latent_interps[:, :, np.newaxis, np.newaxis]


# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def load_vgg16(model_dir):
    """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    vgg = vgg16(pretrained=True)
    return vgg


def vgg_preprocess(batch):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5 # [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size()).cuda()
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean)) # subtract mean
    return batch


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


def load_inception(model_path):
    state_dict = torch.load(model_path)
    model = inception_v3(pretrained=False, transform_input=True)
    model.aux_logits = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, state_dict['fc.weight'].size(0))
    model.load_state_dict(state_dict)
    for param in model.parameters():
        param.requires_grad = False
    return model


CAT_CLASS_IDS = [
                    #id       name            color
                    #─────────────────────────────────────
                    (0,  'background',   (  0,   0,   0)),
                    (1,  'foreground',   (181, 181, 181)),
                    (2,  'face',         (255, 182, 193)),
                    (3,  'left-ear',     ( 39, 105, 105)),
                    (4,  'right-ear',    (139,  54,  38)),
                    (5,  'left-eye',     (135, 206, 255)),
                    (6,  'right-eye',    (  0, 191, 255)),
                    (7,  'nose',         (255, 215,   0)),
                    (8,  'up-lip',       (255, 105, 180)),
                    (9,  'bottom-lip',   (255,  20, 147)),
                    (10, 'mouse',        (  0, 255, 127))
                ]

HUMAN_CLASS_IDS = [
                    #id       name            color
                    #─────────────────────────────────────
                    (0,  'background',   (  0,   0,   0)),
                    (1,  'foreground',   (181, 181, 181)),
                    (2,  'face',         (255, 182, 193)),
                    (3,  'left-eyebow',  ( 39, 105, 105)),
                    (4,  'right-eyebow', (139,  54,  38)),
                    (5,  'left-eye',     (135, 206, 255)),
                    (6,  'right-eye',    (  0, 191, 255)),
                    (7,  'nose',         (255, 215,   0)),
                    (8,  'up-lip',       (255, 105, 180)),
                    (9,  'bottom-lip',   (255,  20, 147)),
                    (10, 'mouse',        (  0, 255, 127))
                 ]


def mask2color(mask, name='human'):
    if name == 'human':
        class_id = HUMAN_CLASS_IDS
    elif name == 'cat':
        class_id = CAT_CLASS_IDS
    else:
        raise NotImplementedError

    c = np.zeros([mask.shape[0], mask.shape[1], 3], np.uint8)
    for _, cid in enumerate(class_id):
        if c[mask == cid[0]].shape[0] > 0:
            c[mask == cid[0]] = np.array(cid[2][::-1], np.uint8)
    return c