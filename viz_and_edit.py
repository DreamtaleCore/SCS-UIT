import os
import cv2
import torch
import argparse
import numpy as np
from trainer import SCS_UIT
from data import ImageFolder
from torch.autograd import Variable
from utils import get_config, get_data_loader_folder


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/animal_face2human_face-Modulation.yaml',
                    help='Path to the config file.')
parser.add_argument('--input_folder', type=str, default='/home/lyf2/dataset/StyleI2I/StyleI2I/animal_face2human_face/testA/',
                    help="input image folder")
parser.add_argument('--output_folder', type=str, default='/home/lyf2/results/StyleI2I/Modulation/a2b/viz_a2b',
                    help="output image folder")
parser.add_argument('--checkpoint', type=str, default='/home/lyf2/checkpoints/iit_ckpts/Modulation/outputs/animal_face2human_face-Modulation/checkpoints/gen_00730000.pt',
                    help="checkpoint of autoencoders")
parser.add_argument('--trainer', type=str, default='Style', help="Style|MUNIT|")
parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--gpu_id', type=int, default=2, help="gpu id")

opts = parser.parse_args()

print(' ######################### Arguments: ############################')
print(' ┌────────────────────────────────────────────────────────────────')
for arg in vars(opts):
    print(' │ {:<12} : {}'.format(arg, getattr(opts, arg)))
print(' └────────────────────────────────────────────────────────────────')

torch.cuda.set_device(opts.gpu_id)

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

# Load experiment setting
config = get_config(opts.config)
input_dim = config['input_dim_a'] if opts.a2b else config['input_dim_b']

# Setup model and data loader
image_names = ImageFolder(opts.input_folder, transform=None, return_paths=True)
data_loader = get_data_loader_folder(opts.input_folder, 1, False, new_size=config['new_size_a'], crop=False, num_workers=0)

trainer = SCS_UIT(config)

state_dict = torch.load(opts.checkpoint)
trainer.gen_a.load_state_dict(state_dict['a'])
trainer.gen_b.load_state_dict(state_dict['b'])

trainer.cuda()
trainer.eval()

encoder = trainer.gen_a.enc if opts.a2b == 1 else trainer.gen_b.enc
img_decoder = trainer.gen_b.dec_img if opts.a2b == 1 else trainer.gen_a.dec_img
sem_encoder = trainer.gen_b.enc if opts.a2b == 1 else trainer.gen_a.enc
sem_decoder = trainer.gen_b.dec_sem if opts.a2b == 1 else trainer.gen_a.dec_sem

# global feature for intermediate features in image decoder
total_features = []


def hook_fn_forward(module, input, output):
    total_features.append(output)


for name, module in img_decoder.named_children():
    print(name)


for name, module in img_decoder.named_children():
    module.register_forward_hook(hook_fn_forward)


def manipulate_intermediate_feature(new_map, index, batch_id, channel_id):
    # locate the feature, replace/change it by new_map
    pass


def find_matched_features(features, semantic, threshold=0.5):
    res = {}
    for ii in range(semantic.shape[0]):
        sem = torch.argmax(semantic[ii], dim=0).cpu().numpy()
        for sem_id in range(sem.min(), sem.max()):
            _sem = sem == sem_id
            for idx, feat in enumerate(features):
                for jj in range(feat.shape[0]):
                    for kk in range(feat.shape[1]):
                        _feat = feat.cpu().numpy()
                        _feat = cv2.resize(_feat, (_sem.shape[0], _sem.shape[1])) > threshold
                        iou = np.sum(np.bitwise_and(_feat, _sem)) / np.sum(np.bitwise_or(_feat, _sem))
                        if iou > threshold:
                            if sem_id not in res:
                                res[sem_id] = []
                            res[sem_id].append((ii, jj, kk, idx, iou, _feat))
    return res


def visual_image(image, bin_map):
    thresh = cv2.Canny(bin_map, 128, 256)
    thresh, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color = np.ones_like(image)
    cv2.rectangle(color, (0, 0), (image.shape[0], image.shape[1]), (212, 255, 127,), -1)
    image[bin_map == 0] = np.uint8(np.float(image[bin_map == 0]) * 0.6 + np.float(color[bin_map == 0]) * 0.4)
    cv2.drawContours(image, contours, -1, (34, 139, 34), 3)
    return image


def visualization_scs_uit():
    if not os.path.exists(opts.output_folder):
        os.makedirs(opts.output_folder)

    for i, (samples, names) in enumerate(list(zip(data_loader, image_names))):
        print(names[1])
        if i > 5: break
        images = Variable(samples[0].cuda(), volatile=True)
        content, style, _ = encoder(images)
        style_new = Variable(torch.randn(images.size(0), config.gen.style_dim, 1, 1).cuda())
        images_trans = img_decoder(content, style_new)
        _, _, features = sem_encoder(images_trans)
        semantic = sem_decoder(features, images_trans.shape[2], images_trans.shape[3])
        res = find_matched_features(total_features, semantic)
        total_features = []     # reset total features to empty

        for k, v in res.items():
            if k == 0:
                # we ignore the background
                continue
            sv = sorted(v, key=lambda v: v[3], reverse=True)
            best_v = sv[0]
            print(f'Best for semantic {k}:, batch_id: {best_v[0]}, channel: {best_v[1]}, idx: {best_v[2]}, IoU: {best_v[3]}.')
            np_image = np.transpose(((images[best_v[0]] + 1) / 2).cpu().numpy() * 255, axes=(1, 2, 0))
            res_vis = visual_image(np_image, best_v[4])

            cv2.imwrite(os.path.join(opts.output_folder, f's_{k}-i_{best_v[2]}-b_{best_v[0]}-c_{best_v[1]}.png'), res_vis)

    print('Done.')


if __name__ == '__main__':
    visualization_scs_uit()
    
