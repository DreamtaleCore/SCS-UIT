import os
import math
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from losses import WassersteinDistance
from networks import MsImageDis, SCSEncoder, Vgg11EncoderMS
from utils import weights_init, get_model_list, vgg_preprocess, get_scheduler, mask2color, visualize_network_params


class SCS_UIT(nn.Module):
    """
    The Style Trainer for channel selection translator, i.e. StyleGen
    """
    def __init__(self, hparams):
        super().__init__()
        lr = hparams.lr
        self.weights = hparams.loss_weights
        # Initiate the networks
        self.gen_a = SCSEncoder(hparams.gen)                               # generator for domain a
        self.gen_b = SCSEncoder(hparams.gen)                               # generator for domain b
        self.dis_a = MsImageDis(hparams.input_dim_a, hparams.dis)          # discriminator for domain a
        self.dis_b = MsImageDis(hparams.input_dim_b, hparams.dis)          # discriminator for domain b

        # visualize_network_params(self.dis_a, (torch.randn(1, 3, 224, 224)))

        self.style_dim    = hparams.gen.style_dim
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.n_res        = hparams.gen.n_res

        display_size = int(hparams.display_size)
        self.s_a     = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_b     = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        beta1      = hparams.beta1
        beta2      = hparams.beta2
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())

        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr * 0.2, betas=(beta1, beta2), weight_decay=hparams.weight_decay)
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hparams.weight_decay)

        self.dis_scheduler = get_scheduler(self.dis_opt, hparams)
        self.gen_scheduler = get_scheduler(self.gen_opt, hparams)

        self.content_criterion  = WassersteinDistance()
        self.semantic_criterion = nn.CrossEntropyLoss(ignore_index=255)

        # Network weight initialization
        self.apply(weights_init(hparams.init))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        if 'vgg_w' in hparams.keys() and hparams['vgg_w'] > 0:
            self.vgg = Vgg11EncoderMS(pretrained=True)

    @torch.no_grad()
    def forward(self, x_a, x_b):
        s_a    = Variable(self.s_a)
        s_b    = Variable(self.s_b)
        c_a, _ = self.gen_a.encode(x_a)
        c_b, _ = self.gen_b.encode(x_b)
        x_ba   = self.gen_a.decode(c_b, s_a)[0]
        x_ab   = self.gen_b.decode(c_a, s_b)[0]
        return x_ab, x_ba

    def recon_criterion(self, input, target):
        ret = torch.mean(torch.abs(input - target))
        if math.isnan(ret.item()):
            ret = 0
        return ret

    def compute_vgg_loss(self, vgg, img, target):
        with torch.no_grad():
            img_vgg    = vgg_preprocess(img)
            target_vgg = vgg_preprocess(target)
            img_fea    = vgg(img_vgg)
            target_fea = vgg(target_vgg)
            loss       = 0
        for k in img_fea.keys():
            loss += torch.mean((self.instancenorm(img_fea[k]) - self.instancenorm(target_fea[k])) ** 2)
        return loss / len(img_fea.keys())

    def get_semantic_color_map(self, sem, is_pred=True, mode='human'):
        colors = []
        for ii in range(sem.shape[0]):
            _sem  = torch.argmax(sem[ii], dim=0).cpu().numpy() if is_pred else sem[ii].cpu().numpy()
            color = mask2color(_sem, mode)
            color = torch.from_numpy(np.transpose(color, (2, 0, 1)))
            colors.append(torch.unsqueeze(color, 0))
        color = torch.cat(colors).cuda()
        color = (color / 255. * 2 - 1).float()
        return color

    def gen_update(self, x_a, x_b, sem_a=None, sem_b=None, weights=None):
        self.gen_opt.zero_grad()
        w = self.weights if weights is None else weights

        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon, x_sem_a = self.gen_a.decode(c_a, s_a_prime)
        x_b_recon, x_sem_b = self.gen_b.decode(c_b, s_b_prime)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)[0]
        x_ab = self.gen_b.decode(c_a, s_b)[0]
        # encode again
        c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
        c_a_recon, s_b_recon = self.gen_b.encode(x_ab)

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)

        self.loss_sem_a = self.semantic_criterion(x_sem_a, sem_a)
        self.loss_sem_b = self.semantic_criterion(x_sem_b, sem_b)

        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if w['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if w['vgg_w'] > 0 else 0

        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)

        # total loss
        self.loss_gen_total = w.gan_w       * self.loss_gen_adv_a + \
                              w.gan_w       * self.loss_gen_adv_b + \
                              w.recon_x_w   * self.loss_gen_recon_x_a + \
                              w.recon_s_w   * self.loss_gen_recon_s_a + \
                              w.recon_c_w   * self.loss_gen_recon_c_a + \
                              w.recon_x_w   * self.loss_gen_recon_x_b + \
                              w.recon_s_w   * self.loss_gen_recon_s_b + \
                              w.recon_c_w   * self.loss_gen_recon_c_b + \
                              w.sem_w       * self.loss_sem_a * 0.2 + \
                              w.sem_w       * self.loss_sem_b + \
                              w.vgg_w       * self.loss_gen_vgg_a + \
                              w.vgg_w       * self.loss_gen_vgg_b

        self.loss_gen_total.backward()
        self.gen_opt.step()

    def dis_update(self, x_a, x_b, weights):
        self.dis_opt.zero_grad()
        w = self.weights if weights is None else weights

        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, _ = self.gen_a.encode(x_a)
        c_b, _ = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba, _ = self.gen_a.decode(c_b, s_a)
        x_ab, _ = self.gen_b.decode(c_a, s_b)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = w.gan_w * self.loss_dis_a + w.gan_w * self.loss_dis_b

        if math.isnan(self.loss_dis_total.item()):
            print('[Warning] Total dis loss is N/A. Skipping')
            return
        self.loss_dis_total.backward()
        self.dis_opt.step()

    @torch.no_grad()
    def sample(self, x_a, x_b, sem_a, sem_b):
        s_a1 = Variable(self.s_a)
        s_b1 = Variable(self.s_b)
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        sem_pred_as, sem_pred_bs, sem_gt_as, sem_gt_bs = [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_rec, pred_sem_a = self.gen_a.decode(c_a, s_a_fake)
            x_b_rec, pred_sem_b = self.gen_b.decode(c_b, s_b_fake)

            x_a_recon.append(x_a_rec)
            x_b_recon.append(x_b_rec)
            sem_gt_as.append(self.get_semantic_color_map(sem_a, is_pred=False, mode='cat'))
            sem_gt_bs.append(self.get_semantic_color_map(sem_b, is_pred=False, mode='human'))
            sem_pred_as.append(self.get_semantic_color_map(pred_sem_a, is_pred=True, mode='cat'))
            sem_pred_bs.append(self.get_semantic_color_map(pred_sem_b, is_pred=True, mode='human'))

            x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0))[0])
            x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0))[0])
            x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0))[0])
            x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0))[0])
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        sem_pred_a, sem_pred_b = torch.cat(sem_pred_as), torch.cat(sem_pred_bs)
        sem_gt_a, sem_gt_b = torch.cat(sem_gt_as), torch.cat(sem_gt_bs)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        return x_a, x_a_recon, sem_pred_a, sem_gt_a, x_ab1, x_ab2, x_b, x_b_recon, sem_pred_b, sem_gt_b, x_ba1, x_ba2

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def set_lr(self, lr):
        for param_group in self.gen_opt.param_groups :
            param_group['lr'] = lr
        for param_group in self.dis_opt.param_groups :
            param_group['lr'] = lr * 0.2

    def print_losses(self):
        def ee(x):
            if type(x) is torch.Tensor:
                return x.item()
            else:
                return x
        
        print("gen lr is %.7f"%self.gen_opt.state_dict()['param_groups'][0]['lr'])
        print("dis lr is %.7f"%self.dis_opt.state_dict()['param_groups'][0]['lr'])

        print('adv_a: %.4f | adv_b: %.4f | rec_xa: %.4f | rec_xb: %.4f | rec_sa: %.4f | rec_sb: %.4f | rec_ca: %.4f '
              '| rec_cb: %.4f | sem_a: %.4f |  sem_b: %.4f | vgg_a: %.4f | vgg_b: %.4f | dis_xa: %.4f | dis_xb: %.4f' % (
            ee(self.loss_gen_adv_a), ee(self.loss_gen_adv_b), ee(self.loss_gen_recon_x_a), ee(self.loss_gen_recon_x_b),
            ee(self.loss_gen_recon_s_a), ee(self.loss_gen_recon_s_b), ee(self.loss_gen_recon_c_a), ee(self.loss_gen_recon_c_b),
            ee(self.loss_sem_a), ee(self.loss_sem_b), ee(self.loss_gen_vgg_a), ee(self.loss_gen_vgg_b), ee(self.loss_dis_a), ee(self.loss_dis_b)))

    def resume(self, checkpoint_dir, hyperparameters):
        params = hyperparameters
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, params, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, params, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)


