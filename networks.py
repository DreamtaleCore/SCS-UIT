"""
Main networks design.
===
Thie file is based on the MUNIT:
https://github.com/NVlabs/MUNIT/blob/master/networks.py

"""
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
from torchvision.models import vgg11, vgg19


##################################################################################
# Discriminator
##################################################################################
class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, params):
        """Mutli-scale discriminator init

        Args:
            input_dim (int):   input channel of input image
            params (EasyDict): parameters for configuration
        """
        super(MsImageDis, self).__init__()
        self.n_layer    = params.n_layer
        self.gan_type   = params.gan_type
        self.dim        = params.dim
        self.norm       = params.norm
        self.activ      = params.activ
        self.num_scales = params.num_scales
        self.pad_type   = params.pad_type
        self.input_dim  = input_dim

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns       = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for _ in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for _, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for _, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1) ** 2)  # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


class EmbDis(nn.Module):
    """
    The discriminator for the invariant embeddings
    Gradient reversal layer is used for the domain adaption
    """
    def __init__(self, params):
        """Embedding discriminator init

        Args:
            params (EasyDict): parameters for configuration
        """
        super().__init__()
        self.n_layer   = params.n_layer
        self.gan_type  = params.gan_type
        self.dim       = params.dim
        self.norm      = params.norm
        self.activ     = params.activ
        self.input_dim = params.input_dim
        self.pad_type  = params.pad_type
        # self.revgrad   = RevGrad.apply

        self.cnn = self._make_net()

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for _ in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        # y = self.revgrad(x)
        return self.cnn(x)

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        out0 = self.forward(input_fake)
        out1 = self.forward(input_real)

        if self.gan_type == 'lsgan':
            loss = torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2)
        elif self.gan_type == 'nsgan':
            all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
            all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
            loss = torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                              F.binary_cross_entropy(F.sigmoid(out1), all1))
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        out0 = self.forward(input_fake)
        if self.gan_type == 'lsgan':
            loss = torch.mean((out0 - 1) ** 2)  # LSGAN
        elif self.gan_type == 'nsgan':
            all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
            loss = torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)

        return loss


class EmbClassify(nn.Module):
    """
    The discriminator for the invariant embeddings
    Gradient reversal layer is used for the domain adaption
    """
    def __init__(self, params):
        """Classifiers for embedding

        Args:
            params (EasyDict): parameters for configuration
        """
        super().__init__()
        self.n_layer    = params.n_layer
        self.gan_type   = params.gan_type
        self.mlp_dim    = params.mlp_dim
        self.norm       = params.norm
        self.activ      = params.activ
        self.input_dim  = params.input_dim
        self.output_dim = params.num_trans
        self.criterion  = nn.CrossEntropyLoss()
        self.ap         = nn.AdaptiveAvgPool2d(1)
        self.mlp        = MLP(self.input_dim, self.output_dim, self.mlp_dim, self.n_layer, norm='none', activ=self.activ)

    def forward(self, x):
        if len(x.shape) > 2 and x.shape[2] > 1:
            y = self.ap(x)
        else:
            y = x
        return torch.softmax(self.mlp(y), dim=1)

    def calc_loss(self, input_embedding, labels):
        # calculate the loss to train D
        out = self.forward(input_embedding)
        loss = self.criterion(out, labels)
        return loss


##################################################################################
# Generator
##################################################################################
class SeparationUnifyGen(nn.Module):
    """
    The unify generator with content-style separator, SCS-Gen
    """
    def __init__(self, params):
        """SCS Generator

        Args:
            params (EasyDict): parameters for configuration
        """
        super().__init__()
        dim            = params.dim
        style_dim      = params.style_dim
        n_style_conv   = params.n_style_conv
        n_downsample   = params.n_downsample
        n_upsample     = params.n_upsample
        n_res          = params.n_res
        activ          = params.activ
        pad_type       = params.pad_type
        mlp_dim        = params.mlp_dim
        input_dim      = params.input_dim
        exfoliate_mode = params.exfoliate_mode
        norm           = params.norm
        res_norm       = params.res_norm
        num_classes    = params.num_classes

        self.enc = SCSEncoder(n_downsample, n_res, input_dim, dim, norm, activ, pad_type, mlp_dim, style_dim,
                                      n_style_conv, exfoliate_mode)
        self.out_feat_dim = self.enc.output_dim
        self.dec_img = ImgDecoder(n_upsample, n_res, self.out_feat_dim, input_dim, res_norm, activ, pad_type)

        self.dec_sem = SemDecoder(self.out_feat_dim, dim, num_classes)

        # ############################################################################
        # from thop import profile
        # from thop import clever_format
        # input_i = torch.randn(1, 3, 224, 224)
        # content, style, features = self.enc(input_i)
        # macs, params = profile(self.dec_sem, inputs=(features['end_feature'], input_i.shape[2], input_i.shape[3]))
        # print('========================')
        # print('MACs: ',   macs)
        # print('PARAMs: ', params)
        # print('------------------------')
        # macs, params = clever_format([macs, params], "%.3f")
        # print('Clever MACs: ',   macs)
        # print('Clever PARAMs: ', params)
        # print('========================')
        # ############################################################################

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec_img), mlp_dim, 3, norm='none',
                       activ=activ)
        self.encode_features = None

    def forward(self, images):
        # reconstruct an image
        content, styles_fake = self.encode(images)
        images_recon = self.decode(content, styles_fake)
        return images_recon

    def encode(self, images):
        # encode an image to its content and style codes
        content, style, features = self.enc(images)
        self.encode_features = features
        return content, style

    def decode(self, content, styles, latent_feature=None):
        # decode content and style codes to an image
        adain_params = self.mlp(styles)
        self.assign_adain_params(adain_params, self.dec_img)
        images = self.dec_img(content)
        latent_feature = latent_feature if latent_feature is not None else self.encode_features['end_feature']
        sem_masks = None
        if latent_feature is not None:
            sem_masks = self.dec_sem(latent_feature, images.shape[2], images.shape[3])
        return images, sem_masks

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)

                # norm the weight and bias to get rid of droplet effect
                m.bias = (m.bias - torch.min(m.bias)) / (torch.max(m.bias) - torch.min(m.bias))
                m.weight = m.weight / torch.sqrt(torch.sum(torch.pow(m.weight, 2)) + 1e-8)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params


##################################################################################
# Encoder and Decoders
##################################################################################

class SCSEncoder(nn.Module):
    """[summary]
    Separating content and style encoder via correlation with DI-HV
    """
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, 
                 pad_type, mlp_dim, style_dim, n_style_conv):
        """Configuration for SCSEncoder

        Args : 
            n_downsample (int): downsample layers
            n_res        (int): number of residual blocks
            input_dim    (int): n channels of input image
            dim          (int): hidden dimensions
            norm         (str): normalization name
            activ        (str): activation name
            pad_type     (str): padding name
            mlp_dim      (int): hidden dimensions for MLP
            style_dim    (int): hidden dimensions for style code
            n_style_conv (int): numbers for style convolutions
        """
        super().__init__()
        self.model_dict      = nn.ModuleDict()
        to_latent            = []
        to_latent           += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # use a cascade downsampling blocks as the to_latent function
        for i in range(n_downsample):
            to_latent += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        self.model_dict['to_latent'] = nn.Sequential(*to_latent)

        # use several ResNet blocks to exfoliate hierarchical style and content
        n_sub_content_fea = dim // n_res
        # residual blocks
        self.n_res = n_res
        for i in range(self.n_res):
            self.model_dict['res_block_{}'.format(i)] = ResBlock(dim, norm=norm, activation=activ, pad_type=pad_type)
            self.model_dict['sep_block_{}'.format(i)] = SeparationBlock(dim, n_sub_content_fea, mlp_dim,
                                                               dim, style_dim, norm, activ, n_style_conv)

        self.output_dim = dim
        self.style_mapping = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                           nn.Conv2d(dim * (n_res - 1), style_dim, 1, 1, 0))

    def forward(self, x):
        contents = []
        styles = []
        latent_emb = self.model_dict['to_latent'](x)
        for i in range(self.n_res):
            latent_emb = self.model_dict['res_block_{}'.format(i)].forward(latent_emb)
            c, s, f    = self.model_dict['exf_block_{}'.format(i)].forward(latent_emb)
            latent_emb = f      # replace the latent emb with excited latent code
            contents.append(c)
            styles.append(s)

        features = {
            'contents': contents, 'styles': styles, 'end_feature': latent_emb
        }
        content = torch.cat(contents, dim=1)    # just concat content codes in channel-wise
        style   = torch.cat(styles, dim=1)
        style   = self.style_mapping(style)     # map style into modulation params for decoder

        return content, style, features


class ImgDecoder(nn.Module):
    """
    Decoders for synthezing image
    """
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        """

        Args: 
            n_upsample (int): 
            n_res      (int): 
            dim        (int): 
            output_dim (int): 
            res_norm   (str, optional): . Defaults to 'adain'.
            activ      (str, optional): . Defaults to 'relu'.
            pad_type   (str, optional): . Defaults to 'zero'.
        """
        super().__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for _ in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class SemDecoder(nn.Module):
    """
    Decoder for semantic segmentation
    """
    def __init__(self, fea_in_dim=64, dim=64, n_classes=9):
        super().__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=4)
        self.cat_conv = nn.Sequential(
            nn.Conv2d(fea_in_dim, dim, 3, 1,
                      padding=1,
                      bias=True),
            nn.InstanceNorm2d(dim),
            # simply the sem decoder to make the result of resnet more like semantic 
            nn.LeakyReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(dim, dim, 3, 1, padding=1, bias=True),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.cls_conv = nn.Conv2d(dim, n_classes, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature, out_h=256, out_w=256):
        result = self.cat_conv(feature)
        result = self.cls_conv(result)

        result = F.interpolate(result, size=(out_h, out_w), mode='nearest', align_corners=None)
        return result


##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class SeparationBlock(nn.Module):
    """
    Separate content code and style code from the given latent code in channel-wise on the basis of correlation to DI-HV
    """
    def __init__(self,
                 in_channels,               # The total number of conv channels, e.g., the last #n channel of res block
                 n_cc,                      # The output number of selected channels for content code (cc)
                 mlp_dim,                   # Number of filters in MLP
                 dim,                       # The dimension of style downsampling layers
                 style_dim,                 # The dimension of output style latent codes
                 norm,                      # The type of normalization in style downsampling layers
                 activ,                     # The name of activation in downsampling layers
                 n_sc):                     # The number of downsampling layers for style encoding
        super().__init__()
        # the content_selector is a based on a modified version of SE layer
        self.content_selector = SSELayer(in_channels, n_cc, mlp_dim)
        style_encoder = []
        for i in range(n_sc):
            in_dim = in_channels - n_cc if i == 0 else dim
            style_encoder += [Conv2dBlock(in_dim, dim, 4, 2, 0, norm=norm, activation=activ, pad_type='zero')]
        style_encoder += [nn.AdaptiveAvgPool2d(1)]                  # global average pooling
        style_encoder += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.style_encoder = nn.Sequential(*style_encoder)

    def forward(self, latent_code):
        b, c, _, _ = latent_code.size()
        # Separation
        channel_weighted, selected_index, exclusive_index = self.content_selector(latent_code)
        # Selection
        selected_feature  = torch.index_select(latent_code, (0, 1), selected_index)
        exclusive_feature = torch.index_select(latent_code, (0, 1), exclusive_index)
        # Excitation
        excited_feature   = latent_code * channel_weighted.view(b, c, 1, 1).expand_as(latent_code)

        return selected_feature, exclusive_feature, excited_feature


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):
        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for _ in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')]  # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


##################################################################################
# Basic Blocks
##################################################################################
class SSELayer(nn.Module):
    """
    A modified Squeeze-and-Excitation layer (https://zhuanlan.zhihu.com/p/65459972) for Selection
    Squeeze-Selection-and-Excitation
    the squeezed weights are also provided for selection, add out_channel to select the top-N channels as out-feature,
    where N = out_channel
    """

    def __init__(self, in_channel, out_channel, n_hidden):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.out_channel = out_channel
        self.fc = nn.Sequential(
            nn.Linear(in_channel, n_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(n_hidden, in_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        y = y.view(b, c, 1, 1)

        sorted_index = torch.argsort(y, dim=1, descending=True) 
        selected_index = np.sort(sorted_index[:, :self.out_channel, :, :], axis=1)
        exclusive_index = np.sort(sorted_index[:, self.out_channel:, :, :], axis=1)

        return y, selected_index, exclusive_index


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


def get_activation(name):
    if name == 'relu':
        act = nn.ReLU(inplace=True)
    elif name == 'lrelu':
        act = nn.LeakyReLU(0.2, inplace=True)
    elif name == 'prelu':
        act = nn.PReLU()
    elif name == 'selu':
        act = nn.SELU(inplace=True)
    elif name == 'tanh':
        act = nn.Tanh()
    elif name == 'none':
        act = None
    else:
        assert 0, "Unsupported activation: {}".format(name)
    return act


def get_normalization(name, dim):
    if name == 'bn':
        norm = nn.BatchNorm2d(dim)
    elif name == 'in':
        # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
        norm = nn.InstanceNorm2d(dim)
    elif name == 'ln':
        norm = LayerNorm(dim)
    elif name == 'pn':
        norm = PixelwiseNorm()
    elif name == 'adain':
        norm = AdaptiveInstanceNorm2d(dim)
    elif name == 'none' or name == 'sn':
        norm = None
    else:
        assert 0, "Unsupported normalization: {}".format(name)

    return norm


def get_padding(name, padding):
    if name == 'reflect':
        pad = nn.ReflectionPad2d(padding)
    elif name == 'replicate':
        pad = nn.ReplicationPad2d(padding)
    elif name == 'zero':
        pad = nn.ZeroPad2d(padding)
    else:
        assert 0, "Unsupported padding type: {}".format(name)
    return pad


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        self.pad = get_padding(pad_type, padding)

        # initialize normalization
        norm_dim = output_dim
        self.norm = get_normalization(norm, norm_dim)

        # initialize activation
        self.activation = get_activation(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            # x = self.norm(x)
            pass
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class RevGrad(Function):
    """
    Refer from https://github.com/janfreyberg/pytorch-revgrad
    """
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output
        return grad_input


class EqualizedConv2D(nn.Module):
    """ conv2d with the concept of equalized learning rate
        Args:
            :param c_in: input channels
            :param c_out:  output channels
            :param k_size: kernel size (h, w) should be a tuple or a single integer
            :param stride: stride for conv
            :param pad: padding
            :param bias: whether to use bias or not
    """

    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, bias=True):
        """ constructor for the class """
        from torch.nn.modules.utils import _pair
        from numpy import sqrt, prod

        super().__init__()

        # define the weight and bias if to be used
        self.weight = nn.Parameter(nn.init.normal_(
            torch.empty(c_out, c_in, *_pair(k_size))
        ))

        self.use_bias = bias
        self.stride   = stride
        self.pad      = pad

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))

        fan_in = prod(_pair(k_size)) * c_in  # value of fan_in
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of the network
        :param x: input
        :return: y => output
        """
        from torch.nn.functional import conv2d

        return conv2d(input=x,
                      weight=self.weight * self.scale,  # scale the weight on runtime
                      bias=self.bias if self.use_bias else None,
                      stride=self.stride,
                      padding=self.pad)

    def extra_repr(self):
        return ", ".join(map(str, self.weight.shape))


class EqualizedDeConv2D(nn.Module):
    """ Transpose convolution using the equalized learning rate
        Args:
            :param c_in: input channels
            :param c_out: output channels
            :param k_size: kernel size
            :param stride: stride for convolution transpose
            :param pad: padding
            :param bias: whether to use bias or not
    """

    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, bias=True):
        """ constructor for the class """
        from torch.nn.modules.utils import _pair
        from numpy import sqrt

        super().__init__()

        # define the weight and bias if to be used
        self.weight = nn.Parameter(nn.init.normal_(
            torch.empty(c_in, c_out, *_pair(k_size))
        ))

        self.use_bias = bias
        self.stride = stride
        self.pad = pad

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))

        fan_in = c_in  # value of fan_in for deconv
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of the layer
        :param x: input
        :return: y => output
        """
        from torch.nn.functional import conv_transpose2d

        return conv_transpose2d(input=x,
                                weight=self.weight * self.scale,  # scale the weight on runtime
                                bias=self.bias if self.use_bias else None,
                                stride=self.stride,
                                padding=self.pad)

    def extra_repr(self):
        return ", ".join(map(str, self.weight.shape))


class EqualizedLinear(nn.Module):
    """ Linear layer using equalized learning rate
        Args:
            :param c_in: number of input channels
            :param c_out: number of output channels
            :param bias: whether to use bias with the linear layer
    """

    def __init__(self, c_in, c_out, bias=True):
        """
        Linear layer modified for equalized learning rate
        """
        from numpy import sqrt

        super().__init__()

        self.weight = nn.Parameter(nn.init.normal_(
            torch.empty(c_out, c_in)
        ))

        self.use_bias = bias

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))

        fan_in = c_in
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of the layer
        :param x: input
        :return: y => output
        """
        from torch.nn.functional import linear
        return linear(x, self.weight * self.scale,
                      self.bias if self.use_bias else None)


class MiniBatchStdDev(nn.Module):
    """
    MiniBatch standard deviation layer for the discriminator
    """

    def __init__(self):
        """
        derived class constructor
        """
        super().__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape

        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)

        # [1 x C x H x W]  Calc standard deviation over batch
        y = torch.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size, 1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)

        # return the computed values:
        return y


##################################################################################
# VGG network definition
##################################################################################
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        # relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        # relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        # relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)
        # relu4_3 = h

        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        relu5_3 = h

        return relu5_3
        # return [relu1_2, relu2_2, relu3_3, relu4_3]


class Vgg11EncoderMS(nn.Module):
    """Vgg encoder wiht multi-scales"""

    def __init__(self, pretrained):
        super().__init__()
        features = list(vgg11(pretrained=pretrained).features)
        self.backbone = nn.ModuleList(features)

    def forward(self, x):
        result_dict = {}
        layer_names = ['conv1_1',
                       'conv2_1',
                       'conv3_1', 'conv3_2',
                       'conv4_1', 'conv4_2',
                       'conv5_1', 'conv5_2']
        idx = 0
        for ii, model in enumerate(self.backbone):
            x = model(x)
            if ii in {0, 3, 6, 8, 11, 13, 16, 18}:
                result_dict[layer_names[idx]] = x
                idx += 1

        return {'conv5_1': result_dict['conv5_1'], 'conv5_2': result_dict['conv5_2']}


class Vgg19EncoderMS(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        features = list(vgg19(pretrained=pretrained).features)
        self.backbone = nn.ModuleList(features)

    def forward(self, x):
        # here we assume x is normalized in [-1, 1]
        result_dict = {}
        layer_names = ['conv1_1', 'conv1_2',
                       'conv2_1', 'conv2_2',
                       'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                       'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
                       'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']
        idx = 0
        for ii, model in enumerate(self.backbone):
            x = model(x)
            if ii in {0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34}:
                result_dict[layer_names[idx]] = x
                idx += 1

        return {'conv5_1': result_dict['conv5_1'], 'conv5_2': result_dict['conv5_2'],
                'conv5_3': result_dict['conv5_3'], 'conv5_4': result_dict['conv5_4']}


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


##################################################################################
# Loss functions for GAN
##################################################################################
# =============================================================
# Interface for the losses
# =============================================================

class GANLoss:
    """ Base class for all losses
        @args:
            dis: Discriminator used for calculating the loss
                 Note this must be a part of the GAN framework
    """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        """
        calculate the discriminator loss using the following data
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, height, alpha):
        """
        calculate the generator loss
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("gen_loss method has not been implemented")


class ConditionalGANLoss:
    """ Base class for all conditional losses """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps, labels, height, alpha):
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, labels, height, alpha):
        raise NotImplementedError("gen_loss method has not been implemented")


##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class PixelwiseNorm(nn.Module):
    """
    Pixelwise feature vector normalization.
    reference: https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120
    """
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y


def test_vgg():
    vgg = Vgg11EncoderMS(pretrained=True).cuda()
    t = torch.rand(size=(1, 3, 256, 256)).cuda()
    ret = vgg(t)
    for k in sorted(ret.keys()):
        print(k, ret[k].shape)


if __name__ == '__main__':
    test_vgg()
