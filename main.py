import numpy as np
import torch
from torchvision import models
import cv2
import torch.nn.functional as F
from utils import calculate_outputs, generate_entrie_images, calculate_outputs_and_gradients_steps
from integrated_gradients import random_baseline_integrated_gradients
from visualization import visualize
import argparse
import os
import torch.nn as nn
from torch import optim
from functions import TernaryTanh
from torch.autograd import Variable
from env_wrapper import atari_wrapper
import tools as tl
from scipy.misc import imresize, imsave
import plot_images_together

parser = argparse.ArgumentParser(description='integrated-gradients')
parser.add_argument('--cuda', action='store_true', help='if use the cuda to do the accelartion')
parser.add_argument('--model-type', type=str, default='inception', help='the type of network')
parser.add_argument('--img', type=str, default='01.jpg', help='the images name')


class MMNet(nn.Module):
    """
    Moore Machine Network(MMNet) definition.
    """
    def __init__(self, net, hx_qbn=None, obs_qbn=None):
        super(MMNet, self).__init__()
        self.bhx_units = hx_qbn.bhx_size if hx_qbn is not None else None
        self.gru_units = net.gru_units
        self.obx_net = obs_qbn
        self.gru_net = net
        self.bhx_net = hx_qbn
        # self.actor_linear = self.gru_net.get_action_linear

    def init_hidden(self, batch_size=1):
        return self.gru_net.init_hidden(batch_size)

    def forward(self, x, inspect=False):
        # x, hx = x
        # critic, actor, hx, (ghx, bhx, input_c, input_x) = self.gru_net(x, input_fn=self.obx_net, hx_fn=self.bhx_net, inspect=True)
        input_c = self.gru_net(x, input_fn=self.obx_net, hx_fn=self.bhx_net, inspect=True)
        # if inspect:
        #     return critic, actor, hx, (ghx, bhx), (input_c, input_x)
        # else:
        #     return critic, actor, hx
        return input_c

    def get_action_linear(self, state, decode=False):
        if decode:
            hx = self.bhx_net.decode(state)
        else:
            hx = state
        return self.actor_linear(hx)

    def transact(self, o_x, hx_x):
        hx_x = self.gru_net.transact(self.obx_net.decode(o_x), self.bhx_net.decode(hx_x))
        _, hx_x = self.bhx_net(hx_x)
        return hx_x

    def state_encode(self, state):
        return self.bhx_net.encode(state)

    def obs_encode(self, obs, hx=None):
        if hx is None:
            hx = Variable(torch.zeros(1, self.gru_units))
            if next(self.parameters()).is_cuda:
                hx = hx.cuda()
        _, _, _, (_, _, _, input_x) = self.gru_net((obs, hx), input_fn=self.obx_net, hx_fn=self.bhx_net, inspect=True)
        return input_x


class ObsQBNet(nn.Module):
    """
    Quantized Bottleneck Network(QBN) for observation features.
    """

    def __init__(self, input_size, x_features):
        super(ObsQBNet, self).__init__()
        self.bhx_size = x_features
        f1 = int(8 * x_features)
        self.encoder = nn.Sequential(nn.Linear(input_size, f1),
                                     nn.Tanh(),
                                     nn.Linear(f1, x_features),
                                     TernaryTanh())

        self.decoder = nn.Sequential(nn.Linear(x_features, f1),
                                     nn.Tanh(),
                                     nn.Linear(f1, input_size),
                                     nn.ReLU6())

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded, encoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class GRUNetConv(nn.Module):
    """
    Gated Recurrent Unit Network(GRUNet) definition.
    """

    def __init__(self, input_size, gru_cells, total_actions):
        super(GRUNetConv, self).__init__()
        self.gru_units = gru_cells
        self.noise = False
        self.conv1 = nn.Conv2d(input_size, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(16, 8, 3, stride=2, padding=1)

        self.input_ff = nn.Sequential(self.conv1, nn.ReLU(),
                                      self.conv2, nn.ReLU(),
                                      self.conv3, nn.ReLU(),
                                      self.conv4, nn.ReLU6())
        self.input_c_features = 8 * 5 * 5
        self.input_c_shape = (8, 5, 5)
        # self.gru = nn.GRUCell(self.input_c_features, gru_cells)

        # self.critic_linear = nn.Linear(gru_cells, 1)
        # self.actor_linear = nn.Linear(gru_cells, total_actions)

        # self.apply(tl.weights_init)
        # self.actor_linear.weight.data = tl.normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        # self.actor_linear.bias.data.fill_(0)
        # self.critic_linear.weight.data = tl.normalized_columns_initializer(self.critic_linear.weight.data, 1.0)
        # self.critic_linear.bias.data.fill_(0)

        # self.gru.bias_ih.data.fill_(0)
        # self.gru.bias_hh.data.fill_(0)

    def forward(self, input, input_fn=None, hx_fn=None, inspect=False):
        # input, hx = input
        c_input = self.input_ff(input)
        c_input = c_input.view(-1, self.input_c_features)
        input, input_x = input_fn(c_input) if input_fn is not None else (c_input, c_input)
        # ghx = self.gru(input, hx)

        # Keep the noise during both training as well as evaluation
        # c_input = gaussian(c_input, self.training, mean=0, std=0.05, one_sided=True)
        # c_input = tl.uniform(c_input, self.noise, low=-0.01, high=0.01, enforce_pos=True)
        # ghx = tl.uniform(ghx, self.noise, low=-0.01, high=0.01)

        # hx, bhx = hx_fn(ghx) if hx_fn is not None else (ghx, ghx)

        # if inspect:
        #     return self.critic_linear(hx), self.actor_linear(hx), hx, (ghx, bhx, c_input, input_x)
        # else:
        #     return self.critic_linear(hx), self.actor_linear(hx), hx
        return c_input, input_x

    # def init_hidden(self, batch_size=1):
    #     return torch.zeros(batch_size, self.gru_units)
    #
    # def get_action_linear(self, state):
    #     return self.actor_linear(state)
    #
    # def transact(self, o_x, hx):
    #     hx = self.gru(o_x, hx)
    #     return hx


if __name__ == '__main__':
    args = parser.parse_args()
    # check if have the space to save the results
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if not os.path.exists('results/' + args.model_type):
        os.mkdir('results/' + args.model_type)
    
    # start to create models...
    if args.model_type == 'inception':
        model = models.inception_v3(pretrained=True)
    elif args.model_type == 'resnet152':
        model = models.resnet152(pretrained=True)
    elif args.model_type == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif args.model_type == 'vgg19':
        model = models.vgg19_bn(pretrained=True)
    elif args.model_type == 'atari':
        env = atari_wrapper("PongDeterministic-v4")
        env.seed(1)
        obs = env.reset()
        gru_net = GRUNetConv(len(obs), 32, int(env.action_space.n))
        ox_net = ObsQBNet(gru_net.input_c_features, 100)
        model = MMNet(gru_net, None, ox_net)
        model_path = "./pongD_bgru_model.p"
        pretrained_ox_dict = {k[8:]: v for k, v in torch.load(model_path, map_location='cpu').items() if k.startswith("obx_net")}
        model.obx_net.load_state_dict(pretrained_ox_dict)
        pretrained_conv_dict = {k[8:]: v for k, v in torch.load(model_path, map_location='cpu').items() if k.startswith("gru_net.conv") or k.startswith("gru_net.input_ff")}
        model.gru_net.load_state_dict(pretrained_conv_dict)

    model.eval()
    if args.cuda:
        model.cuda()
    # read the image
    img = cv2.imread('examples/' + args.img)
    if args.model_type == 'inception':
        # the input image's size is different
        img = cv2.resize(img, (299, 299))
    img = img.astype(np.float32) 
    img = imresize(img[35:195].mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80) / 255.0
    original_image_x = calculate_outputs(img, model, args.cuda)
    attributions = random_baseline_integrated_gradients(img, model, calculate_outputs_and_gradients_steps, original_image_x, steps=50, num_random_trials=1, cuda=args.cuda)
    img_integrated_gradient_overlay = visualize(attributions, img.reshape(80, 80, 1), overlay=True, mask_mode=True)
    img_integrated_gradient = visualize(attributions, img.reshape(80, 80, 1), overlay=False)
    output_img = generate_entrie_images(img.reshape(80, 80, 1), img_integrated_gradient, img_integrated_gradient_overlay)
    plot_images_together.plot_grads("./results/atari/frame0198/blank_baseline/")