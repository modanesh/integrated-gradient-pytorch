import torch
import torch.nn.functional as F
import cv2
import numpy as np
import torch.nn as nn
# from scipy.misc import imshow, imsave
from matplotlib.pyplot import imshow,imsave


# def calculate_outputs_and_gradients(inputs, model, target_label_idx, cuda=False):
def calculate_outputs(input_img, model, cuda=False):
    input = pre_processing(input_img, cuda)
    output_c, output_x = model(input)
    return output_x


def calculate_outputs_and_gradients_steps(inputs, model, original_image_x, cuda=False):
    # do the pre-processing
    gradients = []
    for i in range(len(inputs)):
        if i < 10:
            imsave('results/atari/noise_image_step0' + str(i) +'_frame0354.jpg', inputs[i].reshape(80, 80), vmin=0, vmax=1)
        else:
            imsave('results/atari/noise_image_step' + str(i) + '_frame0354.jpg', inputs[i].reshape(80, 80), vmin=0, vmax=1)
        bits_gradients = []
        input = pre_processing(inputs[i], cuda)
        output_c, output_x = model(input)
        for i in range(len(output_x[0])):
            model.zero_grad()
            loss = nn.MSELoss()(output_x[0][i], original_image_x[0][i])
            loss.backward(retain_graph=True)
            gradient = input.grad.cpu().data.numpy()[0].tolist()
            bits_gradients.append(gradient)
            input.grad.data.fill_(0)
        gradients.append(bits_gradients)
    gradients = np.array(gradients)
    assert gradients[0][0].shape == inputs[0].shape
    return gradients


def pre_processing(obs, cuda):
    mean = np.array([0.485]).reshape([1, 1, 1])
    std = np.array([0.229]).reshape([1, 1, 1])
    obs = (obs - mean) / std
    obs = np.expand_dims(obs, 0)
    obs = np.array(obs)
    if cuda:
        torch_device = torch.device('cuda:0')
    else:
        torch_device = torch.device('cpu')
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=torch_device, requires_grad=True)
    return obs_tensor

# generate the entire images
def generate_entrie_images(img_origin, img_integrad, img_integrad_overlay, bit_values):
    imsave('results/atari/original_image_frame0354.jpg', img_origin.reshape(80, 80))
    for i in range(len(img_integrad)):
        if i < 10:
            imsave('results/atari/integrated_grad_bit0' + str(i) + '_bv' + str(int(bit_values[0][i].data)) + '_frame0354.jpg', img_integrad[i].reshape(80, 80))
            imsave('results/atari/integrated_grad_overlay_bit0' + str(i) + '_bv' + str(int(bit_values[0][i].data)) + '_frame0354.jpg', img_integrad_overlay[i].reshape(80, 80))
        else:
            imsave('results/atari/integrated_grad_bit' + str(i) + '_bv' + str(int(bit_values[0][i].data)) + '_frame0354.jpg', img_integrad[i].reshape(80, 80))
            imsave('results/atari/integrated_grad_overlay_bit' + str(i) + '_bv' + str(int(bit_values[0][i].data)) + '_frame0354.jpg', img_integrad_overlay[i].reshape(80, 80))

