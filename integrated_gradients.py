import numpy as np
import torch

# integrated gradients
def integrated_gradients(inputs, model, predict_and_gradients, original_image_x, baseline, steps=50, cuda=False):
    # if baseline is None:
    #     baseline = 0 * inputs

    # new version
    # baseline = np.zeros(inputs.shape)

    # scale inputs and compute gradients
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(0, steps + 1)]
    grads = predict_and_gradients(scaled_inputs, model, original_image_x, cuda)
    avg_grads = np.average(grads[:-1], axis=0)
    avg_grads = np.transpose(avg_grads, (0, 2, 3, 1))
    integrated_grad = []
    for i in range(len(avg_grads)):
        integrated_grad.append((inputs.reshape(80, 80, 1) - baseline.reshape(80, 80, 1)) * avg_grads[i])

    return integrated_grad

def random_baseline_integrated_gradients(inputs, model, predict_and_gradients, original_image_x, steps, num_random_trials, cuda):
    all_intgrads = []
    for i in range(num_random_trials):
        baseline = np.zeros(inputs.shape, )
        # baseline = np.random.random(inputs.shape)
        integrated_grad = integrated_gradients(inputs, model, predict_and_gradients, original_image_x,\
                                                baseline=baseline, steps=steps, cuda=cuda)
        all_intgrads.append(integrated_grad)
        print('the trial number is: {}'.format(i))
    avg_intgrads = np.average(np.array(all_intgrads), axis=0)
    return avg_intgrads
