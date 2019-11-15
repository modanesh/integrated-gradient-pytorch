import numpy as np
import cv2

# G = [0, 255, 0]
G = [1]
R = [255, 0, 0]

def convert_to_gray_scale(attributions):
    return np.average(attributions, axis=3)

def linear_transform(attributions, clip_above_percentile=99.9, clip_below_percentile=70.0, low=0.2, plot_distribution=False):
    transformed_list = []
    for i in range(len(attributions)):
        m = compute_threshold_by_top_percentage(attributions[i], percentage=100-clip_above_percentile, plot_distribution=plot_distribution)
        e = compute_threshold_by_top_percentage(attributions[i], percentage=100-clip_below_percentile, plot_distribution=plot_distribution)
        transformed = (1 - low) * (np.abs(attributions[i]) - e) / (m - e) + low
        transformed *= np.sign(attributions[i])
        transformed *= (transformed >= low)
        transformed = np.clip(transformed, 0.0, 1.0)
        transformed_list.append(transformed)
    transformed_np = np.array(transformed_list)
    return transformed_np

def compute_threshold_by_top_percentage(attributions, percentage=60, plot_distribution=True):
    if percentage < 0 or percentage > 100:
        raise ValueError('percentage must be in [0, 100]')
    if percentage == 100:
        return np.min(attributions)
    flat_attributions = attributions.flatten()
    attribution_sum = np.sum(flat_attributions)
    sorted_attributions = np.sort(np.abs(flat_attributions))[::-1]
    cum_sum = 100.0 * np.cumsum(sorted_attributions) / attribution_sum
    if (cum_sum >= percentage).min().data:
        threshold_idx = None
        threshold = 0
    else:
        threshold_idx = np.where(cum_sum >= percentage)[0][0]
        threshold = sorted_attributions[threshold_idx]
    if plot_distribution:
        raise NotImplementedError 
    return threshold

def polarity_function(attributions, polarity):
    if polarity == 'positive':
        return np.clip(attributions, 0, 1)
    elif polarity == 'negative':
        return np.clip(attributions, -1, 0)
    else:
        raise NotImplementedError

def overlay_function(attributions, image):
    return np.clip(0.7 * image + 0.5 * attributions, 0, 255)

def visualize(attributions, image, polarity='positive', morphological_cleanup=False, outlines=False, overlay=True, mask_mode=False):
    if polarity == 'positive':
        attributions = polarity_function(attributions, polarity=polarity)
    # convert the attributions to the gray scale
    attributions = convert_to_gray_scale(attributions)
    attributions_mask = attributions.copy()
    if morphological_cleanup:
        raise NotImplementedError
    if outlines:
        raise NotImplementedError
    # attributions = np.expand_dims(attributions, 2) * channel
    attributions = np.expand_dims(attributions, 3)
    if overlay:
        if mask_mode == False:
            attributions = overlay_function(attributions, image)
        else:
            attributions = np.expand_dims(attributions_mask, 3)
            attributions = np.clip(attributions * image, 0, 255)
            # attributions = attributions[:, :, (2, 1, 0)]
    return attributions
