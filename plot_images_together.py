from matplotlib.pyplot import imsave, imshow, imread
import matplotlib.pyplot as plt
import os
import re


def plot_grads(path):
    noise_files = sorted([f for f in os.listdir(path) if re.match(r'noise*', f)])
    grad_overlay_files = sorted([f for f in os.listdir(path) if re.match(r'integrated_grad_overlay*', f)])
    grad_files = sorted([f for f in os.listdir(path) if re.match(r'integrated_grad_bit*', f)])

    # settings
    nrows, ncols = 5, 10  # array of sub-plots
    figsize = [20, 20]     # figure size, inches

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        img = imread(os.path.join(path, noise_files[i]))
        axi.imshow(img)
        # write row/col indices as axes' title for identification
        axi.set_title(noise_files[i].split(".")[:-1][0].split("_")[2], fontsize=10)

    # one can access the axes by ax[row_id][col_id]
    # do additional plotting on ax[row_id][col_id] of your choice
    plt.tight_layout(True)
    plt.savefig(path + "baselines.jpg")

    # settings
    nrows2, ncols2 = 10, 10  # array of sub-plots
    figsize = [20, 20]     # figure size, inches

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows2, ncols=ncols2, figsize=figsize)

    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        img = imread(os.path.join(path, grad_overlay_files[i]))
        axi.imshow(img)
        # write row/col indices as axes' title for identification
        axi.set_title(grad_overlay_files[i].split(".")[:-1][0].split("_")[3] + '_bit value: ' + grad_overlay_files[i].split(".")[:-1][0].split("_")[4][2:], fontsize=10)

    # one can access the axes by ax[row_id][col_id]
    # do additional plotting on ax[row_id][col_id] of your choice
    plt.tight_layout(True)
    plt.savefig(path + "grad_overlays.jpg")

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows2, ncols=ncols2, figsize=figsize)

    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        img = imread(os.path.join(path, grad_files[i]))
        axi.imshow(img)
        # write row/col indices as axes' title for identification
        axi.set_title(grad_files[i].split(".")[:-1][0].split("_")[2] + '_bit value: ' + grad_files[i].split(".")[:-1][0].split("_")[3][2:], fontsize=10)

    # one can access the axes by ax[row_id][col_id]
    # do additional plotting on ax[row_id][col_id] of your choice
    plt.tight_layout(True)
    plt.savefig(path + "grads.jpg")

plot_grads("./results/atari/frame0354/noise_baseline/")
plot_grads("./results/atari/frame0354/blank_baseline/")
plot_grads("./results/atari/frame0198/noise_baseline/")
plot_grads("./results/atari/frame0198/blank_baseline/")
plot_grads("./results/atari/frame0276/noise_baseline/")
plot_grads("./results/atari/frame0276/blank_baseline/")