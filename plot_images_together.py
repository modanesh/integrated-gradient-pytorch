from matplotlib.pyplot import imsave, imshow, imread
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import re
import numpy as np
import cv2

def plot_grads(path):
    noise_files = sorted([f for f in os.listdir(path) if re.match(r'noise*', f)])
    grad_overlay_files = sorted([f for f in os.listdir(path) if re.match(r'integrated_grad_overlay*', f)])
    grad_files = sorted([f for f in os.listdir(path) if re.match(r'integrated_grad_bit*', f)])

    import matplotlib.pyplot as plt
    import numpy as np

    # settings
    h, w = 80, 80        # for raster image
    nrows, ncols = 5, 10  # array of sub-plots
    figsize = [20, 20]     # figure size, inches

    # prep (x,y) for extra plotting on selected sub-plots
    # xs = np.linspace(0, 2*np.pi, 60)  # from 0 to 2pi
    # ys = np.abs(np.sin(xs))           # absolute of sine

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        # img = np.random.randint(10, size=(h,w))
        img = imread(os.path.join(path, noise_files[i]))
        axi.imshow(img)
        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        # write row/col indices as axes' title for identification
        axi.set_title(noise_files[i].split(".")[:-1][0].split("_")[2], fontsize=10)

    # one can access the axes by ax[row_id][col_id]
    # do additional plotting on ax[row_id][col_id] of your choice
    plt.tight_layout(True)
    plt.savefig(path + "baselines.jpg")



    # settings
    nrows2, ncols2 = 10, 10  # array of sub-plots
    figsize = [20, 20]     # figure size, inches

    # prep (x,y) for extra plotting on selected sub-plots
    # xs = np.linspace(0, 2*np.pi, 60)  # from 0 to 2pi
    # ys = np.abs(np.sin(xs))           # absolute of sine

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows2, ncols=ncols2, figsize=figsize)

    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        # img = np.random.randint(10, size=(h,w))
        img = imread(os.path.join(path, grad_overlay_files[i]))
        axi.imshow(img)
        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        # write row/col indices as axes' title for identification
        axi.set_title(grad_overlay_files[i].split(".")[:-1][0].split("_")[3], fontsize=10)

    # one can access the axes by ax[row_id][col_id]
    # do additional plotting on ax[row_id][col_id] of your choice
    plt.tight_layout(True)
    plt.savefig(path + "grad_overlays.jpg")


    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows2, ncols=ncols2, figsize=figsize)

    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        # img = np.random.randint(10, size=(h,w))
        img = imread(os.path.join(path, grad_files[i]))
        axi.imshow(img)
        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        # write row/col indices as axes' title for identification
        axi.set_title(grad_files[i].split(".")[:-1][0].split("_")[2], fontsize=10)

    # one can access the axes by ax[row_id][col_id]
    # do additional plotting on ax[row_id][col_id] of your choice
    plt.tight_layout(True)
    plt.savefig(path + "grads.jpg")

plot_grads("./results/atari/frame0198/noise_baseline/")
plot_grads("./results/atari/frame0276/noise_baseline/")