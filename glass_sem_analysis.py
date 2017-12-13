"""
from http://scipy-lectures.github.com/
"""

from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as pl
import matplotlib.cm as cm
from scipy import ndimage
from datetime import datetime
import os
import sys

parameter_file, img_file = sys.argv[1:3]


parameters = {}
with open(parameter_file) as f:
    exec(f.read(), parameters)

img_id, ext = os.path.splitext(img_file)

timestamp = datetime.now()
output_dir = "Data/%s" % timestamp.strftime("%Y%m%d")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_prefix = os.path.join(output_dir, "%s_%s" % (img_id, timestamp.strftime("%H%M%S")))

def remove_axes():
    ax = pl.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

dat = pl.imread(img_file)

# Crop the image to remove the lower panel with measure information.
dat = dat[60:]

# Slightly filter the image with a median filter in order to refine its histogram.
filter_size = (parameters["filter_size"], parameters["filter_size"])
filtdat = ndimage.median_filter(dat, size=filter_size)
bins_dat, edges_dat = np.histogram(dat, bins=np.arange(256))
bins_filtdat, edges_filtdat = np.histogram(filtdat, bins=np.arange(256))
pl.plot(edges_dat[:-1], bins_dat, 'b-')
pl.plot(edges_filtdat[:-1], bins_filtdat, 'g-')
pl.xlim((-5, 260))
pl.ylim((-1000, 145000))
pl.savefig("%s_histogram.png" % output_prefix)

# Define masks for sand pixels, glass pixels and bubble pixels
void = filtdat <= parameters["bubble_sand_boundary"]
sand = np.logical_and(filtdat > parameters["bubble_sand_boundary"], filtdat <= parameters["sand_glass_boundary"])
glass = filtdat > parameters["sand_glass_boundary"]

# Create image with each phase a different colour
phases = void.astype(np.int) + 2*glass.astype(np.int) + 3*sand.astype(np.int)
pl.clf()
colourmap = getattr(cm, parameters["phases_colourmap"])
pl.imshow(phases, cmap=colourmap, origin="lower")
remove_axes()
pl.colorbar()
pl.savefig("%s_phases.png" % output_prefix)

# Clean the phases
sand_op = ndimage.binary_opening(sand, iterations=parameters["cleaning_iterations"])
sand_labels, sand_nb = ndimage.label(sand_op)
sand_areas = np.array(ndimage.sum(sand_op, sand_labels, np.arange(sand_labels.max()+1)))
mask = sand_areas > parameters["cleaning_threshold"]
remove_small_sand = mask[sand_labels.ravel()].reshape(sand_labels.shape)

pl.clf()
pl.subplot(1, 2, 1)
pl.imshow(sand, cmap=cm.gist_gray, origin="lower")
remove_axes()
pl.subplot(1, 2, 2)
pl.imshow(remove_small_sand, cmap=cm.gist_gray, origin="lower")
remove_axes()
pl.savefig("%s_sand.png" % output_prefix)

# Compute the mean size of bubbles.
bubbles_labels, bubbles_nb = ndimage.label(void)
bubbles_areas = np.bincount(bubbles_labels.ravel())[1:]
mean_bubble_size = bubbles_areas.mean()
median_bubble_size = np.median(bubbles_areas)
print(mean_bubble_size, median_bubble_size)
