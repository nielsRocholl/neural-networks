import numpy as np
from matplotlib import pyplot as plt


def scatterplot_with_imgs(x, y, data, ax=None, zoom=0.1):
    vmax = data.max()
    vmin = data.min()

    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    x, y = np.atleast_1d(x, y)
    artist = []
    i = 0
    for x0, y0 in zip(x, y):
        img = data[i, :, :, 0]
        i += 1
        offset_img = OffsetImage(img, zoom=zoom)
        offset_img.get_children()[0].set_clim(vmin, vmax)
        ab = AnnotationBbox(offset_img, (x0, y0), xycoords='data', frameon=False)
        artist.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return ax
