import logging
from pathlib import Path
import napari
import numpy as np
from skimage import filters
from skimage.measure import label, regionprops
from skimage.feature import match_template
from skimage.morphology import local_maxima, remove_small_objects
from scipy.ndimage import zoom
import imageio


def find_objects_by_template_matching(stitched_ds, object_threshold, template_path, downsampling, well,
                                      n_objects_per_site):
    if template_path is None:
        template_path = Path(r'C:\Users\CVUser\Documents\Python\searchFirst\templates\template_ZE_9x.tif')
    logging.info(f"loading template from {template_path}...")
    template = imageio.imread(template_path)
    template_ds = zoom(template, downsampling)

    match = match_template(stitched_ds, template_ds, pad_input=True, mode='constant', constant_values=100)
    match_thresholded = np.where(match > object_threshold, match, 0)
    if np.sum(match_thresholded) == 0:
        logging.warning(
            f"no matches found in {well}! Try lowering the `object_threshold` if you expected to find matches in this well.")
        return np.zeros_like(stitched_ds), np.zeros_like(stitched_ds)
    maxima = local_maxima(match_thresholded)
    n_objects = np.sum(maxima)
    logging.info(f'{n_objects} objects found...')
    score = match[np.where(maxima)]
    n_actual = n_objects_per_site
    if n_objects < n_objects_per_site:
        logging.warning(f"only {n_objects} objects found instead of {n_objects_per_site}")
        n_actual = n_objects
    nth_largest_score = -np.partition(-score, n_actual - 1)[n_actual - 1]
    weighted_maxima = np.where(maxima, match, 0)
    selected_objects = np.where(weighted_maxima >= nth_largest_score, weighted_maxima, 0)
    unselected_objects = np.where(np.logical_and(weighted_maxima > 0, weighted_maxima < nth_largest_score),
                                  weighted_maxima, 0)
    return selected_objects, unselected_objects


def find_objects_by_multiple_template_matching(stitched_ds, object_threshold,
                                               template_path, downsampling,
                                               well, n_objects_per_site):
    if template_path is None:
        template_path = Path((r'C:\Users\CVUser\Documents\Python\searchFirst'
                              '\templates\template_ZE_9x.tif'))
    logging.info(f"loading template from {template_path}...")

    # get list of template files
    template_path = Path(template_path)
    template_files = template_path.glob('*.tif')

    # initialize arrays for object matches from all templates combined
    all_selected = np.empty(np.shape(stitched_ds))
    all_unselected = np.empty(np.shape(stitched_ds))

    # iterate over templates
    for fyle in template_files:
        template = imageio.imread(fyle)
        template_ds = zoom(template, downsampling)

        match = match_template(stitched_ds, template_ds, pad_input=True,
                               mode='constant', constant_values=100)
        match_thresholded = np.where(match > object_threshold, match, 0)
        if np.sum(match_thresholded) == 0:
            logging.warning(
                f"no matches found in {well} for template {fyle}! '"
                f"'Try lowering the `object_threshold` if you expected to'"
                f"' find matches in this well.")
            continue
        maxima = local_maxima(match_thresholded)
        n_objects = np.sum(maxima)
        logging.info(f'{n_objects} objects found for template {fyle}...')
        score = match[np.where(maxima)]
        n_actual = n_objects_per_site
        if n_objects < n_objects_per_site:
            logging.warning(f"only {n_objects} objects found instead of '"
                            f"'{n_objects_per_site}")
            n_actual = n_objects

        nth_largest_score = -np.partition(-score, n_actual - 1)[n_actual - 1]
        weighted_maxima = np.where(maxima, match, 0)
        selected_objects = np.where(weighted_maxima >= nth_largest_score,
                                    weighted_maxima, 0)
        unselected_objects = np.where(
            np.logical_and(weighted_maxima > 0,
                           weighted_maxima < nth_largest_score),
            weighted_maxima, 0)

        all_selected += selected_objects
        all_unselected += unselected_objects

    return all_selected, all_unselected


def find_objects_by_threshold(stitched_ds, sigma, minimum_object_size):
    # Normalize stitched_ds
    img = stitched_ds / np.amax(stitched_ds)
    # initialize canvas of zeroes
    selected_objects = np.zeros(img.shape)

    gaussian = filters.gaussian(img, sigma=sigma)

    threshold_gaussian = filters.threshold_otsu(gaussian)

    binary_gaussian = gaussian >= threshold_gaussian

    masked = remove_small_objects(binary_gaussian, minimum_object_size)

    labeled_blobs = label(masked)

    props = regionprops(labeled_blobs)

    for props in props:
        a = props.centroid
        selected_objects[int(a[0]), int(a[1])] = 1

    return selected_objects, np.zeros_like(selected_objects)


def find_objects_by_manual_annotation(stitched_ds):
    viewer = napari.Viewer()
    viewer.add_image(stitched_ds)
    # rescale stitched image
    low, high = np.quantile(stitched_ds, [0.0001, 0.9999])
    viewer.layers['stitched_ds'].contrast_limits = [low, high]
    viewer.add_points(None)
    viewer.layers['Points'].mode = 'add'
    viewer.show(block=True)

    # after the viewer is closed, the following will be executed:
    coords = viewer.layers['Points'].data
    n_objects = len(coords)
    if n_objects == 0:
        logging.warning('no coordinates were annotated...')
    else:
        logging.info(f'{n_objects} coordinates were annotated...')
    selected_objects = np.empty(np.shape(stitched_ds))
    selected_objects[coords[:, 0].astype('int'),
                     coords[:, 1].astype('int')] = 1
    unselected_objects = np.empty(np.shape(stitched_ds))

    return selected_objects, unselected_objects


def find_objects_by_semiautomatic_annotation(stitched_ds, sigma, minimum_object_size):
    objects, non_objects = find_objects_by_threshold(stitched_ds,
                                                     sigma,
                                                     minimum_object_size,
                                                     )
    points = np.stack(np.where(objects)).T

    viewer = napari.Viewer()
    viewer.add_image(stitched_ds)
    # rescale stitched image
    low, high = np.quantile(stitched_ds, [0.0001, 0.9999])
    viewer.layers['stitched_ds'].contrast_limits = [low, high]
    viewer.add_points(points, symbol='ring', size=100)
    viewer.layers['Points'].mode = 'add'
    viewer.show(block=True)

    # after the viewer is closed, the following will be executed:
    coords = viewer.layers['Points'].data
    n_objects = len(coords)
    if n_objects == 0:
        logging.warning('no coordinates were annotated...')
    else:
        logging.info(f'{n_objects} coordinates were annotated...')
    selected_objects = np.empty(np.shape(stitched_ds))
    selected_objects[coords[:, 0].astype('int'),
                     coords[:, 1].astype('int')] = 1
    unselected_objects = np.empty(np.shape(stitched_ds))

    return selected_objects, unselected_objects



def find_coordinates_by_overlap(stitched_ds: np.ndarray, overlap: float,
                                downsampling: float):
    '''
    This function places ROI's on the grid based on the user defined
    overlap percentage.

    Parameters:
    -----------
    stitched_ds: np.ndarray
        The stitched image.
    overlap: float
        The overlap in percent (0-1).
    downsampling: float
        The downsampling factor.

    Returns:
    --------
    selected_objects: np.ndarray
        The selected objects.
    unselected_objects: np.ndarray
        The unselected objects.
    '''

    # Get the shape of the stitched image
    shape = stitched_ds.shape
    logging.info(f"stitched image shape: {shape}")

    # define the tile size
    tile_shape = (2160, 2560)
    tile_shape = (int(tile_shape[0] * downsampling),
                  int(tile_shape[1] * downsampling))

    # how many tiles can fit in x and y direction
    n_x = int(np.floor((shape[1] - 2*tile_shape[1]*overlap) / (tile_shape[1]*(1-overlap))))
    n_y = int(np.floor((shape[0] - 2*tile_shape[0]*overlap) / (tile_shape[0]*(1-overlap))))

    logging.info(f"placing a grid with {n_x}*{n_y} tiles and {overlap*100}% overlap.")

    len_x = n_x * tile_shape[1] * (1 - overlap) + tile_shape[1] * overlap
    len_y = n_y * tile_shape[0] * (1 - overlap) + tile_shape[0] * overlap

    # calculate offsets to center the new grid
    offset_x = (shape[1] - len_x) / 2
    offset_y = (shape[0] - len_y) / 2

    # find the centroid of each tile so that they have the correct overlap
    x_centroids = np.linspace(tile_shape[1]/2 + offset_x,
                              (len_x - tile_shape[1]/2) + offset_x, n_x)
    y_centroids = np.linspace(tile_shape[0]/2 + offset_y,
                              (len_y - tile_shape[0]/2) + offset_y, n_y)

    selected_objects = np.empty(np.shape(stitched_ds))

    for i in y_centroids.astype('int'):
        selected_objects[i, x_centroids.astype('int')] = 1
    unselected_objects = np.empty(np.shape(stitched_ds))

    return selected_objects, unselected_objects
