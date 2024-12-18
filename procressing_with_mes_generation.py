import argparse
import logging
import napari
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from processing_methods import find_objects_by_threshold, \
    find_objects_by_template_matching, \
    find_objects_by_multiple_template_matching, \
    find_objects_by_manual_annotation, \
    find_objects_by_semiautomatic_annotation, \
    find_coordinates_by_overlap

import warnings
from utils import available_wells, load_well, stitch_arrays, get_xml_mes_template_from_file, \
    get_xml_action_list_from_file, get_pixel_scale, get_xml_timeline_template, get_xml_point, get_xml_targetwell, \
    XML_NAMESPACES, ET, plot_results, estimate_time_for_timeline, PREFIX

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(r'processing_with_mes_generation.log'),
        logging.StreamHandler()
    ]
)

X_OFFSET_PX = -140
Y_OFFSET_PX = 70


def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(
        description="SearchFirst imaging of wells based on matching a template stitched_ds. ")
    parser.add_argument(dest='folder', type=str, help='Full path to folder of first pass.')
    parser.add_argument(dest='template', type=str,
                        help='Full path to .mes file specifying the action list of the second pass.')
    parser.add_argument('-d', '--downsampling', type=float, default=0.25,
                        help='Downsampling ratio to speed up processing (default 0.25).')
    parser.add_argument('-n', '--n_objects_per_site', type=int, default=6)
    parser.add_argument('-p', '--plot_output', type=bool, default=True,
                        help='Whether to generate plots of the results.')
    parser.add_argument('-m', '--method', type=str, default='template',
                        help="""Method to use for object detection. Either `template`, 'multi-template', `threshold`, 'manual' or 'find_coordinates_by_overlap'. Make sure you
                        specify the required arguments for the method you chose.""")
    parser.add_argument('-nx', '--n_tiles_x', type=int, default=4, help='Number of tiles per well in x.')
    parser.add_argument('-ny', '--n_tiles_y', type=int, default=5, help='Number of tiles per well in y.')
    parser.add_argument('-ch', '--channel', type=str, default='C02', help='Channel name to process.')
    parser.add_argument('-op', '--optimize_time_estimate', type=bool, default=True,
                        help='If set to true the time per well is calculated, can lead to faster acquisitions.')

    # Template machting arguments
    parser.add_argument('-ot', '--object_threshold', type=float, default=0.5,
                        help='Threshold [0.0 - 1.0] for rejecting objects (default 0.5).')
    parser.add_argument('-t', '--template_path', type=str, default=None,
                        help="""Full path to template stitched_ds. Default is to search for `template.tif` in the folder.
                        If method 'multi-template' is used, all tiff files in the folder will be used as template.""")

    # Thresholding arguments
    parser.add_argument('-s', '--sigma', type=float, default=7,
                        help='Sigma of the gaussian filter to apply before thresholding.')
    parser.add_argument('-mos', '--minimum_object_size', type=int, default=1000,
                        help='Minimum object size in pixels in the downsampled stitched_ds.')

    # Find coordinates by Overlap arguments
    parser.add_argument('-o', '--overlap', type=float, default=0.1,
                        help='Overlap between FOVs in percent (0-1).')
    parser.add_argument('-spm', '--second_pass_magnification', type=float, default=40,
                        help='Magnification of the second pass.')
    parser.add_argument('-ps', '--perform_preselection', type=bool, default=True,
                        help='Whether to perform preselection of regions in napari.')

    args = parser.parse_args()

    fld = Path(args.folder)
    if not fld.is_dir():
        raise NotADirectoryError(f"Directory {fld.as_posix()} does not exist!")

    second_pass_template_file = Path(args.template)

    out_file = fld / 'SecondPass.mes'

    template_tree = get_xml_mes_template_from_file(second_pass_template_file)
    template_root = template_tree.getroot()
    timelapse_element = template_root.find('.//bts:Timelapse', XML_NAMESPACES)  # append timelines here
    channellist = template_root.find('.//bts:ChannelList', XML_NAMESPACES)  # used to extract exposure times
    current_start_time = 0  # keep track of the starting times for better time estimates

    action_list = get_xml_action_list_from_file(second_pass_template_file)

    pixel_scale = np.array(get_pixel_scale(fld / 'MeasurementDetail.mrf'))
    pixel_scale = pixel_scale / args.downsampling

    logging.info(f'processing folder {fld.as_posix()}')
    for i, well in enumerate(available_wells(fld)):
        logging.info(f'processing well {well}...')
        logging.info('loading images...')
        names, imgs = load_well(fld, well, channel=args.channel)
        logging.info(f'processing {len(imgs)} images...')
        stitched = stitch_arrays(imgs, ny=args.n_tiles_y, nx=args.n_tiles_x)

        stitched_ds = zoom(stitched, args.downsampling)

        if args.method == 'template':
            objects, non_objects = find_objects_by_template_matching(stitched_ds,
                                                                     object_threshold=args.object_threshold,
                                                                     template_path=args.template_path,
                                                                     downsampling=args.downsampling,
                                                                     n_objects_per_site=args.n_objects_per_site,
                                                                     well=well,
                                                                     )
        elif args.method == 'multi-template':
            objects, non_objects = find_objects_by_multiple_template_matching(stitched_ds,
                                                                              object_threshold=args.object_threshold,
                                                                              template_path=args.template_path,
                                                                              downsampling=args.downsampling,
                                                                              n_objects_per_site=args.n_objects_per_site,
                                                                              well=well,
                                                                              )
        elif args.method == 'threshold':
            objects, non_objects = find_objects_by_threshold(stitched_ds,
                                                             sigma=args.sigma,
                                                             minimum_object_size=args.minimum_object_size,
                                                             )
        elif args.method == 'manual':
            objects, non_objects = find_objects_by_manual_annotation(stitched_ds,
                                                                     )

        elif args.method == 'semiautomatic_threshold':
            objects, non_objects = find_objects_by_semiautomatic_annotation(stitched_ds,
                                                                            sigma=args.sigma,
                                                                            minimum_object_size=args.minimum_object_size,
                                                                            )

        elif args.method == 'find_coordinates_by_overlap':

            if args.perform_preselection:
                if i==0:
                    # in napari, annotate the region that should be imaged
                    viewer = napari.Viewer()
                    viewer.add_image(stitched_ds)
                    # rescale stitched image
                    low, high = np.quantile(stitched_ds, [0.0001, 0.9999])
                    viewer.layers['stitched_ds'].contrast_limits = [low, high]
                    viewer.add_shapes(None)
                    viewer.layers['Shapes'].mode = 'add_rectangle'
                    viewer.show(block=True)

                    # get y and x start and end coordinates
                    selection_coords = viewer.layers['Shapes'].data[0]
                    y_min = int(np.min([x[0] for x in selection_coords]))
                    y_max = int(np.max([x[0] for x in selection_coords]))
                    x_min = int(np.min([x[1] for x in selection_coords]))
                    x_max = int(np.max([x[1] for x in selection_coords]))

                    # Get the shape of the stitched image
                    shape = (y_max - y_min, x_max - x_min)

            else:
                shape = stitched_ds.shape
                x_min = 0
                y_min = 0

            objects, non_objects = find_coordinates_by_overlap(stitched_ds,
                                                               overlap=args.overlap,
                                                               downsampling=args.downsampling,
                                                               second_pass_magnification=args.second_pass_magnification,
                                                               shape=shape,
                                                               y_min=y_min,
                                                               x_min=x_min,
                                                               )

        else:
            raise NotImplementedError(f"Method `{args.method}` is not available. Use either `template` or `threshold`.")

        if args.plot_output:
            plot_results(stitched_ds, objects, non_objects, out_file=fld / f'plot_{well}.png')

        center = np.array(objects.shape) / 2
        object_positions_px = np.stack(np.where(objects)).T - center
        object_positions = (object_positions_px * pixel_scale).round(3)
        object_positions[:, 0] = - object_positions[:, 0]  # invert y coordinates

        timeline = get_xml_timeline_template(name=f'Time Line {i + 1}')
        timeline.append(action_list)
        wellsequence_element = timeline.find('.//bts:WellSequence', XML_NAMESPACES)  # append well here
        wellsequence_element.append(get_xml_targetwell(well))

        pointsequence_element = timeline.find('.//bts:FixedPosition', XML_NAMESPACES)  # append points here
        for y, x in object_positions:
            y = y + Y_OFFSET_PX / pixel_scale[0]
            x = x + X_OFFSET_PX / pixel_scale[1]
            point = get_xml_point(x=x, y=y)
            pointsequence_element.append(point)

        if args.optimize_time_estimate:
            estimated_time = estimate_time_for_timeline(timeline, channellist)
            timeline.set(PREFIX + 'OverrideExpectedTime', 'true')
            timeline.set(PREFIX + 'InitialTime', str(current_start_time))
            timeline.set(PREFIX + 'ExpectedTime', str(estimated_time))
            current_start_time += estimated_time

        timelapse_element.append(timeline)

    try:
        ET.indent(template_tree, space='   ')
    except AttributeError:
        warnings.warn(
            "xml.etree.ElementTree.indent is only available with Python version >= 3.9! Writing unformatted tree...")
    template_tree.write(out_file)


if __name__ == '__main__':
    main()
