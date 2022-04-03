import os.path
from pathlib import Path
from pprint import pformat
import argparse

from ... import extract_features, match_features, whitening
from ... import pairs_from_covisibility, pairs_from_retrieval
from ... import colmap_from_nvm, triangulation, localize_sfm


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=Path, default='datasets/aachen',
                    help='Path to the dataset, default: %(default)s')
parser.add_argument('--outputs', type=Path, default='outputs/aachen',
                    help='Path to the output directory, default: %(default)s')
parser.add_argument('--num_covis', type=int, default=20,
                    help='Number of image pairs for SfM, default: %(default)s')
parser.add_argument('--num_loc', type=int, default=50,
                    help='Number of image pairs for loc, default: %(default)s')
parser.add_argument('--retrieval', type=str, default='netvlad',
                    choices=['netvlad', 'dns', 'geoloc', 'geoloc-rrm'],
                    help='Method used for retrieval: %(default)s')
parser.add_argument('--matching', type=str, default='superglue',
                    choices=['superglue', 'superglue-fast', 'NN-superpoint'],
                    help='Method used for matching: %(default)s')
parser.add_argument('--im_size', type=int, default=None,
                    help='Min size of the smaller dimension of input images, default: %(default)s')
parser.add_argument('--multiscale', type=str, default='[1]',
                    help="Use multiscale vectors for global descriptors, " +
                    " examples: '[1]' | '[1, 1/2**(1/2), 1/2]' | '[1, 2**(1/2), 1/2**(1/2)]' (default: '[1]')")
parser.add_argument('--query_expansion', type=int, default=None,
                    help='Number of target images used for query expansion, default: %(default)s')
parser.add_argument('--whitening', action='store_true',
                    help='Flag indicator for feature whitening')
parser.add_argument('--use_todaygan', action='store_true',
                    help='Flag indicator for use of ToDayGAN')
args = parser.parse_args()

# Setup the paths
dataset = args.dataset
images = dataset / 'images/images_upright/'

outputs = args.outputs  # where everything will be saved
ext = f'_white' if args.whitening else ''
ext += f'_{args.im_size}' if args.im_size is not None else ''
ext += f'_multiscale' if args.multiscale != '[1]' else ''
ext += f'_todaygan' if args.use_todaygan else ''

sift_sfm = outputs / 'sfm_sift'  # from which we extract the reference poses
reference_sfm = outputs / 'sfm_superpoint+superglue'  # the SfM model we will build
sfm_pairs = outputs / f'pairs-db-covis{args.num_covis}.txt'  # top-k most covisible in SIFT model

ext += f'_qe{args.query_expansion}' if args.query_expansion is not None else ''
loc_pairs = outputs / f'pairs-query-{args.retrieval}{args.num_loc}{ext}.txt'
results = outputs / f'Aachen_hloc_superpoint+{args.matching}_{args.retrieval}{args.num_loc}{ext}.txt'

# list the standard configurations available
print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')

# pick one of the configurations for extraction and matching
retrieval_conf = extract_features.confs[args.retrieval]
feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs[args.matching]

features = extract_features.main(feature_conf, images, outputs)

if not os.path.exists(reference_sfm):
    colmap_from_nvm.main(
        dataset / '3D-models/aachen_cvpr2018_db.nvm',
        dataset / '3D-models/database_intrinsics.txt',
        dataset / 'aachen.db',
        sift_sfm)
    pairs_from_covisibility.main(
        sift_sfm, sfm_pairs, num_matched=args.num_covis)
    sfm_matches = match_features.main(
        matcher_conf, sfm_pairs, feature_conf['output'], outputs)

    triangulation.main(
        reference_sfm,
        sift_sfm,
        images,
        sfm_pairs,
        features,
        sfm_matches)

retrieval_conf['preprocessing']['scales'] = list(eval(args.multiscale))
if args.im_size is not None:
    retrieval_conf['preprocessing']['resize_min'] = args.im_size
    retrieval_conf['output'] += f'_{args.im_size}'
global_descriptors = extract_features.main(retrieval_conf, images, outputs, use_todaygan=args.use_todaygan)
if args.whitening:
    global_descriptors = whitening.main(global_descriptors)
pairs_from_retrieval.main(
    global_descriptors, loc_pairs, args.num_loc,
    query_prefix='query', db_model=reference_sfm, query_expansion=args.query_expansion)
loc_matches = match_features.main(
    matcher_conf, loc_pairs, feature_conf['output'], outputs)

localize_sfm.main(
    reference_sfm,
    dataset / 'queries/*_time_queries_with_intrinsics.txt',
    loc_pairs,
    features,
    loc_matches,
    results,
    covisibility_clustering=False)  # not required with SuperPoint+SuperGlue
