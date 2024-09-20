#!/bin/python3

# Copyright 2024 Image Analysis Lab, German Center for Neurodegenerative Diseases
# (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# imports

import argparse
import os
import sys

import nibabel as nib
import numpy as np

# parse arguments

def _parse_arguments():
    """
    Parse input arguments.

    Returns
    -------
    class
        Arguments.
    """

    # setup parser
    parser = argparse.ArgumentParser(
        description="This program conducts FDR and FDR2 corrections for multiple comparisons for vertex-based data.",
        add_help=True,
    )

    # required arguments
    required = parser.add_argument_group("Required arguments")

    required.add_argument(
        "--left",
        dest="left_file",
        help="Filename of a left-hemisphere surface overlay file. The expected format is mgh. Expected values are signed sig values.",
        default=None,
        metavar="<filename>",
        required=False, # set to False, because only one of --left and --right is required; will be checked
    )

    required.add_argument(
        "--right",
        dest="right_file",
        help="Filename of a right-hemisphere surface overlay file. The expected format is mgh. Expected values are signed sig values.",
        default=None,
        metavar="<filename>",
        required=False, # set to False, because only one of --left and --right is required; will be checked
    )

    required.add_argument(
        "--method",
        dest="method",
        help="Which method to apply for multiple comparison correction, can be FDR or FDR2.",
        default=None,
        metavar="<FDR|FDR2>",
        required=False, # set to False, because requirements checking is done separately
    )

    # optional arguments
    optional = parser.add_argument_group("Optional arguments")

    optional.add_argument(
        "--fdr",
        dest="fdr_value",
        help="False discovery rate. Default: 0.05.",
        default=0.05,
        metavar="<float>",
        required=False,
        type=float,
    )

    optional.add_argument(
        "--sided-test",
        dest="sided_test",
        help="Whether directed tests should be computed. \"left\" for left-sided, \"right\" for right-sided, and \"two-sided\" for two-sided (undirected) test. Requires signed sig values as input. Default: two-sided.",
        default="two-sided",
        metavar="<left|right|two-sided>",
        required=False,
    )

    optional.add_argument(
        "--left-mask",
        dest="left_mask",
        help="Filename of a left-hemisphere mask file. The expected format is a label file. Default: None.",
        default=None,
        metavar="<mask>",
        required=False,
    )

    optional.add_argument(
        "--right-mask",
        dest="right_mask",
        help="Filename of a right-hemisphere mask file. The expected format is a label file. Default: None.",
        default=None,
        metavar="<mask>",
        required=False,
    )

    optional.add_argument(
        "--left-output",
        dest="left_output",
        help="Filename to write out thresholded left-hemisphere surface overlay file. Default: None.",
        default=None,
        metavar="<filename>",
        required=False,
    )

    optional.add_argument(
        "--right-output",
        dest="right_output",
        help="Filename to write out thresholded right-hemisphere surface overlay file. Default: None.",
        default=None,
        metavar="<filename>",
        required=False,
    )

    optional.add_argument(
        "--text-output",
        dest="text_output",
        help="Write threshold for left and/or right hemisphere to a text file. Default: None.",
        default=None,
        metavar="<filename>",
        required=False,
    )

    #
    args = parser.parse_args()

    #
    return args

# read mgh

def _read_mgh(fname):
    """
    Reads Freesurfer's .mgh or .mgz data files.

    Parameters
    ----------
    fname : str
        A Freesurfer surface overlay file. Cannot have more than one frame.

    Returns
    -------
    dict
        Image header.
    array-like
        Image data, will be 1D.
    """

    # Load the .mgh or .mgz file
    img = nib.load(fname)
    img_dims = img.header.get_data_shape()
    img_data = img.get_fdata()
    if len(img_dims)>3:
        assert img_dims[3] == 1, "ERROR: 4th dimension of input image must be equal to one"
    data = img_data.flatten()

    return img, data

# read label

def _read_label(fname):
    """
    Reads Freesurfer's label files.

    Parameters
    ----------
    fname : str
        A Freesurfer label file.

    Returns
    -------
    array-like
        Indices of vertices included in label.
    """

    return nib.freesurfer.io.read_label(fname)

# write mgh

def _write_mgh(fname, img, data):
    """
    Writes data to a Freesurfer's .mgh or .mgz file.

    Parameters
    ----------
    fname : str
        Output file name.
    img : dict
        Image header.
    data : array-like
        Image data.
    """

    # Create an MGH image
    new_img = nib.MGHImage(data.astype(np.float32), img.affine)

    # Save the MGH image
    nib.save(img=new_img, filename=fname)

    return

# compute fdr

def compute_fdr(p, sgn, mask_indices=None, fdr=0.05, tail="two-sided"):

    """
    Computes the false discovery rate (FDR).

    Parameters
    ----------
    p : array-like
        p-values.
    sgn : array-like
        Signs of p-values.
    mask_indices : array-like
        Mask indices. Default: None.
    fdr : float
        False discovery rate, between 0 and 1. Default: 0.05.
    tail : str
        "left" for left-sided, "right" for right-sided, or "two-sided" for two-sided hypothesis testing. Default is "two-sided".

    Returns
    -------
    float
        The p-value threshold based on the given FDR. A threshold of p=0 is returned if no data survive the correction.
    """

    # determine length of non-masked p-values
    nv0 = len(p)

    # determine mask_indices if not present
    if mask_indices is None:
        mask_indices = np.arange(nv0)

    # get masked p-values and masked sgn
    masked_p = p[mask_indices]
    masked_sgn = sgn[mask_indices]

    # compute sided p-values for conducting directed tests; these will be un-signed
    sided_masked_p = masked_p.copy()
    if tail == "left":
        sided_masked_p[masked_sgn < 0] *= 0.5
        sided_masked_p[masked_sgn > 0] = 1 - 0.5 * sided_masked_p[masked_sgn > 0]
    elif tail == "right":
        sided_masked_p[masked_sgn > 0] *= 0.5
        sided_masked_p[masked_sgn < 0] = 1 - 0.5 * sided_masked_p[masked_sgn < 0]

    # sort sided, masked p-values
    sorted_sided_masked_p = np.sort(sided_masked_p)

    # compute FDR thresholds
    nvs = len(sorted_sided_masked_p)
    nvs_indices = np.arange(nvs) + 1
    fdr_i = fdr * nvs_indices / nvs

    # compare sorted p-values against FDR thresholds and find argmax
    imax = np.max(np.where(sorted_sided_masked_p <= fdr_i)[0], initial=-1) # -1 will be returned in case np.where(...)[0] is empty and no max can be taken; -1 is to be interpreted as an (impossible) positional index

    # check if comparison has been successful, otherwise return zero
    pthresh = sorted_sided_masked_p[imax] if imax != -1 else 0.0

    # return
    return pthresh

# compute fdr2

def compute_fdr2(p, sgn, mask_indices=None, fdr=0.05, tail="two-sided"):
    """
    Computes the two-stage false discovery rate (FDR2).

    Parameters
    ----------
    p : array-like
        p-values.
    sgn : array-like
        Signs of p-values.
    mask_indices : array-like
        Mask indices. Default: None.
    fdr : float
        False discovery rate, between 0 and 1. Default: 0.05.
    tail : str
        "left" for left-sided, "right" for right-sided, or "two-sided" for two-sided hypothesis testing. Default is "two-sided".

    Returns
    -------
    pth : float
        FDR2 threshold.
    detvtx : array-like
        Detected vertices.
    m0 : int
        Estimated number of null vertices.
    """

    # determine length of non-masked p-values
    nv0 = len(p)

    # determine mask_indices if not present
    if mask_indices is None:
        mask_indices = np.arange(nv0)

    # get signed, masked p-values
    masked_p = p[mask_indices]
    masked_sgn = sgn[mask_indices]

    # compute sided p-values for conducting directed tests; these will be un-signed
    sided_masked_p = masked_p.copy()
    if tail == "left":
        sided_masked_p[masked_sgn < 0] *= 0.5
        sided_masked_p[masked_sgn >= 0] = 1 - 0.5 * sided_masked_p[masked_sgn >= 0] # note: changed > to >= 

    elif tail == "right":
        sided_masked_p[masked_sgn >= 0] *= 0.5 # note: changed > to >= 
        sided_masked_p[masked_sgn < 0] = 1 - 0.5 * sided_masked_p[masked_sgn < 0]

    # determine length of sided, masked p-values
    nv = len(sided_masked_p)

    # -------------------------------------------------------------------------
    # # First stage (m0 estimation)
    # q0     : initial rate
    # pth0   : initial FDR threshold
    # detvtx : detected vertices
    # ndetv0 : number of detected vertices
    # m0     : estimated number of null vertices

    # compute fdr (using q0)
    sorted_sided_masked_p = np.sort(sided_masked_p)
    nvs = len(sorted_sided_masked_p)
    nvs_indices = np.arange(1, nvs + 1)
    q0 = fdr / (1 + fdr)
    imax = np.max(np.where(sorted_sided_masked_p <= q0 * nvs_indices / nvs)[0], initial=-1)
    pth0 = sorted_sided_masked_p[imax] if imax != -1 else 0.0 

    if tail == 'two-sided':
        detv0 = mask_indices[sided_masked_p <= pth0]
    elif tail == 'left':
        vtx = mask_indices[masked_sgn < 0]
        detv0 = vtx[sided_masked_p[masked_sgn < 0] <= pth0]
    elif tail == 'right':
        vtx = mask_indices[masked_sgn > 0]
        detv0 = vtx[sided_masked_p[masked_sgn > 0] <= pth0]

    ndetv0 = len(detv0)
    m0 = nv - ndetv0

    # -------------------------------------------------------------------------
    # Second stage
    if (ndetv0 != 0) and (ndetv0 != nv):

        # compute fdr (using q0 * nv / m0)
        sorted_sided_masked_p = np.sort(sided_masked_p)
        q0 = q0 * nv / m0
        nvs = len(sorted_sided_masked_p)
        nvs_indices = np.arange(1, nvs + 1)
        imax = np.max(np.where(sorted_sided_masked_p <= q0 * nvs_indices / nvs)[0], initial=-1)
        pth = sorted_sided_masked_p[imax] if imax != -1 else np.array(0.0, ndmin=1, dtype='float64')

        if tail == 'two-sided':
            detvtx = mask_indices[sided_masked_p <= pth]
        elif tail == 'left':
            detvtx = vtx[sided_masked_p[masked_sgn < 0] <= pth]
        elif tail == 'right':
            detvtx = vtx[sided_masked_p[masked_sgn > 0] <= pth]
    else:
        detvtx = detv0
        pth = pth0

    # -------------------------------------------------------------------------
    # Output

    # return
    return pth, detvtx, m0

# main

if __name__ == "__main__":

    # message

    print("---------------------------------------------------")
    print("FDR-based correction for multiple comparisons      ")
    print("---------------------------------------------------")
    print()

    # parse args

    args = _parse_arguments()

    # check args

    if args.left_file is None and args.right_file is None:
        print("ERROR: at least one of --left or --right arguments is required, exiting. Use --help to see details.")
        print()
        sys.exit(1)

    if args.method is None:
        print("ERROR: the --method argument is required, exiting. Use --help to see details.")
        print()
        sys.exit(1)

    elif args.method != "FDR" and args.method != "FDR2":
        print("ERROR: the --method argument can only be 'FDR' or 'FDR2', exiting. Use --help to see details.")
        print()
        sys.exit(1)

    # evaluate args

    if args.left_file is not None:
        if os.path.isfile(args.left_file):
            print("Found " + args.left_file)
        else:
            raise RuntimeError("ERROR: Could not find file " + args.left_file)

    if args.right_file is not None:
        if os.path.isfile(args.right_file):
            print("Found " + args.right_file)
        else:
            raise RuntimeError("ERROR: Could not find file " + args.right_file)

    if args.left_mask is not None:
        if args.left_file is None:
            raise RuntimeError("ERROR: --left must be present if using --left-mask")
        if os.path.isfile(args.left_mask):
            print("Found " + args.left_mask)
        else:
            raise RuntimeError("Could not find file " + args.left_mask)

    if args.right_mask is not None:
        if args.right_file is None:
            raise RuntimeError("ERROR: --right must be present if using --right-mask")
        if os.path.isfile(args.right_mask):
            print("Found " + args.right_mask)
        else:
            raise RuntimeError("Could not find file " + args.right_mask)

    if args.left_file is not None and args.right_file is not None:
        if args.left_mask is not None and args.right_mask is None:
            raise RuntimeError("ERROR: --left-mask and --right-mask must be present if using --left and --right")
        elif args.left_mask is None and args.right_mask is not None:
            raise RuntimeError("ERROR: --left-mask and --right-mask must be present if using --left and --right")

    if args.left_output is not None and args.left_file is None:
        raise RuntimeError("ERROR: --left must be present if using --left-output")
    if args.right_output is not None and args.right_file is None:
        raise RuntimeError("ERROR: --right must be present if using --right-output")        

    if args.fdr_value <= 0 or args.fdr_value >= 1:
        raise RuntimeError("ERROR: The --fdr value must be numeric and between 0 and 1.")

    # load data

    if args.left_file is not None:
        left_hdr, left_data = _read_mgh(args.left_file)
    else:
        left_hdr, left_data = None, None

    if args.right_file is not None:
        right_hdr, right_data = _read_mgh(args.right_file)
    else:
        right_hdr, right_data = None, None

    # load mask (if present)

    if args.left_mask is not None:
        left_mask_indices = _read_label(args.left_mask)
    else:
        left_mask_indices = None

    if args.right_mask is not None:
        right_mask_indices = _read_label(args.right_mask)
    else:
        right_mask_indices = None

    # concatenate (if both left and right are present)

    if args.left_file is not None and args.right_file is not None:
        sig_values = np.concatenate((left_data, right_data))
        if left_mask_indices is not None and right_mask_indices is not None:
            mask_indices = np.concatenate((left_mask_indices, right_mask_indices + left_data.size))
        else:
            mask_indices = None
    elif args.left_file is not None and args.right_file is None:
        sig_values = left_data
        mask_indices = left_mask_indices
    elif args.left_file is None and args.right_file is not None:
        sig_values = right_data
        mask_indices = right_mask_indices

    # transform from signed sig values into unsigned p-values; also keep sign;
    # thresholding will be done on p_values, left and right variants are needed for output

    p_values = 10**(-np.abs(sig_values))
    p_values_sign = np.sign(sig_values)
    if args.left_file is not None:
        p_values_left = 10**(-np.abs(left_data))
        p_values_left_sign = np.sign(left_data)
    if args.right_file is not None:
        p_values_right = 10**(-np.abs(right_data))
        p_values_right_sign = np.sign(right_data)

    # determine hemisphere
    if args.left_file is not None and args.right_file is not None:
        hemi = "both hemispheres"
    elif args.left_file is not None and args.right_file is None:
        hemi = "left hemisphere"
    elif args.left_file is None and args.right_file is not None:
        hemi = "right hemisphere"

    # compute FDR
    if args.method == "FDR":
        fdr_thresh = compute_fdr(p=p_values, sgn=p_values_sign, mask_indices=mask_indices, fdr=args.fdr_value, tail=args.sided_test)
        print("%s %s threshold at q=%s for %s: p=%s" % (str.capitalize(args.sided_test), args.method, args.fdr_value, hemi, "{0:.6g}".format(fdr_thresh)))

    # compute FDR2
    if args.method == "FDR2":
        fdr_thresh, _, _ = compute_fdr2(p=p_values, sgn=p_values_sign, mask_indices=mask_indices, fdr=args.fdr_value, tail=args.sided_test)
        print("%s %s threshold at q=%s for %s: p=%s" % (str.capitalize(args.sided_test), args.method, args.fdr_value, hemi, "{0:.6g}".format(fdr_thresh)))

    # write data to text file
    if args.text_output is not None:
        print("Writing output to ")
        with open(args.text_output, "w") as text_file:
            text_file.write("%s %s threshold at q=%s for %s: p=%s" % (str.capitalize(args.sided_test), args.method, args.fdr_value, hemi, "{0:.6g}".format(fdr_thresh)))

    # write data to overlay files
    if args.left_output is not None:
        print("Writing thresholded, signed sig values to %s. Non-significant vertices and vertices outside the mask will be set to zero." % args.left_output)
        # apply mask if given
        if left_mask_indices is not None:
            masked_p_values_left = np.ones_like(p_values_left)
            masked_p_values_left[left_mask_indices] = p_values_left[left_mask_indices]
        else:
            masked_p_values_left = p_values_left
        # threshold
        thr_masked_p_values_left = np.ones_like(p_values_left)
        thr_masked_p_values_left[masked_p_values_left < fdr_thresh] = masked_p_values_left[masked_p_values_left < fdr_thresh]
        # convert to sig
        thr_masked_sig_values_left = -np.log10(thr_masked_p_values_left)
        # apply sign
        signed_thr_masked_sig_values_left = p_values_left_sign * thr_masked_sig_values_left
        _write_mgh(args.left_output, left_hdr, signed_thr_masked_sig_values_left)

    if args.right_output is not None:
        print("Writing thresholded, signed sig values to %s. Non-significant vertices and vertices outside the mask will be set to zero." % args.right_output)
        # apply mask if given
        if right_mask_indices is not None:
            masked_p_values_right = np.ones_like(p_values_right)
            masked_p_values_right[right_mask_indices] = p_values_right[right_mask_indices]
        else:
            masked_p_values_right = p_values_right
        # threshold
        thr_masked_p_values_right = np.ones_like(p_values_right)
        thr_masked_p_values_right[masked_p_values_right < fdr_thresh] = masked_p_values_right[masked_p_values_right < fdr_thresh]
        # convert to sig
        thr_masked_sig_values_right = -np.log10(thr_masked_p_values_right)
        # apply sign
        signed_thr_masked_sig_values_right = p_values_right_sign * thr_masked_sig_values_right
        _write_mgh(args.right_output, right_hdr, signed_thr_masked_sig_values_right)



