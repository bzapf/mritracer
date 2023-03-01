import os
import pathlib
import argparse
import numpy
import nibabel
   
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--aseg', required=True, type=str, 
    help="FreeSurfer segmentation file aseg.mgz")
parser.add_argument('--t1', default=None, type=float, 
    help="Set the voxel valuels inside mask to t1 (create synthetic T1 map). Note: the other scripts use units of seconds")
parserargs = parser.parse_args()
parserargs = vars(parserargs)


# From the FreeSurfer labels
# https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
# we set the mask to 0 for those voxels labeled the CSF:
csf_labels = [4, 5, 14, 15, 24, 43, 44]

aseg = nibabel.load(parserargs["aseg"])
affine = aseg.affine

aseg = aseg.get_fdata().astype(int)

csf_mask = numpy.zeros(tuple(aseg.shape), dtype=bool)

for csf_label in csf_labels:
    csf_mask += (aseg == csf_label) 

brainmask = (aseg > 0) * (~ csf_mask)

outfile = pathlib.Path(parserargs["aseg"]).parent / "parenchyma_only.mgz"
nibabel.save(nibabel.Nifti1Image((brainmask).astype(float), affine), outfile)

print("Created mask, to view run")
print("freeview ", parserargs["aseg"], " ", outfile)

if parserargs["t1"] is not None:
    outfile = pathlib.Path(parserargs["aseg"]).parent / "synthetic_T1_map.mgz"
    nibabel.save(nibabel.Nifti1Image((brainmask * parserargs["t1"]).astype(float), affine), outfile)