import os
import pathlib
import argparse
import numpy
import nibabel
   
parser = argparse.ArgumentParser()
parser.add_argument("--inputfolder", required=True, type=str, 
    help="path to folder in which the registered images are stored.")
parser.add_argument("--exportfolder", required=True, type=str, 
    help="path to folder in which the normalized images will be stored. Will be created if it does not exist.")
parser.add_argument('--refroi', required=True, type=str, 
    help="binary mask given as .mgz file that defines the ROI for normalization.")
parserargs = parser.parse_args()
parserargs = vars(parserargs)
    
inputfolder = pathlib.Path(parserargs["inputfolder"])
assert inputfolder.is_dir()
exportfolder = pathlib.Path(parserargs["exportfolder"])

assert parserargs["inputfolder"] != parserargs["exportfolder"]

os.makedirs(exportfolder, exist_ok=True)

refroi = nibabel.load(parserargs["refroi"])
refroi_affine = refroi.affine
# Convert to numpy bool array
refroi = refroi.get_fdata().astype(bool)

images = sorted([inputfolder / f for f in os.listdir(inputfolder) if f.endswith(".mgz") and not "template" in f])

for imagepath in images:

    image = nibabel.load(imagepath)
    image_affine = image.affine
    image = image.get_fdata()

    normalized_image = image / numpy.median(image[refroi])

    assert numpy.allclose(refroi_affine, refroi_affine), "Affine transformations differ, are you sure the images are registered properly?"

    nibabel.save(nibabel.Nifti1Image(normalized_image, refroi_affine), exportfolder / imagepath.name)

    print("Normalized", imagepath, "stored to", exportfolder / imagepath.name)