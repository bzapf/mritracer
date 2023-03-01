import argparse
import itertools
from time import time
from typing import Tuple
import dolfin
import nibabel
from nibabel.affines import apply_affine
import numpy as np
from tqdm import tqdm

def function_to_image(function, template_image, extrapolation_value, mask) -> Tuple[nibabel.Nifti1Image, np.ndarray]:

    shape = template_image.get_fdata().shape

    output_data = np.zeros(shape) + extrapolation_value

    vox2ras = template_image.header.get_vox2ras_tkr()

    V = function.function_space()

    ## Code to get a bounding box for the mesh, used to not iterate over all the voxels in the image
    if mask is None:
        imap = V.dofmap().index_map()
        num_dofs_local = (imap.local_range()[1] - imap.local_range()[0])
        xyz = V.tabulate_dof_coordinates()
        xyz = xyz.reshape((num_dofs_local, -1))
        image_coords = apply_affine(np.linalg.inv(vox2ras), xyz)
        
        lower_bounds = np.maximum(0, np.floor(image_coords.min(axis=0)).astype(int))
        upper_bounds = np.minimum(shape, np.ceil(image_coords.max(axis=0)).astype(int))
        
        all_relevant_indices = itertools.product(
            *(range(start, stop+1) for start, stop in zip(lower_bounds, upper_bounds))
        )
        num_voxels_in_mask = np.product(1 + upper_bounds - lower_bounds)
        fraction_of_image = num_voxels_in_mask / np.product(shape)
        print(f"Computed mesh bounding box, evaluating {fraction_of_image:.0%} of all image voxels")
        print(f"There are {num_voxels_in_mask} voxels in the bounding box")
    else:
        raise NotImplementedError
        #     raise NotImplementedError
        #     mask = nibabel.load(args.mask).get_fdata()
        #     # breakpoint()
        #     if args.skip_value is None:
        #         skip_value = args.extrapolation_value
        #     else:
        #         skip_value = args.skip_value
        #     if np.isnan(skip_value):
        #         print("Extrapolation value is NaN")
        #         mask = ~np.isnan(mask)
        #     else:
        #         mask = ~np.isclose(mask, skip_value)
        #     nonzeros = np.nonzero(mask)
        #     num_voxels_in_mask = len(nonzeros[0])
        #     all_relevant_indices = zip(*nonzeros)
        #     fraction_of_image = num_voxels_in_mask / np.product(output_data.shape)
        #     print(f"Using mask, evaluating {fraction_of_image:.0%} of all image voxels")
        #     print(f"There are {num_voxels_in_mask} voxels in the mask")
        #     if fraction_of_image > 1 - 1e-10 and not args.allow_full_mask:
        #         raise ValueError("The supplied mask covers the whole image so you are probably doing something wrong." /
        #                         " To allow for this behaviour, run with --allow_full_mask")



    # Populate image
    def eval_fenics(f, coords, extrapolation_value):
        try:
            return f(*coords)
        except RuntimeError:
            return extrapolation_value
    eps = 1e-12

    progress = tqdm(total=num_voxels_in_mask)
    
    for xyz_vox in all_relevant_indices:
        xyz_ras = apply_affine(vox2ras, xyz_vox) # transform_coords(coords, vox2ras, inverse=True)
        output_data[xyz_vox] = eval_fenics(function, xyz_ras, extrapolation_value)
        progress.update(1)

    
    output_data = np.where(output_data < eps, eps, output_data)
    # Save output
    output_nii = nibabel.Nifti1Image(output_data, template_image.affine, template_image.header)

    return output_nii, output_data

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, type=str, help="FEniCS mesh file")
    parser.add_argument("--image", required=True, type=str, help="MRI file to get transformation matrix from")
    parser.add_argument("--hdf5_file", required=True, type=str, help="File storing the FEniCS function")
    parser.add_argument("--hdf5_name", required=True, type=str, help="Name of function inside the HDF5 file")
    parser.add_argument("--output", required=True, type=str, help="MRI file to save the function to (e.g. shear_modulus.nii)")
    parser.add_argument("--function_space", type=str, default="CG")
    parser.add_argument("--function_degree", type=int, default=1)
    parser.add_argument("--extrapolation_value", type=float, default=float('nan'))
    parser.add_argument("--mask", type=str, help="Mask used to specify which image voxels to evaluate")
    parser.add_argument("--skip_value", type=float, help="Voxel value indicating that a voxel should be skipped in the mask. If unspecified, it's the same as the extrapolation value.")

    parserargs = vars(parser.parse_args())

    # Load data
    nii_img = nibabel.load(parserargs["image"])

    if parserargs["mesh"].endswith(".xml"):
        brainmesh = dolfin.Mesh(parserargs["mesh"])
    else:
        brainmesh= dolfin.Mesh()
        hdf = dolfin.HDF5File(brainmesh.mpi_comm(), parserargs["mesh"], "r")
        hdf.read(brainmesh, "/mesh", False)
        hdf.close()

    # Setup function
    V = dolfin.FunctionSpace(brainmesh, parserargs["function_space"], parserargs["function_degree"]) 
    f = dolfin.Function(V)
    hdf5 = dolfin.HDF5File(brainmesh.mpi_comm(), parserargs["hdf5_file"], "r")
    hdf5_name = parserargs["hdf5_name"]

    if not hdf5_name.startswith("/"):
        hdf5_name = "/" + hdf5_name
    hdf5.read(f, hdf5_name)

    
    # function_to_image(function, template_image, extrapolation_value, shape, mask)
    output_volume, output_arry = function_to_image(function=f, template_image=nii_img, 
                                                   extrapolation_value=parserargs["extrapolation_value"], mask=parserargs["mask"])

    # ## Code to get a bounding box for the mesh, used to not iterate over all the voxels in the image
    # if parserargs["mask"] is None:
    #     imap = V.dofmap().index_map()
    #     num_dofs_local = (imap.local_range()[1] - imap.local_range()[0])
    #     xyz = V.tabulate_dof_coordinates()
    #     xyz = xyz.reshape((num_dofs_local, -1))
    #     image_coords = apply_affine(np.linalg.inv(vox2ras), xyz)
        
    #     lower_bounds = np.maximum(0, np.floor(image_coords.min(axis=0)).astype(int))
    #     upper_bounds = np.minimum(output_data.shape, np.ceil(image_coords.max(axis=0)).astype(int))
        
    #     all_relevant_indices = itertools.product(
    #         *(range(start, stop+1) for start, stop in zip(lower_bounds, upper_bounds))
    #     )
    #     num_voxels_in_mask = np.product(1 + upper_bounds - lower_bounds)
    #     fraction_of_image = num_voxels_in_mask / np.product(output_data.shape)
    #     print(f"Computed mesh bounding box, evaluating {fraction_of_image:.0%} of all image voxels")
    #     print(f"There are {num_voxels_in_mask} voxels in the bounding box")

    # else:


    nibabel.save(output_volume, parserargs["output"])
