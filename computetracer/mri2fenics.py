from dolfin import *
# from dolfin_adjoint import *
import nibabel
import numpy
from nibabel.affines import apply_affine
import argparse
import pathlib

def read_image(filename, functionspace, data_filter=None):
    
    print("Loading", filename)
    
    mri_volume = nibabel.load(filename)
    voxeldata = mri_volume.get_fdata()

    c_data = Function(functionspace)
    ras2vox_tkr_inv = numpy.linalg.inv(mri_volume.header.get_vox2ras_tkr())

    xyz = functionspace.tabulate_dof_coordinates()
    ijk = apply_affine(ras2vox_tkr_inv, xyz).T
    i, j, k = numpy.rint(ijk).astype("int")
    
    if data_filter is not None:
        voxeldata = data_filter(voxeldata, ijk, i, j, k)
        c_data.vector()[:] = voxeldata[i, j, k]
    else:
        if numpy.where(numpy.isnan(voxeldata[i, j, k]), 1,0).sum() > 0:
            print("No filter used, setting", numpy.where(numpy.isnan(voxeldata[i, j, k]), 1, 0).sum(), "/", i.size, " nan voxels to 0")
            voxeldata[i, j, k] = numpy.where(numpy.isnan(voxeldata[i, j, k]), 0, voxeldata[i, j, k])
        if numpy.where(voxeldata[i, j, k] < 0, 1,0).sum() > 0:
            print("No filter used, setting", numpy.where(voxeldata[i, j, k] < 0, 1, 0).sum(), "/", i.size, " voxels in mesh have value < 0")

        c_data.vector()[:] = voxeldata[i, j, k]

    return c_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, help="Path to image in mgz format.")
    parser.add_argument("-m", "--mesh", help="Path to mesh.")
    parser.add_argument("--functionspace", type=str, default="CG")
    parser.add_argument("--outputname", type=str, default=None)
    parser.add_argument("--functiondegree", type=int, default=1)

    parserargs = vars(parser.parse_args())

    meshfile = parserargs["mesh"]

    if meshfile.endswith(".xml"):
        brainmesh = Mesh(meshfile)
    else:
        brainmesh = Mesh()
        hdf = HDF5File(brainmesh.mpi_comm(), meshfile, "r")
        hdf.read(brainmesh, "/mesh", False)

    V = FunctionSpace(brainmesh, parserargs["functionspace"], parserargs["functiondegree"])

    c_data_fenics = read_image(filename=parserargs["data"], functionspace=V, data_filter=None)

    if parserargs["outputname"] is None:
        outputname = parserargs["data"]
        
        # outputname = pathlib.Path(parserargs["data"]).name
    else:
        outputname = parserargs["outputname"]
    
    outputname = pathlib.Path(outputname)
    outputname = str(outputname.with_suffix(''))

    File(outputname + ".pvd") << c_data_fenics

    hdf5file = HDF5File(V.mesh().mpi_comm(), outputname + ".hdf", "w")
    hdf5file.write(V.mesh(), "mesh")
    hdf5file.write(c_data_fenics, "c")
        