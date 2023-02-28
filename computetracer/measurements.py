from dolfin import *
# from dolfin_adjoint import *
import pathlib
import nibabel
import numpy
from nibabel.affines import apply_affine
import os
from datetime import datetime


# parameters['allow_extrapolation'] = True



def get_delta_t(f1, f2, file_suffix=""):
    
    frmt = '%Y%m%d_%H%M%S'

    dates = []

    for file in [f1, f2]:

        if "/" in file or "." in file:
            raise ValueError("Expecting string of format " + frmt)

        date_time = file.replace(file_suffix, "")
    
        dates.append(datetime.strptime(date_time, frmt))

    difference = dates[1] - dates[0]
    time = difference.days * 3600 * 24 + difference.seconds

    return time


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

        print("No filter used, setting", numpy.where(numpy.isnan(voxeldata[i, j, k]), 1, 0).sum(), "/", i.size, " nan voxels to 0")
        voxeldata[i, j, k] = numpy.where(numpy.isnan(voxeldata[i, j, k]), 0, voxeldata[i, j, k])
        c_data.vector()[:] = voxeldata[i, j, k]

    return c_data


class MRI_Measurements():

    def __init__(self, datapath, function_space, Tmax = 2.5 * 24 * 3600, data_filter=None, file_suffix=""):
        
        self.function_space = function_space
        self.data_filter = data_filter
        self.datapath = datapath
        self.file_suffix = file_suffix

        files = sorted([self.datapath / x for x in os.listdir(self.datapath) if "template" not in x])

        self.measurements = {}
        self.time_filename_mapping = {}

        for filename in files:

            dt = get_delta_t(files[0].stem, filename.stem, file_suffix=file_suffix)

            if dt > Tmax:
                print("Omit image", filename)
                continue
            
            key = self.timeformat(dt)

            self.time_filename_mapping[key] = filename.name

            print("Added ", key, "=", self.time_filename_mapping[key], "to time_filename_mapping")

            self.measurements[key] = read_image(str(filename), functionspace=self.function_space, data_filter=self.data_filter)

            self.measurements[key].rename("data", "data")

    def tolist(self):
        return [image_function for _, image_function in self.measurements.items()]

    def timeformat(self, t):
        return format(t, ".2f")

    def get_measurement(self, t):

        t = self.timeformat(t)
        return self.measurements[t]           
            

    def measurement_points(self):
        
        return sorted(list(map(lambda x: float(x), list(self.time_filename_mapping.keys()))))

    def dump_pvd(self, vtkpath):
        """Dump all data snapshots to a pvd file

        Args:
            vtkpath (str): Path to export to
        """

        
        vtkfile = File(vtkpath)
        
        for t in self.measurement_points():
            u = self.get_measurement(t)
            vtkfile << u