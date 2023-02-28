from fenics import *
# from fenics_adjoint import *
import pathlib
import os
import numpy as np
import json
from measurements import MRI_Measurements
from tqdm import tqdm
import argparse

class Model(object):
    def __init__(self, dt, V, mris, outfolder: pathlib.Path, mean_diffusivity, diffusion_tensor, dx_SD):
        """
        :param mesh_config: dictionary which contains ds, dx
        :param V: FunctionSpace for state variable
        :param delta: diffusion coefficient
        :param params: dictionary which contains time step size self.dt, regularization parameter alpha
        :param data: class
        """

        self.mean_diffusivity = D_scale * mean_diffusivity
        self.dx_SD = dx_SD
        self.diffusion_tensor = D_scale * diffusion_tensor
        self.V = V

        self.ds = Measure('ds')(domain=self.V.mesh())
        #  ds # mesh_config["ds"]
        self.dx = Measure('dx')(domain=self.V.mesh()) # mesh_config["dx"]
        self.data = mris.tolist()
        self.times = mris.measurement_points()

        self.t = 0
        self.dt = dt
        self.T = max(self.times)

        print("Simulating for", format(self.T / 3600, ".0f"), "hours",
                "(", format(self.T / 1, ".0f"), "seconds)"
                " with time step size", format(self.dt / 60, ".0f"), "minutes")


        self.next_image_index = 0
        
        self.L2_error = 0.0
        self.datanorm = 0.0

        # self.linear_solver_args = ("gmres", "amg")
        self.state_at_measurement_points = []
        self.outfolder = outfolder
        self.brain_volume = assemble(Constant(1) * self.dx)
        self.brain_surface_area = assemble(1 * self.ds)
        self.gray_volume = assemble(1 * dx_SD(1))
        self.white_matter_volume = assemble(1 * dx_SD(2))
        self.brain_stem_volume = assemble(1 * dx_SD(3))

        self.concfile = self.outfolder / 'simulated_concentration.txt'
        with open(self.concfile, 'w') as file:
            file.write('t avg avgds gray white brainstem ')
            file.write('\n')

        self.image_counter = 1
        self.image_counter_prev = 0

        self.linear_interpolation = Function(V)
     
        self.simulated_tracer = []
        
        times = [0]
        t = 0
        while t + self.dt <= self.T:
            t += self.dt
            times.append(t)
            

        times = np.array(times)

        self.checkpoints = []
        for mri_time in self.times:
            self.checkpoints.append(np.round(times[np.argmin(np.abs(times - mri_time))], 0).item())





    def save_predictions(self,):


        pvdfile = File(str(self.outfolder / "simulation.pvd"))
        
        for sim in self.simulated_tracer:
            sim.rename("simulation", "simulation")
            pvdfile << sim

    
        hdf5file = HDF5File(self.V.mesh().mpi_comm(), str(self.outfolder / "simulation.hdf"), "w")
        hdf5file.write(self.V.mesh(), "mesh")
        
        for idx, sim in enumerate(self.simulated_tracer):
            sim.rename("simulation", "simulation")
            hdf5file.write(sim, "simulation" + str(idx))

        hdf5file.close()

        checkpoints = {"simulation": self.checkpoints,
                        "data": self.times}

        with open(self.outfolder / "checkpoints.json", 'w') as outfile:
            json.dump(checkpoints, outfile, sort_keys=True, indent=4)


    def advance_time(self, current_state):
        """
        update of time variables and boundary conditions
        """
        self.t += self.dt     # update time-step

        if self.t > self.times[self.image_counter]:

            self.image_counter_prev = self.image_counter
            
            while self.t > self.times[self.image_counter]:
                
                self.image_counter += 1

        # if not self.next_image_index == len(self.times) and 
        if np.round(self.t, 0) in self.checkpoints:
            
            self.simulated_tracer.append(current_state.copy())
        
            L2_error = assemble((current_state - self.data[self.image_counter]) ** 2 * self.dx) 

            datanorm = assemble((self.data[self.image_counter]) ** 2 * self.dx)
            
            self.datanorm += datanorm
  
            self.L2_error += L2_error


    def boundary_condition(self):
        """
        Linear interpolation c_1 + (c2 - c1) / (t2 - t1) * (t - t1) of image data as boundary condition
        """
        
        def boundary(x, on_boundary):
            return on_boundary
                        
        self.linear_interpolation.vector()[:] = self.data[self.image_counter_prev].vector()[:]
        
        finite_difference = (self.data[self.image_counter].vector()[:] - self.data[self.image_counter_prev].vector()[:])
        finite_difference /= (self.times[self.image_counter] - self.times[self.image_counter_prev])

        self.linear_interpolation.vector()[:] +=  finite_difference * (self.t - self.times[self.image_counter_prev])

        return DirichletBC(self.V, self.linear_interpolation, boundary)

    def return_value(self):
        return self.L2_error / self.datanorm


    def store_values(self, fun):

        with open(self.concfile, 'a') as file:

            file.write('%g ' % self.t)
            average = assemble(fun * self.dx) / self.brain_volume
            average_ds = assemble(fun * self.ds) / self.brain_surface_area
            gray_avg = assemble(fun * self.dx_SD(1)) / self.gray_volume
            white_avg = assemble(fun * self.dx_SD(2)) / self.white_matter_volume
            brain_stem_avg = assemble(fun * self.dx_SD(3)) / self.brain_stem_volume

            file.write('%g ' % average)
            file.write('%g ' % average_ds)
            file.write('%g ' % gray_avg)
            file.write('%g ' % white_avg)
            file.write('%g ' % brain_stem_avg)
            file.write('\n')


    def forward(self, alpha=1, r=None):

        # Define trial and test-functions
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        # Solution at current and previous time
        u_prev = self.data[0] # Function(self.V)
        u_next = Function(self.V)

        self.store_values(fun=u_prev)

        iter_k = 0

        def diffusion(fun):
            term = self.mean_diffusivity * inner(grad(fun), grad(v)) * self.dx_SD(1)
            term += inner(dot(self.diffusion_tensor, grad(fun)), grad(v)) * self.dx_SD(2)
            term += self.mean_diffusivity * inner(grad(fun), grad(v)) * self.dx_SD(3)

            return term

        def reaction(fun):
            return r * inner(fun, v) * dx

        a = inner(u, v) * dx
        # NOTE: if you change this to explicit/Crank-Nicolson, change L in the loop!
        # implicit time stepping:
        a += self.dt * alpha * diffusion(fun=u)
        
        if r is not None:
            a += self.dt * reaction(fun=u)

        A = assemble(a)

        # breakpoint()
        
        # solver = LUSolver()
        # solver.set_operator(A)
        solver = PETScKrylovSolver('gmres', 'amg')
        solver.set_operators(A, A)

        progress = tqdm(total=int(self.T / self.dt))

        while self.t + self.dt / 1 <= self.T:
            
            iter_k += 1

            u_prev.assign(u_next)

            self.advance_time(u_prev)
            
            # get current BC:
            bc = self.boundary_condition()
            # continue
            bc.apply(A)

            # Assemble RHS and apply DirichletBC
            rhs = u_prev * v * dx
            b = assemble(rhs)
            bc.apply(b)

            # Solve A* u_current = b
            solver.solve(u_next.vector(), b)

            # solve(A, U.vector(), b, 'lu')
            
            self.store_values(fun=u_next)


            progress.update(1)


        # return self.return_value()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/basti/Dropbox (UiO)/Sleep/228/FIGURES_CONC_LUT/")
    parser.add_argument("--mesh", default="/home/basti/Dropbox (UiO)/Sleep/228/mesh/parenchyma32_with_DTI.h5")
    parser.add_argument("--outfolder", default="./simulation_outputs/")
    parserargs = vars(parser.parse_args())

    mean_diffusivity_water = 3e-3
    mean_diffusivity_gadovist = 3.8e-4 # (taken from Gd-DPTA)
    scale_diffusion_gad = mean_diffusivity_gadovist / mean_diffusivity_water
    D_scale = Constant(scale_diffusion_gad) 

    outfolder = pathlib.Path(parserargs["outfolder"])

    os.makedirs(outfolder, exist_ok=True)

    datapath = pathlib.Path(parserargs["data"])# )
    
    meshpath =  parserargs["mesh"] # ""

    assert os.path.isfile(meshpath)

    brainmesh = Mesh()
    hdf = HDF5File(brainmesh.mpi_comm(), meshpath, "r")
    hdf.read(brainmesh, "/mesh", False)
    subdomains = MeshFunction("size_t", brainmesh, brainmesh.topology().dim())
    hdf.read(subdomains, "/subdomains")

    # GRAY = 1. WHITE = 2. BRAIN STEM = 3.
    dx_SD = Measure('dx')(domain=brainmesh, subdomain_data=subdomains)

    V = FunctionSpace(brainmesh, "CG", 1)
    diffusiontensor_Space = TensorFunctionSpace(brainmesh, 'DG', 0)
    mean_diffusivitySpace = FunctionSpace(brainmesh, 'DG', 0)
    mean_diffusivity = Function(mean_diffusivitySpace)

    diffusion_tensor = Function(diffusiontensor_Space)
    hdf.read(mean_diffusivity, '/MD')
    hdf.read(diffusion_tensor, '/DTI')

    tmax = 3600 * 60


    mris = MRI_Measurements(datapath=datapath, function_space=V,    Tmax=tmax, file_suffix="_concentration")


    mris.dump_pvd(vtkpath=str(outfolder / "data.pvd"))




    diffusion_model = Model(dt=1800, V=V, mris=mris, dx_SD=dx_SD, outfolder=outfolder,
                            diffusion_tensor=diffusion_tensor, mean_diffusivity=mean_diffusivity)

    diffusion_model.forward()

    mismatch = diffusion_model.L2_error

    print("Simulation done, mismatch=", format(mismatch, ".2f"))

    diffusion_model.save_predictions()