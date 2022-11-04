// The libMesh Finite Element Library.
// Copyright (C) 2002-2017 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

// <h1> Systems Example 7 - Large deformation elasticity (St. Venant-Kirchoff
// material) </h1> \author Lorenzo Zanon \author David Knezevic \date 2014
//
// In this example, we consider an elastic cantilever beam modeled as a St.
// Venant-Kirchoff material (which is an extension of the linear elastic
// material model to the nonlinear regime). The implementation presented here
// uses NonlinearImplicitSystem.
//
// We formulate the PDE on the reference geometry (\Omega) as opposed to the
// deformed geometry (\Omega^deformed). As a result (e.g. see Ciarlet's 3D
// elasticity book, Theorem 2.6-2) the PDE is given as follows:
//
//     \int_\Omega F_im Sigma_mj v_i,j = \int_\Omega f_i v_i + \int_\Gamma g_i
//     v_i ds
//
// where:
//  * F is the deformation gradient, F = I + du/dx (x here refers to reference
//  coordinates).
//  * Sigma is the second Piola-Kirchoff stress, which for the St. Venant
//  Kirchoff model is
//    given by Sigma_ij = C_ijkl E_kl, where E_kl is the strain,
//    E_kl = 0.5 * (u_k,l + u_l,k + u_m,k u_m,l).
//  * f is a body load.
//  * g is a surface traction on the surface \Gamma.
//
// In this example we only consider a body load (e.g. gravity), hence we set g =
// 0.

// C++ include files that we need
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <assert.h>

// Various include files needed for the mesh & solver functionality.
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/dirichlet_boundaries.h"
#include "libmesh/dof_map.h"
#include "libmesh/elem.h"
#include "libmesh/equation_systems.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/fe.h"
#include "libmesh/getpot.h"
#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/mesh_refinement.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/string_to_enum.h"
#include "libmesh/zero_function.h"

#include "libmesh/boundary_info.h"
#include "libmesh/edge_edge2.h"
#include "libmesh/edge_edge3.h"
#include "libmesh/point.h"

#include "libmesh/petsc_matrix.h"

#include "libmesh/tecplot_io.h"

#include "libmesh/linear_implicit_system.h"

// The nonlinear solver and system we will be using
#include "libmesh/nonlinear_implicit_system.h"
#include "libmesh/nonlinear_solver.h"

#include <libmesh/mesh_modification.h>
#include <libmesh/mesh_refinement.h>
#include <libmesh/mesh_tools.h>

#include "libmesh/mesh_tetgen_interface.h"

#include "VesselFlow.h"

using namespace libMesh;
using namespace std;
using namespace std::chrono;

// is it ok ok

void read_mesh(Mesh &mesh)
{
  // mesh.read("mesh1d.e", NULL);
  mesh.read("mesh1d_disc.e", NULL);
  // mesh.read("mesh1d_single.e", NULL);
  // MeshTools::Generation::build_line(mesh, 10, 0.0, 0.04, EDGE3);

  // mesh.write("out_mesh.e");
}

void run_time_step(EquationSystems &es, Mesh &mesh, int rank,
                   LibMeshInit &init)
{
  // ExodusII_IO exo_io(mesh);
  // exo_io.write_timestep("out_time.e", es, 1, 0);
  VesselFlow::time_itr = 0;

  VesselFlow::writeFlowDataTime(es, 0, rank);

  VesselFlow::ttime = 0.0;
  VesselFlow::ttime_dim = 0.0;

  VesselFlow::update_qartvein(rank);

  int count_per = 0;
  for (unsigned int count = 1; count < VesselFlow::N_total; count++)
  {
    VesselFlow::time_itr = count;
    VesselFlow::ttime = count * VesselFlow::dt_v;

    count_per = fmod(count, VesselFlow::N_period);
    VesselFlow::time_itr_per = count_per;
    VesselFlow::ttime_dim = count_per * VesselFlow::dt;

    VesselFlow::n1_old(es);
    VesselFlow::old_new(es);

    auto start = high_resolution_clock::now();

    VesselFlow::solve_flow(es);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "time for solving=" << duration.count() << endl;
    if (VesselFlow::venous_flow == 1)
    {
      auto start = high_resolution_clock::now();

      VesselFlow::update_partvein(es, rank);

      auto stop = high_resolution_clock::now();
      auto duration = duration_cast<microseconds>(stop - start);
      cout << "time taken for part=" << duration.count() << endl;

      if (VesselFlow::st_tree == 1)
      {
        auto start = high_resolution_clock::now();
        VesselFlow::update_qartvein(rank);
        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(stop - start);
        cout << "time taken for qart=" << duration.count() << endl;
      }
    }

    cout << "count=" << count << " count_per=" << count_per << " t=" << VesselFlow::ttime << " t_per=" << count_per * VesselFlow::dt_v << " tdim=" << count * VesselFlow::dt << " tdim_per=" << VesselFlow::ttime_dim << endl;

    if (((count + 1) % 500 == 0))
    {
      VesselFlow::writeFlowDataTime(es, count, rank);
    }

    if (((count + 1) % 10 == 0))
    {

      VesselFlow::writeFlowDataBound(es, count, rank);
    }

    if ((count + 1) % VesselFlow::N_period == 0)
      VesselFlow::write_restart_data(es, VesselFlow::time_itr, rank);
  }
  cout << "before writing restart file" << endl;
  VesselFlow::write_restart_data(es, VesselFlow::time_itr, rank);
  cout << "after writing restart file" << endl;

  MPI_Barrier(init.comm().get());
}

void solve_systems(LibMeshInit &init, int rank, int np)
{

  Mesh mesh(init.comm());
  // Mesh mesh1(init.comm());

  // read_mesh(mesh);
  VesselFlow::initialise_1Dflow(mesh, rank, np, init);
  mesh.print_info();

  EquationSystems equation_systems(mesh);
  VesselFlow::define_systems(equation_systems);

  // PetscMatrix<Number> Matrix(init.comm());
  //
  // MatSetOption(Matrix.mat(), MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

  equation_systems.init();

  // equation_systems.parameters.set<Real>(
  //     "nonlinear solver absolute residual tolerance") = 1.0e-12;
  // equation_systems.parameters.set<Real>(
  //     "nonlinear solver relative residual tolerance") = 1.0e-12;
  equation_systems.parameters.set<unsigned int>(
      "nonlinear solver maximum iterations") = 100;

  // VesselFlow::initialise_area(equation_systems);
  VesselFlow::initialise_flow_data(equation_systems);

  LinearImplicitSystem &flow_system =
      equation_systems.get_system<LinearImplicitSystem>("flowSystem");

  // Matrix.attach_dof_map(flow_system.get_dof_map());
  // Matrix.init();
  // flow_system.matrix = &Matrix;

  // dynamic_cast<PetscMatrix<Number> *>(flow_system.matrix);
  MatSetOption((dynamic_cast<PetscMatrix<Number> *>(flow_system.matrix))->mat(),
               MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

  run_time_step(equation_systems, mesh, rank, init);

  VesselFlow::writeFlowData(equation_systems);

  ExodusII_IO exo_io(mesh);

  exo_io.write_timestep("frame0.e", equation_systems, 1, 0);

  TecplotIO tec_io(mesh);
  tec_io.write_equation_systems("frame0.tec", equation_systems);
}

int main(int argc, char **argv)
{
  LibMeshInit init(argc, argv);

  // This example requires the PETSc nonlinear solvers
  // libmesh_example_requires(libMesh::default_solver_package() ==
  // PETSC_SOLVERS,
  //                          "--enable-petsc");

  // We use a 3D domain.
  libmesh_example_requires(LIBMESH_DIM > 2,
                           "--disable-1D-only --disable-2D-only");

  int rank, np;

  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  auto start = high_resolution_clock::now();
  solve_systems(init, rank, np);
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);

  cout << "Time taken by function: " << duration.count() << " microseconds"
       << endl;

  return 0;
}
