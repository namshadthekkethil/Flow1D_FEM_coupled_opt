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

#ifndef included_Admittance
#define included_Admittance

// C++ include files that we need
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

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
#include "libmesh/edge_edge4.h"
#include "libmesh/point.h"

// The nonlinear solver and system we will be using
#include "libmesh/linear_implicit_system.h"
#include "libmesh/nonlinear_implicit_system.h"
#include "libmesh/nonlinear_solver.h"

#include <libmesh/mesh_modification.h>
#include <libmesh/mesh_tools.h>

using namespace libMesh;
using namespace std;

class Admittance
{
public:
  Admittance();
  ~Admittance();

  static int alphamax, betamax, count, endbranch;

  static void compute_admittance(int n, int m, double omega_k, double fa1, double fa2, double fa3, double fv1,
                                 double fv2, double fv3, double rho, double mu, double r_root, double r_min, double Lr, double q, double g,
                                 double asym, double expo, double lrr_A, double lrr_V, DenseMatrix<double> &Y, DenseMatrix<double> &Y_i);

  static void series(DenseMatrix<double> &YA, DenseMatrix<double> &YA_i,
                     DenseMatrix<double> &YB, DenseMatrix<double> &YB_i,
                     DenseMatrix<double> &Y, DenseMatrix<double> &Y_i);

  static void compute_impedance(int tmstps, double Per, double rho, double mu, double r_root, double r_min,
                                DenseVector<double> &y11, DenseVector<double> &y12, DenseVector<double> &y21, DenseVector<double> &y22,
                                double Lr, double q, double g, double fa1, double fa2, double fa3, double fv1, double fv2, double fv3,
                                double asym, double expo, double lrrA, double lrrV);

  static void fftshift(vector<complex<double>> &data);

  static void bitreverse(vector<complex<double>> &X);

  static void IFFT(vector<complex<double>> &Zhat);
};

#endif
