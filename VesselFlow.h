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

#ifndef included_VesselFlow
#define included_VesselFlow


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

#include "Admittance.h"

using namespace libMesh;
using namespace std;

typedef struct Vess {
  double x1, y1, z1, x2, y2, z2, bl, br, l, r, r1, r2, p1, p2, Q;
  int p, dl, dr, nt, n1, n2, e_id;
  int i1, i2, i3, ter, i_in, inside,ter_num;
  double beta, pext;
  double A1,A2,Q1,Q2,Q3;
  double A1Old,A2Old,Q1Old,Q2Old,Q3Old;
  double A1n1,A2n1,Q1n1,Q2n1,Q3n1;
} Vess;

class VesselFlow {
public:
  VesselFlow();
  ~VesselFlow();

  static Vess vess_i;
  static vector<Vess> vessels, vessels_in;
  static int idx, ide, ivess, vess_start, trans_soln,restart,restart_part_vein;
  static double L_v, mu_v, nu_v, rho_v, alpha_0, beta_0, alpha_v, gamma_v, p_0,
      p_out, c_v,p_in_const,p_out_const,p_ext_const;
  static double ttime, dt_v, dt,ttime_dim;
  static double alpha_t;
  static int time_integ, time_itr, inlet_bc, outlet_bc,time_itr_per;
  static string vessel_file_name, restart_file_name, partvein_file_name;
  static int beta_type, pin_type, pout_type, interface_type, 
      wave_type,pext_type,venous_flow, st_tree;
  static double t_load, time_per;
  static double gamma_perm;
  static DenseVector<vector<double>> pArt,pVein;
  static vector<double> qArt,qVein;
  static vector<int> termNum;
  static int N_period,N_total;
  static DenseVector<DenseVector<double>> pLt,pRt;
  static double qArtTotal,qVeinTotal,pArtTotal,pVeinTotal,pInCur,pOutCur,qInCur,qOutCur;
  static double p_diastole,p_systole;

  static DenseVector<double> y11, y12, y21, y22;


  static void read_vessel_data(int rank, int np, LibMeshInit &init);
  static void read_input();
  static void create_mesh(Mesh &mesh);
  static void create_mesh_3(Mesh &mesh);
  static void initialise_1Dflow(Mesh &mesh, int rank, int np,
                                LibMeshInit &init);
  static void add_element_node(Mesh &mesh, int i);
  static void add_element_node_3(Mesh &mesh, int i);
  static void define_systems(EquationSystems &es);
  static void solve_flow(EquationSystems &es);
  static void compute_jacobian(const NumericVector<Number> &,
                               SparseMatrix<Number> &J,
                               NonlinearImplicitSystem &system);

  static void compute_residual(const NumericVector<Number> &X,
                               NumericVector<Number> &R,
                               NonlinearImplicitSystem &system);
  static void initialise_area(EquationSystems &es);
  static void old_new(EquationSystems &es);
  static void n1_old(EquationSystems &es);
  static void writeFlowData(EquationSystems &es);
  static void update_vessels();
  static void update_beta();
  static void add_vessels(int i);
  static void writeFlowDataTime(EquationSystems &es, int it, int rank);
  static void writeFlowDataBound(EquationSystems &es, int it, int rank);
  static double pofA(double A_cur);
  static double dpofA(double A_cur);
  static double d2pofA(double A_cur);
  static double AInlet(double time_v);

  static double AOutlet(double time_v, int n);
  static void writeUpdatedVessels();
  static double source_q(double Q_cur, double p_cur);
  static double dsourcedQ(double Q_cur, double p_cur);
  static double dsourcedp(double Q_cur, double p_cur);
  static double betaV(int n, double r_cur);
  static double betaPr(int n, double r_cur);
  static double timeDer(double Q_cur, double Q_old, double Q_n1);
  static double DtimeDer();
  static double PInlet(double time_v);
  static double POutlet(double time_v);
  static double PDrain(double time_v);
  static double PExt();
  static void compute_pext(double time_v);

  static void updateImpedance();

  static void initialise_partvein(int rank, int np, LibMeshInit &init);
  static void update_partvein(EquationSystems &es, int rank);
  static void initialise_flow_data(EquationSystems &es);
  static void read_vessel_restart(int rank, int np, LibMeshInit &init);
  static void write_restart_data(EquationSystems &es, int it, int rank);
  static void update_qartvein(int rank);
  static void update_pqbound(EquationSystems & es,int rank);
  static double QInlet();
};

#endif
