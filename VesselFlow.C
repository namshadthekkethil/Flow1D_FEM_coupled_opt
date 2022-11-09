/******* 03/11/2022 In the optimised version intervesstype=0 is removed (we don't use the divergence theorem for the boundary terms*********/
/**********03/11/2022 eigen_bound removed*/
/*****03/11/2022 flow type removed. For poisuille flow (flow_type = 0) use source_codes/Flow1D_FEM_coupled ******/
#include "VesselFlow.h"

using namespace libMesh;
using namespace std;

Vess VesselFlow::vess_i;
vector<Vess> VesselFlow::vessels, VesselFlow::vessels_in;
int VesselFlow::idx, VesselFlow::ide, VesselFlow::ivess, VesselFlow::vess_start,
    VesselFlow::trans_soln, VesselFlow::restart, VesselFlow::restart_part_vein;

double VesselFlow::L_v, VesselFlow::mu_v, VesselFlow::nu_v, VesselFlow::rho_v,
    VesselFlow::alpha_v, VesselFlow::alpha_0, VesselFlow::beta_0,
    VesselFlow::gamma_v, VesselFlow::p_0, VesselFlow::c_v, VesselFlow::p_out,
    VesselFlow::p_in_const, VesselFlow::p_out_const, VesselFlow::p_ext_const;
double VesselFlow::ttime, VesselFlow::dt_v, VesselFlow::dt, VesselFlow::ttime_dim;
double VesselFlow::alpha_t;
int VesselFlow::time_integ,
    VesselFlow::time_itr, VesselFlow::inlet_bc, VesselFlow::outlet_bc, VesselFlow::time_itr_per;

string VesselFlow::vessel_file_name, VesselFlow::restart_file_name, VesselFlow::partvein_file_name;
int VesselFlow::beta_type, VesselFlow::pin_type, VesselFlow::pout_type,
    VesselFlow::interface_type,
    VesselFlow::wave_type, VesselFlow::pext_type, VesselFlow::venous_flow, VesselFlow::st_tree;

DenseVector<DenseVector<double>> VesselFlow::pLt, VesselFlow::pRt;

double VesselFlow::t_load, VesselFlow::time_per;
double VesselFlow::gamma_perm;

DenseVector<vector<double>> VesselFlow::pArt, VesselFlow::pVein;

vector<double> VesselFlow::qArt, VesselFlow::qVein;

vector<int> VesselFlow::termNum;

int VesselFlow::N_period, VesselFlow::N_total;

DenseVector<double> VesselFlow::y11, VesselFlow::y12, VesselFlow::y21, VesselFlow::y22;

double VesselFlow::qArtTotal, VesselFlow::qVeinTotal, VesselFlow::pArtTotal, VesselFlow::pVeinTotal,
    VesselFlow::pInCur, VesselFlow::pOutCur, VesselFlow::qInCur, VesselFlow::qOutCur;

double VesselFlow::p_diastole, VesselFlow::p_systole;

VesselFlow::VesselFlow() {}

VesselFlow::~VesselFlow() {}

void VesselFlow::read_input()
{
    // GetPot infile("input_1D.in");
    // GetPot infile("input_1D_Quart.in");
    // GetPot infile("input_1D_Shervin.in");
    // GetPot infile("input_1D_Lee_2008.in");
    GetPot infile("input_1D_Lee_LV.in");

    trans_soln = infile("trans_soln", 1);
    time_integ = infile("time_integ", 2);
    inlet_bc = infile("inlet_bc", 0);
    outlet_bc = infile("outlet_bc", 0);

    alpha_t = infile("alpha_t", 1.0);
    L_v = infile("L_v", 1.0);
    rho_v = infile("rho_v", 1.05e-6);
    mu_v = infile("mu_v", 3.05e-6);
    alpha_0 = infile("alpha_0", 1.25);
    beta_0 = infile("beta_0", 83.74);
    p_0 = infile("p_0", 1.0e6);
    p_out = infile("p_out", 0.0);
    c_v = infile("c_v", 1.5e4);
    dt = infile("dt", 1.0);
    beta_type = infile("beta_type", 0);
    pin_type = infile("pin_type", 0);
    pout_type = infile("pout_type", 0);
    interface_type = infile("interface_type", 0);
    wave_type = infile("wave_type", 0);

    pext_type = infile("pext_type", 0);
    venous_flow = infile("venous_flow", 0);

    vessel_file_name = infile("vessel_file", "vessels_single_mesh_data_0_1.dat");

    t_load = infile("t_load", 1.0);
    time_per = infile("time_per", 1.0);

    gamma_perm = infile("gamma_perm", 1.0e-7);

    N_period = infile("N_period", 32768);
    N_total = infile("N_total", 32768);

    restart = infile("restart", 0);
    restart_part_vein = infile("restart_part_vein", 0);

    restart_file_name = infile("restart_file_name", "vessels_single_mesh_data_0_1.dat");
    partvein_file_name = infile("partvein_file_name", "partvein_data.dat");

    p_in_const = infile("p_in_const", 0.0);
    p_out_const = infile("p_out_const", 0.0);
    p_ext_const = infile("p_ext_const", 0.0);

    p_diastole = infile("p_diastole", 13.0);
    p_systole = infile("p_systole", 120.0);

    st_tree = infile("st_tree", 0);

    p_out = 0.0;

    dt = time_per / N_period;

    double t_c = sqrt(rho_v / p_0) * L_v;
    dt_v = dt / t_c;

    // beta_0 = (4.0 / 3.0) * 100 * sqrt(M_PI) * 0.05; // for Lee 2008

    nu_v = mu_v / rho_v;
    alpha_v = alpha_0; // alpha_0;
    gamma_v = (2.0 * M_PI * alpha_0 * nu_v * sqrt(rho_v / p_0)) /
              ((alpha_0 - 1.0) * L_v);
}

void VesselFlow::read_vessel_data(int rank, int np, LibMeshInit &init)
{

    for (int rank_i = 0; rank_i < np; rank_i++)
    {
        if (rank == rank_i)
        {
            ifstream file_tree;
            file_tree.open(vessel_file_name);

            int vess_counter = 0;
            while (!file_tree.eof())
            {
                file_tree >> vess_i.x1 >> vess_i.y1 >> vess_i.z1 >> vess_i.x2 >>
                    vess_i.y2 >> vess_i.z2 >> vess_i.l >> vess_i.r1 >> vess_i.r2 >>
                    vess_i.r >> vess_i.p >> vess_i.dl >> vess_i.dr >> vess_i.inside;

                if (file_tree.eof())
                    break;

                else
                {

                    if (vess_i.dl == -10)
                        vess_i.ter = 1;
                    else if (vess_i.p == -10)
                        vess_i.ter = -1;
                    else
                        vess_i.ter = 0;

                    if (vess_i.dl != -10 && vess_i.dr != -10)
                        vess_i.ter = 2;

                    vess_i.A1 = M_PI * pow(vess_i.r1 / L_v, 2);
                    vess_i.A2 = M_PI * pow(vess_i.r2 / L_v, 2);

                    vess_i.A1Old = M_PI * pow(vess_i.r1 / L_v, 2);
                    vess_i.A2Old = M_PI * pow(vess_i.r2 / L_v, 2);

                    vess_i.A1n1 = M_PI * pow(vess_i.r1 / L_v, 2);
                    vess_i.A2n1 = M_PI * pow(vess_i.r2 / L_v, 2);

                    vess_i.Q1 = 0.0;
                    vess_i.Q2 = 0.0;
                    vess_i.Q3 = 0.0;

                    vess_i.Q1Old = 0.0;
                    vess_i.Q2Old = 0.0;
                    vess_i.Q3Old = 0.0;

                    vess_i.Q1n1 = 0.0;
                    vess_i.Q2n1 = 0.0;
                    vess_i.Q3n1 = 0.0;

                    vessels_in.push_back(vess_i);

                    if (vess_i.p == -10)
                        vess_start = vess_counter;

                    vess_counter++;
                }
            }

            for (int i = 0; i < vessels_in.size(); i++)
            {
                if (vessels_in[i].ter == 2)
                {
                    vessels_in[vessels_in[i].dl].ter = 3;
                    vessels_in[vessels_in[i].dr].ter = 3;
                }
            }

            // for (int i = 0; i < vessels_in.size(); i++) {
            //   cout << "i=" << i << " ter=" << vessels_in[i].ter << endl;
            // }

            file_tree.close();
        }
    }

    MPI_Barrier(init.comm().get());
}

void VesselFlow::update_vessels()
{

    for (int i = 0; i < vessels_in.size(); i++)
    {
        vessels.push_back(vessels_in[i]);
    }

    if (venous_flow == 1)
    {
        if (restart == 0)
        {
            for (int i = 0; i < vessels_in.size(); i++)
            {
                vessels.push_back(vessels_in[i]);

                if (vessels_in[i].dl != -10)
                    vessels[i + vessels_in.size()].dl = vessels_in[i].dl + vessels_in.size();

                if (vessels_in[i].dr != -10)
                    vessels[i + vessels_in.size()].dr = vessels_in[i].dr + vessels_in.size();

                if (vessels_in[i].p != -10)
                    vessels[i + vessels_in.size()].p = vessels_in[i].p + vessels_in.size();
            }
        }
        else if (restart == 1)
        {
            vessels_in.resize(0);
            for (int i = 0; i < vessels.size() / 2; i++)
            {
                vessels_in.push_back(vessels[i]);
            }
        }
    }

    writeUpdatedVessels();
}

void VesselFlow::update_beta()
{
    for (int i = 0; i < vessels.size(); i++)
    {
        if (wave_type == 0)
        {
            vessels[i].beta =
                c_v * c_v * 2.0 * rho_v * sqrt(M_PI * vessels[i].r * vessels[i].r);
        }

        else if (wave_type == 1)
        {
            if (beta_type == 0)
                vessels[i].beta = beta_0;
            else if (beta_type == 1)
                vessels[i].beta = beta_0 * vessels[i].r; // for Lee 2008
            else
                vessels[i].beta = beta_0;
        }
    }
}

void VesselFlow::initialise_1Dflow(Mesh &mesh, int rank, int np,
                                   LibMeshInit &init)
{
    read_input();
    if (restart == 0)
        read_vessel_data(rank, np, init);
    else if (restart == 1)
        read_vessel_restart(rank, np, init);

    cout << "READ VESSEL DATA" << endl;
    update_vessels();

    /* for (int i = 0; i < vessels.size(); i++)
    {
        cout << "updated " << vessels[i].x1 << " " << vessels[i].y1 << " "
             << vessels[i].z1 << " " << vessels[i].x2 << " " << vessels[i].y2 << " "
             << vessels[i].z2 << " " << vessels[i].l << " " << vessels[i].r << " "
             << vessels[i].p << " " << vessels[i].dl << " " << vessels[i].dr
             << endl;
    } */

    update_beta();
    cout << "alpha=" << alpha_v << " beta=" << betaV(0, vessels[0].r)
         << " gamma=" << gamma_v << endl;
    create_mesh(mesh);
    // create_mesh_3(mesh);
    // MeshTools::Generation::build_line(mesh, 1000, 0.0, 5.0, EDGE5);

    initialise_partvein(rank, np, init);
    updateImpedance();

    string name = "flow_data_inlet";
    string fileend = ".dat";
    string out_frame;
    out_frame = name + fileend;

    ofstream file_vess;
    if (rank == 0)
    {
        file_vess.open(out_frame, ios::out);
    }
}

void VesselFlow::add_vessels(int i)
{

    cout << "i=" << i << " dl=" << vessels_in[i].dl << endl;
    if (vessels_in[i].dl != -10)
    {

        if (vessels_in[i].dr == -10)
        {

            ivess++;
            vessels_in[vessels_in[i].dl].e_id = ivess;

            // vessels[ivess] = vessels_in[vessels_in[i].dl];
            vessels.push_back(vessels_in[vessels_in[i].dl]);
            vessels[ivess].p = vessels_in[i].e_id;

            vessels[vessels[ivess].p].dl = ivess;

            if (vessels_in[vessels_in[i].dl].dl == -10)
            {
                vessels[ivess].dl = -10;
                vessels[ivess].dr = -10;
            }
            else if (vessels_in[vessels_in[i].dl].dr == -10)
            {
                vessels[ivess].dl = ivess + 1;
                vessels[ivess].dr = -10;
            }

            vessels[ivess].i_in = vessels_in[i].dl;
        }

        else
        {

            ivess++;
            vessels_in[vessels_in[i].dl].e_id = ivess;

            // vessels[ivess] = vessels_in[vessels_in[i].dl];
            vessels.push_back(vessels_in[vessels_in[i].dl]);
            vessels[ivess].p = vessels_in[i].e_id;
            vessels[vessels[ivess].p].dl = ivess;
            vessels[vessels[ivess].p].dr = ivess + 1;

            vessels[vessels[ivess].p].dl = ivess;

            if (vessels_in[vessels_in[i].dl].dl == -10)
            {
                vessels[ivess].dl = -10;
                vessels[ivess].dr = -10;
            }
            else if (vessels_in[vessels_in[i].dl].dr == -10)
            {
                vessels[ivess].dl = ivess + 2;
                vessels[ivess].dr = -10;
            }

            vessels[ivess].i_in = vessels_in[i].dl;

            ivess++;

            vessels_in[vessels_in[i].dr].e_id = ivess;

            // vessels[ivess] = vessels_in[vessels_in[i].dr];
            vessels.push_back(vessels_in[vessels_in[i].dr]);
            vessels[ivess].p = vessels_in[i].e_id;

            vessels[vessels[ivess].p].dr = ivess;

            if (vessels_in[vessels_in[i].dr].dl == -10)
            {
                vessels[ivess].dl = -10;
                vessels[ivess].dr = -10;
            }
            if (vessels_in[vessels_in[i].dr].dr == -10)
            {
                vessels[ivess].dr = -10;
            }

            vessels[ivess].i_in = vessels_in[i].dr;
        }

        add_vessels(vessels_in[i].dl);

        if (vessels_in[i].dr != -10)
            add_vessels(vessels_in[i].dr);
    }
}

void VesselFlow::create_mesh(Mesh &mesh)
{

    mesh.set_mesh_dimension(1);

    Point X1(0.0, 0.0, 0.0);
    Point X2(vessels[0].l, 0.0, 0.0);
    Point X3(0.5 * vessels[0].l, 0.0, 0.0);

    idx = 0;
    mesh.add_point(X1, idx);
    vessels[0].i1 = idx;
    idx++;
    mesh.add_point(X2, idx);
    vessels[0].i2 = idx;
    idx++;
    mesh.add_point(X3, idx);
    vessels[0].i3 = idx;

    ide = 0;
    Elem *elem = mesh.add_elem(new Edge3);
    elem->set_node(0) = mesh.node_ptr(idx - 2);
    elem->set_node(1) = mesh.node_ptr(idx - 1);
    elem->set_node(2) = mesh.node_ptr(idx);

    mesh.boundary_info->add_side(elem, 0, 1000);

    vessels[0].e_id = ide;

    add_element_node(mesh, 0);

    if (venous_flow == 1)
    {

        idx++;
        mesh.add_point(X1, idx);
        vessels[vessels_in.size()].i1 = idx;
        idx++;
        mesh.add_point(X2, idx);
        vessels[vessels_in.size()].i2 = idx;
        idx++;
        mesh.add_point(X3, idx);
        vessels[vessels_in.size()].i3 = idx;

        ide++;
        Elem *elem_new = mesh.add_elem(new Edge3);
        elem_new->set_node(0) = mesh.node_ptr(idx - 2);
        elem_new->set_node(1) = mesh.node_ptr(idx - 1);
        elem_new->set_node(2) = mesh.node_ptr(idx);

        mesh.boundary_info->add_side(elem_new, 0, 1000);

        vessels[vessels_in.size()].e_id = ide;

        add_element_node(mesh, vessels_in.size());
    }

    mesh.prepare_for_use();

    MeshTools::Modification::scale(mesh, 1.0 / L_v);

    mesh.write("vessels_mesh.xda");
}

void VesselFlow::add_element_node(Mesh &mesh, int i)
{

    /*  cout << "i=" << i << " dl=" << vessels[i].dl << " dr=" << vessels[i].dr
          << endl; */
    if (vessels[i].dl != -10)
    {

        if (vessels[i].dr == -10)
        {
            Elem *elem_n = mesh.elem_ptr(i);
            Point pj;
            pj = elem_n->point(1);
            Point X2l(pj(0) + vessels[vessels[i].dl].l, 0.0, 0.0);
            Point X3l(pj(0) + 0.5 * vessels[vessels[i].dl].l, 0.0, 0.0);

            int idx_n = vessels[i].i2; // idx - 1;
            vessels[vessels[i].dl].i1 = idx_n;
            idx++;
            mesh.add_point(X2l, idx);
            vessels[vessels[i].dl].i2 = idx;
            idx++;
            mesh.add_point(X3l, idx);
            vessels[vessels[i].dl].i3 = idx;

            ide++;
            Elem *elem_l = mesh.add_elem(new Edge3);
            elem_l->set_node(0) = mesh.node_ptr(idx_n);
            elem_l->set_node(1) = mesh.node_ptr(idx - 1);
            elem_l->set_node(2) = mesh.node_ptr(idx);

            vessels[vessels[i].dl].e_id = ide;

            if (vessels[vessels[i].dl].dl == -10)
                mesh.boundary_info->add_side(elem_l, 1, 4000);
        }

        else
        {
            Elem *elem_n = mesh.elem_ptr(ide);
            Point pj;
            pj = elem_n->point(1);

            Point X1l(pj(0), 0.0, 0.0);
            Point X2l(pj(0) + vessels[vessels[i].dl].l, 0.0, 0.0);
            Point X3l(pj(0) + 0.5 * vessels[vessels[i].dl].l, 0.0, 0.0);

            idx++;
            mesh.add_point(X1l, idx);
            vessels[vessels[i].dl].i1 = idx;
            idx++;
            mesh.add_point(X2l, idx);
            vessels[vessels[i].dl].i2 = idx;
            idx++;
            mesh.add_point(X3l, idx);
            vessels[vessels[i].dl].i3 = idx;

            int ide_p = i;

            Elem *elem_p = mesh.elem_ptr(ide_p);
            mesh.boundary_info->add_side(elem_p, 1, 2000);

            ide++;
            Elem *elem_l = mesh.add_elem(new Edge3);
            elem_l->set_node(0) = mesh.node_ptr(idx - 2);
            elem_l->set_node(1) = mesh.node_ptr(idx - 1);
            elem_l->set_node(2) = mesh.node_ptr(idx);

            vessels[vessels[i].dl].e_id = ide;

            mesh.boundary_info->add_side(elem_l, 0, 3000);

            if (vessels[vessels[i].dl].dl == -10)
                mesh.boundary_info->add_side(elem_l, 1, 4000);

            Point X1r(pj(0), 0.0, 0.0);
            Point X2r(pj(0) + vessels[vessels[i].dr].l, 0.0, 0.0);
            Point X3r(pj(0) + 0.5 * vessels[vessels[i].dr].l, 0.0, 0.0);

            idx++;
            mesh.add_point(X1r, idx);
            vessels[vessels[i].dr].i1 = idx;
            idx++;
            mesh.add_point(X2r, idx);
            vessels[vessels[i].dr].i2 = idx;
            idx++;
            mesh.add_point(X3r, idx);
            vessels[vessels[i].dr].i3 = idx;

            ide++;
            Elem *elem_r = mesh.add_elem(new Edge3);
            elem_r->set_node(0) = mesh.node_ptr(idx - 2);
            elem_r->set_node(1) = mesh.node_ptr(idx - 1);
            elem_r->set_node(2) = mesh.node_ptr(idx);

            vessels[vessels[i].dr].e_id = ide;

            mesh.boundary_info->add_side(elem_r, 0, 3000);

            if (vessels[vessels[i].dr].dl == -10)
                mesh.boundary_info->add_side(elem_r, 1, 4000);
        }

        add_element_node(mesh, vessels[i].dl);

        if (vessels[i].dr != -10)
            add_element_node(mesh, vessels[i].dr);
    }
}

void VesselFlow::define_systems(EquationSystems &es)
{

    NonlinearImplicitSystem &flow_system =
        es.add_system<NonlinearImplicitSystem>("flowSystem");
    flow_system.add_variable("QVar", SECOND, LAGRANGE);
    flow_system.add_variable("pVar", FIRST, LAGRANGE);

    System &flow_system_old = es.add_system<System>("flowSystemOld");
    flow_system_old.add_variable("QOldVar", SECOND, LAGRANGE);
    flow_system_old.add_variable("pOldVar", FIRST, LAGRANGE);

    System &flow_system_n1 = es.add_system<System>("flowSystemn1");
    flow_system_n1.add_variable("Qn1Var", SECOND, LAGRANGE);
    flow_system_n1.add_variable("pn1Var", FIRST, LAGRANGE);

    auto &nl_solver = *flow_system.nonlinear_solver;
    nl_solver.residual = compute_residual;
    nl_solver.jacobian = compute_jacobian;

    // LinearImplicitSystem &flow_system =
    //     es.add_system<LinearImplicitSystem>("flowSystem");
    // flow_system.add_variable("QVar", SECOND, LAGRANGE);
    // flow_system.add_variable("pVar", FIRST, LAGRANGE);
    //
    // System &flow_system_old = es.add_system<System>("flowSystemOld");
    // flow_system_old.add_variable("QOldVar", SECOND, LAGRANGE);
    // flow_system_old.add_variable("pOldVar", FIRST, LAGRANGE);
    //
    // flow_system.attach_assemble_function(assemble_flow_mod);
}

void VesselFlow::solve_flow(EquationSystems &es)
{
    NonlinearImplicitSystem &flow_system =
        es.add_system<NonlinearImplicitSystem>("flowSystem");

    // LinearImplicitSystem &flow_system =
    //     es.add_system<LinearImplicitSystem>("flowSystem");

    double t_c = sqrt(rho_v / p_0) * L_v;

    cout << "A_in=" << AInlet(ttime)
         << " A_out=" << AOutlet(ttime, vessels.size() - 1) << endl;
    cout << "time_v=" << ttime << " time=" << ttime * t_c
         << " pin=" << PInlet(ttime) << endl;

    compute_pext(ttime);

    flow_system.solve();
}

void VesselFlow::compute_jacobian(const NumericVector<Number> &,
                                  SparseMatrix<Number> &J,
                                  NonlinearImplicitSystem &system)
{
    libmesh_assert_equal_to(system_name, "flowSystem");

    // Get a constant reference to the mesh object.
    const MeshBase &mesh = system.get_mesh();

    // The dimension that we are running
    const unsigned int dim = mesh.mesh_dimension();

    // Numeric ids corresponding to each variable in the system
    const unsigned int u_var = system.variable_number("QVar");
    const unsigned int p_var = system.variable_number("pVar");

    auto &es = system.get_equation_systems();

    System &system_old = es.get_system<System>("flowSystemOld");
    const DofMap &dof_map_old = system_old.get_dof_map();
    std::vector<dof_id_type> dof_indices_u_old;
    std::vector<dof_id_type> dof_indices_p_old;

    System &system_n1 = es.get_system<System>("flowSystemn1");
    const DofMap &dof_map_n1 = system_n1.get_dof_map();
    std::vector<dof_id_type> dof_indices_u_n1;
    std::vector<dof_id_type> dof_indices_p_n1;

    NumericVector<double> &flow_data = *(system.current_local_solution);
    flow_data.close();

    vector<double> flow_vec;
    flow_data.localize(flow_vec);

    // Get the Finite Element type for "u".  Note this will be
    // the same as the type for "v".
    FEType fe_vel_type = system.variable_type(u_var);
    std::unique_ptr<FEBase> fe_vel(FEBase::build(dim, fe_vel_type));
    QGauss qrule(dim, fe_vel_type.default_quadrature_order());
    fe_vel->attach_quadrature_rule(&qrule);

    // Get the Finite Element type for "p".
    FEType fe_pres_type = system.variable_type(p_var);
    std::unique_ptr<FEBase> fe_pres(FEBase::build(dim, fe_pres_type));
    fe_pres->attach_quadrature_rule(&qrule);

    const std::vector<Real> &JxW = fe_vel->get_JxW();

    const std::vector<std::vector<Real>> &phi = fe_vel->get_phi();
    const std::vector<std::vector<RealGradient>> &dphi = fe_vel->get_dphi();

    const std::vector<std::vector<Real>> &psi = fe_pres->get_phi();
    const std::vector<std::vector<RealGradient>> &dpsi = fe_pres->get_dphi();

    UniquePtr<FEBase> fe_face(FEBase::build(dim, fe_vel_type));
    QGauss qface(dim - 1, fe_vel_type.default_quadrature_order());
    fe_face->attach_quadrature_rule(&qface);

    UniquePtr<FEBase> fe_face_n(FEBase::build(dim, fe_vel_type));
    fe_face_n->attach_quadrature_rule(&qface);

    UniquePtr<FEBase> fe_face_p(FEBase::build(dim, fe_pres_type));
    fe_face_p->attach_quadrature_rule(&qface);

    const std::vector<std::vector<RealGradient>> &dphi_face = fe_face->get_dphi();

    const std::vector<std::vector<RealGradient>> &dphi_face_p =
        fe_face_p->get_dphi();

    const DofMap &dof_map = system.get_dof_map();

    DenseMatrix<Number> Ke;

    DenseSubMatrix<Number> Kuu(Ke), Kup(Ke), Kpu(Ke), Kpp(Ke);

    std::vector<dof_id_type> dof_indices;
    std::vector<dof_id_type> dof_indices_u;
    std::vector<dof_id_type> dof_indices_p;

    std::vector<dof_id_type> dof_indices_neighbor_u;
    std::vector<dof_id_type> dof_indices_neighbor_p;
    std::vector<dof_id_type> dof_indices_neighbor_1_u;
    std::vector<dof_id_type> dof_indices_neighbor_1_p;
    std::vector<dof_id_type> dof_indices_neighbor_2_u;
    std::vector<dof_id_type> dof_indices_neighbor_2_p;

    std::vector<dof_id_type> dof_indices_neighbor_out_u;
    std::vector<dof_id_type> dof_indices_neighbor_out_p;

    J.zero();

    MeshBase::const_element_iterator el = mesh.active_local_elements_begin();
    const MeshBase::const_element_iterator end_el =
        mesh.active_local_elements_end();

    int elem_count = 0;
    for (; el != end_el; ++el)
    {
        const Elem *elem = *el;

        DenseMatrix<Number> Kpun1, Kpun2, Kuun1, Kuun2, Kupn, Kuun, Kppn, Kpun, Kpunout, Kppnout;

        const auto elem_id = elem->id();
        int elem_id_n, elem_id_n_out;
        dof_map.dof_indices(elem, dof_indices);
        dof_map.dof_indices(elem, dof_indices_u, u_var);
        dof_map.dof_indices(elem, dof_indices_p, p_var);
        dof_map_old.dof_indices(elem, dof_indices_p_old, p_var);
        dof_map_old.dof_indices(elem, dof_indices_u_old, u_var);

        dof_map_n1.dof_indices(elem, dof_indices_u_n1, u_var);
        dof_map_n1.dof_indices(elem, dof_indices_p_n1, p_var);

        const unsigned int n_dofs = dof_indices.size();
        const unsigned int n_u_dofs = dof_indices_u.size();
        const unsigned int n_p_dofs = dof_indices_p.size();

        fe_vel->reinit(elem);
        fe_pres->reinit(elem);

        Ke.resize(n_dofs, n_dofs);

        Kuu.reposition(u_var * n_u_dofs, u_var * n_u_dofs, n_u_dofs, n_u_dofs);
        Kup.reposition(u_var * n_u_dofs, p_var * n_u_dofs, n_u_dofs, n_p_dofs);

        Kpu.reposition(p_var * n_u_dofs, u_var * n_u_dofs, n_p_dofs, n_u_dofs);
        Kpp.reposition(p_var * n_u_dofs, p_var * n_u_dofs, n_p_dofs, n_p_dofs);

        double A0_elem = M_PI * vessels[elem_id].r * vessels[elem_id].r * L_v * L_v;

        // Now we will build the element matrix.
        for (unsigned int qp = 0; qp < qrule.n_points(); qp++)
        {

            double Q_qp = 0.0;
            double p_qp = 0.0;
            double p_qp_old = 0.0;

            double r_qp = 0.0;
            r_qp += psi[0][qp] * vessels[elem_id].r1;
            r_qp += psi[1][qp] * vessels[elem_id].r2;

            for (unsigned int j = 0; j < n_u_dofs; j++)
            {
                Q_qp += phi[j][qp] * system.current_solution(dof_indices_u[j]);
            }

            for (unsigned int j = 0; j < n_p_dofs; j++)
            {
                p_qp += psi[j][qp] * system.current_solution(dof_indices_p[j]);
                p_qp_old +=
                    psi[j][qp] * system_old.current_solution(dof_indices_p_old[j]);
            }

            double dQdx_qp = 0.0;
            double dpdx_qp = 0.0;

            for (unsigned int j = 0; j < n_u_dofs; j++)
            {
                dQdx_qp += dphi[j][qp](0) * system.current_solution(dof_indices_u[j]);
            }

            for (unsigned int j = 0; j < n_p_dofs; j++)
            {
                dpdx_qp += dpsi[j][qp](0) * system.current_solution(dof_indices_p[j]);
            }

            for (unsigned int i = 0; i < n_u_dofs; i++)
            {

                for (unsigned int j = 0; j < n_u_dofs; j++)
                {

                    Kuu(i, j) += alpha_t * ((2.0 * alpha_v * Q_qp) / p_qp) *
                                 dphi[j][qp](0) * phi[i][qp] * JxW[qp];

                    Kuu(i, j) += alpha_t * ((2.0 * alpha_v) / p_qp) * dQdx_qp *
                                 phi[j][qp] * phi[i][qp] * JxW[qp];

                    Kuu(i, j) += alpha_t * (-(2.0 * alpha_v * Q_qp) / p_qp) * dpdx_qp *
                                 phi[j][qp] * phi[i][qp] * JxW[qp];

                    Kuu(i, j) += alpha_t * dsourcedQ(Q_qp, p_qp) * phi[i][qp] *
                                 phi[j][qp] * JxW[qp];

                    if (trans_soln == 1)
                        Kuu(i, j) += DtimeDer() * phi[i][qp] * phi[j][qp] * JxW[qp];
                }
            }

            for (unsigned int i = 0; i < n_u_dofs; i++)
            {
                for (unsigned int j = 0; j < n_p_dofs; j++)
                {

                    Kup(i, j) += alpha_t * (-((2.0 * alpha_v * Q_qp) / (p_qp * p_qp))) *
                                 dQdx_qp * psi[j][qp] * phi[i][qp] * JxW[qp];

                    Kup(i, j) +=
                        alpha_t *
                        (((2.0 * alpha_v * Q_qp * Q_qp) / (p_qp * p_qp * p_qp))) *
                        dpdx_qp * psi[j][qp] * phi[i][qp] * JxW[qp];

                    Kup(i, j) += alpha_t * betaV(elem_id, r_qp) * d2pofA(p_qp) *
                                 dpdx_qp * psi[j][qp] * phi[i][qp] * JxW[qp];

                    Kup(i, j) += alpha_t *
                                 (-((alpha_v * Q_qp * Q_qp) / (p_qp * p_qp)) +
                                  betaV(elem_id, r_qp) * dpofA(p_qp)) *
                                 dpsi[j][qp](0) * phi[i][qp] * JxW[qp];

                    Kup(i, j) += alpha_t * dsourcedp(Q_qp, p_qp) * psi[j][qp] *
                                 phi[i][qp] * JxW[qp];
                }
            }

            for (unsigned int i = 0; i < n_p_dofs; i++)
                for (unsigned int j = 0; j < n_u_dofs; j++)
                {
                    Kpu(i, j) += alpha_t * JxW[qp] * dphi[j][qp](0) * psi[i][qp];
                }

            for (unsigned int i = 0; i < n_p_dofs; i++)
                for (unsigned int j = 0; j < n_p_dofs; j++)
                {

                    if (trans_soln == 1)
                        Kpp(i, j) += DtimeDer() * psi[i][qp] * psi[j][qp] * JxW[qp];
                }

        } // end of the quadrature point qp-loop

        if (vessels[elem_id].dr != -10) // parent at junction
        {
            const Elem *neighbor1 = mesh.elem_ptr(vessels[elem_id].dl);
            const Elem *neighbor2 = mesh.elem_ptr(vessels[elem_id].dr);

            // cout << "elem_id=" << elem_id << " n1=" << vessels[elem_id].dl
            //      << " n2=" << vessels[elem_id].dr << endl;

            dof_map.dof_indices(neighbor1, dof_indices_neighbor_1_u, u_var);
            dof_map.dof_indices(neighbor2, dof_indices_neighbor_2_u, u_var);

            Kpun1.resize(dof_indices_p.size(), dof_indices_neighbor_1_u.size());
            Kpun2.resize(dof_indices_p.size(), dof_indices_neighbor_2_u.size());

            Kuun1.resize(dof_indices_u.size(), dof_indices_neighbor_1_u.size());
            Kuun2.resize(dof_indices_u.size(), dof_indices_neighbor_2_u.size());
        }

        if (vessels[elem_id].p != -10) // daughter at junction
        {
            if (vessels[vessels[elem_id].p].dr != -10)
            {

                const Elem *neighbor = mesh.elem_ptr(vessels[elem_id].p);

                elem_id_n = neighbor->id();

                dof_map.dof_indices(neighbor, dof_indices_neighbor_u, u_var);
                dof_map.dof_indices(neighbor, dof_indices_neighbor_p, p_var);

                Kuun.resize(dof_indices_u.size(), dof_indices_neighbor_u.size());
                Kupn.resize(dof_indices_u.size(), dof_indices_neighbor_p.size());

                Kppn.resize(dof_indices_p.size(), dof_indices_neighbor_p.size());
                Kpun.resize(dof_indices_p.size(), dof_indices_neighbor_u.size());
            }
        }

        if (venous_flow == 1)
        {
            if (vessels[elem_id].dl == -10)
            {
                if (elem_id < vessels_in.size())
                {
                    const Elem *neighborOut = mesh.elem_ptr(elem_id + vessels_in.size());
                    dof_map.dof_indices(neighborOut, dof_indices_neighbor_out_u, u_var);
                    dof_map.dof_indices(neighborOut, dof_indices_neighbor_out_p, p_var);

                    elem_id_n_out = neighborOut->id();
                }

                else
                {
                    const Elem *neighborOut = mesh.elem_ptr(elem_id - vessels_in.size());
                    dof_map.dof_indices(neighborOut, dof_indices_neighbor_out_u, u_var);
                    dof_map.dof_indices(neighborOut, dof_indices_neighbor_out_p, p_var);

                    elem_id_n_out = neighborOut->id();
                }

                Kppnout.resize(dof_indices_p.size(), dof_indices_neighbor_out_p.size());
                Kpunout.resize(dof_indices_p.size(), dof_indices_neighbor_out_u.size());
            }
        }

        double A_new = 0.0;
        double A_old = 0.0;
        double A_n1 = 0.0;
        double dAdt_b = 0.0;

        double Q_b = 0.0;
        double p_b = 0.0;

        double r_b = 0.0;

        double const_1 = 0.0;
        double const_2 = 0.0;

        double dconst_1_dQ = 0.0;
        double dconst_2_dQ = 0.0;

        double dconst_1_dp = 0.0;
        double dconst_2_dp = 0.0;

        if ((vessels[elem_id].ter == -1) || (vessels[elem_id].ter == 3)) // inlet
        {
            A_new = system.current_solution(dof_indices_p[0]);
            A_old = system_old.current_solution(dof_indices_p[0]);
            A_n1 = system_n1.current_solution(dof_indices_p[0]);
            dAdt_b = timeDer(A_new, A_old, A_n1); // dAInletdt(ttime);

            Q_b = system.current_solution(dof_indices_u[0]);
            p_b = A_new; // AInlet(ttime);
            r_b = vessels[elem_id].r1;
        }

        else if ((vessels[elem_id].ter == 1) ||
                 (vessels[elem_id].ter == 2)) // outlet
        {
            A_new = system.current_solution(dof_indices_p[1]);
            A_old = system_old.current_solution(dof_indices_p[1]);
            A_n1 = system_n1.current_solution(dof_indices_p[1]);
            dAdt_b = timeDer(A_new, A_old, A_n1); // dAInletdt(ttime);

            Q_b = system.current_solution(dof_indices_u[1]);
            p_b = A_new; // AOutlet(ttime);
            r_b = vessels[elem_id].r2;
        }

        if ((vessels[elem_id].ter == -1) || (vessels[elem_id].ter == 1) ||
            (vessels[elem_id].ter == 3) || (vessels[elem_id].ter == 2))
        {
            double const_a = -alpha_v * (Q_b / p_b);
            // double const_b =
            //     sqrt(((Q_b * Q_b * alpha_v * (alpha_v - 1.0)) / (p_b * p_b)) +
            //          betaPr(elem_id, r_b) * (sqrt(p_b) / (2.0 * rho_v *
            //          A0_in)));

            double const_b = 0.0;

            if (wave_type == 0)
                const_b =
                    sqrt(((Q_b * Q_b * alpha_v * (alpha_v - 1.0)) / (p_b * p_b)) +
                         betaPr(elem_id, r_b));
            else if (wave_type == 1)
                const_b =
                    sqrt(((Q_b * Q_b * alpha_v * (alpha_v - 1.0)) / (p_b * p_b)) +
                         betaPr(elem_id, r_b) * sqrt(p_b));

            const_1 = const_a + const_b;
            const_2 = const_a - const_b;

            double dconst_a_dQ = -(alpha_v / p_b);
            double dconst_a_dp = ((alpha_v * Q_b) / (p_b * p_b));

            double dconst_b_dQ =
                (0.5 / const_b) *
                ((2.0 * Q_b * alpha_v * (alpha_v - 1.0)) / (p_b * p_b));
            // double dconst_b_dp =
            //     (0.5 / const_b) *
            //     ((-(2.0 * Q_b * Q_b * alpha_v * (alpha_v - 1.0)) /
            //       (p_b * p_b * p_b)) +
            //      (betaPr(elem_id, r_b) / (4.0 * rho_v * A0_in)) * pow(p_b,
            //      -0.5));

            double dconst_b_dp = 0.0;

            if (wave_type == 0)
                dconst_b_dp = (0.5 / const_b) *
                              ((-(2.0 * Q_b * Q_b * alpha_v * (alpha_v - 1.0)) /
                                (p_b * p_b * p_b)));
            else if (wave_type == 1)
                dconst_b_dp = (0.5 / const_b) *
                              ((-(2.0 * Q_b * Q_b * alpha_v * (alpha_v - 1.0)) /
                                (p_b * p_b * p_b)) +
                               (betaPr(elem_id, r_b) / (2.0 * sqrt(p_b))));

            dconst_1_dQ = dconst_a_dQ + dconst_b_dQ;
            dconst_1_dp = dconst_a_dp + dconst_b_dp;

            dconst_2_dQ = dconst_a_dQ - dconst_b_dQ;
            dconst_2_dp = dconst_a_dp - dconst_b_dp;
        }

        if ((vessels[elem_id].ter == -1)) // inlet
        {
            unsigned int side = 0;
            fe_face->reinit(elem, side);
            fe_face_p->reinit(elem, side);

            double dQdx_b = 0.0;
            double dpdx_b = 0.0;

            for (unsigned int i = 0; i < n_u_dofs; i++)
            {
                dQdx_b +=
                    dphi_face[i][0](0) * system.current_solution(dof_indices_u[i]);
            }

            for (unsigned int i = 0; i < n_p_dofs; i++)
            {
                dpdx_b +=
                    dphi_face_p[i][0](0) * system.current_solution(dof_indices_p[i]);
            }

            for (unsigned int j = 0; j < n_u_dofs; j++)
            {
                Kuu(0, j) = 0.0;
            }

            for (unsigned int j = 0; j < n_p_dofs; j++)
            {
                Kup(0, j) = 0.0;
            }

            for (unsigned int j = 0; j < n_u_dofs; j++)
            {
                Kuu(0, j) += const_2 * dphi_face[j][0](0);
                Kuu(0, j) += ((2.0 * alpha_v * Q_b) / p_b) * dphi_face[j][0](0);
            }

            Kuu(0, 0) += DtimeDer();
            Kuu(0, 0) += ((2.0 * alpha_v) / p_b) * dQdx_b;
            Kuu(0, 0) += (-(2.0 * alpha_v * Q_b) / (p_b * p_b)) * dpdx_b;
            Kuu(0, 0) += gamma_v / p_b;
            Kuu(0, 0) += dconst_2_dQ * (dAdt_b + dQdx_b);

            for (unsigned int j = 0; j < n_p_dofs; j++)
            {
                Kup(0, j) += (-((alpha_v * Q_b * Q_b) / (p_b * p_b)) +
                              betaV(elem_id, r_b) * dpofA(p_b)) *
                             dphi_face_p[j][0](0);
            }

            Kup(0, 0) += const_2 * DtimeDer();
            Kup(0, 0) += -((2.0 * alpha_v * Q_b) / (p_b * p_b)) * dQdx_b;
            Kup(0, 0) +=
                (((2.0 * alpha_v * Q_b * Q_b) / (p_b * p_b * p_b))) * dpdx_b;
            Kup(0, 0) += betaV(elem_id, r_b) * d2pofA(p_b) * dpdx_b;
            Kup(0, 0) += -gamma_v * (Q_b / (p_b * p_b));
            Kup(0, 0) += dconst_2_dp * (dAdt_b + dQdx_b);

            if (inlet_bc == 0)
            {
                for (unsigned int j = 0; j < n_u_dofs; j++)
                {
                    Kpu(0, j) = 0.0;
                }

                for (unsigned int j = 0; j < n_p_dofs; j++)
                {
                    Kpp(0, j) = 0.0;
                }
                Kpp(0, 0) = 1.0;
            }

            else if (inlet_bc == 1)
            {
                for (unsigned int j = 0; j < n_u_dofs; j++)
                {
                    Kpu(0, j) = 0.0;
                }

                for (unsigned int j = 0; j < n_p_dofs; j++)
                {
                    Kpp(0, j) = 0.0;
                }

                Kpp(0, 0) += const_1 * DtimeDer();

                Kpu(0, 0) += DtimeDer();
                Kpu(0, 0) += gamma_v * (1.0 / p_b);

                Kpu(0, 0) += dconst_1_dQ * dAdt_b;
            }

            else if (inlet_bc == 3)
            {
                for (unsigned int j = 0; j < n_u_dofs; j++)
                {
                    Kpu(0, j) = 0.0;
                }

                for (unsigned int j = 0; j < n_p_dofs; j++)
                {
                    Kpp(0, j) = 0.0;
                }

                if (elem_id < vessels_in.size())
                    Kpu(0, 0) = 1.0;
                else
                    Kpp(0, 0) = 1.0;
            }
        }

        else if ((vessels[elem_id].ter == 1)) // outlet
        {
            if (venous_flow == 0)
            {
                fe_face->reinit(elem, 1);
                fe_face_p->reinit(elem, 1);

                double dQdx_b = 0.0;
                double dpdx_b = 0.0;

                for (unsigned int i = 0; i < n_u_dofs; i++)
                {
                    dQdx_b +=
                        dphi_face[i][0](0) * system.current_solution(dof_indices_u[i]);
                }

                for (unsigned int i = 0; i < n_p_dofs; i++)
                {
                    dpdx_b +=
                        dphi_face_p[i][0](0) * system.current_solution(dof_indices_p[i]);
                }

                for (unsigned int j = 0; j < n_u_dofs; j++)
                {
                    Kuu(1, j) = 0.0;
                }

                for (unsigned int j = 0; j < n_p_dofs; j++)
                {
                    Kup(1, j) = 0.0;
                }

                for (unsigned int j = 0; j < n_u_dofs; j++)
                {
                    Kuu(1, j) += const_1 * dphi_face[j][0](0);
                    Kuu(1, j) += ((2.0 * alpha_v * Q_b) / p_b) * dphi_face[j][0](0);
                }

                Kuu(1, 1) += DtimeDer();
                Kuu(1, 1) += ((2.0 * alpha_v) / p_b) * dQdx_b;
                Kuu(1, 1) += (-(2.0 * alpha_v * Q_b) / (p_b * p_b)) * dpdx_b;
                Kuu(1, 1) += gamma_v / p_b;
                Kuu(1, 1) += dconst_1_dQ * (dAdt_b + dQdx_b);

                for (unsigned int j = 0; j < n_p_dofs; j++)
                {
                    Kup(1, j) += (-((alpha_v * Q_b * Q_b) / (p_b * p_b)) +
                                  betaV(elem_id, r_b) * dpofA(p_b)) *
                                 dphi_face_p[j][0](0);
                }

                Kup(1, 1) += const_1 * DtimeDer();
                Kup(1, 1) += -((2.0 * alpha_v * Q_b) / (p_b * p_b)) * dQdx_b;
                Kup(1, 1) +=
                    (((2.0 * alpha_v * Q_b * Q_b) / (p_b * p_b * p_b))) * dpdx_b;
                Kup(1, 1) += betaV(elem_id, r_b) * d2pofA(p_b) * dpdx_b;
                Kup(1, 1) += -gamma_v * (Q_b / (p_b * p_b));
                Kup(1, 1) += dconst_1_dp * (dAdt_b + dQdx_b);

                if (outlet_bc == 0)
                {
                    for (unsigned int j = 0; j < n_u_dofs; j++)
                    {
                        Kpu(1, j) = 0.0;
                    }

                    for (unsigned int j = 0; j < n_p_dofs; j++)
                    {
                        Kpp(1, j) = 0.0;
                    }
                    Kpp(1, 1) = 1.0;
                }

                else if (outlet_bc == 1)
                {
                    for (unsigned int j = 0; j < n_u_dofs; j++)
                    {
                        Kpu(1, j) = 0.0;
                    }

                    for (unsigned int j = 0; j < n_p_dofs; j++)
                    {
                        Kpp(1, j) = 0.0;
                    }

                    Kpp(1, 1) += const_2 * DtimeDer();

                    Kpu(1, 1) += DtimeDer();
                    Kpu(1, 1) += gamma_v * (1.0 / p_b);

                    Kpu(1, 1) += dconst_2_dQ * dAdt_b;
                }

                else if (outlet_bc == 2)
                {
                    for (unsigned int j = 0; j < n_u_dofs; j++)
                    {
                        Kpu(1, j) = 0.0;
                    }

                    for (unsigned int j = 0; j < n_p_dofs; j++)
                    {
                        Kpp(1, j) = 0.0;
                    }

                    double A0_cur = M_PI * pow(vessels[elem_id].r, 2);
                    double A0bybeta = A0_cur / vessels[elem_id].beta;

                    double sqrt_At_cur =
                        sqrt(A0_cur) +
                        A0bybeta * ((POutlet(ttime) - PExt()) +
                                    sqrt(p_0 / rho_v) * ((L_v * L_v) / gamma_perm) *
                                        system.current_solution(dof_indices_u[1]));

                    Kpp(1, 1) = 1.0;
                    Kpu(1, 1) = -2.0 * sqrt_At_cur * A0bybeta * sqrt(p_0 / rho_v) *
                                ((L_v * L_v) / gamma_perm);
                }
            }
            else if (venous_flow == 1)
            {
                if (elem_id < vessels_in.size())
                {
                    fe_face->reinit(elem, 1);
                    fe_face_p->reinit(elem, 1);

                    double dQdx_b = 0.0;
                    double dpdx_b = 0.0;

                    for (unsigned int i = 0; i < n_u_dofs; i++)
                    {
                        dQdx_b +=
                            dphi_face[i][0](0) * system.current_solution(dof_indices_u[i]);
                    }

                    for (unsigned int i = 0; i < n_p_dofs; i++)
                    {
                        dpdx_b +=
                            dphi_face_p[i][0](0) * system.current_solution(dof_indices_p[i]);
                    }

                    for (unsigned int j = 0; j < n_u_dofs; j++)
                    {
                        Kuu(1, j) = 0.0;
                    }

                    for (unsigned int j = 0; j < n_p_dofs; j++)
                    {
                        Kup(1, j) = 0.0;
                    }

                    for (unsigned int j = 0; j < n_u_dofs; j++)
                    {
                        Kuu(1, j) += const_1 * dphi_face[j][0](0);
                        Kuu(1, j) += ((2.0 * alpha_v * Q_b) / p_b) * dphi_face[j][0](0);
                    }

                    Kuu(1, 1) += DtimeDer();
                    Kuu(1, 1) += ((2.0 * alpha_v) / p_b) * dQdx_b;
                    Kuu(1, 1) += (-(2.0 * alpha_v * Q_b) / (p_b * p_b)) * dpdx_b;
                    Kuu(1, 1) += gamma_v / p_b;
                    Kuu(1, 1) += dconst_1_dQ * (dAdt_b + dQdx_b);

                    for (unsigned int j = 0; j < n_p_dofs; j++)
                    {
                        Kup(1, j) += (-((alpha_v * Q_b * Q_b) / (p_b * p_b)) +
                                      betaV(elem_id, r_b) * dpofA(p_b)) *
                                     dphi_face_p[j][0](0);
                    }

                    Kup(1, 1) += const_1 * DtimeDer();
                    Kup(1, 1) += -((2.0 * alpha_v * Q_b) / (p_b * p_b)) * dQdx_b;
                    Kup(1, 1) +=
                        (((2.0 * alpha_v * Q_b * Q_b) / (p_b * p_b * p_b))) * dpdx_b;
                    Kup(1, 1) += betaV(elem_id, r_b) * d2pofA(p_b) * dpdx_b;
                    Kup(1, 1) += -gamma_v * (Q_b / (p_b * p_b));
                    Kup(1, 1) += dconst_1_dp * (dAdt_b + dQdx_b);

                    for (unsigned int j = 0; j < n_u_dofs; j++)
                    {
                        Kpu(1, j) = 0.0;
                    }

                    for (unsigned int j = 0; j < n_p_dofs; j++)
                    {
                        Kpp(1, j) = 0.0;
                    }

                    if (st_tree == 0)
                    {
                        Kpu(1, 1) = 1.0;
                        Kpunout(1, 1) = 1.0;
                    }
                    else if (st_tree == 1)
                    {
                        Kpu(1, 1) = 1.0;
                    }
                }
                else
                {
                    fe_face->reinit(elem, 1);
                    fe_face_p->reinit(elem, 1);

                    double dQdx_b = 0.0;
                    double dpdx_b = 0.0;

                    for (unsigned int i = 0; i < n_u_dofs; i++)
                    {
                        dQdx_b +=
                            dphi_face[i][0](0) * system.current_solution(dof_indices_u[i]);
                    }

                    for (unsigned int i = 0; i < n_p_dofs; i++)
                    {
                        dpdx_b +=
                            dphi_face_p[i][0](0) * system.current_solution(dof_indices_p[i]);
                    }

                    for (unsigned int j = 0; j < n_u_dofs; j++)
                    {
                        Kuu(1, j) = 0.0;
                    }

                    for (unsigned int j = 0; j < n_p_dofs; j++)
                    {
                        Kup(1, j) = 0.0;
                    }

                    for (unsigned int j = 0; j < n_u_dofs; j++)
                    {
                        Kuu(1, j) += const_2 * dphi_face[j][0](0);
                        Kuu(1, j) += ((2.0 * alpha_v * Q_b) / p_b) * dphi_face[j][0](0);
                    }

                    Kuu(1, 1) += DtimeDer();
                    Kuu(1, 1) += ((2.0 * alpha_v) / p_b) * dQdx_b;
                    Kuu(1, 1) += (-(2.0 * alpha_v * Q_b) / (p_b * p_b)) * dpdx_b;
                    Kuu(1, 1) += gamma_v / p_b;
                    Kuu(1, 1) += dconst_2_dQ * (dAdt_b + dQdx_b);

                    for (unsigned int j = 0; j < n_p_dofs; j++)
                    {
                        Kup(1, j) += (-((alpha_v * Q_b * Q_b) / (p_b * p_b)) +
                                      betaV(elem_id, r_b) * dpofA(p_b)) *
                                     dphi_face_p[j][0](0);
                    }

                    Kup(1, 1) += const_2 * DtimeDer();
                    Kup(1, 1) += -((2.0 * alpha_v * Q_b) / (p_b * p_b)) * dQdx_b;
                    Kup(1, 1) +=
                        (((2.0 * alpha_v * Q_b * Q_b) / (p_b * p_b * p_b))) * dpdx_b;
                    Kup(1, 1) += betaV(elem_id, r_b) * d2pofA(p_b) * dpdx_b;
                    Kup(1, 1) += -gamma_v * (Q_b / (p_b * p_b));
                    Kup(1, 1) += dconst_2_dp * (dAdt_b + dQdx_b);

                    double A_n = flow_vec[dof_indices_neighbor_out_p[1]];
                    double A_n0 = M_PI * pow(vessels[elem_id_n_out].r, 2);
                    double A0_b = M_PI * pow(vessels[elem_id].r, 2);

                    double Q_n = flow_vec[dof_indices_neighbor_out_u[1]];

                    for (unsigned int j = 0; j < n_u_dofs; j++)
                    {
                        Kpu(1, j) = 0.0;
                    }

                    for (unsigned int j = 0; j < n_p_dofs; j++)
                    {
                        Kpp(1, j) = 0.0;
                    }

                    if (st_tree == 0)
                    {
                        if (interface_type == 0)
                        {
                            Kpp(1, 1) += ((vessels[elem_id].beta / A0_b) /
                                          ((vessels[0].beta / (M_PI * pow(vessels[0].r, 2))))) *
                                         (0.5 / sqrt(A_new));
                            Kppnout(1, 1) -= ((vessels[elem_id_n_out].beta / A_n0) /
                                              ((vessels[0].beta / (M_PI * pow(vessels[0].r, 2))))) *
                                             (0.5 / sqrt(A_n));
                        }

                        else if (interface_type == 1)
                        {
                            Kpp(1, 1) += (vessels[elem_id].beta / A0_b) * (0.5 / sqrt(A_new));
                            Kpp(1, 1) += -rho_v * (pow(Q_b, 2) / pow(A_new, 3));
                            Kpu(1, 1) += rho_v * (Q_b / pow(A_new, 2));

                            Kppnout(1, 1) -= (vessels[elem_id_n_out].beta / A_n0) * (0.5 / sqrt(A_n));
                            Kppn(1, 1) -= -rho_v * (pow(Q_n, 2) / pow(A_n, 3));
                            Kpunout(1, 1) -= rho_v * (Q_n / pow(A_n, 2));
                        }
                    }
                    else if (st_tree == 1)
                    {
                        Kpu(1, 1) = 1.0;
                    }
                }
            }
        }

        else if ((vessels[elem_id].ter == 3)) // inlet
        {
            unsigned int side = 0;
            fe_face->reinit(elem, side);
            fe_face_p->reinit(elem, side);

            double dQdx_b = 0.0;
            double dpdx_b = 0.0;

            for (unsigned int i = 0; i < n_u_dofs; i++)
            {
                dQdx_b +=
                    dphi_face[i][0](0) * system.current_solution(dof_indices_u[i]);
            }

            for (unsigned int i = 0; i < n_p_dofs; i++)
            {
                dpdx_b +=
                    dphi_face_p[i][0](0) * system.current_solution(dof_indices_p[i]);
            }

            for (unsigned int j = 0; j < n_u_dofs; j++)
            {
                Kuu(0, j) = 0.0;
            }

            for (unsigned int j = 0; j < n_p_dofs; j++)
            {
                Kup(0, j) = 0.0;
            }

            for (unsigned int j = 0; j < n_u_dofs; j++)
            {
                Kuu(0, j) += const_2 * dphi_face[j][0](0);
                Kuu(0, j) += ((2.0 * alpha_v * Q_b) / p_b) * dphi_face[j][0](0);
            }

            Kuu(0, 0) += DtimeDer();
            Kuu(0, 0) += ((2.0 * alpha_v) / p_b) * dQdx_b;
            Kuu(0, 0) += (-(2.0 * alpha_v * Q_b) / (p_b * p_b)) * dpdx_b;
            Kuu(0, 0) += gamma_v / p_b;
            Kuu(0, 0) += dconst_2_dQ * (dAdt_b + dQdx_b);

            for (unsigned int j = 0; j < n_p_dofs; j++)
            {
                Kup(0, j) += (-((alpha_v * Q_b * Q_b) / (p_b * p_b)) +
                              betaV(elem_id, r_b) * dpofA(p_b)) *
                             dphi_face_p[j][0](0);
            }

            Kup(0, 0) += const_2 * DtimeDer();
            Kup(0, 0) += -((2.0 * alpha_v * Q_b) / (p_b * p_b)) * dQdx_b;
            Kup(0, 0) +=
                (((2.0 * alpha_v * Q_b * Q_b) / (p_b * p_b * p_b))) * dpdx_b;
            Kup(0, 0) += betaV(elem_id, r_b) * d2pofA(p_b) * dpdx_b;
            Kup(0, 0) += -gamma_v * (Q_b / (p_b * p_b));
            Kup(0, 0) += dconst_2_dp * (dAdt_b + dQdx_b);

            double A_n = flow_vec
                [dof_indices_neighbor_p
                     [1]]; // system.current_solution(dof_indices_neighbor_p[1]);
            double A_n0 = M_PI * pow(vessels[elem_id_n].r, 2);
            double A0_b = M_PI * pow(vessels[elem_id].r, 2);

            double Q_n = flow_vec
                [dof_indices_neighbor_u
                     [1]]; // system.current_solution(dof_indices_neighbor_u[1]);

            for (unsigned int j = 0; j < n_u_dofs; j++)
            {
                Kpu(0, j) = 0.0;
            }

            for (unsigned int j = 0; j < n_p_dofs; j++)
            {
                Kpp(0, j) = 0.0;
            }

            if (interface_type == 0)
            {
                Kpp(0, 0) += ((vessels[elem_id].beta / A0_b) /
                              ((vessels[0].beta / (M_PI * pow(vessels[0].r, 2))))) *
                             (0.5 / sqrt(A_new));
                Kppn(0, 1) -= ((vessels[elem_id_n].beta / A_n0) /
                               ((vessels[0].beta / (M_PI * pow(vessels[0].r, 2))))) *
                              (0.5 / sqrt(A_n));
            }

            else if (interface_type == 1)
            {
                Kpp(0, 0) += (vessels[elem_id].beta / A0_b) * (0.5 / sqrt(A_new));
                Kpp(0, 0) += -rho_v * (pow(Q_b, 2) / pow(A_new, 3));
                Kpu(0, 0) += rho_v * (Q_b / pow(A_new, 2));

                Kppn(0, 1) -= (vessels[elem_id_n].beta / A_n0) * (0.5 / sqrt(A_n));
                Kppn(0, 1) -= -rho_v * (pow(Q_n, 2) / pow(A_n, 3));
                Kpun(0, 1) -= rho_v * (Q_n / pow(A_n, 2));
            }
        }

        else if ((vessels[elem_id].ter == 2)) // outlet
        {
            fe_face->reinit(elem, 1);
            fe_face_p->reinit(elem, 1);

            double dQdx_b = 0.0;
            double dpdx_b = 0.0;

            for (unsigned int i = 0; i < n_u_dofs; i++)
            {
                dQdx_b +=
                    dphi_face[i][0](0) * system.current_solution(dof_indices_u[i]);
            }

            for (unsigned int i = 0; i < n_p_dofs; i++)
            {
                dpdx_b +=
                    dphi_face_p[i][0](0) * system.current_solution(dof_indices_p[i]);
            }

            for (unsigned int j = 0; j < n_u_dofs; j++)
            {
                Kuu(1, j) = 0.0;
            }

            for (unsigned int j = 0; j < n_p_dofs; j++)
            {
                Kup(1, j) = 0.0;
            }

            for (unsigned int j = 0; j < n_u_dofs; j++)
            {
                Kuu(1, j) += const_1 * dphi_face[j][0](0);
                Kuu(1, j) += ((2.0 * alpha_v * Q_b) / p_b) * dphi_face[j][0](0);
            }

            Kuu(1, 1) += DtimeDer();
            Kuu(1, 1) += ((2.0 * alpha_v) / p_b) * dQdx_b;
            Kuu(1, 1) += (-(2.0 * alpha_v * Q_b) / (p_b * p_b)) * dpdx_b;
            Kuu(1, 1) += gamma_v / p_b;
            Kuu(1, 1) += dconst_1_dQ * (dAdt_b + dQdx_b);

            for (unsigned int j = 0; j < n_p_dofs; j++)
            {
                Kup(1, j) += (-((alpha_v * Q_b * Q_b) / (p_b * p_b)) +
                              betaV(elem_id, r_b) * dpofA(p_b)) *
                             dphi_face_p[j][0](0);
            }

            Kup(1, 1) += const_1 * DtimeDer();
            Kup(1, 1) += -((2.0 * alpha_v * Q_b) / (p_b * p_b)) * dQdx_b;
            Kup(1, 1) +=
                (((2.0 * alpha_v * Q_b * Q_b) / (p_b * p_b * p_b))) * dpdx_b;
            Kup(1, 1) += betaV(elem_id, r_b) * d2pofA(p_b) * dpdx_b;
            Kup(1, 1) += -gamma_v * (Q_b / (p_b * p_b));
            Kup(1, 1) += dconst_1_dp * (dAdt_b + dQdx_b);

            for (unsigned int j = 0; j < n_u_dofs; j++)
            {
                Kpu(1, j) = 0.0;
            }

            for (unsigned int j = 0; j < n_p_dofs; j++)
            {
                Kpp(1, j) = 0.0;
            }
            Kpu(1, 1) = 1.0;
            Kpun1(1, 0) = -1.0;
            Kpun2(1, 0) = -1.0;
        }

        if (vessels[elem_id].p != -10)
        {
            if (vessels[vessels[elem_id].p].dr != -10) // This is the x=+ boundary
            {
                J.add_matrix(Kupn, dof_indices_u, dof_indices_neighbor_p);
                J.add_matrix(Kuun, dof_indices_u, dof_indices_neighbor_u);

                J.add_matrix(Kppn, dof_indices_p, dof_indices_neighbor_p);
                J.add_matrix(Kpun, dof_indices_p, dof_indices_neighbor_u);
            }
        }

        if (vessels[elem_id].dr != -10)
        {

            J.add_matrix(Kpun1, dof_indices_p, dof_indices_neighbor_1_u);
            J.add_matrix(Kpun2, dof_indices_p, dof_indices_neighbor_2_u);

            J.add_matrix(Kuun1, dof_indices_u, dof_indices_neighbor_1_u);
            J.add_matrix(Kuun2, dof_indices_u, dof_indices_neighbor_2_u);
        }

        if (vessels[elem_id].dl == -10)
        {
            J.add_matrix(Kppnout, dof_indices_p, dof_indices_neighbor_out_p);
            J.add_matrix(Kpunout, dof_indices_p, dof_indices_neighbor_out_u);
        }

        // dof_map.constrain_element_matrix(Ke, dof_indices);

        elem_count++;
        J.add_matrix(Ke, dof_indices);

    } // end of element loop

    // cout << "COMPUTED JACOBIAN" << endl;
}

void VesselFlow::compute_residual(const NumericVector<Number> &X,
                                  NumericVector<Number> &R,
                                  NonlinearImplicitSystem &system)
{
    // It is a good idea to make sure we are
    // assembling the proper system.
    libmesh_assert_equal_to(system_name, "flowSystem");

    // Get a constant reference to the mesh
    // object.
    const MeshBase &mesh = system.get_mesh();

    // The dimension that we are running
    const unsigned int dim = mesh.mesh_dimension();

    // Numeric ids corresponding to each
    // variable in the system
    const unsigned int u_var = system.variable_number("QVar");
    const unsigned int p_var = system.variable_number("pVar");

    auto &es = system.get_equation_systems();

    System &system_old = es.get_system<System>("flowSystemOld");
    const DofMap &dof_map_old = system_old.get_dof_map();
    std::vector<dof_id_type> dof_indices_u_old;
    std::vector<dof_id_type> dof_indices_p_old;

    System &system_n1 = es.get_system<System>("flowSystemn1");
    const DofMap &dof_map_n1 = system_n1.get_dof_map();
    std::vector<dof_id_type> dof_indices_u_n1;
    std::vector<dof_id_type> dof_indices_p_n1;

    // Get the Finite Element type for "u".
    // Note this will be the same as the type
    // for "v".
    FEType fe_vel_type = system.variable_type(u_var);
    std::unique_ptr<FEBase> fe_vel(FEBase::build(dim, fe_vel_type));
    QGauss qrule(dim, fe_vel_type.default_quadrature_order());
    fe_vel->attach_quadrature_rule(&qrule);

    // Get the Finite Element type for "p".
    FEType fe_pres_type = system.variable_type(p_var);
    std::unique_ptr<FEBase> fe_pres(FEBase::build(dim, fe_pres_type));
    fe_pres->attach_quadrature_rule(&qrule);

    // system.solution->localize(*system.current_local_solution);
    NumericVector<double> &flow_data = *(system.current_local_solution);
    flow_data.close();

    vector<double> flow_vec;
    flow_data.localize(flow_vec);

    const std::vector<Real> &JxW = fe_vel->get_JxW();

    const std::vector<std::vector<Real>> &phi = fe_vel->get_phi();
    const std::vector<std::vector<RealGradient>> &dphi = fe_vel->get_dphi();

    const std::vector<Point> &Xref_el = fe_vel->get_xyz();

    const std::vector<std::vector<Real>> &psi = fe_pres->get_phi();
    const std::vector<std::vector<RealGradient>> &dpsi = fe_pres->get_dphi();

    UniquePtr<FEBase> fe_face(FEBase::build(dim, fe_vel_type));
    QGauss qface(dim - 1, fe_vel_type.default_quadrature_order());
    fe_face->attach_quadrature_rule(&qface);

    UniquePtr<FEBase> fe_face_p(FEBase::build(dim, fe_pres_type));
    fe_face_p->attach_quadrature_rule(&qface);

    const std::vector<std::vector<RealGradient>> &dphi_face = fe_face->get_dphi();

    const std::vector<std::vector<RealGradient>> &dphi_face_p =
        fe_face_p->get_dphi();

    const DofMap &dof_map = system.get_dof_map();

    DenseVector<Number> Fe;

    DenseSubVector<Number> Fu(Fe), Fp(Fe);

    std::vector<dof_id_type> dof_indices;
    std::vector<dof_id_type> dof_indices_u;
    std::vector<dof_id_type> dof_indices_p;

    std::vector<dof_id_type> dof_indices_neighbor_p;
    std::vector<dof_id_type> dof_indices_neighbor_u;
    std::vector<dof_id_type> dof_indices_neighbor_1_u;
    std::vector<dof_id_type> dof_indices_neighbor_1_p;
    std::vector<dof_id_type> dof_indices_neighbor_2_u;
    std::vector<dof_id_type> dof_indices_neighbor_2_p;

    std::vector<dof_id_type> dof_indices_neighbor_out_u;
    std::vector<dof_id_type> dof_indices_neighbor_out_p;

    R.zero();

    MeshBase::const_element_iterator el = mesh.active_local_elements_begin();
    const MeshBase::const_element_iterator end_el =
        mesh.active_local_elements_end();

    int elem_count = 0;
    for (; el != end_el; ++el)
    {
        const Elem *elem = *el;

        const auto elem_id = elem->id();
        int elem_id_n, elem_id_n_out;
        dof_map.dof_indices(elem, dof_indices);
        dof_map.dof_indices(elem, dof_indices_u, u_var);
        dof_map.dof_indices(elem, dof_indices_p, p_var);

        dof_map_old.dof_indices(elem, dof_indices_u_old, u_var);
        dof_map_old.dof_indices(elem, dof_indices_p_old, p_var);

        dof_map_n1.dof_indices(elem, dof_indices_u_n1, u_var);
        dof_map_n1.dof_indices(elem, dof_indices_p_n1, p_var);

        const unsigned int n_dofs = dof_indices.size();
        const unsigned int n_u_dofs = dof_indices_u.size();
        const unsigned int n_p_dofs = dof_indices_p.size();

        fe_vel->reinit(elem);
        fe_pres->reinit(elem);

        Fe.resize(n_dofs);

        Fu.reposition(u_var * n_u_dofs, n_u_dofs);
        Fp.reposition(p_var * n_u_dofs, n_p_dofs);

        double A0_elem = M_PI * vessels[elem_id].r * vessels[elem_id].r * L_v * L_v;

        // cout << "elem_id=" << elem_id << " ter=" << vessels[elem_id].ter << endl;

        // Now we will build the element matrix.
        for (unsigned int qp = 0; qp < qrule.n_points(); qp++)
        {

            double Q_qp = 0.0;
            double p_qp = 0.0;

            double Q_qp_old = 0.0;
            double p_qp_old = 0.0;

            double Q_qp_n1 = 0.0;
            double p_qp_n1 = 0.0;

            double r_qp = 0.0;
            r_qp += psi[0][qp] * vessels[elem_id].r1;
            r_qp += psi[1][qp] * vessels[elem_id].r2;

            for (unsigned int j = 0; j < n_u_dofs; j++)
            {
                Q_qp += phi[j][qp] * system.current_solution(dof_indices_u[j]);
                Q_qp_old +=
                    phi[j][qp] * system_old.current_solution(dof_indices_u_old[j]);

                Q_qp_n1 += phi[j][qp] * system_n1.current_solution(dof_indices_u_n1[j]);
            }

            for (unsigned int j = 0; j < n_p_dofs; j++)
            {
                p_qp += psi[j][qp] * system.current_solution(dof_indices_p[j]);

                p_qp_old +=
                    psi[j][qp] * system_old.current_solution(dof_indices_p_old[j]);

                p_qp_n1 += psi[j][qp] * system_n1.current_solution(dof_indices_p_n1[j]);
            }

            double dQdx_qp = 0.0;
            double dQdx_qp_old = 0.0;
            double dpdx_qp = 0.0;
            double dpdx_qp_old = 0.0;

            for (unsigned int j = 0; j < n_u_dofs; j++)
            {
                dQdx_qp += dphi[j][qp](0) * system.current_solution(dof_indices_u[j]);
                dQdx_qp_old +=
                    dphi[j][qp](0) * system_old.current_solution(dof_indices_u_old[j]);
            }

            for (unsigned int j = 0; j < n_p_dofs; j++)
            {
                dpdx_qp += dpsi[j][qp](0) * system.current_solution(dof_indices_p[j]);
                dpdx_qp_old +=
                    dpsi[j][qp](0) * system_old.current_solution(dof_indices_p_old[j]);
            }

            for (unsigned int i = 0; i < n_u_dofs; i++)
            {

                Fu(i) += alpha_t * ((2.0 * alpha_v * Q_qp) / p_qp) * dQdx_qp *
                         phi[i][qp] * JxW[qp];
                Fu(i) += (1.0 - alpha_t) * ((2.0 * alpha_v * Q_qp_old) / p_qp_old) *
                         dQdx_qp_old * phi[i][qp] * JxW[qp];

                Fu(i) += alpha_t *
                         (-(alpha_v * ((Q_qp * Q_qp) / (p_qp * p_qp))) +
                          betaV(elem_id, r_qp) * dpofA(p_qp)) *
                         dpdx_qp * phi[i][qp] * JxW[qp];
                Fu(i) +=
                    (1.0 - alpha_t) *
                    (-(alpha_v * ((Q_qp_old * Q_qp_old) / (p_qp_old * p_qp_old))) +
                     betaV(elem_id, r_qp) * dpofA(p_qp)) *
                    dpdx_qp_old * phi[i][qp] * JxW[qp];

                Fu(i) += alpha_t * source_q(Q_qp, p_qp) * phi[i][qp] * JxW[qp];

                Fu(i) += (1.0 - alpha_t) * source_q(Q_qp_old, p_qp_old) * phi[i][qp] *
                         JxW[qp];

                if (trans_soln == 1)
                    Fu(i) += timeDer(Q_qp, Q_qp_old, Q_qp_n1) * phi[i][qp] * JxW[qp];
            }

            for (unsigned int i = 0; i < n_p_dofs; i++)
            {
                Fp(i) += alpha_t * dQdx_qp * psi[i][qp] * JxW[qp];
                Fp(i) += (1.0 - alpha_t) * dQdx_qp_old * psi[i][qp] * JxW[qp];

                if (trans_soln == 1)
                    Fp(i) += timeDer(p_qp, p_qp_old, p_qp_n1) * psi[i][qp] * JxW[qp];
            }

        } // end of the quadrature point qp-loop

        if (vessels[elem_id].dr != -10)
        {

            const Elem *neighbor1 = mesh.elem_ptr(vessels[elem_id].dl);
            const Elem *neighbor2 = mesh.elem_ptr(vessels[elem_id].dr);

            dof_map.dof_indices(neighbor1, dof_indices_neighbor_1_u, u_var);
            dof_map.dof_indices(neighbor2, dof_indices_neighbor_2_u, u_var);
        }

        if (vessels[elem_id].p != -10)
        {
            if (vessels[vessels[elem_id].p].dr != -10)
            {
                const Elem *neighbor = mesh.elem_ptr(vessels[elem_id].p);

                elem_id_n = neighbor->id();

                dof_map.dof_indices(neighbor, dof_indices_neighbor_u, u_var);
                dof_map.dof_indices(neighbor, dof_indices_neighbor_p, p_var);
            }
        }

        if (venous_flow == 1)
        {
            if (vessels[elem_id].dl == -10)
            {
                if (elem_id < vessels_in.size())
                {
                    const Elem *neighborOut = mesh.elem_ptr(elem_id + vessels_in.size());
                    dof_map.dof_indices(neighborOut, dof_indices_neighbor_out_u, u_var);
                    dof_map.dof_indices(neighborOut, dof_indices_neighbor_out_p, p_var);

                    elem_id_n_out = neighborOut->id();
                }

                else
                {
                    const Elem *neighborOut = mesh.elem_ptr(elem_id - vessels_in.size());
                    dof_map.dof_indices(neighborOut, dof_indices_neighbor_out_u, u_var);
                    dof_map.dof_indices(neighborOut, dof_indices_neighbor_out_p, p_var);

                    elem_id_n_out = neighborOut->id();
                }
            }
        }

        double A_new = 0.0;
        double A_old = 0.0;
        double A_n1 = 0.0;

        double dAdt_b = 0.0;

        double Q_b = 0.0;
        double Q_b_old = 0.0;
        double Q_b_n1 = 0.0;
        double dQdt_b = 0.0;
        double p_b = 0.0;
        double r_b = 0.0;

        double const_1 = 0.0;
        double const_2 = 0.0;

        if ((vessels[elem_id].ter == -1) || (vessels[elem_id].ter == 3)) // inlet
        {
            A_new = system.current_solution(dof_indices_p[0]);
            A_old = system_old.current_solution(dof_indices_p_old[0]);
            A_n1 = system_n1.current_solution(dof_indices_p_n1[0]);

            dAdt_b = timeDer(A_new, A_old, A_n1); // dAInletdt(ttime);

            Q_b = system.current_solution(dof_indices_u[0]);
            Q_b_old = system_old.current_solution(dof_indices_u_old[0]);
            Q_b_n1 = system_n1.current_solution(dof_indices_u_n1[0]);
            dQdt_b = timeDer(Q_b, Q_b_old, Q_b_n1);
            p_b = A_new; // AInlet(ttime)
            r_b = vessels[elem_id].r1;
        }

        else if ((vessels[elem_id].ter == 1) ||
                 (vessels[elem_id].ter == 2)) // outlet
        {
            A_new = system.current_solution(dof_indices_p[1]);
            A_old = system_old.current_solution(dof_indices_p_old[1]);
            A_n1 = system_n1.current_solution(dof_indices_p_n1[1]);

            dAdt_b = timeDer(A_new, A_old, A_n1); // dAOutletdt(ttime);
            Q_b = system.current_solution(dof_indices_u[1]);
            Q_b_old = system_old.current_solution(dof_indices_u_old[1]);
            Q_b_n1 = system_n1.current_solution(dof_indices_u_n1[1]);
            dQdt_b = timeDer(Q_b, Q_b_old, Q_b_n1);
            p_b = A_new; // AOutlet(ttime);
            r_b = vessels[elem_id].r2;
        }

        if ((vessels[elem_id].ter == -1) || (vessels[elem_id].ter == 1) ||
            (vessels[elem_id].ter == 3) || (vessels[elem_id].ter == 2))
        {
            double const_a = -alpha_v * (Q_b / p_b);
            // double const_b =
            //     sqrt(((Q_b * Q_b * alpha_v * (alpha_v - 1.0)) / (p_b * p_b)) +
            //          betaPr(elem_id, r_b) * (sqrt(p_b) / (2.0 * rho_v *
            //          A0_in)));

            double const_b = 0.0;

            if (wave_type == 0)
                const_b =
                    sqrt(((Q_b * Q_b * alpha_v * (alpha_v - 1.0)) / (p_b * p_b)) +
                         betaPr(elem_id, r_b));
            else if (wave_type == 1)
                const_b =
                    sqrt(((Q_b * Q_b * alpha_v * (alpha_v - 1.0)) / (p_b * p_b)) +
                         betaPr(elem_id, r_b) * sqrt(p_b));

            const_1 = const_a + const_b;
            const_2 = const_a - const_b;
        }

        if ((vessels[elem_id].ter == -1)) // inlet
        {
            unsigned int side = 0;

            fe_face->reinit(elem, side);
            fe_face_p->reinit(elem, side);

            double dQdx_b = 0.0;
            double dpdx_b = 0.0;

            for (unsigned int i = 0; i < n_u_dofs; i++)
            {
                dQdx_b +=
                    dphi_face[i][0](0) * system.current_solution(dof_indices_u[i]);
            }

            double dQdx_b_old = 0.0;
            for (unsigned int i = 0; i < n_u_dofs; i++)
            {
                dQdx_b_old += dphi_face[i][0](0) *
                              system_old.current_solution(dof_indices_u_old[i]);
            }

            for (unsigned int i = 0; i < n_p_dofs; i++)
            {
                dpdx_b +=
                    dphi_face_p[i][0](0) * system.current_solution(dof_indices_p[i]);
            }

            double dpdx_b_old = 0.0;
            for (unsigned int i = 0; i < n_p_dofs; i++)
            {
                dpdx_b_old += dphi_face_p[i][0](0) *
                              system_old.current_solution(dof_indices_p_old[i]);
            }

            Fu(0) = 0.0;
            Fu(0) += const_2 * (dAdt_b + dQdx_b);
            Fu(0) += dQdt_b;
            Fu(0) += ((2.0 * alpha_v * Q_b) / p_b) * dQdx_b;
            Fu(0) += (-((alpha_v * Q_b * Q_b) / (p_b * p_b)) +
                      betaV(elem_id, r_b) * dpofA(p_b)) *
                     dpdx_b;
            Fu(0) += gamma_v * (Q_b / p_b);

            if (inlet_bc == 0)
            {
                if (venous_flow == 0)
                    Fp(0) = system.current_solution(dof_indices_p[0]) - AInlet(ttime);

                else if (venous_flow == 1)
                {
                    if (elem_id < vessels_in.size())
                        Fp(0) = system.current_solution(dof_indices_p[0]) - AInlet(ttime);

                    else
                        Fp(0) = system.current_solution(dof_indices_p[0]) - AOutlet(ttime, elem_id);
                }
            }

            else if (inlet_bc == 1)
            {
                Fp(0) = 0.0;

                Fp(0) += const_1 * dAdt_b;

                Fp(0) += dQdt_b;
                Fp(0) += gamma_v * (Q_b / p_b);
            }

            else if (inlet_bc == 3)
            {
                Fp(0) = 0.0;

                if (venous_flow == 0)
                    Fp(0) = system.current_solution(dof_indices_u[0]) - QInlet();

                else if (venous_flow == 1)
                {
                    if (elem_id < vessels_in.size())
                        Fp(0) = system.current_solution(dof_indices_u[0]) - QInlet();

                    else
                        Fp(0) = system.current_solution(dof_indices_p[0]) - AOutlet(ttime, elem_id);
                }
            }
        }

        else if ((vessels[elem_id].ter == 1)) // outlet
        {
            if (venous_flow == 0)
            {
                fe_face->reinit(elem, 1);
                fe_face_p->reinit(elem, 1);

                double dQdx_b = 0.0;
                double dpdx_b = 0.0;

                for (unsigned int i = 0; i < n_u_dofs; i++)
                {
                    dQdx_b +=
                        dphi_face[i][0](0) * system.current_solution(dof_indices_u[i]);
                }

                double dQdx_b_old = 0.0;
                for (unsigned int i = 0; i < n_u_dofs; i++)
                {
                    dQdx_b_old += dphi_face[i][0](0) *
                                  system_old.current_solution(dof_indices_u_old[i]);
                }

                for (unsigned int i = 0; i < n_p_dofs; i++)
                {
                    dpdx_b +=
                        dphi_face_p[i][0](0) * system.current_solution(dof_indices_p[i]);
                }

                double dpdx_b_old = 0.0;
                for (unsigned int i = 0; i < n_p_dofs; i++)
                {
                    dpdx_b_old += dphi_face_p[i][0](0) *
                                  system_old.current_solution(dof_indices_p_old[i]);
                }

                Fu(1) = 0.0;
                Fu(1) += const_1 * (dAdt_b + dQdx_b);
                Fu(1) += dQdt_b;
                Fu(1) += ((2.0 * alpha_v * Q_b) / p_b) * dQdx_b;
                Fu(1) += (-((alpha_v * Q_b * Q_b) / (p_b * p_b)) +
                          betaV(elem_id, r_b) * dpofA(p_b)) *
                         dpdx_b;
                Fu(1) += gamma_v * (Q_b / p_b);

                if (outlet_bc == 0)
                    Fp(1) = system.current_solution(dof_indices_p[1]) -
                            AOutlet(ttime, elem_id);
                else if (outlet_bc == 1)
                {
                    Fp(1) = 0.0;

                    Fp(1) += const_2 * dAdt_b;

                    Fp(1) += dQdt_b;
                    Fp(1) += gamma_v * (Q_b / p_b);
                }

                else if (outlet_bc == 2)
                {
                    Fp(1) = 0.0;

                    double A0_cur = M_PI * pow(vessels[elem_id].r, 2);
                    double A0bybeta = A0_cur / vessels[elem_id].beta;
                    double At_cur = pow(
                        sqrt(A0_cur) +
                            A0bybeta * ((POutlet(ttime) - PExt()) +
                                        sqrt(p_0 / rho_v) * ((L_v * L_v) / gamma_perm) *
                                            system.current_solution(dof_indices_u[1])),
                        2);

                    Fp(1) += system.current_solution(dof_indices_p[1]);
                    Fp(1) += -At_cur;
                }
            }
            else if (venous_flow == 1)
            {
                if (elem_id < vessels_in.size())
                {
                    fe_face->reinit(elem, 1);
                    fe_face_p->reinit(elem, 1);

                    double dQdx_b = 0.0;
                    double dpdx_b = 0.0;

                    for (unsigned int i = 0; i < n_u_dofs; i++)
                    {
                        dQdx_b +=
                            dphi_face[i][0](0) * system.current_solution(dof_indices_u[i]);
                    }

                    double dQdx_b_old = 0.0;
                    for (unsigned int i = 0; i < n_u_dofs; i++)
                    {
                        dQdx_b_old += dphi_face[i][0](0) *
                                      system_old.current_solution(dof_indices_u_old[i]);
                    }

                    for (unsigned int i = 0; i < n_p_dofs; i++)
                    {
                        dpdx_b +=
                            dphi_face_p[i][0](0) * system.current_solution(dof_indices_p[i]);
                    }

                    double dpdx_b_old = 0.0;
                    for (unsigned int i = 0; i < n_p_dofs; i++)
                    {
                        dpdx_b_old += dphi_face_p[i][0](0) *
                                      system_old.current_solution(dof_indices_p_old[i]);
                    }

                    Fu(1) = 0.0;
                    Fu(1) += const_1 * (dAdt_b + dQdx_b);
                    Fu(1) += dQdt_b;
                    Fu(1) += ((2.0 * alpha_v * Q_b) / p_b) * dQdx_b;
                    Fu(1) += (-((alpha_v * Q_b * Q_b) / (p_b * p_b)) +
                              betaV(elem_id, r_b) * dpofA(p_b)) *
                             dpdx_b;
                    Fu(1) += gamma_v * (Q_b / p_b);

                    // Fp(1) = system.current_solution(dof_indices_u[1]) -
                    //         (system.current_solution(dof_indices_neighbor_1_u[0]) +
                    //          system.current_solution(dof_indices_neighbor_2_u[0]));

                    if (st_tree == 0)
                    {
                        Fp(1) = system.current_solution(dof_indices_u[1]) +
                                flow_vec[dof_indices_neighbor_out_u[1]];
                    }
                    else if (st_tree == 1)
                    {
                        int vessel_ter_num = vessels[elem_id].ter_num;

                        if (vessel_ter_num == -10)
                            cout << "error in terminal number" << endl;
                        else
                            Fp(1) = system.current_solution(dof_indices_u[1]) - qArt[vessel_ter_num];
                    }
                }
                else
                {

                    fe_face->reinit(elem, 1);
                    fe_face_p->reinit(elem, 1);

                    double dQdx_b = 0.0;
                    double dpdx_b = 0.0;

                    for (unsigned int i = 0; i < n_u_dofs; i++)
                    {
                        dQdx_b +=
                            dphi_face[i][0](0) * system.current_solution(dof_indices_u[i]);
                    }

                    double dQdx_b_old = 0.0;
                    for (unsigned int i = 0; i < n_u_dofs; i++)
                    {
                        dQdx_b_old += dphi_face[i][0](0) *
                                      system_old.current_solution(dof_indices_u_old[i]);
                    }

                    for (unsigned int i = 0; i < n_p_dofs; i++)
                    {
                        dpdx_b +=
                            dphi_face_p[i][0](0) * system.current_solution(dof_indices_p[i]);
                    }

                    double dpdx_b_old = 0.0;
                    for (unsigned int i = 0; i < n_p_dofs; i++)
                    {
                        dpdx_b_old += dphi_face_p[i][0](0) *
                                      system_old.current_solution(dof_indices_p_old[i]);
                    }

                    Fu(1) = 0.0;
                    Fu(1) += const_2 * (dAdt_b + dQdx_b);
                    Fu(1) += dQdt_b;
                    Fu(1) += ((2.0 * alpha_v * Q_b) / p_b) * dQdx_b;
                    Fu(1) += (-((alpha_v * Q_b * Q_b) / (p_b * p_b)) +
                              betaV(elem_id, r_b) * dpofA(p_b)) *
                             dpdx_b;
                    Fu(1) += gamma_v * (Q_b / p_b);

                    double A_n = flow_vec[dof_indices_neighbor_out_p[1]];
                    double A_n0 = M_PI * pow(vessels[elem_id_n_out].r, 2);
                    double A0_b = M_PI * pow(vessels[elem_id].r, 2);

                    double Q_n = flow_vec[dof_indices_neighbor_out_u[1]];

                    if (st_tree == 0)
                    {
                        if (interface_type == 0)
                        {
                            Fp(1) = ((vessels[elem_id].beta / A0_b) /
                                     ((vessels[0].beta / (M_PI * pow(vessels[0].r, 2))))) *
                                    (sqrt(A_new) - sqrt(A0_b));
                            Fp(1) -= ((vessels[elem_id_n_out].beta / A_n0) /
                                      ((vessels[0].beta / (M_PI * pow(vessels[0].r, 2))))) *
                                     (sqrt(A_n) - sqrt(A_n0));

                            /* Fp(1) += (vessels[elem_id].pext - vessels[elem_id_n_out].pext) /
                                     ((vessels[0].beta / (M_PI * pow(vessels[0].r, 2)))); */
                        }

                        else if (interface_type == 1)
                        {
                            Fp(1) = (vessels[elem_id].beta / A0_b) * (sqrt(A_new) - sqrt(A0_b));
                            Fp(1) += 0.5 * rho_v * pow(Q_b / A_new, 2);
                            Fp(1) -= (vessels[elem_id_n_out].beta / A_n0) * (sqrt(A_n) - sqrt(A_n0));
                            Fp(1) -= 0.5 * rho_v * pow(Q_n / A_n, 2);

                            /* Fp(0) += (vessels[elem_id].pext - vessels[elem_id_n_out].pext); */
                        }
                    }

                    else if (st_tree == 1)
                    {
                        int vessel_ter_num = vessels[elem_id].ter_num;

                        if (vessel_ter_num == -10)
                            cout << "error in terminal number" << endl;
                        else
                            Fp(1) = system.current_solution(dof_indices_u[1]) - qVein[vessel_ter_num];
                    }
                }
            }
        }

        else if ((vessels[elem_id].ter == 3)) // inlet
        {
            unsigned int side = 0;

            fe_face->reinit(elem, side);
            fe_face_p->reinit(elem, side);

            double dQdx_b = 0.0;
            double dpdx_b = 0.0;

            for (unsigned int i = 0; i < n_u_dofs; i++)
            {
                dQdx_b +=
                    dphi_face[i][0](0) * system.current_solution(dof_indices_u[i]);
            }

            double dQdx_b_old = 0.0;
            for (unsigned int i = 0; i < n_u_dofs; i++)
            {
                dQdx_b_old += dphi_face[i][0](0) *
                              system_old.current_solution(dof_indices_u_old[i]);
            }

            for (unsigned int i = 0; i < n_p_dofs; i++)
            {
                dpdx_b +=
                    dphi_face_p[i][0](0) * system.current_solution(dof_indices_p[i]);
            }

            double dpdx_b_old = 0.0;
            for (unsigned int i = 0; i < n_p_dofs; i++)
            {
                dpdx_b_old += dphi_face_p[i][0](0) *
                              system_old.current_solution(dof_indices_p_old[i]);
            }

            Fu(0) = 0.0;
            Fu(0) += const_2 * (dAdt_b + dQdx_b);
            Fu(0) += dQdt_b;
            Fu(0) += ((2.0 * alpha_v * Q_b) / p_b) * dQdx_b;
            Fu(0) += (-((alpha_v * Q_b * Q_b) / (p_b * p_b)) +
                      betaV(elem_id, r_b) * dpofA(p_b)) *
                     dpdx_b;
            Fu(0) += gamma_v * (Q_b / p_b);

            double A_n = flow_vec
                [dof_indices_neighbor_p
                     [1]]; // system.current_solution(dof_indices_neighbor_p[1]);
            double A_n0 = M_PI * pow(vessels[elem_id_n].r, 2);
            double A0_b = M_PI * pow(vessels[elem_id].r, 2);

            double Q_n = flow_vec
                [dof_indices_neighbor_u
                     [1]]; // system.current_solution(dof_indices_neighbor_u[1]);

            if (interface_type == 0)
            {
                Fp(0) = ((vessels[elem_id].beta / A0_b) /
                         ((vessels[0].beta / (M_PI * pow(vessels[0].r, 2))))) *
                        (sqrt(A_new) - sqrt(A0_b));
                Fp(0) -= ((vessels[elem_id_n].beta / A_n0) /
                          ((vessels[0].beta / (M_PI * pow(vessels[0].r, 2))))) *
                         (sqrt(A_n) - sqrt(A_n0));

                Fp(0) += (vessels[elem_id].pext - vessels[elem_id_n].pext) /
                         ((vessels[0].beta / (M_PI * pow(vessels[0].r, 2))));
            }

            else if (interface_type == 1)
            {
                Fp(0) = (vessels[elem_id].beta / A0_b) * (sqrt(A_new) - sqrt(A0_b));
                Fp(0) += 0.5 * rho_v * pow(Q_b / A_new, 2);
                Fp(0) -= (vessels[elem_id_n].beta / A_n0) * (sqrt(A_n) - sqrt(A_n0));
                Fp(0) -= 0.5 * rho_v * pow(Q_n / A_n, 2);

                Fp(0) += (vessels[elem_id].pext - vessels[elem_id_n].pext);
            }
        }

        else if ((vessels[elem_id].ter == 2)) // outlet
        {
            fe_face->reinit(elem, 1);
            fe_face_p->reinit(elem, 1);

            double dQdx_b = 0.0;
            double dpdx_b = 0.0;

            for (unsigned int i = 0; i < n_u_dofs; i++)
            {
                dQdx_b +=
                    dphi_face[i][0](0) * system.current_solution(dof_indices_u[i]);
            }

            double dQdx_b_old = 0.0;
            for (unsigned int i = 0; i < n_u_dofs; i++)
            {
                dQdx_b_old += dphi_face[i][0](0) *
                              system_old.current_solution(dof_indices_u_old[i]);
            }

            for (unsigned int i = 0; i < n_p_dofs; i++)
            {
                dpdx_b +=
                    dphi_face_p[i][0](0) * system.current_solution(dof_indices_p[i]);
            }

            double dpdx_b_old = 0.0;
            for (unsigned int i = 0; i < n_p_dofs; i++)
            {
                dpdx_b_old += dphi_face_p[i][0](0) *
                              system_old.current_solution(dof_indices_p_old[i]);
            }

            Fu(1) = 0.0;
            Fu(1) += const_1 * (dAdt_b + dQdx_b);
            Fu(1) += dQdt_b;
            Fu(1) += ((2.0 * alpha_v * Q_b) / p_b) * dQdx_b;
            Fu(1) += (-((alpha_v * Q_b * Q_b) / (p_b * p_b)) +
                      betaV(elem_id, r_b) * dpofA(p_b)) *
                     dpdx_b;
            Fu(1) += gamma_v * (Q_b / p_b);

            // Fp(1) = system.current_solution(dof_indices_u[1]) -
            //         (system.current_solution(dof_indices_neighbor_1_u[0]) +
            //          system.current_solution(dof_indices_neighbor_2_u[0]));

            Fp(1) = system.current_solution(dof_indices_u[1]) -
                    (flow_vec[dof_indices_neighbor_1_u[0]] +
                     flow_vec[dof_indices_neighbor_2_u[0]]);
        }

        // dof_map.constrain_element_vector(Fe, dof_indices);

        elem_count++;

        R.add_vector(Fe, dof_indices);
    } // end of element loop

    // cout << "R=" << R << endl;
}

void VesselFlow::initialise_area(EquationSystems &es)
{
    const MeshBase &mesh = es.get_mesh();
    const unsigned int dim = mesh.mesh_dimension();

    System &system = es.get_system<System>("flowSystem");

    const DofMap &dof_map = system.get_dof_map();
    std::vector<dof_id_type> dof_indices_p;

    MeshBase::const_element_iterator el = mesh.active_local_elements_begin();
    const MeshBase::const_element_iterator end_el =
        mesh.active_local_elements_end();

    for (; el != end_el; ++el)
    {
        const Elem *elem = *el;
        dof_map.dof_indices(elem, dof_indices_p, 1);

        const auto elem_id = elem->id();

        vessels[elem_id].x1 /= L_v;
        vessels[elem_id].y1 /= L_v;
        vessels[elem_id].z1 /= L_v;
        vessels[elem_id].x2 /= L_v;
        vessels[elem_id].y2 /= L_v;
        vessels[elem_id].z2 /= L_v;
        vessels[elem_id].l /= L_v;
        vessels[elem_id].r /= L_v;

        const unsigned int n_p_dofs = dof_indices_p.size();

        for (unsigned int j = 0; j < n_p_dofs; j++)
        {

            double A_vess = (M_PI * vessels[elem_id].r * vessels[elem_id].r);

            if (j == 0)
                A_vess = (M_PI * vessels[elem_id].r1 * vessels[elem_id].r1);
            else if (j == 1)
                A_vess = (M_PI * vessels[elem_id].r2 * vessels[elem_id].r2);

            system.solution->set(dof_indices_p[j], A_vess);
        }
    }
    system.solution->close();
    system.solution->localize(*system.current_local_solution);
}

void VesselFlow::old_new(EquationSystems &es)
{
    System &system_new = es.get_system<System>("flowSystem");
    System &system_old = es.get_system<System>("flowSystemOld");

    system_old.solution->zero();
    system_old.solution->add(1.0, *system_new.current_local_solution);
    system_old.solution->close();
    system_old.solution->localize(*system_old.current_local_solution);
}

void VesselFlow::n1_old(EquationSystems &es)
{
    System &system_new = es.get_system<System>("flowSystemOld");
    System &system_old = es.get_system<System>("flowSystemn1");

    system_old.solution->zero();
    system_old.solution->add(1.0, *system_new.current_local_solution);
    system_old.solution->close();
    system_old.solution->localize(*system_old.current_local_solution);
}

void VesselFlow::writeFlowData(EquationSystems &es)
{
    ofstream file_vess;
    file_vess.open("vessels_1d_flow_data.dat", ios::out);

    const MeshBase &mesh = es.get_mesh();

    LinearImplicitSystem &system =
        es.get_system<LinearImplicitSystem>("flowSystem");

    // Numeric ids corresponding to each variable in the system
    const unsigned int u_var = system.variable_number("QVar");
    const unsigned int p_var = system.variable_number("pVar");

    const DofMap &dof_map = system.get_dof_map();
    std::vector<dof_id_type> dof_indices_u;
    std::vector<dof_id_type> dof_indices_p;

    MeshBase::const_element_iterator el = mesh.active_local_elements_begin();
    const MeshBase::const_element_iterator end_el =
        mesh.active_local_elements_end();

    for (; el != end_el; ++el)
    {
        const Elem *elem = *el;

        const auto n = elem->id();
        dof_map.dof_indices(elem, dof_indices_u, u_var);
        dof_map.dof_indices(elem, dof_indices_p, p_var);

        double Q1_prime = system.current_solution(dof_indices_u[0]);
        double Q2_prime = system.current_solution(dof_indices_u[1]);

        double A1_prime = system.current_solution(dof_indices_p[0]);
        double A2_prime = system.current_solution(dof_indices_p[1]);

        double Q1 = Q1_prime * sqrt(p_0 / rho_v) * L_v * L_v;
        double Q2 = Q2_prime * sqrt(p_0 / rho_v) * L_v * L_v;

        double A1 = A1_prime * L_v * L_v;
        double A2 = A2_prime * L_v * L_v;

        double beta_c = (2.0 * rho_v * c_v * c_v) /
                        sqrt(M_PI * vessels[n].r * vessels[n].r * L_v * L_v);

        double p1 = beta_c * (sqrt(A1) -
                              sqrt(M_PI * vessels[n].r * vessels[n].r * L_v * L_v));
        double p2 = beta_c * (sqrt(A2) -
                              sqrt(M_PI * vessels[n].r * vessels[n].r * L_v * L_v));

        double rad1 = sqrt(A1 / M_PI);
        double rad2 = sqrt(A2 / M_PI);

        // rad1 = vessels[n].r;
        // rad2 = vessels[n].r;

        file_vess << vessels[n].x1 * L_v << "," << vessels[n].y1 * L_v << ","
                  << vessels[n].z1 * L_v << "," << vessels[n].x2 * L_v << ","
                  << vessels[n].y2 * L_v << "," << vessels[n].z2 * L_v << ","
                  << vessels[n].l * L_v << "," << rad1 << "," << rad2 << ","
                  << vessels[n].p << "," << vessels[n].dl << "," << vessels[n].dr
                  << "," << Q1 << "," << Q2 << "," << p1 << "," << p2 << ","
                  << Q1_prime << "," << Q2_prime << "," << A1_prime << ","
                  << A2_prime << endl;
    }
    file_vess.close();
}

void VesselFlow::writeFlowDataTime(EquationSystems &es, int it, int rank)
{
    string name = "flow_data/vessels_1d_flow_data_";
    string fileend = ".csv";
    string out_frame;
    char numstr[21];
    sprintf(numstr, "%d", it);
    out_frame = name + numstr + fileend;

    ofstream file_vess;
    if (rank == 0)
    {
        file_vess.open(out_frame, ios::out);

        file_vess << "\"x1\""
                  << ",\"y1\""
                  << ",\"z1\""
                  << ",\"x2\""
                  << ",\"y2\""
                  << ",\"z2\""
                  << ",\"l\""
                  << ",\"r1\""
                  << ",\"r2\""
                  << ",\"p\""
                  << ",\"dl\""
                  << ",\"dr\""
                  << ",\"Q1\""
                  << ",\"Q2\""
                  << ",\"p1\""
                  << ",\"p2\""
                  << ",\"Q1p\""
                  << ",\"Q2p\""
                  << ",\"A1p\""
                  << ",\"A2p\"" << endl;
    }

    const MeshBase &mesh = es.get_mesh();

    LinearImplicitSystem &system =
        es.get_system<LinearImplicitSystem>("flowSystem");

    // Numeric ids corresponding to each variable in the system
    const unsigned int u_var = system.variable_number("QVar");
    const unsigned int p_var = system.variable_number("pVar");

    NumericVector<double> &flow_data = *(system.current_local_solution);
    flow_data.close();

    vector<double> flow_vec;
    flow_data.localize(flow_vec);

    const DofMap &dof_map = system.get_dof_map();
    std::vector<dof_id_type> dof_indices_u;
    std::vector<dof_id_type> dof_indices_p;

    // MeshBase::const_element_iterator el = mesh.active_local_elements_begin();
    // const MeshBase::const_element_iterator end_el =
    //     mesh.active_local_elements_end();

    if (rank == 0)
    {
        MeshBase::const_element_iterator el = mesh.active_elements_begin();
        const MeshBase::const_element_iterator end_el = mesh.active_elements_end();

        for (; el != end_el; ++el)
        {
            const Elem *elem = *el;

            const auto n = elem->id();
            dof_map.dof_indices(elem, dof_indices_u, u_var);
            dof_map.dof_indices(elem, dof_indices_p, p_var);

            double Q1_prime = flow_vec
                [dof_indices_u[0]]; // system.current_solution(dof_indices_u[0]);
            double Q2_prime = flow_vec
                [dof_indices_u[1]]; // system.current_solution(dof_indices_u[1]);

            double A1_prime = flow_vec
                [dof_indices_p[0]]; // system.current_solution(dof_indices_p[0]);
            double A2_prime = flow_vec
                [dof_indices_p[1]]; // system.current_solution(dof_indices_p[1]);

            double Q1 = Q1_prime * sqrt(p_0 / rho_v) * L_v * L_v;
            double Q2 = Q2_prime * sqrt(p_0 / rho_v) * L_v * L_v;

            double A1 = A1_prime * L_v * L_v;
            double A2 = A2_prime * L_v * L_v;

            double p1 = vessels[n].pext +
                        (vessels[n].beta / (M_PI * vessels[n].r * vessels[n].r)) *
                            (sqrt(A1) - sqrt(M_PI * vessels[n].r * vessels[n].r));
            double p2 = vessels[n].pext +
                        (vessels[n].beta / (M_PI * vessels[n].r * vessels[n].r)) *
                            (sqrt(A2) - sqrt(M_PI * vessels[n].r * vessels[n].r));

            // if (n == 500)
            //   cout << "p1=" << p1 << " p2=" << p2 << " A1=" << A1 << " A2=" << A2
            //   << " "
            //        << " " << vessels[n].beta << " " << vessels[n].r << " "
            //        << (M_PI * vessels[n].r * vessels[n].r) << " "
            //        << (vessels[n].beta / (M_PI * vessels[n].r * vessels[n].r)) <<
            //        endl;

            double rad1 = sqrt(A1 / M_PI);
            double rad2 = sqrt(A2 / M_PI);

            // rad1 = vessels[n].r;
            // rad2 = vessels[n].r;

            double vess_x1 = vessels[n].x1 * L_v;
            double vess_y1 = vessels[n].y1 * L_v;
            double vess_z1 = vessels[n].z1 * L_v;

            double vess_x2 = vessels[n].x2 * L_v;
            double vess_y2 = vessels[n].y2 * L_v;
            double vess_z2 = vessels[n].z2 * L_v;

            if (n >= vessels_in.size())
            {
                vess_y1 += 40.0;
                vess_y2 += 40.0;
            }

            file_vess << vess_x1 << "," << vess_y1 << ","
                      << vess_z1 << "," << vess_x2 << ","
                      << vess_y2 << "," << vess_z2 << ","
                      << vessels[n].l * L_v << "," << rad1 << "," << rad2 << ","
                      << vessels[n].p << "," << vessels[n].dl << "," << vessels[n].dr
                      << "," << Q1 << "," << Q2 << "," << p1 << "," << p2 << ","
                      << Q1_prime << "," << Q2_prime << "," << A1_prime << ","
                      << A2_prime << endl;
        }
        file_vess.close();
    }
}

double VesselFlow::pofA(double A_cur)
{
    if (wave_type == 0)
        return A_cur;
    else if (wave_type == 1)
        return pow(A_cur, 1.5);
}

double VesselFlow::dpofA(double A_cur)
{
    if (wave_type == 0)
        return 1.0;
    else if (wave_type == 1)
        return (1.5 * pow(A_cur, 0.5));
}

double VesselFlow::d2pofA(double A_cur)
{
    if (wave_type == 0)
        return 0.0;
    else if (wave_type == 1)
        return (1.5 * 0.5 * pow(A_cur, -0.5));
}

double VesselFlow::PExt()
{
    double pext_cur = 0.0;

    if (pext_type == 0)
    {
        pext_cur = 0.0;
    }

    else if (pext_type == 1)
    {
        pext_cur = p_ext_const;
        pext_cur *= 0.13332;
    }

    else if (pext_type == 2)
    {

        if (ttime_dim < t_load)
            pext_cur = p_ext_const * (ttime_dim / t_load);
        else
            pext_cur = p_ext_const;

        pext_cur *= 0.13332;
    }

    else if (pext_type == 3)
    {
        double p_sys_max = p_diastole + p_systole * (1.0 - exp(-pow((0.7) - 0.5, 2) / 0.007));

        if (ttime_dim < 0.2)
            pext_cur = p_diastole * ((ttime_dim) / 0.2);
        else if ((ttime_dim) < 0.5)
            pext_cur = p_diastole;
        else if ((ttime_dim) < 0.7)
            pext_cur = p_diastole + p_systole * (1.0 - exp(-pow((ttime_dim)-0.5, 2) / 0.007));
        else if ((ttime_dim) < 0.8)
            pext_cur = p_sys_max * (1.0 - exp(-pow(0.8 - (ttime_dim), 2) / 0.0015));
        else
            pext_cur = 0.0;

        pext_cur *= 0.13332;
    }

    else if (pext_type == 4)
    {
        double p_sys_max = -p_diastole + p_systole * (1.0 - exp(-pow((0.7) - 0.5, 2) / 0.007));

        if (ttime_dim < 0.2)
            pext_cur = -p_diastole * ((ttime_dim) / 0.2);
        else if ((ttime_dim) < 0.5)
            pext_cur = -p_diastole;
        else if ((ttime_dim) < 0.7)
            pext_cur = -p_diastole + p_systole * (1.0 - exp(-pow((ttime_dim)-0.5, 2) / 0.007));
        else if ((ttime_dim) < 0.8)
            pext_cur = p_sys_max * (1.0 - exp(-pow(0.8 - (ttime_dim), 2) / 0.0015));
        else
            pext_cur = 0.0;

        pext_cur *= 0.13332;
    }

    else if (pext_type == 5)
    {
        double p_sys_max = -p_diastole + p_systole * (1.0 - exp(-pow((0.65) - 0.5, 2) / 0.004));

        if (ttime_dim < 0.2)
            pext_cur = -p_diastole * ((ttime_dim) / 0.2);
        else if ((ttime_dim) < 0.5)
            pext_cur = -p_diastole;
        else if ((ttime_dim) < 0.65)
            pext_cur = -p_diastole + p_systole * (1.0 - exp(-pow((ttime_dim)-0.5, 2) / 0.004));
        else if ((ttime_dim) < 0.8)
            pext_cur = p_sys_max * (1.0 - exp(-pow(0.8 - (ttime_dim), 2) / 0.004));
        else
            pext_cur = 0.0;

        pext_cur *= 0.13332;
    }

    return pext_cur;
}

void VesselFlow::compute_pext(double time_v)
{
    double pext_cur = PExt();

    for (int n = 0; n < vessels.size(); n++)
    {
        if (vessels[n].inside == 1)
        {
            vessels[n].pext = pext_cur;
        }
        else
            vessels[n].pext = 0.0;
    }
}

double VesselFlow::PInlet(double time_v)
{
    double t_c = sqrt(rho_v / p_0) * L_v;
    double p_inlet = 0.0;

    if (pin_type == 0)
    {
        p_inlet = 0.0;
    }

    else if (pin_type == 1)
    {
        p_inlet = p_in_const;
        p_inlet *= 0.13332;
    }

    else if (pin_type == 2)
    {

        if (ttime_dim < t_load)
            p_inlet = (p_in_const / t_load) * ttime_dim;
        else
            p_inlet = p_in_const;

        p_inlet *= 0.13332;
    }

    else if (pin_type == 3)
    {

        if (ttime_dim < 0.2)
            p_inlet = 1730 * pow(ttime_dim + 0.6, 3) - 3625 * pow(ttime_dim + 0.6, 2) +
                      2454 * (ttime_dim + 0.6) - 440;
        else if (ttime_dim < 0.575)
            p_inlet = -48.68 * (ttime_dim - 0.2) + 87.75;
        else if (ttime_dim < 0.7)
            p_inlet = 13.0 + 120.0 * (1.0 - exp(-pow((ttime_dim)-0.5, 2) / 0.007));
        else if (ttime_dim < 0.75)
            p_inlet = 132.53 * (1.0 - exp(-pow(0.8 - (ttime_dim), 2) / 0.0015));
        else if (ttime_dim < 0.8)
            p_inlet = 113328 * pow(ttime_dim, 3) - 271559 * pow(ttime_dim, 2) + 216817 * ttime_dim - 57578;

        p_inlet *= 0.13332;
    }

    else if (pin_type == 4)
    {

        if (ttime_dim < 0.55)
            p_inlet = 101.776 - (101.776 - 70.712) * (ttime_dim / 0.55);
        else if (ttime_dim < 0.7)
            p_inlet = 70.712 + (132.59 - 70.712) * (1 - exp(-(pow(ttime_dim - 0.55, 2)) / 0.004));
        else if (ttime_dim < 0.76)
            p_inlet = 132.53 * (1 - exp(-(pow(0.8 - (ttime_dim), 2)) / 0.0011));
        else if (ttime_dim < 0.8)
            p_inlet = 113328 * pow(ttime_dim, 3) - 271559 * pow(ttime_dim, 2) + 216817 * ttime_dim - 57578;

        p_inlet *= 0.13332;
    }

    else if (pin_type == 5)
    {

        if (ttime_dim < 0.55)
            p_inlet = 101.776 - (101.776 - 70.712) * (ttime_dim / 0.55);
        else if (ttime_dim < 0.7)
            p_inlet = 70.712 + (132.59 - 70.712) * (1 - exp(-(pow(ttime_dim - 0.55, 2)) / 0.004));
        else
            p_inlet = 101.776+(132.53-101.776) * (1 - exp(-(pow(0.8 - (ttime_dim), 2)) / 0.0005));

        p_inlet *= 0.13332;
    }

    return p_inlet;
}

double VesselFlow::POutlet(double time_v)
{
    double t_c = sqrt(rho_v / p_0) * L_v;
    double p_outlet = 0.0;

    if (pout_type == 0)
    {
        p_outlet = 0.0;
    }

    else if (pout_type == 1)
    {
        p_outlet = p_out_const;
        p_outlet *= 0.13332;
    }

    else if (pout_type == 2)
    {

        if (ttime_dim < t_load)
            p_outlet = p_out_const * (ttime_dim / t_load);
        else
            p_outlet = p_out_const;

        p_outlet *= 0.13332;
    }

    else if (pout_type == 3)
    {

        double p_sys_max = p_diastole + p_systole * (1.0 - exp(-pow((0.7) - 0.5, 2) / 0.007));

        if (ttime_dim < 0.2)
            p_outlet = p_diastole * ((ttime_dim) / 0.2);
        else if ((ttime_dim) < 0.5)
            p_outlet = p_diastole;
        else if ((ttime_dim) < 0.7)
            p_outlet = p_diastole + p_systole * (1.0 - exp(-pow((ttime_dim)-0.5, 2) / 0.007));
        else if ((ttime_dim) < 0.8)
            p_outlet = p_sys_max * (1.0 - exp(-pow(0.8 - (ttime_dim), 2) / 0.0015));
        else
            p_outlet = 0.0;

        p_outlet *= 0.13332;
    }

    else if (pout_type == 4)
    {
        double p_sys_max = -p_diastole + p_systole * (1.0 - exp(-pow((0.7) - 0.5, 2) / 0.007));

        if (ttime_dim < 0.2)
            p_outlet = -p_diastole * ((ttime_dim) / 0.2);
        else if ((ttime_dim) < 0.5)
            p_outlet = -p_diastole;
        else if ((ttime_dim) < 0.7)
            p_outlet = -p_diastole + p_systole * (1.0 - exp(-pow((ttime_dim)-0.5, 2) / 0.007));
        else if ((ttime_dim) < 0.8)
            p_outlet = p_sys_max * (1.0 - exp(-pow(0.8 - (ttime_dim), 2) / 0.0015));
        else
            p_outlet = 0.0;

        p_outlet *= 0.13332;
    }

    else if (pout_type == 5)
    {
        double p_sys_max = -p_diastole + p_systole * (1.0 - exp(-pow((0.65) - 0.5, 2) / 0.004));

        if (ttime_dim < 0.2)
            p_outlet = -p_diastole * ((ttime_dim) / 0.2);
        else if ((ttime_dim) < 0.5)
            p_outlet = -p_diastole;
        else if ((ttime_dim) < 0.65)
            p_outlet = -p_diastole + p_systole * (1.0 - exp(-pow((ttime_dim)-0.5, 2) / 0.004));
        else if ((ttime_dim) < 0.8)
            p_outlet = p_sys_max * (1.0 - exp(-pow(0.8 - (ttime_dim), 2) / 0.004));
        else
            p_outlet = 0.0;

        p_outlet *= 0.13332;
    }

    return p_outlet;
}

double VesselFlow::PDrain(double time_v)
{
    double t_c = sqrt(rho_v / p_0) * L_v;

    double pdrain_cur = 0.0;

    if (time_v * t_c < 0.1)
        pdrain_cur = 0.5 * ((time_v * t_c) / 0.1);
    else
        pdrain_cur = 0.5;

    return pdrain_cur;
}

double VesselFlow::AInlet(double time_v)
{
    double A0_inlet = M_PI * pow(vessels[0].r, 2);

    double AIn =
        pow(((PInlet(time_v) * A0_inlet) / vessels[0].beta) + sqrt(A0_inlet), 2);

    return AIn;
}

double VesselFlow::AOutlet(double time_v, int n)
{
    double A0_outlet = M_PI * pow(vessels[n].r, 2);
    double AOut = pow(
        ((POutlet(time_v) * A0_outlet) / vessels[0].beta) + sqrt(A0_outlet), 2);

    return AOut;
}

double VesselFlow::QInlet()
{
    double qin_cur = 0.0;

    if (ttime_dim < 0.2)
        qin_cur = 5595.0 + (4616.0 - 5595.0) * (ttime_dim / 0.2);
    else if (ttime_dim < 0.5)
        qin_cur = 4616.0 + (3592.0 - 4616.0) * ((ttime_dim - 0.2) / 0.3);
    else if (ttime_dim < 0.575)
        qin_cur = -218846.0 * pow(ttime_dim, 2) + 216100.0 * ttime_dim - 49776.0;
    else if (ttime_dim < 0.75)
        qin_cur = -21096262.0 * pow(ttime_dim, 4) + 56320676.0 * pow(ttime_dim, 3) - 56391356.0 * pow(ttime_dim, 2) + 25093342.0 * ttime_dim - 4182792.0;
    else
        qin_cur = -1560149.0 * pow(ttime_dim, 2) + 2517185.0 * ttime_dim - 1009693.0;

    return qin_cur/(sqrt(p_0 / rho_v) * L_v * L_v);
}

void VesselFlow::writeUpdatedVessels()
{

    ofstream file_vess;
    file_vess.open("updated_vessels.csv", ios::out);

    ofstream file_vess_dat;
    file_vess_dat.open("updated_vessels_data.dat", ios::out);

    file_vess << "\"x1\""
              << ",\"y1\""
              << ",\"z1\""
              << ",\"x2\""
              << ",\"y2\""
              << ",\"z2\""
              << ",\"l\""
              << ",\"r1\""
              << ",\"r2\""
              << ",\"p\""
              << ",\"dl\""
              << ",\"dr\""
              << ",\"Q1\""
              << ",\"Q2\""
              << ",\"p1\""
              << ",\"p2\""
              << ",\"Q1p\""
              << ",\"Q2p\""
              << ",\"A1p\""
              << ",\"A2p\"" << endl;

    for (unsigned int n = 0; n < vessels.size(); n++)
    {

        double Q1_prime = 0.0; // system.current_solution(dof_indices_u[0]);
        double Q2_prime = 0.0; // system.current_solution(dof_indices_u[1]);

        double A1_prime = 0.0; // system.current_solution(dof_indices_p[0]);
        double A2_prime = 0.0; // system.current_solution(dof_indices_p[1]);

        double Q1 = Q1_prime; // * sqrt(p_0 / rho_v) * L_v * L_v;
        double Q2 = Q2_prime; // * sqrt(p_0 / rho_v) * L_v * L_v;

        double A1 = A1_prime; // * L_v * L_v;
        double A2 = A2_prime; // * L_v * L_v;

        double beta_c = beta_0;
        // (2.0 * rho_v * c_v * c_v) / (sqrt(M_PI * vessels[n].r * vessels[n].r));

        double p1 = beta_c * (sqrt(A1) - sqrt(M_PI * vessels[n].r * vessels[n].r));
        double p2 = beta_c * (sqrt(A2) - sqrt(M_PI * vessels[n].r * vessels[n].r));

        // double rad1 = sqrt(A1 / M_PI);
        // double rad2 = sqrt(A2 / M_PI);

        double rad1 = vessels[n].r;
        double rad2 = vessels[n].r;

        file_vess << vessels[n].x1 << "," << vessels[n].y1 << "," << vessels[n].z1
                  << "," << vessels[n].x2 << "," << vessels[n].y2 << ","
                  << vessels[n].z2 << "," << vessels[n].l << "," << rad1 << ","
                  << rad2 << "," << vessels[n].p << "," << vessels[n].dl << ","
                  << vessels[n].dr << "," << Q1 << "," << Q2 << "," << p1 << ","
                  << p2 << "," << Q1_prime << "," << Q2_prime << "," << A1_prime
                  << "," << A2_prime << endl;

        file_vess_dat << vessels[n].x1 << " " << vessels[n].y1 << " "
                      << vessels[n].z1 << " " << vessels[n].x2 << " "
                      << vessels[n].y2 << " " << vessels[n].z2 << " "
                      << vessels[n].l << " " << vessels[n].r << " " << vessels[n].p
                      << " " << vessels[n].dl << " " << vessels[n].dr << " "
                      << vessels[n].i_in << " " << -10 << endl;
    }
    file_vess.close();
    file_vess_dat.close();
}

double VesselFlow::source_q(double Q_cur, double p_cur)
{
    double source_cur = 0.0;
    source_cur = gamma_v * (Q_cur / p_cur);
    return source_cur;
}

double VesselFlow::dsourcedQ(double Q_cur, double p_cur)
{
    double dSdQ = 0.0;
    dSdQ = gamma_v / p_cur;
    return dSdQ;
}

double VesselFlow::dsourcedp(double Q_cur, double p_cur)
{
    double dSdp = 0.0;
    dSdp = -gamma_v * (Q_cur / (p_cur * p_cur));
    return dSdp;
}

double VesselFlow::betaV(int n, double r_cur)
{

    double beta_cur = 0.0;
    if (wave_type == 0)
        beta_cur = c_v * c_v * (rho_v / p_0);
    else if (wave_type == 1)
        beta_cur = (vessels[n].beta * L_v) / (3.0 * p_0 * M_PI * pow(r_cur, 2));
    return beta_cur;
}

double VesselFlow::betaPr(int n, double r_cur)
{
    double beta_cur = 0.0;
    if (wave_type == 0)
        beta_cur = c_v * c_v * (rho_v / p_0);
    else if (wave_type == 1)
        beta_cur = (vessels[n].beta * L_v) / (2.0 * p_0 * M_PI * pow(r_cur, 2));
    return beta_cur;
}

double VesselFlow::timeDer(double Q_cur, double Q_old, double Q_n1)
{
    double dQdt_cur = 0.0;
    if (time_integ == 1)
        dQdt_cur = (Q_cur - Q_old) / dt_v;
    else if (time_integ == 2)
    {
        if (time_itr < 2)
            dQdt_cur = (Q_cur - Q_old) / dt_v;
        else
            dQdt_cur = (3.0 * Q_cur - 4.0 * Q_old + Q_n1) / (2.0 * dt_v);
    }

    return dQdt_cur;
}

double VesselFlow::DtimeDer()
{
    double ddQdt_cur = 0.0;
    if (time_integ == 1)
        ddQdt_cur = 1.0 / dt_v;
    else if (time_integ == 2)
    {
        if (time_itr < 2)
            ddQdt_cur = 1.0 / dt_v;
        else
            ddQdt_cur = 3.0 / (2.0 * dt_v);
    }

    return ddQdt_cur;
}

void VesselFlow::create_mesh_3(Mesh &mesh)
{

    mesh.set_mesh_dimension(1);

    Point X1(0.0, 0.0, 0.0);
    Point X2(vessels[0].l, 0.0, 0.0);
    Point X3((1.0 / 3.0) * vessels[0].l, 0.0, 0.0);
    Point X4((2.0 / 3.0) * vessels[0].l, 0.0, 0.0);

    idx = 0;
    mesh.add_point(X1, idx);
    vessels[0].i1 = idx;
    idx++;
    mesh.add_point(X2, idx);
    vessels[0].i2 = idx;
    idx++;
    mesh.add_point(X3, idx);
    vessels[0].i3 = idx;
    idx++;
    mesh.add_point(X4, idx);

    ide = 0;
    Elem *elem = mesh.add_elem(new Edge4);
    elem->set_node(0) = mesh.node_ptr(idx - 3);
    elem->set_node(1) = mesh.node_ptr(idx - 2);
    elem->set_node(2) = mesh.node_ptr(idx - 1);
    elem->set_node(3) = mesh.node_ptr(idx);

    mesh.boundary_info->add_side(elem, 0, 1000);

    vessels[0].e_id = ide;

    add_element_node_3(mesh, 0);

    mesh.prepare_for_use();

    MeshTools::Modification::scale(mesh, 1.0 / L_v);

    mesh.write("vessels_mesh.xda");
}

void VesselFlow::add_element_node_3(Mesh &mesh, int i)
{

    cout << "i=" << i << " dl=" << vessels[i].dl << " dr=" << vessels[i].dr
         << endl;
    if (vessels[i].dl != -10)
    {

        if (vessels[i].dr == -10)
        {
            Elem *elem_n = mesh.elem_ptr(i);
            Point pj;
            pj = elem_n->point(1);
            Point X2l(pj(0) + vessels[vessels[i].dl].l, 0.0, 0.0);
            Point X3l(pj(0) + (1.0 / 3.0) * vessels[vessels[i].dl].l, 0.0, 0.0);
            Point X4l(pj(0) + (2.0 / 3.0) * vessels[vessels[i].dl].l, 0.0, 0.0);

            int idx_n = vessels[i].i2; // idx - 1;
            vessels[vessels[i].dl].i1 = idx_n;
            idx++;
            mesh.add_point(X2l, idx);
            vessels[vessels[i].dl].i2 = idx;
            idx++;
            mesh.add_point(X3l, idx);
            vessels[vessels[i].dl].i3 = idx;
            idx++;
            mesh.add_point(X4l, idx);

            ide++;
            Elem *elem_l = mesh.add_elem(new Edge4);
            elem_l->set_node(0) = mesh.node_ptr(idx_n);
            elem_l->set_node(1) = mesh.node_ptr(idx - 2);
            elem_l->set_node(2) = mesh.node_ptr(idx - 1);
            elem_l->set_node(3) = mesh.node_ptr(idx);

            vessels[vessels[i].dl].e_id = ide;

            if (vessels[vessels[i].dl].dl == -10)
                mesh.boundary_info->add_side(elem_l, 1, 4000);
        }

        else
        {
            Elem *elem_n = mesh.elem_ptr(ide);
            Point pj;
            pj = elem_n->point(1);

            Point X1l(pj(0), 0.0, 0.0);
            Point X2l(pj(0) + vessels[vessels[i].dl].l, 0.0, 0.0);
            Point X3l(pj(0) + (1.0 / 3.0) * vessels[vessels[i].dl].l, 0.0, 0.0);
            Point X4l(pj(0) + (2.0 / 3.0) * vessels[vessels[i].dl].l, 0.0, 0.0);

            idx++;
            mesh.add_point(X1l, idx);
            vessels[vessels[i].dl].i1 = idx;
            idx++;
            mesh.add_point(X2l, idx);
            vessels[vessels[i].dl].i2 = idx;
            idx++;
            mesh.add_point(X3l, idx);
            vessels[vessels[i].dl].i3 = idx;
            idx++;
            mesh.add_point(X4l, idx);

            int ide_p = i;

            Elem *elem_p = mesh.elem_ptr(ide_p);
            mesh.boundary_info->add_side(elem_p, 1, 2000);

            ide++;
            Elem *elem_l = mesh.add_elem(new Edge4);
            elem_l->set_node(0) = mesh.node_ptr(idx - 3);
            elem_l->set_node(1) = mesh.node_ptr(idx - 2);
            elem_l->set_node(2) = mesh.node_ptr(idx - 1);
            elem_l->set_node(3) = mesh.node_ptr(idx);

            vessels[vessels[i].dl].e_id = ide;

            mesh.boundary_info->add_side(elem_l, 0, 3000);

            if (vessels[vessels[i].dl].dl == -10)
                mesh.boundary_info->add_side(elem_l, 1, 4000);

            Point X1r(pj(0), 0.0, 0.0);
            Point X2r(pj(0) + vessels[vessels[i].dr].l, 0.0, 0.0);
            Point X3r(pj(0) + (1.0 / 3.0) * vessels[vessels[i].dr].l, 0.0, 0.0);
            Point X4r(pj(0) + (2.0 / 3.0) * vessels[vessels[i].dr].l, 0.0, 0.0);

            idx++;
            mesh.add_point(X1r, idx);
            vessels[vessels[i].dr].i1 = idx;
            idx++;
            mesh.add_point(X2r, idx);
            vessels[vessels[i].dr].i2 = idx;
            idx++;
            mesh.add_point(X3r, idx);
            vessels[vessels[i].dr].i3 = idx;
            idx++;
            mesh.add_point(X4r, idx);

            ide++;
            Elem *elem_r = mesh.add_elem(new Edge4);
            elem_r->set_node(0) = mesh.node_ptr(idx - 3);
            elem_r->set_node(1) = mesh.node_ptr(idx - 2);
            elem_r->set_node(2) = mesh.node_ptr(idx - 1);
            elem_r->set_node(3) = mesh.node_ptr(idx);

            vessels[vessels[i].dr].e_id = ide;

            mesh.boundary_info->add_side(elem_r, 0, 3000);

            if (vessels[vessels[i].dr].dl == -10)
                mesh.boundary_info->add_side(elem_r, 1, 4000);
        }

        add_element_node_3(mesh, vessels[i].dl);

        if (vessels[i].dr != -10)
            add_element_node_3(mesh, vessels[i].dr);
    }
}

void VesselFlow::writeFlowDataBound(EquationSystems &es, int it, int rank)
{
    string name = "flow_data_inlet";
    string fileend = ".dat";
    string out_frame;
    // char numstr[21];
    // sprintf(numstr, "%d", it);
    // out_frame = name + numstr + fileend;
    out_frame = name + fileend;

    ofstream file_vess;

    update_pqbound(es, rank);

    if (rank == 0)
    {

        file_vess.open(out_frame, ios::app);

        file_vess << ttime * sqrt(rho_v / p_0) * L_v << " " << qInCur << " " << pInCur << " "
                  << qArtTotal << " " << pArtTotal << " " << qVeinTotal << " " << pVeinTotal << " "
                  << " " << qOutCur << " " << pOutCur << endl;
        file_vess.close();
    }
}

void VesselFlow::updateImpedance()
{
    int tmstps = N_period; // 26668;
    double Period = time_per;
    double rho = rho_v;
    double mu_pl = mu_v;
    double rbot = 0.1;
    double rmin = 0.05;

    double Lr = 2.5;            // 0.25;
    double q = 100.0 * Lr * Lr; // 10 * Lr * Lr;
    double g = 9810.0;          // 981.0;

    double fa1 = 0;
    double fa2 = 1;
    double fa3 = 100.0; // 1000000;

    double fv1 = fa1;
    double fv2 = fa2;
    double fv3 = fa3;
    double asym = 0.41;
    double expo = 2.76;
    double lrrA = 36;
    double lrrV = 36;

    cout << "before impedance" << endl;

    Admittance::compute_impedance(tmstps, Period, rho, mu_pl, rbot, rmin,
                                  y11, y12, y21, y22, Lr, q, g,
                                  fa1, fa2, fa3, fv1, fv2, fv3, asym, expo, lrrA, lrrV);

    cout << "after impedance" << endl;

    for (int j = 0; j < y11.size(); j++)
    {
        y11(j) = y11(j) * (q / (rho * g * Lr));
        y12(j) = y12(j) * (q / (rho * g * Lr));
        y21(j) = y21(j) * (q / (rho * g * Lr));
        y22(j) = y22(j) * (q / (rho * g * Lr));
    }

    cout << "0 " << y11(0) << " " << y12(0) << " " << y21(0) << " " << y22(0) << endl;
    cout << "1 " << y11(1) << " " << y12(1) << " " << y21(1) << " " << y22(1) << endl;
    cout << "10 " << y11(10) << " " << y12(10) << " " << y21(10) << " " << y22(10) << endl;
    cout << "1000 " << y11(1000) << " " << y12(1000) << " " << y21(1000) << " " << y22(1000) << endl;
}

void VesselFlow::initialise_partvein(int rank, int np, LibMeshInit &init)
{
    int terminal_num = -1;
    for (int i = 0; i < vessels_in.size(); i++)
    {
        vessels_in[i].ter_num = -10;
        vessels[i].ter_num = -10;
        if (vessels[i].dl == -10)
        {
            termNum.push_back(i);

            qArt.push_back(0);
            qVein.push_back(0);

            terminal_num++;
            vessels_in[i].ter_num = terminal_num;
            vessels[i].ter_num = terminal_num;
            if (venous_flow == 1)
                vessels[i + vessels_in.size()].ter_num = terminal_num;
        }
    }

    pArt.resize(N_period);
    pVein.resize(N_period);

    y11.resize(N_period);
    y12.resize(N_period);
    y21.resize(N_period);
    y22.resize(N_period);

    pLt.resize(termNum.size());
    pRt.resize(termNum.size());

    for (int i = 0; i < termNum.size(); i++)
    {
        pLt(i).resize(N_period);
        pRt(i).resize(N_period);
    }

    if (restart_part_vein == 0)
    {
        for (int j = 0; j < N_period; j++)
        {
            for (int i = 0; i < vessels_in.size(); i++)
            {
                if (vessels[i].dl == -10)
                {
                    pArt(j).push_back(0.0);
                    pVein(j).push_back(0.0);
                }
            }
        }
    }

    else if (restart_part_vein == 1)
    {
        for (int rank_i = 0; rank_i < np; rank_i++)
        {
            if (rank == rank_i)
            {
                ifstream file_partvein;
                file_partvein.open(partvein_file_name);
                for (int i = 0; i < N_period; i++)
                {
                    for (int j = 0; j < termNum.size(); j++)
                    {
                        double part_data = 0.0, pvein_data = 0.0;
                        file_partvein >> part_data >> pvein_data;

                        pArt(i).push_back(part_data);
                        pVein(i).push_back(pvein_data);
                    }
                }
                file_partvein.close();
            }
        }

        MPI_Barrier(init.comm().get());
    }

    for (int i = 0; i < N_period; i++)
    {
        for (int j = 0; j < termNum.size(); j++)
        {
            int tmtau_itr = fmod(-i + N_period, N_period);

            pLt(j)(i) = pArt(tmtau_itr)[j];
            pRt(j)(i) = pVein(tmtau_itr)[j];
        }
    }
}

void VesselFlow::update_partvein(EquationSystems &es, int rank)
{

    const MeshBase &mesh = es.get_mesh();

    LinearImplicitSystem &system =
        es.get_system<LinearImplicitSystem>("flowSystem");

    // Numeric ids corresponding to each variable in the system
    const unsigned int u_var = system.variable_number("QVar");
    const unsigned int p_var = system.variable_number("pVar");

    NumericVector<double> &flow_data = *(system.current_local_solution);
    flow_data.close();

    vector<double> flow_vec;
    flow_data.localize(flow_vec);

    const DofMap &dof_map = system.get_dof_map();
    std::vector<dof_id_type> dof_indices_u;
    std::vector<dof_id_type> dof_indices_p;

    if (rank == 0)
    {

        for (int i = 0; i < pArt(0).size(); i++)
        {
            for (int j = N_period - 1; j > 0; j--)
            {

                pLt(i)(j) = pLt(i)(j - 1);
                pRt(i)(j) = pRt(i)(j - 1);
            }
        }
        for (int i = 0; i < pArt(0).size(); i++)
        {
            int n = termNum[i];
            const Elem *elem = mesh.elem_ptr(n);

            dof_map.dof_indices(elem, dof_indices_u, u_var);
            dof_map.dof_indices(elem, dof_indices_p, p_var);

            double A2_prime = flow_vec[dof_indices_p[1]];
            double A2 = A2_prime * L_v * L_v;
            /* double p2 = vessels[n].pext +
                        (vessels[n].beta / (M_PI * vessels[n].r * vessels[n].r)) *
                            (sqrt(A2) - sqrt(M_PI * vessels[n].r * vessels[n].r)); */

            double p2 = (vessels[n].beta / (M_PI * vessels[n].r * vessels[n].r)) *
                        (sqrt(A2) - sqrt(M_PI * vessels[n].r * vessels[n].r));

            pArt(time_itr_per)[i] = p2;
            pLt(i)(0) = p2;

            n = n + vessels_in.size();
            const Elem *elem_n = mesh.elem_ptr(n);

            dof_map.dof_indices(elem_n, dof_indices_u, u_var);
            dof_map.dof_indices(elem_n, dof_indices_p, p_var);

            A2_prime = flow_vec[dof_indices_p[1]];
            A2 = A2_prime * L_v * L_v;
            /* p2 = vessels[n].pext +
                 (vessels[n].beta / (M_PI * vessels[n].r * vessels[n].r)) *
                     (sqrt(A2) - sqrt(M_PI * vessels[n].r * vessels[n].r)); */

            p2 = (vessels[n].beta / (M_PI * vessels[n].r * vessels[n].r)) *
                 (sqrt(A2) - sqrt(M_PI * vessels[n].r * vessels[n].r));

            pVein(time_itr_per)[i] = p2;
            pRt(i)(0) = p2;
        }
    }

    for (int i = 0; i < pArt(0).size(); i++)
    {
        MPI_Bcast(&pArt(time_itr_per)[i], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&pVein(time_itr_per)[i], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Bcast(&pLt(i)(time_itr_per), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&pRt(i)(time_itr_per), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}

void VesselFlow::update_qartvein(int rank)
{

    if (rank == 0)
    {
        for (int i = 0; i < pArt(0).size(); i++)
        {

            qArt[i] = y11.dot(pLt(i)) + y12.dot(pRt(i));
            qVein[i] = y21.dot(pLt(i)) + y22.dot(pRt(i));

            qArt[i] /= sqrt(p_0 / rho_v) * L_v * L_v;
            qVein[i] /= sqrt(p_0 / rho_v) * L_v * L_v;
        }
    }

    for (int i = 0; i < qArt.size(); i++)
    {
        MPI_Bcast(&qArt[i], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&qVein[i], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}

void VesselFlow::update_pqbound(EquationSystems &es, int rank)
{
    const MeshBase &mesh = es.get_mesh();

    LinearImplicitSystem &system =
        es.get_system<LinearImplicitSystem>("flowSystem");

    // Numeric ids corresponding to each variable in the system
    const unsigned int u_var = system.variable_number("QVar");
    const unsigned int p_var = system.variable_number("pVar");

    NumericVector<double> &flow_data = *(system.current_local_solution);
    flow_data.close();

    vector<double> flow_vec;
    flow_data.localize(flow_vec);

    const DofMap &dof_map = system.get_dof_map();
    std::vector<dof_id_type> dof_indices_u;
    std::vector<dof_id_type> dof_indices_p;

    if (rank == 0)
    {
        pArtTotal = 0.0, qArtTotal = 0.0;
        pVeinTotal = 0.0, qVeinTotal = 0.0;
        pInCur = 0.0, qInCur = 0.0;
        pOutCur = 0.0, qOutCur = 0.0;

        int n = 0;
        const Elem *elem_in = mesh.elem_ptr(n);

        dof_map.dof_indices(elem_in, dof_indices_u, u_var);
        dof_map.dof_indices(elem_in, dof_indices_p, p_var);

        double A2_prime_in = flow_vec[dof_indices_p[1]];
        double A2_in = A2_prime_in * L_v * L_v;
        double p2_in = vessels[n].pext +
                       (vessels[n].beta / (M_PI * vessels[n].r * vessels[n].r)) *
                           (sqrt(A2_in) - sqrt(M_PI * vessels[n].r * vessels[n].r));

        pInCur = p2_in;
        qInCur = flow_vec[dof_indices_u[1]] * sqrt(p_0 / rho_v) * L_v * L_v;

        if (venous_flow == 1)
        {
            n = vessels_in.size();
            const Elem *elem_out = mesh.elem_ptr(n);

            dof_map.dof_indices(elem_out, dof_indices_u, u_var);
            dof_map.dof_indices(elem_out, dof_indices_p, p_var);

            double A2_prime_out = flow_vec[dof_indices_p[1]];
            double A2_out = A2_prime_out * L_v * L_v;
            double p2_out = vessels[n].pext +
                            (vessels[n].beta / (M_PI * vessels[n].r * vessels[n].r)) *
                                (sqrt(A2_out) - sqrt(M_PI * vessels[n].r * vessels[n].r));

            pOutCur = p2_out;
            qOutCur = flow_vec[dof_indices_u[1]] * sqrt(p_0 / rho_v) * L_v * L_v;
        }

        for (int i = 0; i < pArt(0).size(); i++)
        {
            n = termNum[i];
            const Elem *elem = mesh.elem_ptr(n);

            dof_map.dof_indices(elem, dof_indices_u, u_var);
            dof_map.dof_indices(elem, dof_indices_p, p_var);

            double A2_prime = flow_vec[dof_indices_p[1]];
            double A2 = A2_prime * L_v * L_v;
            double p2 = vessels[n].pext +
                        (vessels[n].beta / (M_PI * vessels[n].r * vessels[n].r)) *
                            (sqrt(A2) - sqrt(M_PI * vessels[n].r * vessels[n].r));

            pArtTotal += p2;
            qArtTotal += flow_vec[dof_indices_u[1]] * sqrt(p_0 / rho_v) * L_v * L_v;

            if (venous_flow == 1)
            {
                n = termNum[i] + vessels_in.size();
                const Elem *elem_v = mesh.elem_ptr(n);

                dof_map.dof_indices(elem_v, dof_indices_u, u_var);
                dof_map.dof_indices(elem_v, dof_indices_p, p_var);

                double A2_prime_v = flow_vec[dof_indices_p[1]];
                double A2_v = A2_prime_v * L_v * L_v;
                double p2_v = vessels[n].pext +
                              (vessels[n].beta / (M_PI * vessels[n].r * vessels[n].r)) *
                                  (sqrt(A2_v) - sqrt(M_PI * vessels[n].r * vessels[n].r));

                pVeinTotal += p2_v;
                qVeinTotal += flow_vec[dof_indices_u[1]] * sqrt(p_0 / rho_v) * L_v * L_v;
            }
        }

        pArtTotal /= (double)pArt(0).size();
        pVeinTotal /= (double)pArt(0).size();
    }
    MPI_Bcast(&pInCur, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&qInCur, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&pOutCur, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&qOutCur, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&pArtTotal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pVeinTotal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&qArtTotal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&qVeinTotal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void VesselFlow::initialise_flow_data(EquationSystems &es)
{
    const MeshBase &mesh = es.get_mesh();

    LinearImplicitSystem &system =
        es.get_system<LinearImplicitSystem>("flowSystem");

    System &system_old =
        es.get_system<System>("flowSystemOld");

    System &system_n1 =
        es.get_system<System>("flowSystemn1");

    // Numeric ids corresponding to each variable in the system
    const unsigned int u_var = system.variable_number("QVar");
    const unsigned int p_var = system.variable_number("pVar");

    const DofMap &dof_map = system.get_dof_map();
    std::vector<dof_id_type> dof_indices_u;
    std::vector<dof_id_type> dof_indices_p;

    MeshBase::const_element_iterator el = mesh.active_local_elements_begin();
    const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end();

    for (; el != end_el; ++el)
    {
        const Elem *elem = *el;

        const auto n = elem->id();
        dof_map.dof_indices(elem, dof_indices_u, u_var);
        dof_map.dof_indices(elem, dof_indices_p, p_var);

        system.solution->set(dof_indices_u[0], vessels[n].Q1);
        system.solution->set(dof_indices_u[1], vessels[n].Q2);
        system.solution->set(dof_indices_u[2], vessels[n].Q3);

        system_old.solution->set(dof_indices_u[0], vessels[n].Q1Old);
        system_old.solution->set(dof_indices_u[1], vessels[n].Q2Old);
        system_old.solution->set(dof_indices_u[2], vessels[n].Q3Old);

        system_n1.solution->set(dof_indices_u[0], vessels[n].Q1n1);
        system_n1.solution->set(dof_indices_u[1], vessels[n].Q2n1);
        system_n1.solution->set(dof_indices_u[2], vessels[n].Q3n1);

        system.solution->set(dof_indices_p[0], vessels[n].A1);
        system.solution->set(dof_indices_p[1], vessels[n].A2);

        system_old.solution->set(dof_indices_p[0], vessels[n].A1Old);
        system_old.solution->set(dof_indices_p[1], vessels[n].A2Old);

        system_n1.solution->set(dof_indices_p[0], vessels[n].A1n1);
        system_n1.solution->set(dof_indices_p[1], vessels[n].A2n1);
    }

    system.solution->close();
    system.solution->localize(*system.current_local_solution);

    system_old.solution->close();
    system_old.solution->localize(*system_old.current_local_solution);

    system_n1.solution->close();
    system_n1.solution->localize(*system_n1.current_local_solution);
}

void VesselFlow::read_vessel_restart(int rank, int np, LibMeshInit &init)
{

    for (int rank_i = 0; rank_i < np; rank_i++)
    {
        if (rank == rank_i)
        {
            ifstream file_tree;
            file_tree.open(restart_file_name);

            int vess_counter = 0;
            while (!file_tree.eof())
            {
                file_tree >> vess_i.x1 >> vess_i.y1 >> vess_i.z1 >> vess_i.x2 >>
                    vess_i.y2 >> vess_i.z2 >> vess_i.l >> vess_i.r1 >> vess_i.r2 >>
                    vess_i.r >> vess_i.p >> vess_i.dl >> vess_i.dr >> vess_i.inside >>
                    vess_i.A1 >> vess_i.A2 >> vess_i.A1Old >> vess_i.A2Old >> vess_i.A1n1 >> vess_i.A2n1 >>
                    vess_i.Q1 >> vess_i.Q2 >> vess_i.Q3 >>
                    vess_i.Q1Old >> vess_i.Q2Old >> vess_i.Q3Old >>
                    vess_i.Q1n1 >> vess_i.Q2n1 >> vess_i.Q3n1;

                if (file_tree.eof())
                    break;

                else
                {

                    if (vess_i.dl == -10)
                        vess_i.ter = 1;
                    else if (vess_i.p == -10)
                        vess_i.ter = -1;
                    else
                        vess_i.ter = 0;

                    if (vess_i.dl != -10 && vess_i.dr != -10)
                        vess_i.ter = 2;

                    vessels_in.push_back(vess_i);

                    if (vess_i.p == -10)
                        vess_start = vess_counter;

                    vess_counter++;
                }
            }

            for (int i = 0; i < vessels_in.size(); i++)
            {
                if (vessels_in[i].ter == 2)
                {
                    vessels_in[vessels_in[i].dl].ter = 3;
                    vessels_in[vessels_in[i].dr].ter = 3;
                }
            }

            // for (int i = 0; i < vessels_in.size(); i++) {
            //   cout << "i=" << i << " ter=" << vessels_in[i].ter << endl;
            // }

            file_tree.close();
        }
    }

    MPI_Barrier(init.comm().get());
}

void VesselFlow::write_restart_data(EquationSystems &es, int it, int rank)
{
    string name = "restart_data_";
    string fileend = ".dat";
    string out_frame;
    char numstr[21];
    sprintf(numstr, "%d", it);
    out_frame = name + numstr + fileend;

    string name_1 = "partvein_data_";
    string out_frame_1;
    out_frame_1 = name_1 + numstr + fileend;

    ofstream file_vess;

    const MeshBase &mesh = es.get_mesh();

    LinearImplicitSystem &system =
        es.get_system<LinearImplicitSystem>("flowSystem");

    System &system_old = es.get_system<System>("flowSystemOld");

    System &system_n1 = es.get_system<System>("flowSystemn1");

    // Numeric ids corresponding to each variable in the system
    const unsigned int u_var = system.variable_number("QVar");
    const unsigned int p_var = system.variable_number("pVar");

    NumericVector<double> &flow_data = *(system.current_local_solution);
    flow_data.close();

    NumericVector<double> &flow_data_old = *(system_old.current_local_solution);
    flow_data_old.close();

    NumericVector<double> &flow_data_n1 = *(system_n1.current_local_solution);
    flow_data_n1.close();

    vector<double> flow_vec;
    flow_data.localize(flow_vec);

    vector<double> flow_vec_old;
    flow_data_old.localize(flow_vec_old);

    vector<double> flow_vec_n1;
    flow_data_n1.localize(flow_vec_n1);

    const DofMap &dof_map = system.get_dof_map();
    std::vector<dof_id_type> dof_indices_u;
    std::vector<dof_id_type> dof_indices_p;

    if (rank == 0)
    {
        file_vess.open(out_frame, ios::out);
        MeshBase::const_element_iterator el = mesh.active_elements_begin();
        const MeshBase::const_element_iterator end_el = mesh.active_elements_end();

        for (; el != end_el; ++el)
        {
            const Elem *elem = *el;

            const auto n = elem->id();
            dof_map.dof_indices(elem, dof_indices_u, u_var);
            dof_map.dof_indices(elem, dof_indices_p, p_var);

            double Q1_cur = flow_vec[dof_indices_u[0]];
            double Q2_cur = flow_vec[dof_indices_u[1]];
            double Q3_cur = flow_vec[dof_indices_u[2]];

            double Q1_old = flow_vec_old[dof_indices_u[0]];
            double Q2_old = flow_vec_old[dof_indices_u[1]];
            double Q3_old = flow_vec_old[dof_indices_u[2]];

            double Q1_n1 = flow_vec_n1[dof_indices_u[0]];
            double Q2_n1 = flow_vec_n1[dof_indices_u[1]];
            double Q3_n1 = flow_vec_n1[dof_indices_u[2]];

            double A1_cur = flow_vec[dof_indices_p[0]];
            double A2_cur = flow_vec[dof_indices_p[1]];

            double A1_old = flow_vec_old[dof_indices_p[0]];
            double A2_old = flow_vec_old[dof_indices_p[1]];

            double A1_n1 = flow_vec_n1[dof_indices_p[0]];
            double A2_n1 = flow_vec_n1[dof_indices_p[1]];

            file_vess << setprecision(10) << vessels[n].x1 << " " << vessels[n].y1 << " " << vessels[n].z1 << " "
                      << vessels[n].x2 << " " << vessels[n].y2 << " " << vessels[n].z2 << " "
                      << vessels[n].l << " " << vessels[n].r1 << " " << vessels[n].r2 << " "
                      << vessels[n].r << " " << vessels[n].p << " " << vessels[n].dl << " "
                      << vessels[n].dr << " " << vessels[n].inside << " "
                      << A1_cur << " " << A2_cur << " "
                      << A1_old << " " << A2_old << " "
                      << A1_n1 << " " << A2_n1 << " "
                      << Q1_cur << " " << Q2_cur << " " << Q3_cur << " "
                      << Q1_old << " " << Q2_old << " " << Q3_old << " "
                      << Q1_n1 << " " << Q2_n1 << " " << Q3_n1 << endl;
        }
        file_vess.close();

        ofstream file_partvein;
        file_partvein.open(out_frame_1, ios::out);
        for (int i = 0; i < N_period; i++)
        {
            for (int j = 0; j < pArt(0).size(); j++)
            {
                file_partvein << pArt(i)[j] << " " << pVein(i)[j] << endl;
            }
            file_partvein << endl;
        }
        file_partvein.close();
    }
}