#include "Admittance.h"

using namespace libMesh;
using namespace std;

int Admittance::alphamax, Admittance::betamax, Admittance::count, Admittance::endbranch;

Admittance::Admittance() {}

Admittance::~Admittance() {}

void Admittance::compute_impedance(int tmstps, double Per, double rho, double mu, double r_root, double r_min,
                                   DenseVector<double> &y11, DenseVector<double> &y12, DenseVector<double> &y21, DenseVector<double> &y22,
                                   double Lr, double q, double g, double fa1, double fa2, double fa3, double fv1, double fv2, double fv3,
                                   double asym, double expo, double lrrA, double lrrV)
{

    double df;
    DenseVector<double> Freq(tmstps + 1), Omega(tmstps + 1);
    vector<DenseMatrix<double>> Y, Y_i;
    DenseMatrix<double> tempY(2, 2), tempY_i(2, 2);
    vector<complex<double>> YHAT11(tmstps), YHAT12(tmstps), YHAT21(tmstps), YHAT22(tmstps);

    Y.resize(tmstps);
    Y_i.resize(tmstps);

    df = 1.0 / Per; //                               ! Frequency interval.
    for (int j = 0; j < tmstps + 1; j++)
    {
        Freq(j) = ((double)j - 0.5 * (double)tmstps) * df;
        Omega(j) = 2.0 * M_PI * Freq(j);
    }

    alphamax = 0;
    betamax = 0;
    count = 0;
    endbranch = 0;

    for (int j = (tmstps / 2); j < tmstps + 1; j++)
    {
        compute_admittance(0, 0, Omega(j), fa1, fa2, fa3, fv1, fv2, fv3, rho, mu, r_root, r_min, Lr, q, g, asym, expo,
                           lrrA, lrrV, Y[j - 1], Y_i[j - 1]);
    }

    tempY.zero();
    tempY.add(1.0, Y[(tmstps / 2) - 1]);

    tempY_i.zero();
    tempY_i.add(1.0, Y_i[(tmstps / 2) - 1]);

    for (int j = 0; j < tmstps / 2; j++)
    {
        Y[j].resize(2, 2);
        Y[j].add(1.0, Y[tmstps - 1 - j]);

        Y_i[j].resize(2, 2);
        Y_i[j].add(-1.0, Y_i[tmstps - 1 - j]);
    }

    Y[0](0, 0) = Y[tmstps - 1](0, 0);
    Y[0](0, 1) = Y[tmstps - 1](0, 1);
    Y[0](1, 0) = Y[tmstps - 1](1, 0);
    Y[0](1, 1) = Y[tmstps - 1](1, 1);

    for (int j = (tmstps / 2) + 1; j < tmstps; j++)
    {
        Y[j].resize(2, 2);
        Y[j].add(1.0, Y[j - 1]);

        Y_i[j].resize(2, 2);
        Y_i[j].add(1.0, Y_i[j - 1]);
    }

    Y[tmstps / 2].resize(2, 2);
    Y[tmstps / 2].add(1.0, tempY);

    Y_i[tmstps / 2].resize(2, 2);
    Y_i[tmstps / 2].add(1.0, tempY_i);

    for (int j = 0; j < tmstps; j++)
    {
        double real_m = Y[j](0, 0);
        double imag_m = Y_i[j](0, 0);
        complex<double> num_i = 0.0 + 1i;
        complex<double> comp_m = real_m + imag_m * num_i;

        YHAT11[j] = comp_m * (1.0 / Per);

        real_m = Y[j](0, 1);
        imag_m = Y_i[j](0, 1);
        comp_m = real_m + imag_m * num_i;
        YHAT12[j] = comp_m * (1.0 / Per);

        real_m = Y[j](1, 0);
        imag_m = Y_i[j](1, 0);
        comp_m = real_m + imag_m * num_i;
        YHAT21[j] = comp_m * (1.0 / Per);

        real_m = Y[j](1, 1);
        imag_m = Y_i[j](1, 1);
        comp_m = real_m + imag_m * num_i;
        YHAT22[j] = comp_m * (1.0 / Per);
    }

    /* for (int j = 0; j < tmstps; j++)
    {
        double y11_j = 0;
        double y12_j = 0;
        double y21_j = 0;
        double y22_j = 0;
        for (int k = 0; k < tmstps; k++)
        {
            int theta = Omega(k)*j*(0.8/32768);
            y11_j = y11_j + Y[j](0,0) * cos(theta) + Y_i[j](0,0) * sin(theta);
            y12_j = y12_j + Y[j](0,1) * cos(theta) + Y_i[j](0,1) * sin(theta);
            y21_j = y21_j + Y[j](1,0) * cos(theta) + Y_i[j](1,0) * sin(theta);
            y22_j = y22_j + Y[j](1,1) * cos(theta) + Y_i[j](1,1) * sin(theta);
        }
        y11(j) = y11_j / tmstps;
        y12(j) = y12_j / tmstps;
        y21(j) = y21_j / tmstps;
        y22(j) = y22_j / tmstps;
        
    } */

    cout<<"t="<<tmstps/2<<" "<<YHAT11[tmstps/2]<<" "<<YHAT12[tmstps/2]<<" "<<YHAT21[tmstps/2]<<" "<<YHAT22[tmstps/2]<<endl;

    fftshift(YHAT11);
    fftshift(YHAT12);
    fftshift(YHAT21);
    fftshift(YHAT22);

    cout<<"AFTER FFT SHIFT"<<endl;
    cout<<"t="<<tmstps/2<<" "<<YHAT11[tmstps/2]<<" "<<YHAT12[tmstps/2]<<" "<<YHAT21[tmstps/2]<<" "<<YHAT22[tmstps/2]<<endl;

    bitreverse(YHAT11);
    bitreverse(YHAT12);
    bitreverse(YHAT21);
    bitreverse(YHAT22);

    cout<<"AFTER BITREVERSE"<<endl;
    cout<<"t="<<tmstps/2<<" "<<YHAT11[tmstps/2]<<" "<<YHAT12[tmstps/2]<<" "<<YHAT21[tmstps/2]<<" "<<YHAT22[tmstps/2]<<endl;

    IFFT(YHAT11);
    IFFT(YHAT12);
    IFFT(YHAT21);
    IFFT(YHAT22);

    cout<<"AFTER IFFT"<<endl;
    cout<<"t="<<tmstps/2<<" "<<YHAT11[tmstps/2]<<" "<<YHAT12[tmstps/2]<<" "<<YHAT21[tmstps/2]<<" "<<YHAT22[tmstps/2]<<endl;

    for (int j = 0; j < YHAT11.size(); j++)
    {
        y11(j) = real(YHAT11[j])/tmstps;
        y12(j) = real(YHAT12[j])/tmstps;
        y21(j) = real(YHAT21[j])/tmstps;
        y22(j) = real(YHAT22[j])/tmstps;
    }

    cout << "EVERYTHING IS FINE" << endl;
    cout << "t=" << 0 << y11(0) << " " << y12(0) << " " << y21(0) << " " << y22(0) << endl;
}

void Admittance::fftshift(vector<complex<double>> &data)
{
    int k = 0;
    int count = data.size();
    int c = (int)floor((float)count / 2);
    // For odd and for even numbers of element use different algorithm
    if (count % 2 == 0)
    {
        complex<double> dummy1;
        for (k = 0; k < c; k++)
        {
            dummy1 = data[k];
            data[k] = data[k + c];
            data[k + c] = dummy1;
        }
    }
    else
    {
        complex<double> tmp = data[0];
        for (k = 0; k < c; k++)
        {
            data[k] = data[c + k + 1];
            data[c + k + 1] = data[k + 1];
        }
        data[c] = tmp;
    }
}

void Admittance::bitreverse(vector<complex<double>> &X)
{

    int N = X.size();
    int j = 0, k;

    vector<complex<double>> Y(N);

    for (int l = 0; l < N - 1; l++)
    {
        Y[j] = X[l];
        k = (N / 2) - 1;

        while (k < j)
        {
            j = j - k - 1;
            k = (k - 1) / 2;
        }

        j = j + k + 1;
    }
    Y[N - 1] = X[N - 1];

    for (int j = 0; j < N; j++)
    {
        X[j] = Y[j];
    }
}

void Admittance::IFFT(vector<complex<double>> &Zhat)
{
    int N = Zhat.size();
    complex<double> u, w, t;
    vector<complex<double>> Z(N);

    int le, le1, i, j, ip;

    int M = log(double(N)) / log(2.0);

    // cout<<"Z0="<<Zhat[0]<<endl;

    for (int ii = 0; ii < N; ii++)
    {
        Z[ii] = Zhat[ii];
    }

    for (int l = 0; l < M; l++)
    {
        le = pow(2, l + 1) - 1;
        le1 = (le - 1) / 2;
        u = 1.0 + 0i;
        complex<double> num_i = 0.0 + 1i;
        w = cos(M_PI / (le1 + 1)) + sin(M_PI / (le1 + 1)) * num_i;

        cout << "le1=" << le1 << endl;

        for (int j = 0; j < le1 + 1; j++)
        {
            for (int i = j; i < N; i = i + le + 1)
            {
                ip = i + le1 + 1;

                // cout<<"j="<<j<<" i="<<i<<" ip="<<ip<<endl;

                if (i == 0)
                    cout << "l=" << l << " j=" << j << " i=" << i << " Z=" << Z[i] << " ip=" << ip << " Zip=" << Z[ip] << endl;

                t = Z[ip] * u;
                Z[ip] = Z[i] - t;
                Z[i] = Z[i] + t;
            }
            u = u * w;
        }
    }

    for (int j = 0; j < N; j++)
    {
        Zhat[j] = Z[j];
    }
    // cout<<"AFTER IFFT INSIDE"<<endl;
    // cout<<"Z0after="<<Zhat[0]<<endl;
}

void Admittance::compute_admittance(int n, int m, double omega_k, double fa1, double fa2, double fa3, double fv1,
                                    double fv2, double fv3, double rho, double mu, double r_root, double r_min, double Lr, double q, double g,
                                    double asym, double expo, double lrr_A, double lrr_V, DenseMatrix<double> &Y, DenseMatrix<double> &Y_i)
{

    DenseMatrix<double> YA(2, 2), YV(2, 2), Ymid(2, 2), d1(2, 2), d2(2, 2);
    DenseMatrix<double> YA_i(2, 2), YV_i(2, 2), Ymid_i(2, 2), d1_i(2, 2), d2_i(2, 2);
    /* complex<double> YA[2][2],YV[2][2],Ymid[2][2], d1[2][2],d2[2][2]; */
    complex<double> kappa_A, kappa_V, Fk_A, Fk_V, g_omega_A, c_omega_A, g_omega_V, c_omega_V;
    double alpha, beta, nu, r, r_d, l_A, l_V, A, A_d, D_A, D_V, wom, F0_A, F0_V;

    count = count + 1;

    Y.resize(2, 2);
    Y_i.resize(2, 2);

    alpha = pow(pow(pow(asym, expo / 2) + 1.0, -1 / expo), n);
    beta = pow(sqrt(asym) * pow(pow(asym, expo / 2) + 1.0, -1 / expo), m);

    if (n >= alphamax)
        alphamax = n;

    if (m >= betamax)
        betamax = m;

    r_d = alpha * beta * r_root; //     ! Radius at root, cm.
    A_d = M_PI * pow(r_d, 2);    //            ! Cross-sectional area, cm^2.
    r = r_d;                     //                ! Radius at root, dimension-less.
    A = A_d;                     //

    if (r < 0.005)
        l_A = 1.88 * (pow(r, 0.47));
    else
        l_A = lrr_A * pow(r, 1.10); //     ! Length of arterial vessel segment.

    l_V = lrr_V * r;                                      //                           ! Length of venous vessel segment.
    nu = mu / rho;                                        //                            ! Kinematic blood viscosity,cm^2/s.
    D_A = 1 / (fa1 * exp(fa2 * r_d) + fa3) * 3 * A_d / 2; //  ! Distensibility.
    D_V = 1 / (fv1 * exp(fv2 * r_d) + fv3) * 3 * A_d / 2; //
    wom = r_d * sqrt(omega_k / nu);                       //              ! Womersleys parameter.

    complex<double> num_i, num1_i, num2_i, denom1_i;

    num_i = 0.0 + 1i;

    num1_i = 1.0 + 0i;
    num2_i = 2.0 + 0i;
    denom1_i = wom * (1.0 + 1.0 / 2.0 / wom) + 0i;

    if (wom > 3.0)
    {
        complex<double> const_i = sqrt(1.0 - 2.0 / sqrt(num_i) / wom * (1.0 + 1.0 / 2.0 / wom));

        g_omega_A = sqrt(D_A * A / rho) * const_i;

        c_omega_A = sqrt(A / D_A / rho) * const_i;

        g_omega_V = sqrt(D_V * A / rho) * const_i;

        c_omega_V = sqrt(A / D_V / rho) * const_i;
    }

    else if (wom > 2.0)
    {
        complex<double> const_i = (3.0 - wom) * sqrt(num_i * pow(wom, 2) / 8.0 + pow(wom, 4) / 48.0) +
                                  (wom - 2.0) * sqrt(1.0 - 2.0 / sqrt(num_i) / wom * (1.0 + 1.0 / 2.0 / wom));

        g_omega_A = sqrt(D_A * A / rho) * const_i;
        c_omega_A = sqrt(A / D_A / rho) * const_i;

        g_omega_V = sqrt(D_V * A / rho) * const_i;

        c_omega_V = sqrt(A / D_V / rho) * const_i;
    }

    else if (wom == 0)
    {
        g_omega_A = 0.0;
        c_omega_A = 0.0;
        g_omega_V = 0.0;
        c_omega_V = 0.0;
    }

    else
    {
        complex<double> const_i = sqrt(num_i * pow(wom, 2) / 8 + pow(wom, 4) / 48);

        g_omega_A = sqrt(D_A * A / rho) * const_i;
        c_omega_A = sqrt(A / D_A / rho) * const_i;
        g_omega_V = sqrt(D_V * A / rho) * const_i;
        c_omega_V = sqrt(A / D_V / rho) * const_i;
    }

    if (omega_k != 0)
    {
        kappa_A = omega_k * l_A / c_omega_A;
        kappa_V = omega_k * l_V / c_omega_V;
    }
    else
    {
        kappa_A = 0.0;
        kappa_V = 0.0;
    }

    if (kappa_A == 0)
    {
        F0_A = (M_PI * rho * g * Lr * pow(r, 4)) / (8 * mu * l_A * q);
        YA(0, 0) = real(F0_A);
        YA(1, 0) = real(-F0_A);
        YA(0, 1) = real(-F0_A);
        YA(1, 1) = real(F0_A);

        YA_i(0, 0) = imag(F0_A);
        YA_i(1, 0) = imag(-F0_A);
        YA_i(0, 1) = imag(-F0_A);
        YA_i(1, 1) = imag(F0_A);
    }
    else
    {
        Fk_A = num_i * g_omega_A * rho * g * Lr / (q * sin(kappa_A));
        YA(0, 0) = real(-Fk_A * cos(kappa_A));
        YA(1, 0) = real(Fk_A);
        YA(0, 1) = real(Fk_A);
        YA(1, 1) = real(-Fk_A * cos(kappa_A));

        YA_i(0, 0) = imag(-Fk_A * cos(kappa_A));
        YA_i(1, 0) = imag(Fk_A);
        YA_i(0, 1) = imag(Fk_A);
        YA_i(1, 1) = imag(-Fk_A * cos(kappa_A));
    }

    if (kappa_V == 0)
    {
        F0_V = (M_PI * rho * g * Lr * pow(r, 4)) / (8 * mu * l_V * q);
        YV(0, 0) = real(F0_V);
        YV(1, 0) = real(-F0_V);
        YV(0, 1) = real(-F0_V);
        YV(1, 1) = real(F0_V);

        YV_i(0, 0) = imag(F0_V);
        YV_i(1, 0) = imag(-F0_V);
        YV_i(0, 1) = imag(-F0_V);
        YV_i(1, 1) = imag(F0_V);
    }
    else
    {
        Fk_V = num_i * g_omega_V * rho * g * Lr / (q * sin(kappa_V));
        YV(0, 0) = real(-Fk_V * cos(kappa_V));
        YV(1, 0) = real(Fk_V);
        YV(0, 1) = real(Fk_V);
        YV(1, 1) = real(-Fk_V * cos(kappa_V));

        YV_i(0, 0) = imag(-Fk_V * cos(kappa_V));
        YV_i(1, 0) = imag(Fk_V);
        YV_i(0, 1) = imag(Fk_V);
        YV_i(1, 1) = imag(-Fk_V * cos(kappa_V));
    }


    if (r < r_min)
    {
        endbranch = endbranch + 1;
        series(YA, YA_i, YV, YV_i, Y, Y_i);
    }
    else
    {
        compute_admittance(n + 1, m, omega_k, fa1, fa2, fa3, fv1, fv2, fv3, rho, mu, r_root, r_min, Lr, q, g, asym, expo, lrr_A, lrr_V, d1, d1_i);
        compute_admittance(n, m + 1, omega_k, fa1, fa2, fa3, fv1, fv2, fv3, rho, mu, r_root, r_min, Lr, q, g, asym, expo, lrr_A, lrr_V, d2, d2_i);

        Ymid.zero();
        Ymid.add(1.0, d1);
        Ymid.add(1.0, d2);

        Ymid_i.zero();
        Ymid_i.add(1.0, d1_i);
        Ymid_i.add(1.0, d2_i);

        DenseMatrix<double> Yinter(2, 2), Yinter_i(2, 2);
        series(YA, YA_i, Ymid, Ymid_i, Yinter, Yinter_i);
        series(Yinter, Yinter_i, YV, YV_i, Y, Y_i);
    }
}

void Admittance::series(DenseMatrix<double> &YA, DenseMatrix<double> &YA_i,
                        DenseMatrix<double> &YB, DenseMatrix<double> &YB_i,
                        DenseMatrix<double> &Y, DenseMatrix<double> &Y_i)
{

    complex<double> YA_m[2][2], YB_m[2][2], DA, DB, E;

    for (int j = 0; j < 2; j++)
    {
        for (int k = 0; k < 2; k++)
        {
            complex<double> num_i = 0.0 + 1i;

            double real_m, imag_m;
            real_m = YA(j, k);
            imag_m = YA_i(j, k);

            complex<double> comp_m = real_m + imag_m * num_i;
            YA_m[j][k] = comp_m;

            real_m = YB(j, k);
            imag_m = YB_i(j, k);
            comp_m = real_m + imag_m * num_i;
            YB_m[j][k] = comp_m;
        }
    }

    DA = YA_m[0][0] * YA_m[1][1] - YA_m[0][1] * YA_m[1][0];
    DB = YB_m[0][0] * YB_m[1][1] - YB_m[0][1] * YB_m[1][0];

    E = YA_m[1][1] + YB_m[0][0];

    Y(0, 0) = real((DA + YA_m[0][0] * YB_m[0][0]) / E);
    Y(0, 1) = real(-(YA_m[0][1] * YB_m[0][1]) / E);
    Y(1, 0) = real(-(YA_m[1][0] * YB_m[1][0]) / E);
    Y(1, 1) = real((DB + YA_m[1][1] * YB_m[1][1]) / E);

    Y_i(0, 0) = imag((DA + YA_m[0][0] * YB_m[0][0]) / E);
    Y_i(0, 1) = imag(-(YA_m[0][1] * YB_m[0][1]) / E);
    Y_i(1, 0) = imag(-(YA_m[1][0] * YB_m[1][0]) / E);
    Y_i(1, 1) = imag((DB + YA_m[1][1] * YB_m[1][1]) / E);
}