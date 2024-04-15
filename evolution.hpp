//
// Created by polya on 4/2/24.
//

#ifndef OS_DCE_CPP_EVOLUTION_HPP
#define OS_DCE_CPP_EVOLUTION_HPP
#include <string>
#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <cstdio>
#include <regex>
#include <array>
#include <boost/filesystem.hpp>
#include <armadillo>
#include <fftw3.h>
#include <boost/json.hpp>
#include <fstream>

namespace fs = boost::filesystem;
using namespace std::complex_literals;
const auto PI=std::numbers::pi;

//This subroutine computes evolution using operator splitting and particle number
// and verifies the solution
class os_DCE_Evolution {


public:
    /// This constructor initializes all parameters
    /// @param group group number of parameters
    /// @param row row number of parameters
    os_DCE_Evolution(const int &group, const int &row) {
        //
        this->groupNum = group;
        this->rowNum = row;

        this->parseCSV(group, row);
        for (int n1 =0;n1<N1;n1++){
            this->x1ValsAll.push_back(-L1+dx1*n1);
        }
        for (int n2=0;n2<N2;n2++){
            this->x2ValsAll.push_back(-L2+dx2*n2);
        }

        for(const auto& val: x1ValsAll){
            x1ValsAllSquared.push_back(std::pow(val,2));
        }
        for(const auto &val:x2ValsAll){
            x2ValsAllSquared.push_back(std::pow(val,2));
        }

        for(int n1=0;n1<static_cast<int>(N1/2);n1++){
            k1ValsAll.push_back(2*PI*static_cast<double >(n1)/(2.0*L1));
        }
        for(int n1=static_cast<int>(N1/2);n1<N1;n1++){
            k1ValsAll.push_back(2*PI*static_cast<double >(n1-N1)/(2.0*L1));
        }

        for(const auto&val: k1ValsAll){
            k1ValsAllSquared.push_back(std::pow(val,2));
        }

        for(int n2=0;n2<static_cast<int>(N2/2);n2++){
            k2ValsAll.push_back(2*PI*static_cast<double >(n2)/(2.0*L2));
        }
        for(int n2=static_cast<int >(N2/2);n2<N2;n2++){
            k2ValsAll.push_back(2*PI*static_cast<double >(n2-N2)/(2.0*L2));
        }
        for(const auto &val:k2ValsAll){
            k2ValsAllSquared.push_back(std::pow(val,2));
        }


        for(int fls=0;fls<flushNum;fls++){
            int startingInd=fls*stepsPerFlush;
            for(int j=0;j<stepsPerFlush;j++){
                double indTmp=static_cast<double >(startingInd+j);
                this->timeValsAll.push_back(indTmp*dt);
            }
        }

        arma::cx_drowvec vec_x2Row(N2);
        for (int n2=0;n2<N2;n2++){
            vec_x2Row(n2)=x2ValsAll[n2];
        }
        arma::cx_dcolvec vec_x1SquaredCol(N1);
        for(int n1=0;n1<N1;n1++){
            vec_x1SquaredCol(n1)=x1ValsAllSquared[n1];
        }
        this->U14Exp=arma::kron(vec_x1SquaredCol,vec_x2Row);

        this->r=std::log(er);

        this->k2Row=arma::cx_drowvec(N2);
        for(int n2=0;n2<N2;n2++){
            k2Row(n2)=k2ValsAll[n2];
        }

        E1=0.5*omegac*(2.0*static_cast<double >(jH1)+1.0);

        E2=Deltam/(std::cosh(2.0*r))*(static_cast<double >(jH2)+0.5);


        Y=new std::complex<double>[N1*N2];
        Z=new std::complex<double>[N1*N2];
        W=new std::complex<double>[N1*N2];
        psiTmp=new std::complex<double>[N1*N2];
        MArray=new std::complex<double>[N1*N2];

        //row fft
        int rowfft_rank=1;
        int rowfft_n[]={N2};
        int rowfft_howmany=N1;
        int rowfft_istride=1;
        int rowfft_ostride=1;
        int rowfft_idist=N2;
        int rowfft_odist=N2;
        int *rowfft_inembed = rowfft_n, *rowfft_onembed = rowfft_n;
        //psi and Z
        plan_psi2Z=fftw_plan_many_dft(rowfft_rank,rowfft_n,rowfft_howmany,
                                                reinterpret_cast<fftw_complex*>(psiTmp),rowfft_inembed,
                                                rowfft_istride,rowfft_idist,reinterpret_cast<fftw_complex*>(Z),
                                                rowfft_onembed,rowfft_ostride,rowfft_odist,FFTW_FORWARD,FFTW_MEASURE);

        plan_Z2psi=fftw_plan_many_dft(rowfft_rank,rowfft_n,rowfft_howmany,
                                                reinterpret_cast<fftw_complex*>(Z),rowfft_inembed,
                                                rowfft_istride,rowfft_idist,reinterpret_cast<fftw_complex*>(psiTmp),
                                                rowfft_onembed,rowfft_ostride,rowfft_odist,FFTW_BACKWARD,FFTW_MEASURE);

        //psi and W
        plan_psi2W=fftw_plan_many_dft(rowfft_rank,rowfft_n,rowfft_howmany,
                                                reinterpret_cast<fftw_complex*>(psiTmp),rowfft_inembed,
                                                rowfft_istride,rowfft_idist,reinterpret_cast<fftw_complex*>(W),
                                                rowfft_onembed,rowfft_ostride,rowfft_odist,FFTW_FORWARD,FFTW_MEASURE);


        plan_W2psi=fftw_plan_many_dft(rowfft_rank,rowfft_n,rowfft_howmany,
                                                reinterpret_cast<fftw_complex*>(W),rowfft_inembed,
                                                rowfft_istride,rowfft_idist,reinterpret_cast<fftw_complex*>(psiTmp),
                                                rowfft_onembed,rowfft_ostride,rowfft_odist,FFTW_BACKWARD,FFTW_MEASURE);



        //col fft
        //psi and Y
        int colfft_rank=1;
        int colfft_n[]={N1};
        int colfft_howmany=N2;
        int colfft_idist=1;
        int colfft_odist=1;
        int colfft_istride=N2;
        int colfft_ostride=N2;
        int *colfft_inembed = colfft_n, *colfft_onembed = colfft_n;

        plan_psi2Y= fftw_plan_many_dft(colfft_rank,colfft_n,colfft_howmany,
                                                 reinterpret_cast<fftw_complex*>(psiTmp),colfft_inembed,
                                                 colfft_istride,colfft_idist,reinterpret_cast<fftw_complex*>(Y),
                                                 colfft_onembed,colfft_ostride,colfft_odist,FFTW_FORWARD,FFTW_MEASURE);

        plan_Y2psi=fftw_plan_many_dft(colfft_rank,colfft_n,colfft_howmany,
                                                reinterpret_cast<fftw_complex*>(Y),colfft_inembed,
                                                colfft_istride,colfft_idist,reinterpret_cast<fftw_complex*>(psiTmp),
                                                colfft_onembed,colfft_ostride,colfft_odist,FFTW_BACKWARD,FFTW_MEASURE);





        this->outDir="./groupNew"+std::to_string(groupNum)+"/row"+std::to_string(rowNum)+"/";
        if(!fs::is_directory(outDir) || !fs::exists(outDir)){
            fs::create_directories(outDir);
        }

    }//end of constructor

    ~os_DCE_Evolution(){

        delete[] Y;
        delete[]Z;
        delete[]W;
        delete[]psiTmp;
        delete[]MArray;
        fftw_destroy_plan(plan_psi2Y);
        fftw_destroy_plan(plan_Y2psi);
        fftw_destroy_plan(plan_psi2Z);
        fftw_destroy_plan(plan_Z2psi);
        fftw_destroy_plan(plan_psi2W);
        fftw_destroy_plan(plan_W2psi);

    }




public:
    int jH1 = -1;
    int jH2 = -1;
    double g0 = 0;
    double omegam = 0;
    double omegap=0;
    double omegac = 0;
    double er = 0;
    double thetaCoef = 0;
    int groupNum = -1;
    int rowNum = -1;
    double theta=0;
    double lmd=0;
    double Deltam=0;
    double r=0;
    double e2r=0;

    int N1=100;
    int N2=500;

    double L1=1;
    double L2=5;
    double dx1=0;
    double dx2=0;

    double dtEst=0.0005;
    double tFlushStart=0;
    double tFlushStop=0.1;
    double tTotPerFlush=tFlushStop-tFlushStart;
    int flushNum=3;
    int stepsPerFlush=static_cast<int>(std::ceil(tTotPerFlush/dtEst));
    double dt=tTotPerFlush/static_cast<double >(stepsPerFlush);
    std::vector<double> timeValsAll;


    std::vector<double> x1ValsAll;
    std::vector<double> x2ValsAll;
    std::vector<double> k1ValsAll;
    std::vector<double> k2ValsAll;
    std::vector<double> x1ValsAllSquared;
    std::vector<double> x2ValsAllSquared;
    std::vector<double> k1ValsAllSquared;
    std::vector<double> k2ValsAllSquared;

    ///the following values Y,Z,W, psiTmp are pointers used in fft
    std::complex<double> *Y;
    std::complex<double> *Z;
    std::complex<double> *W;
    std::complex<double> *psiTmp;
    std::complex<double> *MArray;
    fftw_plan plan_psi2Y;
    fftw_plan plan_Y2psi;
    fftw_plan plan_psi2Z;
    fftw_plan plan_Z2psi;
    fftw_plan plan_psi2W;
    fftw_plan plan_W2psi;



//    arma::cx_dvec psi0;
    arma::cx_dmat psi0;
    arma::cx_dmat U14Exp;
    arma::cx_drowvec k2Row;
    arma::cx_dmat psiSpace;

    //matrices for computing particle numbers
    arma::sp_cx_dmat H6;
    arma::sp_cx_dmat NcMat1;
    arma::sp_cx_dmat NmPart1;
    arma::sp_cx_dmat NmPart2;

    std::string outDir;
     double E1=0;
     double E2=0;
public:

    /// @param group group number
    /// @param row row number
    ///parse csv with group number group
    void parseCSV(const int &group, const int &row);

    ///
    /// @param cmd python execution string
    /// @return signal from the python
    static std::string execPython(const char *cmd);

    template<class T>
    static void printVec(const std::vector <T> &vec) {
        for (int i = 0; i < vec.size() - 1; i++) {
            std::cout << vec[i] << ",";
        }
        std::cout << vec[vec.size() - 1] << std::endl;
    }

    ///initialize wavefunction serially
    void initPsiSerial();



    ///
    /// @param n1 index of x1
    /// @return wavefunction of photon at n1
    double f1(int n1);

    ///
    /// @param n2 index of x2
    /// @return wavefunction of phonon at n2
    double f2(int n2);

    ///
    /// @param n1 index for x1n1
    /// @param t time
    /// @return coefficient for evolution using H3
    double f(int n1, double t);

    ///
    /// @param j time step
    /// @param psi wavefunction at the beginning of the time step j
    /// @return
    arma::cx_dmat evolution1Step(const int&j, const arma::cx_dmat& psi);

    ///initialize matrices for computing particle numbers
    void popolateMatrices();

    ///
    /// @param psi wavefunction
    /// @return photon number
    double avgNc(const arma::cx_dmat& psi);

    ///
    /// @param psi wavefunction
    /// @return phonon number
    double avgNm(const arma::cx_dmat& psi);


    ///
    /// @param psiIn starting value of the wavefunction in one flush
    /// @param fls flush number
    /// @return starting value of the wavefunction in the next flush
    arma::cx_dmat oneFlush(const arma::cx_dmat& psiIn, const int& fls);

    ///evolution of wavefunctuion
    void evolution();

    ///
    /// @param j time ind
    /// @return analytical solution for g0=0
    arma::cx_dmat  psit(const int &j);

    double funcf(int n1);

    double funcg(int n2);

};


#endif //OS_DCE_CPP_EVOLUTION_HPP
