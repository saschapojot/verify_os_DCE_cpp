//
// Created by polya on 4/2/24.
//

#include "evolution.hpp"

/// @param group group number
/// @param row row number
///parse csv with group number group
void os_DCE_Evolution::parseCSV(const int &group, const int &row){
    std::string commandToReadCSV="python3 readCSV.py "+std::to_string(group)+" "+std::to_string(row);

    std::string result=this->execPython(commandToReadCSV.c_str());
//    std::cout<<result<<std::endl;

    std::regex pattern_j1H("j1H(\\d+)j2H");
    std::smatch  match_j1H;
    if (std::regex_search(result,match_j1H,pattern_j1H)){
        this->jH1=std::stoi(match_j1H[1].str());
    }

    std::regex pattern_j2H("j2H(\\d+)g0");
    std::smatch match_j2H;
    if (std::regex_search(result,match_j2H,pattern_j2H)){
        this->jH2=std::stoi(match_j2H[1].str());
    }

    std::regex pattern_g0("g0([+-]?\\d+(\\.\\d+)?)omegam");
    std::smatch match_g0;
    if (std::regex_search(result,match_g0,pattern_g0)){
        this->g0=std::stod(match_g0[1].str());
    }


    std::regex pattern_omegam("omegam([+-]?\\d+(\\.\\d+)?)omegap");
    std::smatch match_omegam;
    if (std::regex_search(result,match_omegam,pattern_omegam)){
        this->omegam=std::stod(match_omegam[1].str());
    }


    std::regex pattern_omegap("omegap([+-]?\\d+(\\.\\d+)?)omegac");
    std::smatch match_omegap;
    if (std::regex_search(result,match_omegap,pattern_omegap)){
        this->omegap=std::stod(match_omegap[1].str());
    }

    std::regex pattern_omegac("omegac([+-]?\\d+(\\.\\d+)?)er");
    std::smatch match_omegac;
    if (std::regex_search(result,match_omegac,pattern_omegac)){
        this->omegac=std::stod(match_omegac[1].str());
    }

    std::regex pattern_er("er([+-]?\\d+(\\.\\d+)?)thetaCoef");
    std::smatch match_er;
    if(std::regex_search(result,match_er,pattern_er)){
        this->er=std::stod(match_er[1].str());
    }

    std::regex pattern_thetaCoef("thetaCoef([+-]?\\d+(\\.\\d+)?)");
    std::smatch  match_thetaCoef;
    if (std::regex_search(result,match_thetaCoef,pattern_thetaCoef)){
        this->thetaCoef=std::stod(match_thetaCoef[1].str());
    }
//    std::cout<<"jH1="<<jH1<<std::endl;
//
//    std::cout<<"jH2="<<jH2<<std::endl;
//
//    std::cout<<"g0="<<g0<<std::endl;
//
//    std::cout<<"omegam="<<omegam<<std::endl;
//
//    std::cout<<"omegac="<<omegac<<std::endl;
//
//    std::cout<<"omegap="<<omegap<<std::endl;
//
//    std::cout<<"er="<<er<<std::endl;
//
//    std::cout<<"thetaCoef="<<thetaCoef<<std::endl;
    e2r=std::pow(er,2);
    r=std::log(er);
    double eM2r=1/e2r;
    this->Deltam=this->omegam-this->omegap;
    this->lmd=(e2r-eM2r)/(e2r+eM2r)*Deltam;
    this->theta=thetaCoef*PI;
//      std::cout<<"lambda="<<lmd<<std::endl;
//      std::cout<<"theta="<<theta<<std::endl;
//      std::cout<<"Deltam="<<Deltam<<std::endl;
    double height1=0.5;

    double width1=std::sqrt(-2.0*std::log(height1)/omegac);
//    std::cout<<"width1="<<std::to_string(width1)<<std::endl;
    double minGrid1=width1/20.0;
//    std::cout<<"minGrid1="<<std::to_string(minGrid1)<<std::endl;
    this->N1=static_cast<int>(std::ceil(L1*2/minGrid1));
    if(N1%2==1){
        N1+=1;//make sure N1 is even
    }
//    if(N1<9000){
//        N1=9000;
//    }
//for inParamsNew6, N1=6000
//for inParamsNew7, N1=9000

    std::cout<<"N1="<<N1<<std::endl;
    dx1=2*L1/(static_cast<double>(N1));
    dx2=2*L2/(static_cast<double >(N2));
//    std::cout<<"dt="<<dt<<std::endl;
}

///
/// @param cmd python execution string
/// @return signal from the python
 std::string os_DCE_Evolution::execPython(const char *cmd){
    std::array<char, 4096> buffer; // Buffer to store command output
    std::string result; // String to accumulate output

    // Open a pipe to read the output of the executed command
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }

    // Read the output a chunk at a time and append it to the result string
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }

    return result; // Return the accumulated output


}




///
/// @param n1 index of x1
/// @return wavefunction of photon at n1
double os_DCE_Evolution::f1(int n1){
    double x1TmpSquared=x1ValsAllSquared[n1];
    double x1Tmp=x1ValsAll[n1];

    double valTmp = std::exp(-0.5 * omegac * x1TmpSquared)
                    * std::hermite(this->jH1, std::sqrt(omegac) * x1Tmp);


    return valTmp;


}


///
/// @param n2 index of x2
/// @return wavefunction of phonon at n2
double os_DCE_Evolution::f2(int n2){
    double x2TmpSquared=x2ValsAllSquared[n2];
    double x2Tmp=x2ValsAll[n2];

    double valTmp=std::exp(-0.5 * omegam*std::exp(-2.0*r) * x2TmpSquared)
                  *std::hermite(this->jH2,std::sqrt(omegam*std::exp(-2.0*r))*x2Tmp);

    return valTmp;

}


///initialize wavefunction serially
void os_DCE_Evolution::initPsiSerial(){

    arma::cx_dcolvec vec1(N1);
    arma::cx_drowvec vec2(N2);
    for(int n1=0;n1<N1;n1++){
        vec1(n1)= f1(n1);
    }
    for(int n2=0;n2<N2;n2++){
        vec2(n2)= f2(n2);
    }
    this->psi0=arma::kron(vec1,vec2);
    this->psi0/=arma::norm(psi0,2);
//    std::cout<<"finish init"<<std::endl;





//    printVec(v2);
    this->psiSpace=arma::kron(vec1,vec2);
//    std::cout<<psiSpace<<std::endl;
//    std::cout<<"psiSpace norm="<<arma::norm(psiSpace,2)<<std::endl;
    psiSpace/=arma::norm(psiSpace,2);


//    std::cout<<psi0<<std::endl;


}


///
/// @param n1 index for x1n1
/// @param t time
/// @return coefficient for evolution using H3
double os_DCE_Evolution::f(int n1, double t){

double x1n1Squared=x1ValsAllSquared[n1];

double val= -g0*omegac*std::sqrt(2.0/omegam)*std::sin(omegap*t)*x1n1Squared\
            +0.5*g0*std::sqrt(2.0/omegam)*std::sin(omegap*t);
    return val;


}


///
/// @param j time step
/// @param psi wavefunction at the beginning of the time step j
/// @return
arma::cx_dmat os_DCE_Evolution::evolution1Step(const int&j, const arma::cx_dmat& psi){
    arma::cx_dmat psiCurr(psi);

    double tj=timeValsAll[j];
//    std::cout<<"tj="<<tj<<std::endl;

    ///////////////////operator exp(-idt H1)
    //operator U15, for each column n2
    for (int n2=0;n2<N2;n2++){
        double x2n2=x2ValsAll[n2];
        psiCurr.col(n2)*=std::exp(1i*dt*0.5*g0*std::sqrt(2.0*omegam)*std::cos(omegap*tj)*x2n2);
    }

    //operator U14
    //construct U14
    //construct the exponent part
    arma::cx_dmat U14=-1i*dt*g0*omegac*std::sqrt(2.0*omegam)*std::cos(omegap*tj)*this->U14Exp;


    U14=arma::exp(U14);
    psiCurr=psiCurr %U14;

    //operator U13
    psiCurr*=std::exp(1i*dt*0.5*Deltam+1i*dt*0.5*omegac);

    //operator U12, for each column n2
    for(int n2=0;n2<N2;n2++){
        double x2n2Squared=x2ValsAllSquared[n2];
        psiCurr.col(n2)*=std::exp(-1i*dt*Deltam*omegam/(2.0*std::cosh(2.0*r))
                *std::exp(-2.0*r)*x2n2Squared);
    }

    //operator U11, for each row n1
    for(int n1=0;n1<N1;n1++){
        double x1n1Squared=x1ValsAllSquared[n1];
        psiCurr.row(n1)*=std::exp(-1i*dt*0.5*std::pow(omegac,2)*x1n1Squared);
    }

    double dbN1=static_cast<double >(N1);
    double dbN2=static_cast<double >(N2);
    //////////////////////////exp(-idt H2)
    // \partial_{x_{1}}^{2}
    arma::cx_drowvec psiRow=psiCurr.as_row();
    for(int i=0;i<N1*N2;i++){
        psiTmp[i]=psiRow(i);
    }
    //psi2Y
    fftw_execute(plan_psi2Y);
    //for each row n1
    for(int n1=0;n1<N1;n1++){
        double kn1Squared=k1ValsAllSquared[n1];
        for(int n2=0;n2<N2;n2++){
            Y[n1*N2+n2]*=std::exp(-1i*0.5*kn1Squared*dt);
        }
    }
    //Y2psi
    fftw_execute(plan_Y2psi);
    //normalization
    for(int i=0;i<N1*N2;i++){
        psiTmp[i]/=dbN1;
    }

    //\partial_{x_{2}}^{2}
    //psi2Z
    fftw_execute(plan_psi2Z);
    //for each col n2
    for(int n2=0;n2<N2;n2++){
        double kn2Squared=k2ValsAllSquared[n2];
        for(int n1=0;n1<N1;n1++){
            Z[n2+n1*N2]*=std::exp(-1i*Deltam/(2.0*omegam*std::cosh(2.0*r))
                    *e2r*kn2Squared*dt);
        }
    }
    //Z2psi
    fftw_execute(plan_Z2psi);
    //normalization
    for(int i=0;i<N1*N2;i++){
        psiTmp[i]/=dbN2;
    }

    ////////////////////exp(-idt H3)

    //psi2W
    fftw_execute(plan_psi2W);

    arma::cx_dcolvec fx1n1Vec(N1);
    for(int n1=0;n1<N1;n1++){
        fx1n1Vec(n1)=this->f(n1,tj);
    }
    arma::cx_dmat matTmp=arma::kron(fx1n1Vec,k2Row);
    matTmp*=-1i*dt;

    arma::cx_dmat M=arma::exp(matTmp);

    for(int n1=0;n1<N1;n1++){
        for(int n2=0;n2<N2;n2++){
            MArray[n1*N2+n2]=M(n1,n2);
        }
    }

    for(int i=0;i<N1*N2;i++){
        W[i]*=MArray[i];
    }
    //W2psi
    fftw_execute(plan_W2psi);

    //normalization
    for(int i=0;i<N1*N2;i++){
        psiTmp[i]/=dbN2;
    }


    arma::cx_dmat psiNext(N1,N2);
    for(int n1=0;n1<N1;n1++){
        for(int n2=0;n2<N2;n2++){
            psiNext(n1,n2)=psiTmp[n1*N2+n2];
        }
    }

    return psiNext;





}


///initialize matrices for computing particle numbers
void os_DCE_Evolution::popolateMatrices(){
    //construct H6

    arma::cx_dmat  V0(N1, 3);
    for(int n1=0;n1<N1;n1++){
        V0(n1,0)=1.0;
        V0(n1,1)=-2.0;
        V0(n1,2)=1.0;
    }

    arma::ivec D0 = {-1, 0, +1};

    arma::sp_cx_dmat leftMat = arma::spdiags(V0, D0, N1, N1);

    arma::sp_cx_dmat IN2=arma::speye<arma::sp_cx_dmat>(N2,N2);

    H6=-1/(2*std::pow(dx1,2))*arma::kron(leftMat,IN2);

    arma::cx_dmat  V1(N1, 1);
    for(int n1=0;n1<N1;n1++){
        V1(n1,0)=x1ValsAllSquared[n1];
    }
    arma::ivec D1 = {0};
    arma::sp_cx_dmat tmp0=arma::spdiags(V1,D1,N1,N1);

    NcMat1=arma::kron(tmp0,IN2);

    arma::cx_dmat  V2(N2, 1);
    for(int n2=0;n2<N2;n2++){
        V2(n2,0)=x2ValsAllSquared[n2];
    }
    arma::ivec D2 = {0};
    arma::sp_cx_dmat S2=arma::spdiags(V2,D2,N2,N2);



    arma::cx_dmat  V3(N2, 3);
    arma::ivec D3 = {-1, 0, +1};
    for(int n2=0;n2<N2;n2++){
        V3(n2,0)=1.0;
        V3(n2,1)=-2.0;
        V3(n2,2)=1.0;
    }

    arma::sp_cx_dmat Q2 = arma::spdiags(V3,D3,N2,N2);
    arma::sp_cx_dmat IN1=arma::speye<arma::sp_cx_dmat>(N1,N1);
    NmPart1=arma::kron(IN1,S2);
    NmPart2=arma::kron(IN1,Q2);

//    std::cout<<arma::cx_dmat (Q2)<<std::endl;

}

///
/// @param psi wavefunction
/// @return photon number
double os_DCE_Evolution::avgNc(const arma::cx_dmat& psi){

    arma::cx_drowvec PsiRow = psi.as_row();
    arma::cx_dcolvec Psi=PsiRow.t();
std::complex<double> val=0.5*omegac*arma::cdot(Psi,NcMat1*Psi)-0.5*arma::cdot(Psi,Psi)+1/omegac*arma::cdot(Psi,H6*Psi);

    return std::abs(val);
}


///
/// @param psi wavefunction
/// @return phonon number
double os_DCE_Evolution::avgNm(const arma::cx_dmat& psi) {

    arma::cx_drowvec PsiRow = psi.as_row();
    arma::cx_dcolvec Psi=PsiRow.t();
    std::complex<double>val=0.5*omegam*arma::cdot(Psi,NmPart1*Psi)-0.5*arma::cdot(Psi,Psi)-1/(2.0*omegam*std::pow(dx2,2))*arma::cdot(Psi,NmPart2*Psi);

    return std::abs(val);

}


///
/// @param psiIn starting value of the wavefunction in one flush
/// @param fls flush number
/// @return starting value of the wavefunction in the next flush
arma::cx_dmat os_DCE_Evolution::oneFlush(const arma::cx_dmat& psiIn, const int& fls){
    int startingInd=fls * stepsPerFlush;
    arma::cx_dmat psiCurr(psiIn);
    arma::cx_dmat psiNext;
    std::vector<double> photonPerFlush;
    std::vector<double> phononPerFlush;
    std::vector<double> analytical_photonPerFlush;
    std::vector<double> analytical_phononPerFlush;
    photonPerFlush.push_back(avgNc(psiCurr));
    phononPerFlush.push_back(avgNm(psiCurr));

   std::vector<double> diffPerFlush;
    auto analytical_start= psit(startingInd);

    analytical_photonPerFlush.push_back(avgNc(analytical_start));
    analytical_phononPerFlush.push_back(avgNm(analytical_start));

    diffPerFlush.push_back(arma::norm(analytical_start-psiCurr,2));
    for(int j=0;j<stepsPerFlush;j++){
        int indCurr=startingInd+j;
        psiNext= evolution1Step(indCurr,psiCurr);
        psiCurr=psiNext;
        photonPerFlush.push_back(avgNc(psiCurr));
        phononPerFlush.push_back(avgNm(psiCurr));

        analytical_start= psit(indCurr+1);
        diffPerFlush.push_back(arma::norm(analytical_start-psiCurr,2));

        analytical_photonPerFlush.push_back(avgNc(analytical_start));
        analytical_phononPerFlush.push_back(avgNm(analytical_start));

    }

    //to json
    std::string suffix="flush"+std::to_string(fls)+"N1"+std::to_string(N1)+"N2"+std::to_string(N2)+"L1"+std::to_string(L1)+"L2"+std::to_string(L2);
    std::string outNumFileName=this->outDir+"Num"+suffix+".json";



    boost::json::object objNum;
    boost::json::array arrPhoton;
    for(const auto&val:photonPerFlush){
        arrPhoton.push_back(val);
    }
    objNum["photonNum"]=arrPhoton;

    boost::json::array arrPhonon;
    for(const auto&val:phononPerFlush){
        arrPhonon.push_back(val);
    }
    objNum["phononNum"]=arrPhonon;

    boost::json::array arrDiff;
    for(const auto&val:diffPerFlush){
        arrDiff.push_back(val);
    }
    objNum["diff"]=arrDiff;

    boost::json::array analy_photon;
    for(const auto &val:analytical_photonPerFlush){
        analy_photon.push_back(val);
    }
objNum["ana_photon"]=analy_photon;

    boost::json::array analy_phonon;
    for(const auto& val:analytical_phononPerFlush){
        analy_phonon.push_back(val);
    }
    objNum["ana_phonon"]=analy_phonon;

    std::ofstream ofsNum(outNumFileName);
    std::string num_str=boost::json::serialize(objNum);
    ofsNum<<num_str<<std::endl;
    ofsNum.close();

    return psiCurr;





}


///evolution of wavefunctuion
void os_DCE_Evolution::evolution(){
    arma::cx_dmat psiStart(psi0);
    arma::cx_dmat psiFinal;
    std::string suffix_wv="N1"+std::to_string(N1)+"N2"+std::to_string(N2)+"L1"+std::to_string(L1)+"L2"+std::to_string(L2);
    std::string initWvName=this->outDir+"initWvFunction"+suffix_wv+".txt";
    psiStart.save(initWvName,arma::raw_ascii);
    for(int fls=0;fls<flushNum;fls++){
        const auto tFlushStart{std::chrono::steady_clock::now()};
        psiFinal= oneFlush(psiStart,fls);
        psiStart=psiFinal;
        const auto tFlushEnd{std::chrono::steady_clock::now()};
        const std::chrono::duration<double> elapsed_secondsAll{tFlushEnd - tFlushStart};
        std::cout<<"Flush"+std::to_string(fls)+" time: "<< elapsed_secondsAll.count()  << " s" << std::endl;

    }
    std::string finalWvName=this->outDir+"finalWvFunction"+suffix_wv+".txt";
    psiFinal.save(finalWvName,arma::raw_ascii);

}
double os_DCE_Evolution::funcf(int n1){
    double x1TmpSquared=x1ValsAllSquared[n1];
    double x1Tmp=x1ValsAll[n1];

    double valTmp = std::exp(-0.5 * omegac * x1TmpSquared)
                    * std::hermite(this->jH1, std::sqrt(omegac) * x1Tmp);
//    std::cout<<valTmp<<std::endl;

    return valTmp;

}

double os_DCE_Evolution::funcg(int n2) {
    double x2TmpSquared=x2ValsAllSquared[n2];
    double x2Tmp=x2ValsAll[n2];
    double coef=omegam*std::exp(-2.0*r);

    double valTmp=std::exp(-0.5*coef*x2TmpSquared)*std::hermite(this->jH2,std::sqrt(coef)*x2Tmp);
//    std::cout<<valTmp<<std::endl;

    return valTmp;
}


///
/// @param j time ind
/// @return analytical solution for g0=0
arma::cx_dmat  os_DCE_Evolution::psit(const int &j){

    double tjminus1=timeValsAll[j-1];
    double tj=tjminus1+dt;
    if (j==0){
        tj=0;
    }

    return psiSpace*std::exp(-1i*E1*tj)*std::exp(-1i*E2*tj)
    *std::exp(1i*0.5*Deltam*tj+1i*0.5*omegam*tj);
}