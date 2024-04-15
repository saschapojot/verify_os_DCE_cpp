#include "evolution.hpp"

int main(int argc, char *argv[]) {
    if(argc!=3){
        std::cerr<<"wrong number of arguments"<<std::endl;
        exit(1);
    }
    int groupNum=std::stoi(argv[1]);
    int rowNum=std::stoi(argv[2]);
    auto evo=os_DCE_Evolution(groupNum,rowNum);
    evo.initPsiSerial();



    evo.popolateMatrices();
    evo.evolution();

//    arma::mat A(3,3);
//    for (int i=0;i<3;i++){
//        for (int j=0;j<3;j++){
//            A(i,j)=3*i+j;
//        }
//    }
//
//    auto v1=A.as_row();
//    auto v2=A.as_row();
//    std::cout<<arma::cdot(v1,v2)<<std::endl;
//
//    std::cout<<"A="<<A<<std::endl;
//
//    std::cout<<"row="<<A.as_row()<<std::endl;
//
//    std::cout<<"A="<<A<<std::endl;


    return 0;
}
