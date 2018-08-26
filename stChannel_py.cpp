#include "FADBAD++/fadiff.h"
#include "FADBAD++/interval.hpp"
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

using namespace boost::python;

typedef std::vector<double> MyList;
typedef std::vector<MyList> MyListOfList;

typedef mc::Interval I;

struct StMosfet{

std::map<std::string, double> model_params(const char mType){
	std::map<std::string, double> input_params;
	std::string versionS = "version", mTypeS = "mType", WS = "W", LgdrS = "Lgdr", dLgS = "dLg", CgS = "Cg";
	std::string etovS = "etov", deltaS = "delta", n0S = "n0", Rs0S = "Rs0", Rd0S = "Rd0", CifS = "Cif", CofS = "Cof", vxoS = "vxo";
	std::string muS = "mu", betaS = "beta", TjunS = "Tjun", phibS = "phib", gammaS = "gamma", Vt0S = "Vt0", alphaS = "alpha", mcS = "mc";
	std::string CTM_selectS = "CTM_select", CCS = "CC", ndS = "nd", zetaS = "zeta";
	if (mType == 'n'){
		input_params[versionS] = 1.10;
		input_params[mTypeS] = 1;
		input_params[WS] = 450e-7;
		input_params[LgdrS] = 45e-7;
		input_params[dLgS] = 3.75e-7;
		input_params[CgS] = 2.837e-6;
		input_params[etovS] = 1.67e-7;
		input_params[deltaS] = 0.1370; 
		input_params[n0S] = 1.4075;
		input_params[Rs0S] = 100.5558;
		input_params[Rd0S] = 100.5558;
		input_params[CifS] = 1.9856e-12;
		input_params[CofS] = 1.9856e-12;
		input_params[vxoS] = 1.2277;
		input_params[muS] = 641.7222;
		input_params[betaS] = 1.8;
		input_params[TjunS] = 298;
		input_params[phibS] =1.2;
		input_params[gammaS] = 0.2;
		input_params[Vt0S] = 0.5496;
		input_params[alphaS] = 3.5;
		input_params[mcS] = 0.2;
		input_params[CTM_selectS] = 1;
		input_params[CCS] = 0;
		input_params[ndS] = 3.053e-14;
		input_params[zetaS] = 1.0;
	}
	else if (mType == 'p'){
		input_params[versionS] = 1.10;
		input_params[mTypeS] = -1;
		input_params[WS] = 450e-7;
		input_params[LgdrS] = 45e-7;
		input_params[dLgS] = 3.75e-7;
		input_params[CgS] = 2.837e-6;
		input_params[etovS] = 1.67e-7;
		input_params[deltaS] = 0.1637;
		input_params[n0S] = 1.4001;
		input_params[Rs0S] = 153.8674;
		input_params[Rd0S] = 153.8674;
		input_params[CifS] = 1.9856e-12;
		input_params[CofS] = 1.9856e-12;
		input_params[vxoS] = 1.0751;
		input_params[muS] = 285.6056;
		input_params[betaS] = 1.6;
		input_params[TjunS] = 298;
		input_params[phibS] = 1.2;
		input_params[gammaS] = 0.2;
		input_params[Vt0S] = 0.6252;
		input_params[alphaS] = 3.5; 
		input_params[mcS] = 0.2;
		input_params[CTM_selectS] = 1;
		input_params[CCS] = 0;
		input_params[ndS] = 0.0235;
		input_params[zetaS] = 1.0;
	}

	else{
		std::string TNOMS = "TNOM", WintS = "Wint", LgdrS = "Lgdr", dLgS = "dLg";
		std::string XlS = "Xl", XwS = "Xw", XjS = "Xj", CjdTnomS = "CjdTnom";
		std::string TjunS = "Tjun", TcjS = "Tcj", PbdTnomS = "PbdTnom", TpbS = "Tpb";
		std::string CjswdTnomS = "CjswdTnom", TcjswS = "Tcjsw", PbswdTnomS = "PbswdTnom", TpbswS = "Tpbsw";
		std::string CjswgdTnomS = "CjswgdTnom", TcjswgS = "Tcjswg", PbswgdTnomS = "PbswgdTnom", TpbswgS = "Tpbswg";
		std::string MjdS = "Mjd", MjswdS = "Mjswd", MjswgdS = "Mjswgd", WS = "W";
		input_params[TNOMS] = 300;
		input_params[WintS] = 5e-7;
		input_params[LgdrS] = 45e-7;
		input_params[dLgS] = 3.75e-7;
		input_params[XlS] = -20e-9;
		input_params[XwS] = 0;
		input_params[XjS] = 1.4e-6; 
		input_params[CjdTnomS] = 0.0005;
		input_params[TjunS] = 298; 
		input_params[TcjS] = 0.001;
		input_params[PbdTnomS] = 1.0; 
		input_params[TpbS] = 0.005;
		input_params[CjswdTnomS] = 5e-10; 
		input_params[TcjswS] = 0.001;
		input_params[PbswdTnomS] = 1.0; 
		input_params[TpbswS] = 0.005;
		input_params[CjswgdTnomS] = 5e-10; 
		input_params[TcjswgS] = 0.001;
		input_params[PbswgdTnomS] = 1.0; 
		input_params[TpbswgS] = 0.005;
		input_params[MjdS] = 0.5;
		input_params[MjswdS] = 0.33;
		input_params[MjswgdS] = 0.33; 
		input_params[WS] = 450e-7;
	}

	return input_params;
}

// bias voltages are Vd, Vg, Vs, Vb
F<I> mvs_id(const std::map<std::string, double>& params, 
						const F<I>& Vds, const F<I>& Vgs,  
						const F<I>& Vbs, const int iDir){
	std::string versionS = "version", mTypeS = "mType", WS = "W", LgdrS = "Lgdr", dLgS = "dLg", CgS = "Cg";
	std::string etovS = "etov", delS = "delta", n0S = "n0", Rs0S = "Rs0", Rd0S = "Rd0", CifS = "Cif", CofS = "Cof", vxoS = "vxo";
	std::string muS = "mu", betaS = "beta", TjunS = "Tjun", phibS = "phib", gammaS = "gamma", Vt0S = "Vt0", alphaS = "alpha", mcS = "mc";
	std::string CTM_selectS = "CTM_select", CCS = "CC", ndS = "nd", zetaS = "zeta";
	double version = params.at(versionS);
	double mType = params.at(mTypeS);
	double W = params.at(WS);
	double Lgdr = params.at(LgdrS);
	double dLg = params.at(dLgS);
	double Cg = params.at(CgS);
	double etov = params.at(etovS);
	double delta = params.at(delS);
	double n0 = params.at(n0S);
	double Rs0 = params.at(Rs0S);
	double Rd0 = params.at(Rd0S);
	double Cif = params.at(CifS);
	double Cof = params.at(CofS);
	double vxo = params.at(vxoS)*1e+7;
	double mu = params.at(muS);
	double beta = params.at(betaS);
	double Tjun = params.at(TjunS);
	double phib = params.at(phibS);
	double gamma = params.at(gammaS);
	double Vt0 = params.at(Vt0S);
	double alpha = params.at(alphaS);
	double mc = params.at(mcS);
	double CTM_select = params.at(CTM_selectS);
	double CC = params.at(CCS);
	double nd = params.at(ndS);
	double zeta = params.at(zetaS);

	// SMALL_VALUE
	double SMALL_VALUE = 1e-10;
	// LARGE_VALUE
	double LARGE_VALUE = 40;

	/*int iDir = 1;
	if (Vdsraw.x().l() < 0 && Vdsraw.x().u() < 0){
		//swap VsTemp and VdTemp
		int iDir = -1;
	}*/
	F<I> Vdsi = Vds;
	F<I> Vgsi = Vgs;
	F<I> Vbsi = Vbs;

    //std::cout << "Vds " << Vds << " Vgs " << Vgs << " Vbs " << Vbs << "\n";
    //std::cout << "Vdsi " << Vdsi.x() << " Vgsi " << Vgsi.x() << " Vbsi " << Vbsi.x() << "\n";

    ////  Virtual source not included here
    double Cofs = 0*( 0.345e-12/ etov ) * dLg/ 2.0 + Cof;  // s-terminal outer fringing cap [F/cm]
    double Cofd = 0*( 0.345e-12/ etov ) * dLg/ 2.0 + Cof;  // d-terminal outer fringing cap [F/cm]
    double Leff = Lgdr - dLg;                            // Effective channel length [cm]. After subtracting overlap lengths on s and d side

    double kB = 8.617e-5;                                // Boltzmann constant [eV/K]
    double phit = kB*Tjun;                               // Thermal voltage, kT/q [V]
    double me = (9.1e-31) * mc;                          // Carrier mass [Kg]
    F<I> n = n0 + nd * Vdsi;                           // Total subthreshold swing factor taking punchthrough into account [unit-less]
    F<I> nphit = n * phit;                             // Product of n and phit [used as one variable]
    double aphit = alpha * phit;                         // Product of alpha and phit [used as one variable]

    //std::cout << "Check1\n";
    // Correct Vgsi and Vbsi
    // Vcorr is computed using external Vbs and Vgs but internal Vdsi, Qinv and Qinv_corr are computed with uncorrected Vgs, Vbs and corrected Vgs, Vbs respectively.
    F<I> phibVbs = fabs(phib - Vbs);
    /*if (phibVbs.x().l() < 0 or phibVbs.x().u() < 0){
    	phibVbs = -phibVbs;
    }*/
    F<I> Vtpcorr = Vt0 + gamma * (sqrt(phibVbs)- sqrt(phib))- Vdsi * delta; // Calculated from extrinsic Vbs
    F<I> eVgpre = exp(( Vgs - Vtpcorr )/ ( aphit * 1.5 ));                          // Calculated from extrinsic Vgs
    F<I> FFpre = 1.0/( 1.0 + eVgpre );                                            // Only used to compute the correction factor
    F<I> ab = 2 * ( 1 - 0.99 * FFpre ) * phit;
    F<I> Vcorr = ( 1.0 + 2.0 * delta ) * ( ab/ 2.0 ) * ( exp( -Vdsi/ ab ));         // Correction to intrinsic Vgs
    F<I> Vgscorr = Vgsi + Vcorr;                                             // Intrinsic Vgs corrected (to be used for charge and current computation)
    F<I> Vbscorr = Vbsi + Vcorr;
    F<I> phibVbscorr = fabs(phib - Vbscorr);
  	                                         // Intrinsic Vgs corrected (to be used for charge and current computation)
    //std::cout << "Check2\n";
    F<I> Vt0bs = Vt0 + gamma * (sqrt(phibVbscorr) - sqrt( phib )); // Computed from corrected intrinsic Vbs
    //std::cout << "Vt0bs " << Vt0bs << "\n";
    F<I> phibVbsi = fabs(phib - Vbsi);
   
    F<I> Vt0bs0 = Vt0 + gamma * (sqrt( phibVbsi) - sqrt( phib ));   // Computed from uncorrected intrinsic Vbs
    //std::cout << "Vt0bs0 " << Vt0bs0 << "\n";
    //std::cout << "Vdsi " << Vdsi << " delta " << delta << " aphit " << aphit << "\n";
    F<I> Vtp = Vt0bs - Vdsi * delta - 0.5 * aphit;                           // Computed from corrected intrinsic Vbs and intrinsic Vds
    //std::cout << "Vtp " << Vtp << "\n";
    F<I> Vtp0 = Vt0bs0 - Vdsi * delta - 0.5 * aphit;                         // Computed from uncorrected intrinsic Vbs and intrinsic Vds
    //std::cout << "Vtp0 " << Vtp0 << "\n";
    F<I> eVg = exp(( Vgscorr - Vtp )/ ( aphit ));                            // Compute eVg factor from corrected intrinsic Vgs
    //std::cout << "eVg " << eVg << "\n"; 
    F<I> FF = 1.0/ ( 1.0 + eVg );
    F<I> eVg0 = exp(( Vgsi - Vtp0 )/ ( aphit ));                             // Compute eVg factor from uncorrected intrinsic Vgs
    F<I> FF0 = 1.0/ ( 1.0 + eVg0 );

    F<I> Qref = Cg * nphit;                                                  // F/cm^2*V=Coulombs/cm^2,   F = Coulombs/V
    F<I> eta = ( Vgscorr - ( Vt0bs - Vdsi * delta - FF * aphit ))/ ( nphit ); // Compute eta factor from corrected intrinsic Vgs and intrinsic Vds
    F<I> eta0 = ( Vgsi - ( Vt0bs0 - Vdsi * delta - FF * aphit ))/ ( nphit );  // Compute eta0 factor from uncorrected intrinsic Vgs and internal Vds.
                                                                         // Using FF instead of FF0 in eta0 gives smoother capacitances.
    //std::cout << "Check3\n";
    //Charge at VS in saturation (Qinv)
    F<I> Qinv_corr = Qref*log(1.0 + exp(eta));
    //std::cout << "Check4\n";
    //std::cout << "Check5\n";
    //Transport equations
    double vx0 = vxo;
    double Vdsats = vx0 * Leff/mu;
    //std::cout << "Check6\n";
    F<I> Vdsat = Vdsats * ( 1.0 - FF ) + phit * FF;   // Saturation drain voltage for current    
   	//std::cout << "Check7\n";
    //std::cout << pow(I(0.1, 0.2), 1.8) << "\n";
    F<I> VdsiVdsat =  Vdsi/Vdsat;
    //F<I> VdsiVdsat = VdsiVdsatTemp;
    //double del = 1e-14;
    //std::cout << "VdsiVdsatTemp" << "\n";
    //std::cout << VdsiVdsatTemp.x() << "\n";
    /*if (VdsiVdsatTemp.x().l() <= del && VdsiVdsatTemp.x().u() <= del){
    	//std::cout << "here?\n";
    	VdsiVdsat = VdsiVdsatTemp + del;
    }
    else if(VdsiVdsatTemp.x().l() <= del){
    	VdsiVdsat = VdsiVdsatTemp + del;
    }*/
    //std::cout << "Check8\n";
    //std::cout << "VdsiVdsat " << VdsiVdsat << "\n";
    //F<double> powVal1 = pow(VdsiVdsat.x().l(), beta);
    //F<double> powVal2 = pow(VdsiVdsat.x().u(), beta);
    //F<I> powVal = I(min(powVal1, powVal2), max(powVal1, powVal2));
   
    F<I> powVal = pow(VdsiVdsat, beta);
    //std::cout << "Check9\n";
    F<I> Fsat=(Vdsi/Vdsat)/(pow((1+powVal), (1.0/beta)));      // Transition function from linear to saturation.
                                                 // Fsat = 1 when Vds>>Vdsat; Fsat= Vds when Vds<<Vdsat

    //Total drain current
    //std::cout << "Check10\n";
    F<I> Id = Qinv_corr * vx0 * Fsat * W * mType * iDir;  // Coulombs/s = A
   	//std::cout << "Check11\n";
   	return Id;

}


// calculate current for nfet
F<I> mvs_ids(const F<I>& Vd, const F<I>& Vg,  
			const F<I>& Vs, const F<I>& Vb, const char& fetType){

	int mType;
	if (fetType == 'n'){
		mType = 1;
	}
	else if(fetType == 'p'){
		mType = -1;
	}
	std::map<std::string, double> mosType = model_params(fetType);
	F<I> Vds = mType*(Vd - Vs);
	F<I> Vgs = mType*(Vg - Vs);
	F<I> Vbs = mType*(Vb - Vs);
	//std::cout << "Vds" << Vds << "\n";
	//std::cout << "BEFORE\n";
	//std::cout << "Vds " << Vds << " Vgs " << Vgs << " Vbs " << Vbs << "\n";

	if (Vds.x().l() >= 0.0 and Vds.x().u() >= 0){
		int iDir = 1;
		return mvs_id(mosType, Vds, Vgs, Vbs, iDir);
	}

	if (Vds.x().l() < 0.0 and Vds.x().u() < 0){
		int iDir = -1;
		Vds = -Vds;
		Vgs = mType*(Vg - Vd);
		Vbs = mType*(Vb - Vd);
		return mvs_id(mosType, Vds, Vgs, Vbs, iDir);
	}

	F<I> Vsx;
	if (fetType == 'n'){
		Vsx = I(Vs.v.l(), fmin(Vd.v.u(), Vs.v.u()));
	}
	else if(fetType == 'p'){
		Vsx = I(fmax(Vd.v.u(), Vs.v.u()), Vs.v.u());
	}
	int iDir = 1;
	F<I> Vds1 = I(0.0, Vds.v.u());
	F<I> Vgs1 = mType*(Vg - Vsx);
	F<I> Vbs1 = mType*(Vb - Vsx);	
	F<I> Ids1 = mvs_id(mosType, Vds1, Vgs1, Vbs1, iDir);
	//std::cout << "Ids1 " << Ids1.x() << "\n";

	F<I> Vdx;
	if (fetType == 'n'){
		Vdx = I(Vd.v.l(), fmin(Vd.v.u(), Vs.v.u()));
	}
	else if(fetType == 'p'){
		Vdx = I(fmax(Vd.v.u(), Vs.v.u()), Vd.v.u());
	}
	iDir = -1;
	F<I> Vds2 = (-1)*I(Vds.v.l(), 0.0);
	F<I> Vgs2 = mType*(Vg - Vdx);
	F<I> Vbs2 = mType*(Vb - Vdx);
	F<I> Ids2 = mvs_id(mosType, Vds2, Vgs2, Vbs2, iDir);
	//std::cout << "Ids2 " << Ids2.x() << "\n";
	F<I> IdsUnion = I(fmin(Ids1.v.l(), Ids2.v.l()),
	 					fmax(Ids1.v.u(), Ids2.v.u()));
	return IdsUnion;
}

std::vector<I> mvs_idJacHelp(const std::map<std::string, double>& params, 
								const F<I>& Vds, const F<I>& Vgs, const F<I>& Vbs, const int iDir){
	std::map<std::string, double> nmos = model_params('n');
	F<I> Vdsf, Vgsf, Vbsf;
	Vdsf = Vds;
	Vgsf = Vgs;
	Vbsf = Vbs;

	Vdsf.diff(0,3);
	Vgsf.diff(1,3);
	Vbsf.diff(2,3);

	F<I> ff = mvs_id(params, Vdsf, Vgsf, Vbsf, iDir);

	// Use chain rule to convert derivatives with respect to V - Vs to V
	std::vector<I> jac;
	jac.push_back(ff.d(0));
	jac.push_back(ff.d(1));
	jac.push_back(ff.d(2));
	return jac;

}

std::vector<I> mvs_idsJac(I Vd, I Vg, I Vs, I Vb, char& fetType){
	std::map<std::string, double> mosType = model_params(fetType);
	int mType;
	if (fetType == 'n'){
		mType = 1;
	}
	else if(fetType == 'p'){
		mType = -1;
	}
	I Vds = mType*(Vd - Vs);
	I Vgs = mType*(Vg - Vs);
	I Vbs = mType*(Vb - Vs);
	//std::cout << "Vds" << Vds << "\n";
	//std::cout << "BEFORE\n";
	//std::cout << "Vds " << Vds << " Vgs " << Vgs << " Vbs " << Vbs << "\n";

	
	if (Vds.l() >= 0.0 and Vds.u() >= 0){
		int iDir = 1;
		std::vector<I> jac = mvs_idJacHelp(mosType, Vds, Vgs, Vbs, iDir);
		std::vector<I> jacA;
		jacA.push_back(mType*jac[0]);
		jacA.push_back(mType*jac[1]);
		jacA.push_back(mType*(-jac[0] - jac[1] - jac[2]));
		jacA.push_back(mType*jac[2]);
		return jacA;
	}

	if (Vds.l() < 0.0 and Vds.u() < 0){
		int iDir = -1;
		Vds = -Vds;
		Vgs = mType*(Vg - Vd);
		Vbs = mType*(Vb - Vd);
		std::vector<I> jac = mvs_idJacHelp(mosType, Vds, Vgs, Vbs, iDir);
		std::vector<I> jacA;
		jacA.push_back(mType*(-jac[0] - jac[1] - jac[2]));
		jacA.push_back(mType*jac[1]);
		jacA.push_back(mType*jac[0]);
		jacA.push_back(mType*jac[2]);
		return jacA;
	}

	I Vsx;
	if (fetType == 'n'){
		Vsx = I(Vs.l(), fmin(Vd.u(), Vs.u()));
	}
	else if(fetType == 'p'){
		Vsx = I(fmax(Vd.u(), Vs.u()), Vs.u());
	}
	int iDir = 1;
	I Vds1 = I(0.0, Vds.u());
	I Vgs1 = mType*(Vg - Vsx);
	I Vbs1 = mType*(Vb - Vsx);
	std::vector<I> jac = mvs_idJacHelp(mosType, Vds1, Vgs1, Vbs1, iDir);
	std::vector<I> jac1;
	jac1.push_back(mType*(jac[0]));
	jac1.push_back(mType*(jac[1]));
	jac1.push_back(mType*(-jac[0] - jac[1] - jac[2]));
	jac1.push_back(mType*(jac[2]));
	//std::cout << "Ids1 " << Ids1.x() << "\n";

	I Vdx;
	if (fetType == 'n'){
		Vdx = I(Vd.l(), fmin(Vd.u(), Vs.u()));
	}
	else if(fetType == 'p'){
		Vdx = I(fmax(Vd.u(), Vs.u()), Vd.u());
	}
	iDir = -1;
	I Vds2 = (-1)*I(Vds.l(), 0.0);
	I Vgs2 = mType*(Vg - Vdx);
	I Vbs2 = mType*(Vb - Vdx);
	jac = mvs_idJacHelp(mosType, Vds2, Vgs2, Vbs2, iDir);
	std::vector<I> jac2;
	jac2.push_back(mType*(jac[0]));
	jac2.push_back(mType*(jac[1]));
	jac2.push_back(mType*(-jac[0] - jac[1] - jac[2]));
	jac2.push_back(mType*(jac[2]));
	//std::cout << "Ids2 " << Ids2.x() << "\n";
	std::vector<I> jacUnion;

	for(std::vector<I>::iterator it1 = jac1.begin(), it2 = jac2.begin(); 
		(it1 != jac1.end() && it2 != jac2.end()); ++it1, ++it2) {
		jacUnion.push_back(I(fmin((*it1).l(), (*it2).l()),
	 					fmax((*it1).u(), (*it2).u())));
	}
	return jacUnion;
}

MyListOfList mvs_idnJacPy(const MyList& Vd, 
							const MyList& Vg,
							const MyList& Vs,
							const MyList& Vb){
	I Vdf = I(Vd[0], Vd[1]);
	I Vgf = I(Vg[0], Vg[1]);
	I Vsf = I(Vs[0], Vs[1]);
	I Vbf = I(Vb[0], Vb[1]);
	char fetType = 'n';
	std::vector<I> jac = mvs_idsJac(Vdf, Vgf, Vsf, Vbf, fetType);
	double jacAr0[] = {jac[0].l(), jac[0].u()};
	MyList jac0(jacAr0, jacAr0 + 2);
	double jacAr1[] = {jac[1].l(), jac[1].u()};
	MyList jac1(jacAr1, jacAr1 + 2);
	double jacAr2[] = {jac[2].l(), jac[2].u()};
	MyList jac2(jacAr2, jacAr2 + 2);
	double jacAr3[] = {jac[3].l(), jac[3].u()};
	MyList jac3(jacAr3, jacAr3 + 2);

	MyList overallJacAr[] = {jac0, jac1, jac2, jac3};
	MyListOfList overallJac (overallJacAr, overallJacAr+4);
	return overallJac;

}



MyList mvs_idnPy(const MyList& Vd,
					const MyList& Vg,
					const MyList& Vs,
					const MyList& Vb){
	F<I> Vdf = I(Vd[0], Vd[1]);
	F<I> Vgf = I(Vg[0], Vg[1]);
	F<I> Vsf = I(Vs[0], Vs[1]);
	F<I> Vbf = I(Vb[0], Vb[1]);
	char fetType = 'n';
	F<I> ids = mvs_ids(Vdf, Vgf, Vsf, Vbf, fetType);
	MyList idsVec;
	idsVec.push_back(ids.v.l());
	idsVec.push_back(ids.v.u());
	return idsVec;

}

MyList mvs_idpPy(const MyList& Vd,
					const MyList& Vg,
					const MyList& Vs,
					const MyList& Vb){
	F<I> Vdf = I(Vd[0], Vd[1]);
	F<I> Vgf = I(Vg[0], Vg[1]);
	F<I> Vsf = I(Vs[0], Vs[1]);
	F<I> Vbf = I(Vb[0], Vb[1]);
	char fetType = 'p';
	F<I> ids = mvs_ids(Vdf, Vgf, Vsf, Vbf, fetType);
	MyList idsVec;
	idsVec.push_back(ids.v.l());
	idsVec.push_back(ids.v.u());
	return idsVec;

}


MyListOfList mvs_idpJacPy(const MyList& Vd, 
							const MyList& Vg,
							const MyList& Vs,
							const MyList& Vb){
	I Vdf = I(Vd[0], Vd[1]);
	I Vgf = I(Vg[0], Vg[1]);
	I Vsf = I(Vs[0], Vs[1]);
	I Vbf = I(Vb[0], Vb[1]);
	char fetType = 'p';
	std::vector<I> jac = mvs_idsJac(Vdf, Vgf, Vsf, Vbf, fetType);
	double jacAr0[] = {jac[0].l(), jac[0].u()};
	MyList jac0(jacAr0, jacAr0 + 2);
	double jacAr1[] = {jac[1].l(), jac[1].u()};
	MyList jac1(jacAr1, jacAr1 + 2);
	double jacAr2[] = {jac[2].l(), jac[2].u()};
	MyList jac2(jacAr2, jacAr2 + 2);
	double jacAr3[] = {jac[3].l(), jac[3].u()};
	MyList jac3(jacAr3, jacAr3 + 2);

	MyList overallJacAr[] = {jac0, jac1, jac2, jac3};
	MyListOfList overallJac (overallJacAr, overallJacAr+4);
	return overallJac;
}


};


BOOST_PYTHON_MODULE(stChannel_py){
	class_<MyList>("MyList")
        .def(vector_indexing_suite<MyList>() );

    class_<MyListOfList>("MyListOfList")
        .def(vector_indexing_suite<MyListOfList>() );

	class_<StMosfet>("StMosfet")
		.def("mvs_idn", &StMosfet::mvs_idnPy)
		.def("mvs_idnJac", &StMosfet::mvs_idnJacPy)
		.def("mvs_idp", &StMosfet::mvs_idpPy)
		.def("mvs_idpJac", &StMosfet::mvs_idpJacPy);
}

