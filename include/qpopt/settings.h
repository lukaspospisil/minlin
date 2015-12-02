#ifndef QPSETTINGS_H
#define	QPSETTINGS_H

#include <iostream>
#include <time.h>

#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

#include "minlin/vector.h"
#include "minlin/matrix.h"

using namespace minlin::threx;


namespace minlin {

namespace QPOpt {
	struct QPSettings {
		/* algorithm settings */
		double my_eps;
		double Gamma;
		double eta;
		double betaM;
		double M0;
		double norm_A; 
		double norm_BTB;
		int maxit;
		double rho_coeff;
		
		bool smalbe;
		bool mprgp;
		bool cg;
		
		/* results */
		double norm_g; /* norm of gradient in solution */
		double norm_gp; /* norm of projected gradient in solution */
		double norm_Bx; /* norm of Bx=0 feasibility */
		int hess_mult; /* number of performed hessian multiplications */
		int it_cg; /* number of CG iterations */
		int it_mprgp; /* number of MPRGP iterations */
		int it_smalbe; /* number of SMALBE iterations */

		/* timer */
		double t_start;
		double time;
	};

	/* set default settings */
	void QPSettings_default(QPSettings *settings){
		settings->my_eps = 0.001;
		settings->Gamma = 1.0;
		settings->eta = 1.0;
		settings->betaM = 2.0;
		settings->M0 = 1.0;
		settings->smalbe = false;
		settings->mprgp = false;
		settings->cg = false;
		settings->maxit = 10000;
		settings->rho_coeff = 2;
		
		settings->hess_mult = 0;
		settings->it_mprgp = 0;
		settings->it_smalbe = 0;

		settings->time = 0.0;

		/* default values of norms */
		settings->norm_Bx = -1.0;
		settings->norm_gp = -1.0;
		settings->norm_g = -1.0;

	}

	/* print results */
	void QPSettings_print(QPSettings settings){
		std::cout << "-----------------------------------------------" << std::endl;
		std::cout << "SETTINGS " << std::endl;
		std::cout << "  - eps = \t\t\t" << settings.my_eps << std::endl;
		std::cout << "  - max_it = \t\t\t" << settings.maxit << std::endl;
		
		if(settings.smalbe){
			std::cout << "  SMALBE" << std::endl;
			std::cout << "   - eta = \t\t\t" << settings.eta << std::endl;
			std::cout << "   - betaM = \t\t\t" << settings.betaM << std::endl;
			std::cout << "   - M0 = \t\t\t" << settings.M0 << std::endl;
		}

		if(settings.mprgp){
			std::cout << "  MPRGP" << std::endl;
			std::cout << "   - Gamma = \t\t\t" << settings.Gamma << std::endl;
		}

		if(settings.cg){
			std::cout << "  CG" << std::endl;
		}

		std::cout << std::endl;
		std::cout << "RESULTS " << std::endl;
		std::cout << "  - mat mults = \t\t" << settings.hess_mult << std::endl;
		std::cout << "  - time = \t\t\t" << settings.time << " [s]" << std::endl;

		if(settings.smalbe){
			std::cout << "  SMALBE" << std::endl;
			std::cout << "   - it = \t\t\t" << settings.it_smalbe << std::endl;
			std::cout << "   - ||B*x|| = \t\t\t" << settings.norm_Bx << std::endl;
			std::cout << "   - norm(B^T*B) = \t\t" << settings.norm_BTB << std::endl;
		}

		if(settings.mprgp){
			std::cout << "  MPRGP" << std::endl;
			std::cout << "   - it = \t\t\t" << settings.it_mprgp << std::endl;
			std::cout << "   - ||gP(x)|| = \t\t" << settings.norm_gp << std::endl;
			std::cout << "   - norm(A) = \t\t\t" << settings.norm_A << std::endl;
		}

		if(settings.cg){
			std::cout << "  CG" << std::endl;
			std::cout << "   - it = \t\t\t" << settings.it_cg << std::endl;
			std::cout << "   - ||g(x)|| = \t\t" << settings.norm_g << std::endl;
		}

		std::cout << std::endl;

	}

	double getUnixTime(void){
		struct timespec tv;
		if(clock_gettime(CLOCK_REALTIME, &tv) != 0) return 0;
		return (((double) tv.tv_sec) + (double) (tv.tv_nsec / 1000000000.0));
	}

	void QPSettings_starttimer(QPSettings *settings){
		settings->t_start = getUnixTime();
	}
	
	void QPSettings_stoptimer(QPSettings *settings){
		double t_end = getUnixTime();
		
		settings->time = double(t_end - settings->t_start);
	}

}
}



#endif
