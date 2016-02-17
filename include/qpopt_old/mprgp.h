/*******************************************************************************
QP OPTIMAL SOLVERS for MINLIN library
Lukas Pospisil, 2016
lukas.pospisil@vsb.cz

* using MINLIN library (CSCS Lugano - Timothy Moroney, Ben Cumming)
* based on algorithms from Zdenek Dostal (VSB-TU Ostrava)
* created during HPCCausality project (USI Lugano - Illia Horenko, Patrick Gagliardini, Will Sawyer)
*******************************************************************************/

#ifndef QPOPT_MPRGP_H
#define	QPOPT_MPRGP_H

#include <iostream>
#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

#include "minlin/vector.h"
#include "minlin/matrix.h"
#include "qpopt/settings.h"

using namespace minlin::threx;


namespace minlin {

namespace QPOpt {

	/* compute projection to feasible set x >= l */
	template<typename Expression>
	Vector<Expression> project_bound(Vector<Expression> x, Vector<Expression> l){
		Vector<Expression> Px = x;

		int i;
		for(i=0; i < x.size();i++){
			if( Px(i) < l(i)){
				Px(i) = l(i);		
			}
		}

		return Px;
	}	

	/* compute free gradient */
	template<typename Expression>
	Vector<Expression> compute_fi(Vector<Expression> x, Vector<Expression> g, Vector<Expression> l){
		Vector<Expression> fi = g;
		
		int i;
		for(i=0; i < x.size();i++){
			if(x(i) > l(i)){
				/* free */
				fi(i) = g(i);
			} else {
				/* active */
				fi(i) = 0.0;
			}
		}

		return fi;
	}	

	/* compute chopped gradient */
	template<typename Expression>
	Vector<Expression> compute_beta(Vector<Expression> x, Vector<Expression> g, Vector<Expression> l){
		Vector<Expression> beta = g;
		
		int i;
		for(i=0; i < x.size();i++){
			if(x(i) > l(i)){
				/* free */
				beta(i) = 0.0;
			} else {
				/* active */
				if(g(i) < 0.0){
				 beta(i) = g(i);
				} else {
				 beta(i) = 0.0;
				}
			}
		}

		return beta;
	}	
	
	/* compute max feasible step-size */
	template<typename Expression>
	double compute_alpha_f(Vector<Expression> x, Vector<Expression> p, Vector<Expression> l){
		double alpha_f = std::numeric_limits<double>::max(); /* inf */
		double alpha_f_temp;
		
		int i;
		for(i=0; i < x.size();i++){
			if(p(i) > 0.0){
				alpha_f_temp = (x(i) - l(i))/p(i);

				if(alpha_f_temp < alpha_f){
					alpha_f = alpha_f_temp;
				}
			}
		}

		return alpha_f;
	}	
	

	/* general MPRGP implementation, could be used directly as inner solver of smalbe */
	template<typename Expression>
	Vector<Expression> mprgp_for_smalbe(QPSettings *settings, Matrix<Expression> A, Vector<Expression> b, Vector<Expression> l, Vector<Expression> x0, Matrix<Expression> B, double rho, double M){
		settings->mprgp = true; /* we used mprgp to solve this problem */

		/* set initial approximation */
		Vector<Expression> x = x0;

		/* MPRGP method */
		Vector<Expression> g; /* gradient */
		Vector<Expression> p; /* A-conjugate vector */
		Vector<Expression> gp; /* projected gradient */
		Vector<Expression> fi; /* free gradient */
		Vector<Expression> beta; /* beta gradient */
		Vector<Expression> Ap; /* A*p */
		int it = 0; /* iteration counter */
		int hess_mult = 0; /* number of hessian multiplications */

		double normgp;
		double betaTbeta, fiTfi, gTp, pAp, fiAp;
		double alpha_cg, alpha_f, betaa;
		double normA;
		
		if(settings->smalbe){
			normA = settings->norm_A + rho*settings->norm_BTB; /* this is wrong! use power method */
		} else {
			normA = settings->norm_A;
		}

		/* algorithm parameters */
		double Gamma = settings->Gamma;
		double alpha_bar = 1.9/normA; /* 0 < alpha_bar <= 2*norm(A) */

		/* project initial approximation to feasible set */
		x = project_bound(x, l);	
		
		/* compute gradients */
		g = A*x; g -= b; hess_mult += 1;
		fi = compute_fi(x, g, l);
		beta = compute_beta(x, g, l);
		gp = fi+beta;
		normgp = norm(gp);

		/* set first orthogonal vector */
		p = fi;

		fiTfi = dot(fi,fi);
		betaTbeta = dot(beta,beta);

		bool solved = false;
		Vector<Expression> Bx; /* B*x */
		double norm_Bx;

		/* compute stopping criteria */
		if(settings->smalbe){
			Bx = B*x;
			norm_Bx = norm(Bx);
//			solved = (normgp > std::max(settings->my_eps,std::min(norm_Bx,settings->eta)));
			if(norm_Bx < settings->eta){
			 solved = (normgp > norm_Bx);
			} else {
			 solved = (normgp > settings->eta);
			}
		} else {
			solved = (normgp > settings->my_eps);
		}	

		/* main cycle */
		while(solved && it < settings->maxit){
			if(betaTbeta <= Gamma*Gamma*fiTfi){
				/* 1. Proportional x_k. Trial conjugate gradient step */
				Ap = A*p; hess_mult += 1;
        
				gTp = dot(g,p);
				pAp = dot(Ap,p);
				alpha_cg = gTp/pAp;
				alpha_f = compute_alpha_f(x,p,l);
        
				if (alpha_cg <= alpha_f){
					/* 2. Conjugate gradient step */

					#ifdef QPOPT_DEBUG2
						std::cout << "  CG step, alpha_cg = " << alpha_cg << ", ";
						std::cout << "alpha_f = " << alpha_f << std::endl;
					#endif	

					x -= alpha_cg*p;
					g -= alpha_cg*Ap;
//					g = A*x; g -= b; hess_mult += 1;

					fi = compute_fi(x, g, l);
					beta = compute_beta(x, g, l);
					
					fiAp = dot(fi,Ap);
					betaa = fiAp/pAp;
					
					p = -betaa*p;
					p += fi;
					
				} else {
					/* 3. Expansion step */

					#ifdef QPOPT_DEBUG2
						std::cout << "  CG half-step, alpha_f = " << alpha_f << ", alpha_bar = " << alpha_bar << std::endl;
					#endif	

					x -= alpha_f*p;
					g -= alpha_f*Ap;
					
					beta = x; // temp use
					beta -= alpha_bar*g; 
					
					x = project_bound(beta,l);
					
					g = A*x; g -= b; hess_mult += 1;
					fi = compute_fi(x, g, l);
					beta = compute_beta(x, g, l);
					
					/* restart cg */
					p = fi;

				}
			} else {
				/* 4. Proportioning step */

				Ap = A*beta; hess_mult += 1;
        
				pAp = dot(Ap,beta);
				gTp = dot(g,beta);
				alpha_cg = gTp/pAp;

				#ifdef QPOPT_DEBUG2
					std::cout << "  Proportioning step, alpha_sd = " << alpha_cg << std::endl;
				#endif	
				
				x -= alpha_cg*beta;
				g -= alpha_cg*Ap;
				
				fi = compute_fi(x, g, l);
				beta = compute_beta(x, g, l);

				/* restart cg */
				p = fi;
				
			}

			fiTfi = dot(fi,fi);
			betaTbeta = dot(beta,beta);
			gp = fi + beta;
			normgp = norm(gp);
			
			/* update stopping criteria */
			if(settings->smalbe){
				Bx = B*x;
				norm_Bx = norm(Bx);
				//solved = (normgp > thrust::maximum<double>(settings->my_eps,thrust::minimum<double>(norm_Bx,settings->eta)));
				if(norm_Bx < settings->eta){
					solved = (normgp > norm_Bx);
				} else {
					solved = (normgp > settings->eta);
				} 
			} else {
				solved = (normgp > settings->my_eps);
			}	

			#ifdef QPOPT_DEBUG
				std::cout << "it " << it << ": ";
				#ifdef QPOPT_DEBUG_F
					/* compute f=0.5*x^TAx-b^Tx */
					Ap = A*x;
					pAp = dot(Ap,x);
					gTp = dot(b,x);
					f = 0.5*pAp - gTp;
					std::cout << "f = " << f << ", ";	
				#endif 
				std::cout << "||gP|| = " << normgp << ", ||fi|| = " << std::sqrt(fiTfi) << ", ||beta|| = " << std::sqrt(betaTbeta) << std::endl;
			#endif	
			
			it += 1;
		}

		/* set output values */
		settings->it_mprgp = it;
		settings->norm_gp = normgp;
		settings->hess_mult = hess_mult;

		return x;
	}

	/* only with bound constraints - use mprgp */
	template<typename Expression>
	Vector<Expression> mprgp(QPSettings *settings, Matrix<Expression> A, Vector<Expression> b, Vector<Expression> l, Vector<Expression> x0){
		settings->smalbe = false;
		
		return mprgp_for_smalbe(settings, A, b, l, x0, A, 0.0, 0.0);

	}

	/* bound constrained without initial approximation, init approximation = 0 */
	template<typename Expression>
	Vector<Expression> mprgp(QPSettings *settings, Matrix<Expression> A, Vector<Expression> b, Vector<Expression> l){
		Vector<Expression> x0 = b;
		x0(minlin::all) = 0.0; /* default initial approximation */  
		return mprgp(settings, A, b, l, x0);
	}




}


} // end namespace minlin	




#endif
