/*******************************************************************************
QP OPTIMAL SOLVERS for MINLIN library
Lukas Pospisil, 2015-2016
lukas.pospisil@vsb.cz

* using MINLIN library (CSCS Lugano - Timothy Moroney, Ben Cumming)
* based on algorithms from Zdenek Dostal (VSB-TU Ostrava)
* created during HPCCausality project (USI Lugano - Illia Horenko, Patrick Gagliardini, Will Sawyer)
*******************************************************************************/

#ifndef QPOPT_H
#define	QPOPT_H

#include <iostream>
#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

#include "minlin/vector.h"
#include "minlin/matrix.h"
#include "qpopt/settings.h"

using namespace minlin::threx;


namespace minlin {

namespace QPOpt {

	/* compute projection - feasible set */
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
				beta(i) = std::min(g(i),0.0);
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
	


	/* unconstrained with initial approximation */
	template<typename Expression>
	Vector<Expression> solve_unconstrained(QPSettings *settings, Matrix<Expression> A, Vector<Expression> b, Vector<Expression> x0){
		settings->cg = true;

		Vector<Expression> x = x0;

		/* CG method */
		Vector<Expression> g; /* gradient */
		Vector<Expression> p; /* A-conjugate vector */
		Vector<Expression> Ap; /* A*p */
		int it = 0; /* iteration counter */
		double normb, normg, alpha, beta, pAp, gg, gg_old;
	
		g = A*x; g -= b; /* compute gradient */
		p = g; /* initial conjugate gradient */

		normb = norm(b);
		gg = dot(g,g);
		normg = std::sqrt(gg);
		while(normg > settings->my_eps && it < 10000){
			/* compute new approximation */
			Ap = A*p;
			pAp = dot(Ap,p);
			alpha = gg/pAp;
			x -= alpha*p;

			g -= alpha*Ap; /* compute gradient recursively */
			gg_old = gg;
			gg = dot(g,g);
			normg = std::sqrt(gg);
			
			/* compute new A-orthogonal vector */
			beta = gg/gg_old;
			p = beta*p;
			p += g;
		
			#ifdef QPOPT_DEBUG
				std::cout << "it " << it << ": ||g|| = " << normg << std::endl;
			#endif	
				
			it += 1;

		}
		
		/* set oputput */
		settings->it_cg = it; 
		settings->norm_g = normg; 
		
		return x;
	}

	/* unconstrained without initial approximation, init approximation = 0 */
	template<typename Expression>
	Vector<Expression> solve_unconstrained(QPSettings *settings, Matrix<Expression> A, Vector<Expression> b){
		Vector<Expression> x0 = b;
		x0(minlin::all) = 0.0; /* default initial approximation */  

		return solve_unconstrained(settings, A, b, x0);
	}

	/* with bound constraints */
	template<typename Expression>
	Vector<Expression> solve_bound_smalbe(QPSettings *settings, Matrix<Expression> A, Vector<Expression> b, Vector<Expression> l, Vector<Expression> x0, Matrix<Expression> B, double rho, double M){
		settings->mprgp = true;

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

		double normgp, f;
		double betaTbeta, fiTfi, gTp, pAp, fiAp;
		double alpha_cg, alpha_f, betaa;
		double normA;
		
		if(settings->smalbe){
			normA = settings->norm_A + rho*settings->norm_BTB;
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
			solved = (normgp > std::min(norm_Bx,settings->eta));
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
				solved = (normgp > std::max(settings->my_eps,std::min(norm_Bx,settings->eta)));
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

	/* with bound constraints */
	template<typename Expression>
	Vector<Expression> solve_bound(QPSettings *settings, Matrix<Expression> A, Vector<Expression> b, Vector<Expression> l, Vector<Expression> x0){
		settings->smalbe = false;
		
		return solve_bound_smalbe(settings, A, b, l, x0, A, 0.0, 0.0);

	}

	/* bound constrained without initial approximation, init approximation = 0 */
	template<typename Expression>
	Vector<Expression> solve_bound(QPSettings *settings, Matrix<Expression> A, Vector<Expression> b, Vector<Expression> l){
		Vector<Expression> x0 = b;
		x0(minlin::all) = 0.0; /* default initial approximation */  
		return solve_bound(settings, A, b, l, x0);
	}
	
	/* with equality and bound constraints */
	template<typename Expression>
	Vector<Expression> solve_eqbound(QPSettings *settings, Matrix<Expression> A, Vector<Expression> b, Vector<Expression> l, Matrix<Expression> B, Vector<Expression> x0){
		Vector<Expression> x = x0;

		/* SMALBE method */
		settings->smalbe = true;
		
		int it = 1;
		int it_mprgp_all = 0;
		int hess_mult_all = 0;
		double rho = settings->rho_coeff*settings->norm_A;
		double M = settings->M0;
		double betaM = settings->betaM;

		Vector<Expression> Ax;
		Matrix<Expression> ArhoBTB;
		Matrix<Expression> BT(B.cols(),B.rows());
		Vector<Expression> binner;

		double xAx,xTb,lambdaBx,xBBx;
		double f,L,L_old;
		
		Vector<Expression> lambda(B.rows()); /* lagrange multipliers of equality constraints */
		lambda(minlin::all) = 0.0;

		//BT = transpose(B); // TODO: this is not working, I dont know why :(
		for(int i=0; i < BT.rows(); i++){
			for(int j=0; j < BT.cols(); j++){
				BT(i,j) = B(j,i); // TODO: sorry for this
			}
		}

		Vector<Expression> Bx; /* B*x - feasibility */
		double norm_Bx;

		double norm_gp;
		ArhoBTB = BT*B;
		ArhoBTB = rho*ArhoBTB;
		ArhoBTB += A;
		binner = BT*lambda;
		binner = (-1)*binner;
		binner += b;
		x = solve_bound_smalbe(settings, ArhoBTB, binner, l, x, B, rho, M);
		it_mprgp_all += settings->it_mprgp;
		hess_mult_all += settings->hess_mult;

		/* compute gp */
		norm_gp = settings->norm_gp;
		
		Bx = B*x;
		norm_Bx = norm(Bx);

		/* compute original function value */
		Ax = A*x;
		xAx = dot(Ax,x);
		xTb = dot(b,x);
		f = 0.5*xAx - xTb;			
		/* compute lagrangian */
		L = f + rho*0.5*dot(Bx,Bx);
		
		/* main cycle */
		while((norm_Bx > settings->my_eps || norm_gp > settings->my_eps) && it < 1000){
			
			/* solve inner problem */
			binner = BT*lambda;
			binner = (-1)*binner;
			binner += b;
			x = solve_bound_smalbe(settings,ArhoBTB, binner, l, x, B, rho, M);
			it_mprgp_all += settings->it_mprgp;
			hess_mult_all += settings->hess_mult;
		
			/* compute gp */
			norm_gp = settings->norm_gp;			
			
			/* update lagrange multipliers (Uzawa) */
			Bx = B*x;
			lambda += rho*Bx;
			
			/* compute original function value */
			Ax = A*x;
			xAx = dot(Ax,x);
			xTb = dot(b,x);
			f = 0.5*xAx - xTb;			
			lambdaBx = dot(lambda,Bx);
			xBBx = dot(Bx,Bx);
			norm_Bx = std::sqrt(xBBx);

			/* compute lagrangian */
			L_old = L;
			L = f + lambdaBx + rho*0.5*dot(Bx,Bx); // TODO: rho*0.5*dot(Bx,Bx) could be eliminated
			
			/* update M */
			if (it > 0 && L < L_old + rho*0.5*xBBx){
				M = M/betaM; 
			}

			#ifdef QPOPT_DEBUG
				std::cout << " ----------- SMALBE iteration ---------- " << std::endl;
				std::cout << "it " << it << ": ";
				std::cout << "f = " << f << ", ||gP|| = " << norm_gp << ", ||Bx|| = " << norm_Bx << std::endl;
			#endif	
			
			it += 1;
		}
		
		/* set return values */
		settings->it_smalbe = it;
		settings->it_mprgp = it_mprgp_all;
		settings->hess_mult = hess_mult_all;
		settings->norm_Bx = norm_Bx;	
				
		return x;
	}

	/* equality and bound constrained without initial approximation, init approximation = 0 */
	template<typename Expression>
	Vector<Expression> solve_eqbound(QPSettings *settings, Matrix<Expression> A, Vector<Expression> b, Vector<Expression> l, Matrix<Expression> B){
		Vector<Expression> x0 = b;
		x0(minlin::all) = 0.0; /* default initial approximation */  

		return solve_eqbound(settings, A, b, l, B, x0);
	}



}


} // end namespace minlin	




#endif
