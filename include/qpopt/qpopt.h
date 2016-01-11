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
