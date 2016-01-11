/*******************************************************************************
QP OPTIMAL SOLVERS for MINLIN library
Lukas Pospisil, 2016
lukas.pospisil@vsb.cz

* using MINLIN library (CSCS Lugano - Timothy Moroney, Ben Cumming)
* created during HPCCausality project (USI Lugano - Illia Horenko, Patrick Gagliardini, Will Sawyer)
*******************************************************************************/

#ifndef QPOPT_CG_H
#define	QPOPT_CG_H

#include <iostream>
#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

#include "minlin/vector.h"
#include "minlin/matrix.h"
#include "qpopt/settings.h"

using namespace minlin::threx;


namespace minlin {

namespace QPOpt {

	/* unconstrained with initial approximation */
	template<typename Expression>
	Vector<Expression> cg(QPSettings *settings, Matrix<Expression> A, Vector<Expression> b, Vector<Expression> x0){
		settings->cg = true; /* this problem is computed using CG method */
		
		Vector<Expression> x = x0; /* set aprroximation as initial */

		/* CG method */
		Vector<Expression> g; /* gradient */
		Vector<Expression> p; /* A-conjugate vector */
		Vector<Expression> Ap; /* A*p */
		int it = 0; /* iteration counter */
		int hess_mult = 0; /* number of hessian multiplications */
		double normg, alpha, beta, pAp, gg, gg_old;
	
		g = A*x; hess_mult += 1; g -= b; /* compute gradient */
		p = g; /* initial conjugate gradient */

		gg = dot(g,g);
		normg = std::sqrt(gg);
		while(normg > settings->my_eps && it < settings->maxit){
			/* compute new approximation */
			Ap = A*p; hess_mult += 1;
			pAp = dot(Ap,p);
			alpha = gg/pAp; /* compute step-size */
			x -= alpha*p; /* set new approximation */

			/* compute gradient recursively */
			g -= alpha*Ap; 
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
		settings->hess_mult = hess_mult;		

		return x;
	}

	/* unconstrained without initial approximation, init approximation = 0 */
	template<typename Expression>
	Vector<Expression> cg(QPSettings *settings, Matrix<Expression> A, Vector<Expression> b){
		Vector<Expression> x0 = b;
		x0(minlin::all) = 0.0; /* default initial approximation */  

		return cg(settings, A, b, x0);
	}

}


} // end namespace minlin	




#endif
