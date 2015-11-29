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

//#define MINLIN_DEBUG
#define QPOPT_DEBUG

#include <iostream>
#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

#include "minlin/vector.h"

using namespace minlin::threx;

template<class Expression>
typename Expression::value_type

class QPOpt {

	public:
		static void solve_unconstrained(const minlin::threx::Vector<Expression>& x){
			
			std::cout << x << std::endl;
		
		}
};




#endif
