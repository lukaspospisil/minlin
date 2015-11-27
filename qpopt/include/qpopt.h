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

#include <thrust/functional.h>
#include <iostream>
#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

class QPOpt {
	public:
		void solve_unconstrained();
};



#endif
