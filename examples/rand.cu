/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

#include <iostream>

int main() {

    std::cout << minlin::threx::rand<double>(5) << std::endl;

    std::cout << minlin::threx::rand<double>(10) << std::endl;
    
    std::cout << minlin::threx::rand<double>(5) << std::endl;
    
}
