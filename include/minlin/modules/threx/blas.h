#pragma once


namespace minlin {
namespace threx {

/************************************************************************
 * singleton for cublas initialization
 ************************************************************************/
// don't need if only using OMP backend everywhere

#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP && THRUST_HOST_SYSTEM==THRUST_HOST_SYSTEM_OMP
#define MINLIN_INIT
#else
#include <cublas_v2.h>

class CublasState {
public:
    static CublasState* instance();
    CublasState() {
        cublasStatus_t stat = cublasCreate(&handle_);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "CUBLAS initialization failed" << std::endl;
            exit(1);
        }
    }
    cublasHandle_t& handle(){
        return handle_;
    }
private :
    static CublasState* instance_;
    cublasHandle_t handle_;

    // hide copy constructor and assignment out of harms way
    CublasState& operator=(const CublasState&);
    CublasState(const CublasState&);
};

#define MINLIN_INIT CublasState* CublasState::instance_  = 0;   \
CublasState* CublasState::instance() {                          \
    if( instance_==0 ) {                                        \
        instance_ = new CublasState;                            \
    }                                                           \
    return instance_;                                           \
}

#endif

}
}
