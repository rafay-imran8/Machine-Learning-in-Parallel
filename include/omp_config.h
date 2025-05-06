/**
 * omp_config.h - OpenMP configuration for the hybrid parallel ML system
 *
 * This header centralizes OpenMP thread management across all models
 * ensuring each MPI rank uses a consistent number of threads.
 */

 #ifndef OMP_CONFIG_H
 #define OMP_CONFIG_H
 
 #include <omp.h>
 #include <iostream>
 
 // Default to 5 threads unless overridden at compile time
 #ifndef OMP_NUM_THREADS
 #define OMP_NUM_THREADS 5
 #endif
 
 /**
  * Sets up OpenMP threads according to the OMP_NUM_THREADS macro
  * Call this function at the start of model training/prediction
  */
 inline void setup_openmp_threads() {
     // Explicitly set thread count regardless of environment
     omp_set_num_threads(OMP_NUM_THREADS);
     
     // Verify thread count (debug only, can be removed in production)
     #pragma omp parallel
     {
         #pragma omp master
         {
             std::cout << "OpenMP using " << omp_get_num_threads() 
                       << " threads (max: " << omp_get_max_threads() << ")" << std::endl;
         }
     }
 }
 
 #endif // OMP_CONFIG_H