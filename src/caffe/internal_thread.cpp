#include <thread>
#include <exception>

#include "caffe/internal_thread.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

InternalThread::~InternalThread() {
  StopInternalThread();
}

bool InternalThread::is_started() const {
  return thread_ && thread_->joinable();
}

bool InternalThread::must_stop() {
  return thread_ && interruption_requested_;
}

void InternalThread::StartInternalThread() {
  CHECK(!is_started()) << "Threads should persist and not be restarted.";

  int device = 0;
#ifndef CPU_ONLY
  CUDA_CHECK(cudaGetDevice(&device));
#endif
  Caffe::Brew mode = Caffe::mode();
  int rand_seed = caffe_rng_rand();
  int solver_count = Caffe::solver_count();
  int solver_rank = Caffe::solver_rank();
  bool multiprocess = Caffe::multiprocess();

  interruption_requested_ = false;

  try {
    thread_.reset(new std::thread(&InternalThread::entry, this, device, mode,
          rand_seed, solver_count, solver_rank, multiprocess));
  } catch (std::exception& e) {
    LOG(FATAL) << "Thread exception: " << e.what();
  }
}

void InternalThread::entry(int device, Caffe::Brew mode, int rand_seed,
    int solver_count, int solver_rank, bool multiprocess) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaSetDevice(device));
#endif
  Caffe::set_mode(mode);
  Caffe::set_random_seed(rand_seed);
  Caffe::set_solver_count(solver_count);
  Caffe::set_solver_rank(solver_rank);
  Caffe::set_multiprocess(multiprocess);

  try{
    InternalThreadEntry();
  } catch (operation_aborted& e) {
  }
}

void InternalThread::StopInternalThread() {
  if (is_started()) {
    interruption_requested_ = true;
    try {
      thread_->join();
    } catch (std::exception& e) {
      LOG(FATAL) << "Thread exception: " << e.what();
    }
  }
}

}  // namespace caffe
