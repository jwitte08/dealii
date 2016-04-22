#include <deal.II/base/multithread_info.h>
#include <deal.II/base/utilities.h>
#include <MyLaplace.h>

#include <benchmark/benchmark_api.h>

void laplace_meshworker2d(benchmark::State &state)
{
  while(state.KeepRunning())
    {
      dealii::MultithreadInfo::set_thread_limit(state.range_x());

      MPI_Comm mpi_communicator (MPI_COMM_WORLD);
      std::ofstream   fout("/dev/null");
      std::cout.rdbuf(fout.rdbuf());
      dealii::ConditionalOStream pcout(std::cout,
                                       dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0);
      std::ofstream logfile("deallog");
      dealii::deallog.attach(logfile);
      // of output only on the first process                                                                                        
      
      if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
	dealii::deallog.depth_console (0);
      dealii::TimerOutput timer (mpi_communicator, pcout,
                                 dealii::TimerOutput::never,
                                 dealii::TimerOutput::wall_times);
      MyLaplace<2,false,1> dgmethod(timer, mpi_communicator, pcout);
      dgmethod.run ();
    }
}

BENCHMARK(laplace_meshworker2d)
->Threads(1)
->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)
//->ArgPair(1,2)->ArgPair(2,2)->ArgPair(4,2)->ArgPair(8,2)->ArgPair(16,2)->ArgPair(32,2)
//->ArgPair(1,4)->ArgPair(2,4)->ArgPair(4,4)->ArgPair(8,4)->ArgPair(16,4)->ArgPair(32,4)                                              
//->ArgPair(1,6)->ArgPair(2,6)->ArgPair(4,6)->ArgPair(8,6)->ArgPair(16,6)->ArgPair(32,6)                                              
//->ArgPair(1,8)->ArgPair(2,8)->ArgPair(4,8)->ArgPair(8,8)->ArgPair(16,8)->ArgPair(32,8)
->UseRealTime();

// BENCHMARK(laplace_meshworker3d)
// ->Threads(1)
// ->ArgPair(1,2)->ArgPair(2,2)->ArgPair(4,2)->ArgPair(8,2)->ArgPair(16,2)->ArgPair(32,2)
//->ArgPair(1,6)->ArgPair(2,6)->ArgPair(4,6)->ArgPair(8,6)->ArgPair(16,6)->ArgPair(32,6)
//->ArgPair(1,8)->ArgPair(2,8)->ArgPair(4,8)->ArgPair(8,8)->ArgPair(16,8)->ArgPair(32,8)
//->UseRealTime();
                                                                                                                      

BENCHMARK_MAIN()
