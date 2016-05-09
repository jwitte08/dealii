//dealii
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/template_constraints.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/lac/vector.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/local_integrator.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/simple.h>

//dealii::workstream
#include <deal.II/base/config.h>
#include <deal.II/base/graph_coloring.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/template_constraints.h>
#include <deal.II/base/std_cxx11/function.h>
#include <deal.II/base/std_cxx11/bind.h>
#include <deal.II/base/thread_local_storage.h>
#include <deal.II/base/parallel.h>
#ifdef DEAL_II_WITH_THREADS
#  include <deal.II/base/thread_management.h>
#  include <tbb/pipeline.h>
#endif
#include <utility>
#include <memory>

//benchmark
#include <benchmark/benchmark_api.h>

//c++
#include <iostream>
#include <vector>
#include <cstdio>
#include <functional>

// RHS INTEGRATOR                                                                                                                    
template <int dim>
class RHSIntegrator final : public dealii::MeshWorker::LocalIntegrator<dim>
{
 public:
  RHSIntegrator() : dealii::MeshWorker::LocalIntegrator<dim>() {};
  RHSIntegrator(bool use_cell_, bool use_bdry_, bool use_face_) 
    : dealii::MeshWorker::LocalIntegrator<dim>(use_cell_, use_bdry_, use_face_) {};
  void cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const override;
};

// IMPLEMENTATION: RHS_INTEGRATOR                                                                                                    
template <int dim>
void RHSIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  std::vector<double> f;
  f.resize(info.fe_values(0).n_quadrature_points,1.);
  dealii::LocalIntegrators::L2::L2(dinfo.vector(0).block(0),
                                   info.fe_values(0),
                                   f);
}

// WORKSTREAM
namespace WorkStream
{
  using namespace dealii;

  namespace internal
  {
    namespace Implementation3
    {
      /**                                                                                                                            
       * A structure that contains a pointer to scratch and copy data objects                                                        
       * along with a flag that indicates whether this object is currently in                                                        
       * use.                                                                                                                        
       */
    
      template <typename Iterator,
		typename ScratchData,
		typename CopyData>
      struct ScratchAndCopyDataObjects
      {
	std_cxx11::shared_ptr<ScratchData> scratch_data;
	std_cxx11::shared_ptr<CopyData>    copy_data;
	bool                               currently_in_use;

	/**                                                                                                                          
	 * Default constructor.                                                                                                      
	 */
	ScratchAndCopyDataObjects ()
	  :
	  currently_in_use (false)
	{}

	ScratchAndCopyDataObjects (ScratchData *p,
				   CopyData *q,
				   const bool in_use)
	  :
	  scratch_data (p),
	  copy_data (q),
	  currently_in_use (in_use)
	{}

	//TODO: when we push back an object to the list of scratch objects, in                                                
	//      Worker::operator(), we first create an object and then copy
	//      it to the end of this list. this involves having two objects
	//      of the current type having pointers to it, each with their own                                                       
	//      currently_in_use flag. there is probably little harm in this because                                                 
	//      the original one goes out of scope right away again, but it's                                                        
	//      certainly awkward. one way to avoid this would be to use unique_ptr                                                  
	//      but we'd need to figure out a way to use it in non-C++11 mode                                                        
         
	ScratchAndCopyDataObjects (const ScratchAndCopyDataObjects &o)
	  :
	  scratch_data (o.scratch_data),
	  copy_data (o.copy_data),
	  currently_in_use (o.currently_in_use)
	{}
      };

      /**                                                                                                                            
       * A class that manages calling the worker and copier functions. Unlike                                                        
       * the other implementations, parallel_for is used instead of a                                                                
       * pipeline.                                                                                                                   
       */
      template <typename Iterator,
		typename ScratchData,
		typename CopyData>
      class WorkerAndCopier
      {
      public:
	/**                                                                                                                          
	 * Constructor.                                                                                                              
	 */
	WorkerAndCopier (const std_cxx11::function<void (const Iterator &,
							 ScratchData &,
							 CopyData &)> &worker,
			 const std_cxx11::function<void (const CopyData &)> &copier,
			 const ScratchData    &sample_scratch_data,
			 const CopyData       &sample_copy_data)
	  :
	  worker (worker),
	  copier (copier),
	  sample_scratch_data (sample_scratch_data),
	  sample_copy_data (sample_copy_data)
	{}

	/**                                                                                                                          
	 * The function that calls the worker and the copier functions on a                                                          
	 * range of items denoted by the two arguments.                                                                              
	 */
	void operator() (const tbb::blocked_range<typename std::vector<Iterator>::const_iterator> &range)
	{
	  // we need to find an unused scratch and corresponding copy                                                                
	  // data object in the list that corresponds to the current                                                                 
	  // thread and then mark it as used. If we can't find one,                                                                  
	  // create one as discussed in the discussion of the documentation                                                          
	  // of the IteratorRangeToItemStream::scratch_data variable,                                                                
	  // there is no need to synchronize access to this variable                                                                 
	  // using a mutex as long as we have no yield-point in between.                                                             
	  // This means that we can't take an iterator into the list                                                                 
	  // now and expect it to still be valid after calling the worker,                                                           
	  // but we at least do not have to lock the following section.                                                              
 
	  ScratchData *scratch_data = 0;
	  CopyData    *copy_data    = 0;
	  {
	    ScratchAndCopyDataList &scratch_and_copy_data_list = data.get();

	    // see if there is an unused object. if so, grab it and mark                                                             
	    // it as used                                                                                                            
	    for (typename ScratchAndCopyDataList::iterator
		   p = scratch_and_copy_data_list.begin();
		 p != scratch_and_copy_data_list.end(); ++p)
	      if (p->currently_in_use == false)
		{
		  scratch_data = p->scratch_data.get();
		  copy_data    = p->copy_data.get();
		  p->currently_in_use = true;
		  break;
		}

	    // if no element in the list was found, create one and mark it as used                                                   
	    if (scratch_data == 0)
	      {
		Assert (copy_data==0, ExcInternalError());
		scratch_data = new ScratchData(sample_scratch_data);
		copy_data    = new CopyData(sample_copy_data);

		typename ScratchAndCopyDataList::value_type
		  new_scratch_object (scratch_data, copy_data, true);
		scratch_and_copy_data_list.push_back (new_scratch_object);
	      }
	  }

	  // then call the worker and copier functions on each                                                                       
	  // element of the chunk we were given.                                                                                     
	  for (typename std::vector<Iterator>::const_iterator p=range.begin();
	       p != range.end(); ++p)
	    {
	      try
		{
		  if (worker)
		    worker (*p,
			    *scratch_data,
			    *copy_data);
		  if (copier)
		    copier (*copy_data);
		}
	      catch (const std::exception &exc)
		{
		  Threads::internal::handle_std_exception (exc);
		}
	      catch (...)
		{
		  Threads::internal::handle_unknown_exception ();
		}
	    }

	  // finally mark the scratch object as unused again. as above, there                                                        
	  // is no need to lock anything here since the object we work on                                                            
	  // is thread-local                                                                                                         
	  {
	    ScratchAndCopyDataList &scratch_and_copy_data_list = data.get();

	    for (typename ScratchAndCopyDataList::iterator p =
		   scratch_and_copy_data_list.begin(); p != scratch_and_copy_data_list.end();
		 ++p)
	      if (p->scratch_data.get() == scratch_data)
		{
		  Assert(p->currently_in_use == true, ExcInternalError());
		  p->currently_in_use = false;
		}
	  }

	}
      private:
	typedef
	typename Implementation3::ScratchAndCopyDataObjects<Iterator,ScratchData,CopyData>
	ScratchAndCopyDataObjects;

	/**                                                                                                                          
	 * Typedef to a list of scratch data objects. The rationale for this                                                         
	 * list is provided in the variables that use these lists.                                                                   
	 */
	typedef std::list<ScratchAndCopyDataObjects> ScratchAndCopyDataList;

	Threads::ThreadLocalStorage<ScratchAndCopyDataList> data;

	/**                                                                                                                          
	 * Pointer to the function that does the assembling on the sequence of                                                       
	 * cells.                                                                                                                    
	 */
	const std_cxx11::function<void (const Iterator &,
					ScratchData &,
					CopyData &)> worker;

	/**                                                                                                                          
	 * Pointer to the function that does the copying from local                                                                  
	 * contribution to global object.                                                                                            
	 */
	const std_cxx11::function<void (const CopyData &)> copier;

	/**                                                                                                                          
	 * References to sample scratch and copy data for when we need them.                                                         
	 */
	const ScratchData    &sample_scratch_data;
	const CopyData       &sample_copy_data;
      };
    } // end namespace Implementation3
  } // end namespace internal
} // end namespace WorkStream

// CellWorkerAndCopier
template <typename INTEGRATOR,
	  typename ASSEMBLER,
	  typename DOFINFO,
	  int dim,
	  typename ITERATOR,
	  typename SDATA>
class CellWorkerAndCopier
{
public:
  CellWorkerAndCopier (ASSEMBLER  &assembler, 
		       const SDATA  &sample_scratch_data_,
		       const dealii::MeshWorker::DoFInfoBox<dim,DOFINFO>  &sample_copy_data_)
    :
    sample_scratch_data (sample_scratch_data_),
    sample_copy_data (sample_copy_data_)
  {}

  void operator() (const tbb::blocked_range<typename std::vector<ITERATOR>::const_iterator> &range) const
  {
	  // // we need to find an unused scratch and corresponding copy                                                                
	  // // data object in the list that corresponds to the current                                                                 
	  // // thread and then mark it as used. If we can't find one,                                                                  
	  // // create one as discussed in the discussion of the documentation                                                          
	  // // of the IteratorRangeToItemStream::scratch_data variable,                                                                
	  // // there is no need to synchronize access to this variable                                                                 
	  // // using a mutex as long as we have no yield-point in between.                                                             
	  // // This means that we can't take an iterator into the list                                                                 
	  // // now and expect it to still be valid after calling the worker,                                                           
	  // // but we at least do not have to lock the following section.                                                              
 
	  // ScratchData *scratch_data = 0;
	  // CopyData    *copy_data    = 0;
	  // {
	  //   ScratchAndCopyDataList &scratch_and_copy_data_list = data.get();

	  //   // see if there is an unused object. if so, grab it and mark                                                             
	  //   // it as used                                                                                                            
	  //   for (typename ScratchAndCopyDataList::iterator
	  // 	   p = scratch_and_copy_data_list.begin();
	  // 	 p != scratch_and_copy_data_list.end(); ++p)
	  //     if (p->currently_in_use == false)
	  // 	{
	  // 	  scratch_data = p->scratch_data.get();
	  // 	  copy_data    = p->copy_data.get();
	  // 	  p->currently_in_use = true;
	  // 	  break;
	  // 	}

	  //   // if no element in the list was found, create one and mark it as used                                                   
	  //   if (scratch_data == 0)
	  //     {
	  // 	Assert (copy_data==0, ExcInternalError());
	  // 	scratch_data = new ScratchData(sample_scratch_data);
	  // 	copy_data    = new CopyData(sample_copy_data);

	  // 	typename ScratchAndCopyDataList::value_type
	  // 	  new_scratch_object (scratch_data, copy_data, true);
	  // 	scratch_and_copy_data_list.push_back (new_scratch_object);
	  //     }
	  // }

	  // // then call the worker and copier functions on each                                                                       
	  // // element of the chunk we were given.                                                                                     
	  // for (typename std::vector<Iterator>::const_iterator p=range.begin();
	  //      p != range.end(); ++p)
	  //   {
	  //     try
	  // 	{
	  // 	  if (worker)
	  // 	    worker (*p,
	  // 		    *scratch_data,
	  // 		    *copy_data);
	  // 	  if (copier)
	  // 	    copier (*copy_data);
	  // 	}
	  //     catch (const std::exception &exc)
	  // 	{
	  // 	  Threads::internal::handle_std_exception (exc);
	  // 	}
	  //     catch (...)
	  // 	{
	  // 	  Threads::internal::handle_unknown_exception ();
	  // 	}
	  //   }

	  // // finally mark the scratch object as unused again. as above, there                                                        
	  // // is no need to lock anything here since the object we work on                                                            
	  // // is thread-local                                                                                                         
	  // {
	  //   ScratchAndCopyDataList &scratch_and_copy_data_list = data.get();

	  //   for (typename ScratchAndCopyDataList::iterator p =
	  // 	   scratch_and_copy_data_list.begin(); p != scratch_and_copy_data_list.end();
	  // 	 ++p)
	  //     if (p->scratch_data.get() == scratch_data)
	  // 	{
	  // 	  Assert(p->currently_in_use == true, ExcInternalError());
	  // 	  p->currently_in_use = false;
	  // 	}
	  // }
    
  }
private:
  typedef typename dealii::MeshWorker::DoFInfoBox<dim,DOFINFO> CDATA;
 
  typedef
  typename WorkStream::internal::Implementation3::ScratchAndCopyDataObjects<ITERATOR,SDATA,CDATA>
  ScratchAndCopyDataObjects;

  typedef std::list<ScratchAndCopyDataObjects> ScratchAndCopyDataList;

  dealii::Threads::ThreadLocalStorage<ScratchAndCopyDataList> data;
  
  const INTEGRATOR  integrator;
  ASSEMBLER  &assembler;
  const SDATA  &sample_scratch_data;
  const CDATA  &sample_copy_data;
};

// MAIN
void mw_constrhs(benchmark::State &state)
{
  dealii::MultithreadInfo::set_thread_limit (state.range_y());  
  const unsigned int dim = 2;
  const unsigned int degree = 1;

  dealii::Triangulation<2> tr;
  const dealii::FE_DGQ<dim> fe{degree};
  dealii::DoFHandler<dim> dof_handler{tr};
  const dealii::MappingQ1<dim> mapping;

  const RHSIntegrator<dim> rhs_integrator{true,false,false};
  dealii::MeshWorker::IntegrationInfoBox<dim> info_box;
  std::unique_ptr<dealii::MeshWorker::DoFInfo<dim> > dof_info;
  const dealii::MeshWorker::LoopControl lctrl{} ;
  
  dealii::GridGenerator::hyper_cube (tr,0.,1.);
  tr.refine_global(state.range_x());
  dof_handler.distribute_dofs(fe);

  info_box.initialize_gauss_quadrature(degree+1, 0, 0);
  dealii::UpdateFlags update_flags = dealii::update_quadrature_points | dealii::update_values ;
  info_box.add_update_flags(update_flags, true, false, false, false);
  info_box.initialize(fe, mapping);
  dof_info.reset(new dealii::MeshWorker::DoFInfo<dim> {dof_handler});

  while(state.KeepRunning())
    {
      dealii::Vector<double> rhs;
      rhs.reinit(dof_handler.n_dofs());
      
      dealii::MeshWorker::Assembler::ResidualSimple<dealii::Vector<double> > rhs_assembler;
      dealii::AnyData data;
      data.add<dealii::Vector<double>* >(&rhs, "RHS");
      rhs_assembler.initialize(data);
      
      //template definitions
      typedef dealii::MeshWorker::DoFInfo<dim> DOFINFO ;
      typedef dealii::MeshWorker::IntegrationInfoBox<dim> INFOBOX ;
      typedef dealii::MeshWorker::Assembler::ResidualSimple<dealii::Vector<double> > ASSEMBLER ;
      typedef typename dealii::DoFHandler<dim>::active_cell_iterator ITERATOR ;
      typedef RHSIntegrator<dim> INTEGRATOR;

      // ITERATOR begin{dof_handler.begin_active()} ;
      // typename dealii::identity<ITERATOR>::type end{dof_handler.end()};
      // DOFINFO &dinfo = *dof_info ;
      // INFOBOX &info = info_box  ;
      // ASSEMBLER &assembler = rhs_assembler ;

      // Loop over all cells                                                                                                         
#ifdef DEAL_II_MESHWORKER_PARALLEL
      
      dealii::MeshWorker::DoFInfoBox<dim,DOFINFO> dinfo_box(*dof_info);
      
      rhs_assembler.initialize_info(dinfo_box.cell, false);
      for (unsigned int i=0; i<dealii::GeometryInfo<dim>::faces_per_cell; ++i)
        {
          rhs_assembler.initialize_info(dinfo_box.interior[i], true);
          rhs_assembler.initialize_info(dinfo_box.exterior[i], true);
        }

      // OLD...
      // dealii::WorkStream::run(dof_handler.begin_active(), dof_handler.end(),
      // 			      std::bind(&dealii::MeshWorker::cell_action<INFOBOX,DOFINFO,dim,dim,ITERATOR>,
      // 					std::placeholders::_1,  std::placeholders::_3,  std::placeholders::_2,
      // 					[&rhs_integrator](DOFINFO& dinfo, dealii::MeshWorker::IntegrationInfo<dim,dim>& info)
      // 					{rhs_integrator.cell(dinfo,info);},
      // 					nullptr, nullptr, lctrl),
      // 			      std::bind(&dealii::internal::assemble<dim,DOFINFO,ASSEMBLER>, std::placeholders::_1, &rhs_assembler),
      // 			      info_box, dinfo_box) ;

      // NEW... WorkStream::run

      //template definitions
      typedef dealii::MeshWorker::DoFInfoBox<dim,DOFINFO> CopyData ; // 
      typedef dealii::MeshWorker::IntegrationInfoBox<dim> ScratchData ; // INFOBOX
      typedef typename dealii::DoFHandler<dim>::active_cell_iterator Iterator ; // ITERATOR
      
      // build_colored_iterators
      Iterator begin{dof_handler.begin_active()} ;
      typename dealii::identity<Iterator>::type end{dof_handler.end()};
      // only one color
      std::vector<std::vector<Iterator> >  all_iterators(1);
      for (Iterator p=begin; p!=end; ++p)
	all_iterators[0].push_back (p);

      // WorkStream::run() arguments
      const std::vector<std::vector<Iterator> >& colored_iterators = all_iterators;
      auto worker = std::bind(&dealii::MeshWorker::cell_action<INFOBOX,DOFINFO,dim,dim,ITERATOR>,                   
			      std::placeholders::_1,  std::placeholders::_3,  std::placeholders::_2,     
			      [&rhs_integrator](DOFINFO& dinfo, dealii::MeshWorker::IntegrationInfo<dim,dim>& info)
			      {rhs_integrator.cell(dinfo,info);},                                        
			      nullptr, nullptr, lctrl);
      auto copier = std::bind(&dealii::internal::assemble<dim,DOFINFO,ASSEMBLER>, std::placeholders::_1, &rhs_assembler);
      const ScratchData& sample_scratch_data = info_box;
      const CopyData& sample_copy_data = dinfo_box;
      const unsigned int  queue_length = 2 * dealii::MultithreadInfo::n_threads();
      const unsigned int  chunk_size = 8;

      // NEW... templated::run
      
      CellWorkerAndCopier<INTEGRATOR,
			  ASSEMBLER,
			  DOFINFO,
			  dim,
			  ITERATOR,
			  ScratchData>
	cellworker_and_copier{sample_scratch_data,sample_copy_data};

      {
	Assert (queue_length > 0,
		ExcMessage ("The queue length must be at least one, and preferably "
			    "larger than the number of processors on this system."));
	(void)queue_length; // removes -Wunused-parameter warning in optimized mode                                                  
     
	Assert (chunk_size > 0,
		ExcMessage ("The chunk_size must be at least one."));
	(void)chunk_size; // removes -Wunused-parameter warning in optimized mode 

	// we want to use TBB if we have support and if it is not disabled at                                                        
	// runtime:                                                                                                                  
#ifdef DEAL_II_WITH_THREADS
	if (dealii::MultithreadInfo::n_threads()==1)
#endif
	  {
	    // need to copy the sample since it is marked const                                                                      
	    ScratchData scratch_data = sample_scratch_data;
	    CopyData    copy_data    = sample_copy_data;

	    for (unsigned int color=0; color<colored_iterators.size(); ++color)
	      for (typename std::vector<Iterator>::const_iterator p = colored_iterators[color].begin();
		   p != colored_iterators[color].end(); ++p)
		{
		  // need to check if the function is not the zero function. To                                                      
		  // check zero-ness, create a C++ function out of it and check that                                                 
		  if (static_cast<const dealii::std_cxx11::function<void (const Iterator &,
								  ScratchData &,
								  CopyData &)>& >(worker))
		    worker (*p, scratch_data, copy_data);
		  if (static_cast<const dealii::std_cxx11::function<void (const CopyData &)>& >(copier))
		    copier (copy_data);
		}
	  }

#ifdef DEAL_II_WITH_THREADS
	else // have TBB and use more than one thread                                                                                
	  {
	    // loop over the various colors of what we're given                                                                      
	    for (unsigned int color=0; color<colored_iterators.size(); ++color)
	      if (colored_iterators[color].size() > 0)
		{
              typedef
		WorkStream::internal::Implementation3::WorkerAndCopier<Iterator,ScratchData,CopyData>
		WorkerAndCopier;

              typedef
		typename std::vector<Iterator>::const_iterator
		RangeType;

              WorkerAndCopier worker_and_copier (worker,
                                                 copier,
                                                 sample_scratch_data,
                                                 sample_copy_data);

	      tbb::parallel_for (tbb::blocked_range<RangeType>
                                 (colored_iterators[color].begin(),
                                  colored_iterators[color].end(),
                                  /*grain_size=*/chunk_size),
                                 dealii::std_cxx11::bind (&WorkerAndCopier::operator(),
							  dealii::std_cxx11::ref(worker_and_copier),
							  dealii::std_cxx11::_1),
                                 tbb::auto_partitioner());
		}
	  }
#endif
      } // end of myWorkStream::run()

#else
      // for (ITERATOR cell = begin; cell != end; ++cell)
      //   {
      // 	  dealii::WorkStream::cell_action<INFOBOX,DOFINFO,dim,spacedim>(cell,
      // 									dinfo_box, info_box,
      // 									nullptr, nullptr, nullptr,
      // 									lctrl);
      //     dinfo_box.assemble(assembler);
      //   }
#endif
      //output
      //rhs.print(std::cout);
      //std::cout << rhs[0] << std::endl;
    }
}

BENCHMARK(mw_constrhs)
->Threads(1)
->ArgPair(11,4)
->UseRealTime();

BENCHMARK_MAIN()
