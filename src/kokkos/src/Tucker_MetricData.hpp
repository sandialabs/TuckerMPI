#ifndef TUCKER_KOKKOS_METRIC_DATA_HPP_
#define TUCKER_KOKKOS_METRIC_DATA_HPP_

#include <Kokkos_StdAlgorithms.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_Core.hpp>

namespace Tucker {
enum class Metric {
  MIN, MAX, SUM, NORM1, NORM2, MEAN, VARIANCE
};
}

namespace TuckerOnNode {

// fwd declaration
template<class ScalarType, class MemorySpace> class MetricData;

template<class ScalarType, class MemorySpace>
auto create_mirror(MetricData<ScalarType, MemorySpace> d){
  using T = MetricData<ScalarType, MemorySpace>;
  using T_mirror = typename T::HostMirror;

  auto vals = d.getValues();
  auto map  = d.getMap();
  auto vals_h = Kokkos::create_mirror(vals);
  typename T_mirror::map_t map_h(map.capacity());
  // we need this or deep copy below won't work
  Kokkos::deep_copy(map_h, map);

  return T_mirror(map_h, vals_h);
}

template<class ScalarType, class MemorySpaceFrom, class MemorySpaceDest>
void deep_copy(const MetricData<ScalarType, MemorySpaceDest> & dest,
	       const MetricData<ScalarType, MemorySpaceFrom> & from)
{
  auto vals_dest = dest.getValues();
  auto map_dest  = dest.getMap();
  auto vals_from = from.getValues();
  auto map_from  = from.getMap();
  Kokkos::deep_copy(map_dest, map_from);
  Kokkos::deep_copy(vals_dest, vals_from);
}

template<
  class ScalarType,
  class MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
class MetricData
{

public:
  using map_t = Kokkos::UnorderedMap<Tucker::Metric, int, MemorySpace>;
  using HostMirror = MetricData<ScalarType, Kokkos::HostSpace>;

  template<class MapType, class ValuesType>
  MetricData(MapType map, ValuesType values)
    : values_(values), metricToColumnIndex_(map){}

  MetricData(const std::vector<Tucker::Metric> & metrics, const int numValues)
    : values_("values", numValues, metrics.size()),
      metricToColumnIndex_(metrics.size() /*this is just a hint for the capacity*/)
  {

    using host_map_t = typename map_t::HostMirror;
    host_map_t map_h(metrics.size());
    for (std::size_t i=0; i<metrics.size(); ++i){
      const auto currMetric = metrics[i];

      auto r = map_h.insert(currMetric, i);
      if (!r.success()){
	throw std::runtime_error("MetricData: failure initialing map");
      }

      if(currMetric==Tucker::Metric::MIN){
	auto column = Kokkos::subview(values_, Kokkos::ALL, i);
	Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), column,
				   std::numeric_limits<ScalarType>::max());
      }
      if(currMetric==Tucker::Metric::MAX){
	auto column = Kokkos::subview(values_, Kokkos::ALL, i);
	Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), column,
				   std::numeric_limits<ScalarType>::min());
      }
    }
    Kokkos::deep_copy(metricToColumnIndex_, map_h);
  }

  KOKKOS_FUNCTION bool contains(Tucker::Metric key) const{
    const uint32_t ind = metricToColumnIndex_.find(key);
    return metricToColumnIndex_.valid_at(ind);
  }

  KOKKOS_FUNCTION auto get(Tucker::Metric key) const {
    auto ind = metricToColumnIndex_.find(key);
    const int colIndex = metricToColumnIndex_.value_at(ind);
    return Kokkos::subview(values_, Kokkos::ALL, colIndex);
  }

  //FIXME: these need to be private and make deep_copy/create_mirror friends
  auto getValues() const { return values_; }
  auto getMap() const { return metricToColumnIndex_; }

private:
  // each column contains all values of a given metric computed for all "slices"
  Kokkos::View<ScalarType**, Kokkos::LayoutLeft, MemorySpace> values_ = {};

  map_t metricToColumnIndex_ = {};
};

}

#endif
