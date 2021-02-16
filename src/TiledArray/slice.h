#ifndef TILEDARRAY_SLICE_H__INCLUDED
#define TILEDARRAY_SLICE_H__INCLUDED

#include <TiledArray/dist_array.h>

#include <vector>
#include <stdint.h>

namespace TiledArray {

  class slice {
  public:
    using index_t = int64_t;
    const index_t start, stop;
    const int rank;
    slice(index_t idx)
      : start(idx), stop(idx+1), rank(0) {}
    slice(index_t start, index_t stop)
      : start(start), stop(stop), rank(1) {}
    // index_t begin() const { return this->start; }
    // index_t end() const { return this->stop; }
  };

  void make_array(std::vector<slice> s);

  template<class ... Args>
  auto make_array(const DistArray<Args...> &A, std::vector<slice> s, const World &world);

  template<class ... Args>
  auto make_array(const DistArray<Args...> &A, std::vector<slice> s);

}  // namespace TiledArray

using namespace TiledArray;
int main() {
  slice(0);
  slice({1,2});
  make_array({0, 4, {0,0}});
}

#endif  // TILEDARRAY_SLICE_H__INCLUDED
