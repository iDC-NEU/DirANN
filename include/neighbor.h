#pragma once

#include <cstddef>
#include <mutex>
#include <vector>
#include <limits>
#include "utils.h"

namespace pipeann {

  struct Neighbor {
    unsigned id;
    float distance;
    bool flag;
    bool visited;
    int idx;

    Neighbor() = default;
    Neighbor(unsigned id, float distance, bool f) : id{id}, distance{distance}, flag(f), visited(false), idx(0) {
    }

    inline bool operator<(const Neighbor &other) const {
      return (distance < other.distance) || (distance == other.distance && id < other.id);
    }
    inline bool operator==(const Neighbor &other) const {
      return (id == other.id);
    }
    inline bool operator>(const Neighbor &other) const {
      return (distance > other.distance) || (distance == other.distance && id > other.id);
    }
  };

  template<typename TagT = int>
  struct NeighborTag {
    TagT tag;
    float dist;
    NeighborTag() = default;

    NeighborTag(TagT tag, float dist) : tag{tag}, dist{dist} {
    }
    inline bool operator<(const NeighborTag &other) const {
      return (dist < other.dist) || (dist == other.dist && tag < other.tag);
    }
    inline bool operator==(const NeighborTag &other) const {
      return (tag == other.tag);
    }
  };

  static inline unsigned InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn) {
    // find the location to insert
    unsigned left = 0, right = K - 1;
    if (addr[left].distance > nn.distance) {
      memmove((char *) &addr[left + 1], &addr[left], K * sizeof(Neighbor));
      addr[left] = nn;
      return left;
    }
    if (addr[right].distance < nn.distance) {
      addr[K] = nn;
      return K;
    }
    while (right > 1 && left < right - 1) {
      unsigned mid = (left + right) / 2;
      if (addr[mid].distance > nn.distance)
        right = mid;
      else
        left = mid;
    }
    // check equal ID

    while (left > 0) {
      if (addr[left].distance < nn.distance)
        break;
      if (addr[left].id == nn.id)
        return K + 1;
      left--;
    }
    if (addr[left].id == nn.id || addr[right].id == nn.id)
      return K + 1;
    memmove((char *) &addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
    addr[right] = nn;
    return right;
  }
}  // namespace pipeann
