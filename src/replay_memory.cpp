
#include <algorithm>
#include <iterator>
#include <random>

#include "replay_memory.hpp"

ReplayMemory::ReplayMemory() = default;

ReplayMemory::~ReplayMemory() = default;

ReplayMemory::ReplayMemory(std::size_t capacity) : capacity(capacity) {};

void ReplayMemory::add(replay_t replay)
{
    // remove last if at capacity
    if(memory.size() == capacity)
        memory.pop_front();

    // add replay to memory
    memory.push_back(replay);
}

std::vector<ReplayMemory::replay_t> ReplayMemory::sample(int size)
{
    std::vector<replay_t> v(size);
    std::sample(memory.begin(), memory.end(), v.begin(), v.size(),
                std::mt19937{std::random_device{}()});
    return v;
}

int64_t ReplayMemory::size()
{
    return memory.size();
}
