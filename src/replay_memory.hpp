#ifndef REPLAY_MEMORY_HPP
#define REPLAY_MEMORY_HPP

#include <torch/torch.h>

class ReplayMemory
{
    public:
        struct Replay
        {
            torch::Tensor state;
            torch::Tensor action;
            torch::Tensor reward;
            torch::Tensor done;
            torch::Tensor state_next;
        } typedef replay_t;

        /**
         * @brief Construct a new ReplayMemory object, empty/unused
         */
        ReplayMemory();

        /**
         * @brief Construct a new ReplayMemory object
         * 
         * @param capacity the capacity of the replay memory
         */
        ReplayMemory(std::size_t capacity);

        /**
         * @brief Destroy the ReplayMemory, empty/unused
         */
        virtual ~ReplayMemory();

        /**
         * @brief Add replay to memory
         * 
         * @param replay a replay instance to add to memory
         */
        void add(replay_t replay);

        /**
         * @brief Sample replays from memory
         * 
         * @param size size of replay sample
         * 
         * @return vector of replays
         */
        std::vector<replay_t> sample(int  size);

        /**
         * @brief Get the size of replay memory in use
         * 
         * @return size of replay memory in use
         */
        int64_t size();

    private:
        std::size_t capacity;
        std::deque<replay_t> memory;

};

#endif