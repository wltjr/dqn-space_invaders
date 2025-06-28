#include <algorithm>
#include <iterator>
#include <random>

#include "state_history.hpp"

StateHistory::StateHistory() = default;

StateHistory::~StateHistory() = default;

StateHistory::StateHistory(std::size_t capacity) : capacity(capacity) {};

void StateHistory::add(cv::Mat &state)
{
        // remove last if at capacity
        if(states.size() == capacity)
            states.pop_front();

        // add state to states
        states.push_back(state);
}

torch::Tensor StateHistory::getStates()
{
    std::vector<float> pixels;
    std::size_t size;
    cv::Size state_size;

    state_size = states[0].size();
    size = (state_size.width * state_size.height);
    pixels.reserve(states.size());

    for (const auto &state : states)
        for (long unsigned int i = 0; i < size; i++)
            pixels.emplace_back(state.data[i] / 255.0);

    return torch::from_blob(pixels.data(),
                            {1, static_cast<int64_t>(states.size()), state_size.width, state_size.height});
}

int64_t StateHistory::size()
{
    return states.size();
}
