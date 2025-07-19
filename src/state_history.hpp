#ifndef STATE_HISTORY_HPP
#define STATE_HISTORY_HPP

#include <opencv4/opencv2/opencv.hpp>
#include <torch/torch.h>

class StateHistory
{
    public:
        /**
         * @brief Construct a new StateHistory object, empty/unused
         */
        StateHistory();

        /**
         * @brief Construct a new StateHistory object
         * 
         * @param capacity the capacity/length of the state history
         */
        explicit StateHistory(std::size_t capacity);

        /**
         * @brief Destroy the StateHistory, empty/unused
         */
        virtual ~StateHistory();

        /**
         * @brief Add state to history
         * 
         * @param state reference to ale screen/opencv mat image data
         */
        void add(cv::Mat &state);

        /**
         * @brief Get states from history
         * 
         * @return tensor of states in history
         */
        torch::Tensor getStates();

        /**
         * @brief Get the size of state history in use
         * 
         * @return size of state history in use
         */
        int64_t size();

    private:
        std::size_t capacity;
        std::deque<cv::Mat> states;

};

#endif