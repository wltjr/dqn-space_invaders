#pragma once

#include <torch/torch.h>

/**
 * @brief Deep Q-Network
 */
struct BaseDQNImpl : torch::nn::Module
{
    /**
     * @brief Construct a new BaseDQNImpl object, empty/unused
     */
    BaseDQNImpl();

    /**
     * @brief Destroy the BaseDQNImpl, empty/unused
     */
    virtual ~BaseDQNImpl();

    /**
     * @brief Perform forward pass of input data through network
     *
     * @param x tensor of states up to history size
     *
     * @return torch::Tensor q-values from the network
     */
    virtual torch::Tensor forward(torch::Tensor x) = 0;

    /**
     * @brief Get action from network
     *
     * @param state tensor of current state
     *
     * @return torch::Tensor scalar of the action to be taken
     */
    torch::Tensor act(torch::Tensor state);
};

TORCH_MODULE(BaseDQN);

/**
 * @brief Deep Q-Network
 */
struct DQNImpl : BaseDQNImpl
{
    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::Conv2d conv3;
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};

    /**
     * @brief Construct a new DQNImpl object
     *
     * @param frames number of stacked frames (not layers) per state tensor
     * @param actions number of actions available
     */
    DQNImpl(int64_t frames, int64_t actions);

    /**
     * @brief Perform forward pass of input data through network
     *
     * @param x tensor of states up to history size
     *
     * @return torch::Tensor q-values from the network
     */
    torch::Tensor forward(torch::Tensor x) override;
};

TORCH_MODULE(DQN);

/**
 * @brief Dueling Deep Q-Network
 */
struct DuelingDQNImpl : BaseDQNImpl
{
    int64_t actions;
    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::Conv2d conv3;
    torch::nn::Linear fc1_adv{nullptr};
    torch::nn::Linear fc1_val{nullptr};
    torch::nn::Linear fc2_adv{nullptr};
    torch::nn::Linear fc2_val{nullptr};

    /**
     * @brief Construct a new DuelingDQNImpl object
     *
     * @param frames number of stacked frames (not layers) per state tensor
     * @param actions number of actions available
     */
    DuelingDQNImpl(int64_t frames, int64_t actions);

    /**
     * @brief Perform forward pass of input data through network
     *
     * @param x tensor of states up to history size, shape: (bs,f,h,w)
     *
     * @return torch::Tensor q-values from the network, shape: (bs,actions)
     */
    torch::Tensor forward(torch::Tensor x) override;
};

TORCH_MODULE(DuelingDQN);
