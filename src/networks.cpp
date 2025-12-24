
#include "networks.hpp"

BaseDQNImpl::BaseDQNImpl() = default;

BaseDQNImpl::~BaseDQNImpl() = default;

torch::Tensor BaseDQNImpl::act(torch::Tensor state)
{
    torch::Tensor q_value = forward(state);
    torch::Tensor action = std::get<1>(q_value.max(1));

    return action;
}

DQNImpl::DQNImpl(int64_t frames, int64_t actions)
    : conv1(torch::nn::Conv2dOptions(frames, 32, 8).stride(4)), //  N , 4 x 8
      conv2(torch::nn::Conv2dOptions(32, 64, 4).stride(2)),     // 32 , 8 x 8
      conv3(torch::nn::Conv2dOptions(64, 64, 3).stride(1)),     // 64 , 4 x 4
      fc1(3136, 512), // 64 x 7 x 7
      fc2(512, actions)
{
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
}

torch::Tensor DQNImpl::forward(torch::Tensor x)
{
    x = torch::relu(conv1(x));
    x = torch::relu(conv2(x));
    x = torch::relu(conv3(x));
    x = torch::flatten(x,1);
    x = torch::relu(fc1(x));

    return fc2(x);
}

DuelingDQNImpl::DuelingDQNImpl(int64_t frames, int64_t actions)
    : actions(actions),
      conv1(torch::nn::Conv2dOptions(frames, 32, 8).stride(4)), //  N , 4 x 8
      conv2(torch::nn::Conv2dOptions(32, 64, 4).stride(2)),     // 32 , 8 x 8
      conv3(torch::nn::Conv2dOptions(64, 64, 3).stride(1)),     // 64 , 4 x 4
      fc1_adv(3136, 512), // 64 x 7 x 7
      fc1_val(3136, 512), // 64 x 7 x 7
      fc2_adv(512, actions),
      fc2_val(512, 1)
{
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("fc1_adv", fc1_adv);
    register_module("fc1_val", fc1_val);
    register_module("fc2_adv", fc2_adv);
    register_module("fc2_val", fc2_val);
}

torch::Tensor DuelingDQNImpl::forward(torch::Tensor x)
{
    torch::Tensor adv;
    torch::Tensor val;

    x = torch::relu(conv1(x));
    x = torch::relu(conv2(x));
    x = torch::relu(conv3(x));
    x = torch::flatten(x,1);

    adv = torch::relu(fc1_adv(x));
    val = torch::relu(fc1_val(x));

    adv = fc2_adv(adv);
    val = fc2_val(val);

    return val + adv - adv.mean();
}
