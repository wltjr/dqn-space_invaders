
#include <argp.h>

#include <algorithm>
#include <chrono>
#include <format>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include <ale/ale_interface.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <torch/torch.h>

#include "argp.hpp"
#include "networks.hpp"
#include "replay_memory.hpp"
#include "state_history.hpp"

const char *argp_program_version = "Version 0.1";
const char *argp_program_bug_address = "w@wltjr.com";

const int ACTIONS = 6;
const int HEIGHT = 210;
const int WIDTH = 160;
const int CROP_X = 13; // (110 - 84) / 2
const int CROP_HEIGHT = 84;
const int CROP_WIDTH = 110;

/**
 * @brief Copy policy model network to target model network
 * 
 * @param policy model network
 * @param target model network
 */
void clone_network(torch::nn::Module const &policy,
                   torch::nn::Module const &target)
{
    // enable parameters copying
    torch::autograd::GradMode::set_enabled(false);
    auto params = policy.named_parameters(true);
    auto buffers = policy.named_buffers(true);
    auto new_params = target.named_parameters();

    for (auto &param : new_params)
    {
        auto name = param.key();
        const auto *t = params.find(name);
        if (t != nullptr)
            t->copy_(param.value());
        else
        {
            t = buffers.find(name);
            if (t != nullptr)
                t->copy_(param.value());
        }
    }
}


/**
 * @brief Convert int range 0-5 value to ALE action
 * 
 * @param i integer
 * 
 * @return ale::Action ALE action
 */
ale::Action int_to_action(int i)
{
    ale::Action a;

    if(i == 2)
        a = ale::Action::PLAYER_A_RIGHT;
    else if(i == 3)
        a = ale::Action::PLAYER_A_LEFT;
    else if(i == 4)
        a = ale::Action::PLAYER_A_RIGHTFIRE;
    else if(i == 5)
        a = ale::Action::PLAYER_A_LEFTFIRE;
    else
        a = static_cast<ale::Action>(i);

    return a;
}


/**
 * @brief Scale and crop the screen
 * 
 * @param ale reference to arcade learning environment
 * @param state reference to opencv mat/image
 * 
 * @return state scaled and cropped screen
 */
cv::Mat scale_crop_screen(ale::ALEInterface &ale, cv::Mat &state)
{
    std::vector<unsigned char> screen;
    cv::Mat orig;
    cv::Size scale;

    // prepare current game screen for opencv
    ale.getScreenGrayscale(screen);
    orig = cv::Mat(HEIGHT, WIDTH, CV_8UC1, &screen[0]);
    scale.height = CROP_HEIGHT;
    scale.width = CROP_WIDTH;
    cv::resize(orig, state, scale);
    return cv::Mat(state, cv::Rect(CROP_X, 0, CROP_HEIGHT, CROP_HEIGHT));
}


/**
 * @brief Convert ale screen/opencv mat image to tensor
 * 
 * @param state reference to ale screen/opencv mat image data
 * 
 * @return tensor representation of the ale screen/opencv mat image
 */
torch::Tensor state_to_tensor(cv::Mat &state)
{
    std::vector<float> pixels;
    std::size_t size;
    cv::Size state_size;

    state_size = state.size();
    size = (state_size.width * state_size.height); // 84 x 84
    pixels.reserve(size);

    for (long unsigned int i = 0; i < size; i++)
        pixels.emplace_back(state.data[i] / 255.0);

    return torch::from_blob(pixels.data(), {1, state_size.width, state_size.height}).clone();
}


/**
 * @brief Stack state frame tensors into groups based on history length
 * 
 * @param history_size reference to history length
 * @param states reference to a vector of state frame tensors
 * @param device reference to torch  hardware device
 * 
 * @return vector of state frame tensors in groups
 */
torch::Tensor stack_state_tensors(int const &history_size,
                                  std::vector<torch::Tensor> &states,
                                  torch::Device &device)
{
    int count;
    c10::IntArrayRef state_size;
    std::vector<torch::Tensor> frames;
    std::vector<torch::Tensor> state_frames;

    count = 1;
    frames.reserve(states.size() /  history_size);
    state_size = states[0].sizes();

    for (const auto &state : states)
    {
        state_frames.emplace_back(state);

        if(count == history_size)
        {
            frames.emplace_back(torch::cat(state_frames).unsqueeze(0).to(device));
            state_frames.clear();
            count = 0;
        }

        count++;
    }

    return torch::cat(frames).to(device);
}


/**
 * @brief Initialize weights & bias in a neural network modules
 * 
 * @param model reference to neural network module
 * @param init_weights weight initialize method
 */
void init_nn_modules(torch::nn::Module const &model, int init_weights)
{
    torch::NoGradGuard no_grad;

    auto iwm = static_cast<InitWeightMethod>(init_weights);

    for (auto &p : model.named_parameters()) {
        std::string y = p.key();
        auto z = p.value();
        auto s = y.find(".",0) + 1;

        if (y.compare(s, 6, "weight") == 0)
        {
            if(iwm == InitWeightMethod::kaiming_normal)
                torch::nn::init::kaiming_normal_(z);
            else if(iwm == InitWeightMethod::xavier_normal)
                torch::nn::init::xavier_normal_(z);
            else if(iwm == InitWeightMethod::xavier_uniform)
                torch::nn::init::xavier_uniform_(z);
            else
                torch::nn::init::kaiming_uniform_(z);
        }
        else if (y.compare(s, 4, "bias") == 0)
            z.fill_(0);
    }
}


/**
 * @brief Train agent using deep q-network
 * 
 * @param args reference to args structure
 * @param ale reference to arcade learning environment
 * @param model reference to libtorch model
 * @param device reference to torch  hardware device
 */
template <typename T>
void train(args &args, 
           ale::ALEInterface &ale,
           std::shared_ptr<T> model,
           torch::Device &device)
{
    int update;
    int max_episode;
    ale::reward_t max_score;
    int total_steps;
    ReplayMemory memory(args.memory);
    T policy(args.history_size, ACTIONS);
    torch::optim::Adam optimizer(policy.parameters(),
                                 torch::optim::AdamOptions(args.alpha));

    // initialize random device
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> rand_action(0, ACTIONS-1);
    std::uniform_real_distribution<> rand_epsilon(0.0, args.epsilon);

    auto start = std::chrono::high_resolution_clock::now();

    // limit max lives to game limit
    if(args.lives > ale.lives())
        args.lives = ale.lives();

    max_episode = -1;
    max_score = -1;
    total_steps = 0;
    update = args.update_freq - 1;

    if(args.train)
    {
        // init local policy and set device
        init_nn_modules(policy, args.init_weights);
        policy.to(device);
        policy.train();
        model->train();
    }
    else
        model->eval();

    for(int i = 0; i < args.episodes ;i++)
    {
        ale::reward_t total_reward;
        int ai;
        int steps;
        int lives;
        int lives_game;
        int random;
        int trained;
        double loss_episode;
        StateHistory state_history(args.history_size);

        ai = 0;
        lives = args.lives;
        lives_game = ale.lives();
        loss_episode = 0.0;
        steps = 0;
        random = 0;
        total_reward = 0;
        trained = 0;

        if(args.train)
        {
            int skip;

            // reduce skips for history priming
            skip = args.noop - args.history_size;

            // skip initial frames with noop action
            for(; steps < skip; steps++)
                ale.act(ale::Action::PLAYER_A_NOOP);
        }

        // prime history less one for first state in next loop
        for(int c = 1; c < args.history_size; c++, steps++)
        {
            cv::Mat state;

            ale.act(ale::Action::PLAYER_A_NOOP);
            state = scale_crop_screen(ale, state);
            state_history.add(state);
        }

        // update total steps
        total_steps += steps;

        for(; !ale.game_over() && lives > 0; steps++, total_steps++)
        {
            float reward;
            ale::Action action;
            cv::Mat state;
            torch::Tensor state_tensor;

            state = scale_crop_screen(ale, state);
            state_history.add(state);
            state_tensor = state_to_tensor(state).to(device);

            // random action
            if(args.train && rand_epsilon(gen) < args.epsilon)
            {
                action = int_to_action(rand_action(gen));
                random++;
            }
            else
            // action from model
            {
                torch::Tensor action_tensor;
                torch::Tensor states_tensor;

                states_tensor = state_history.getStates().to(device);
                if(args.train)
                    action_tensor = policy.act(states_tensor).to(device);
                else
                    action_tensor = model->act(states_tensor).to(device);
                action = int_to_action(action_tensor[0].item<int>());
                ai++;
            }

            // take action & collect reward
            reward = ale.act(action);
            total_reward += static_cast<int64_t>(reward);

            if(args.train)
            {
                int size;
                cv::Mat next;
                torch::Tensor action_tensor;
                torch::Tensor reward_tensor;
                torch::Tensor done_tensor;
                torch::Tensor next_tensor;
                torch::Tensor states_tensor;
                torch::Tensor actions_tensor;
                torch::Tensor rewards_tensor;
                torch::Tensor dones_tensor;
                torch::Tensor state_nexts_tensor;
                torch::Tensor q_values;
                torch::Tensor next_target_q_values;
                torch::Tensor next_q_values;
                torch::Tensor q_value;
                torch::Tensor maximum;
                torch::Tensor next_q_value;
                torch::Tensor expected_q_value;
                torch::Tensor loss;
                torch::TensorOptions options;
                std::vector<torch::Tensor> states;
                std::vector<int64_t> actions;
                std::vector<float> rewards;
                std::vector<int64_t> dones;
                std::vector<torch::Tensor> state_nexts;
                std::vector<ReplayMemory::replay_t> batch;

                // decay epsilon
                args.epsilon = std::max(args.epsilon_min, 
                                        args.epsilon * args.epsilon_decay);

                // reward -1, -10, 0, or 1/1000
                if(reward > 0)
                    reward /= 1000;

                // skip k frames, repeat action
                for(int k = 0; k < args.skip; k++, steps++, total_steps++)
                    total_reward += ale.act(action);

                // penalty for dying
                if(lives_game > ale.lives())
                {
                    reward = -10;
                    lives--;
                    lives_game--;
                }
                // penalty for noop
                else if(action == ale::Action::PLAYER_A_NOOP)
                    reward = -1;

                // next state for memory
                next = scale_crop_screen(ale, next);

                options = torch::TensorOptions().dtype(torch::kInt64);
                action_tensor = torch::tensor(action, options).to(device);
                reward_tensor = torch::tensor(reward, torch::kFloat32).to(device);
                done_tensor = torch::tensor(ale.game_over(), options).to(device);
                next_tensor = state_to_tensor(next).to(device);

                // add to memory/replay
                memory.add({state_tensor, action_tensor, reward_tensor, done_tensor, next_tensor});

                // minimum replay memory size
                if(memory.size() < args.memory_min)
                    continue;

                // samples from replay memory
                size = args.batch_size * args.history_size;
                batch = memory.sample(size);
                states.reserve(size);
                state_nexts.reserve(size);
                actions.reserve(args.batch_size);
                rewards.reserve(args.batch_size);
                dones.reserve(args.batch_size);

                // add to individual vectors
                for (int a = 1; const auto &b : batch)
                {
                    // add args.batch_size * args.history_size states
                    states.emplace_back(b.state);
                    state_nexts.emplace_back(b.state_next);

                    // add args.batch_size the rest
                    if (a % args.history_size == 0)
                    {
                        actions.emplace_back(b.action.item().to<int64_t>());
                        rewards.emplace_back(b.reward.item().to<float>());
                        dones.emplace_back(b.done.item().to<int64_t>());
                    }

                    a++;
                }

                // stack frames for processing
                states_tensor = stack_state_tensors(args.history_size, states, device);
                state_nexts_tensor = stack_state_tensors(args.history_size, state_nexts, device);

                // convert vectors to tensors
                actions_tensor = torch::from_blob(actions.data(),
                                                  { static_cast<int64_t>(actions.size()), 1 },
                                                  options).to(device);
                rewards_tensor = torch::from_blob(rewards.data(),
                                                  { static_cast<int64_t>(rewards.size()), 1 },
                                                  torch::kFloat32).to(device);
                dones_tensor = torch::from_blob(dones.data(),
                                                { static_cast<int64_t>(dones.size()), 1 },
                                                options).to(device);

                // get q-values from policy and target/model
                // Q(st,a)
                q_values = policy.forward(states_tensor).to(device);
                q_value = q_values.gather(0, actions_tensor).to(device);
                next_target_q_values = model->forward(state_nexts_tensor).to(device);

                // calculate targets for q-learning update 
                if(args.double_dqn)
                {
                    // Q(st + 1, a)
                    next_q_values = policy.forward(state_nexts_tensor).to(device);
                    // argmax Q(st + 1, a)
                    maximum = std::get<1>(next_q_values.max(1)).unsqueeze(1).to(device);
                    // Q'(st + 1, argmax_a Q(st + 1, a))
                    next_q_value = next_target_q_values.gather(1, maximum).to(device);
                }
                else
                {
                    // max_a Q'(st + 1, a)
                    next_q_value = std::get<1>(next_target_q_values.max(1)).unsqueeze(1).to(device);
                }

                expected_q_value = (rewards_tensor + args.gamma * next_q_value * (1 - dones_tensor)).to(device);
                loss = torch::smooth_l1_loss(q_value, expected_q_value).to(device);
                loss.requires_grad_(true);
                loss_episode += loss.item().to<double>();

                // zero gradients, back propagation, & gradient descent
                optimizer.zero_grad();
                loss.backward();
                optimizer.step();
                trained++;

                // clone policy network to target
                if (total_steps == update)
                {
                    update += args.update_freq;
                    clone_network(policy, *model);
                }
            }
        }

        // final clone of trained policy to target model
        if(args.train)
            clone_network(policy, *model);

        // track max episode & score
        if(total_reward > max_score)
        {
            max_episode = i;
            max_score = total_reward;
        }

        // save final episode results to file
        if(args.png)
            ale.saveScreenPNG(std::format("episode-{}.png", i));

        std::cout << std::format("Episode {:>4} score: {:>4} steps: {:>4}",
                                 i, total_reward, steps);
        // output only when training
        if(args.train)
            std::cout << std::format(" ai: {:>4} random: {:>4} epsilon: {:>0.7f} avg loss: {:>0.8f}",
                                     ai, random, args.epsilon, (loss_episode / trained));
        std::cout << std::endl;
        ale.reset_game();
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);

    std::cout << std::endl
              << std::format("Elapsed Time: {}s ", duration.count());
    // output only when training
    if(args.train)
        std::cout << std::format("Total Steps: {} ", total_steps);
    std::cout << std::format("- Episode {} Max Score: {}", max_episode, max_score)
              << std::endl;
}

/**
 * @brief Network wrapper, allows use of different NN via template
 *
 * @param args reference to args structure
 * @param ale reference to arcade learning environment
 * @param device reference to torch  hardware device
 */
template <typename T>
void select_nn(args &args,
               ale::ALEInterface &ale,
               torch::Device &device)
{
    std::shared_ptr<T> model = std::make_shared<T>(args.history_size, ACTIONS);

    // load model
    if(args.load)
        torch::load(model, args.load_file);
    else
        init_nn_modules(*model, args.init_weights);

    // set model device
    model->to(device);

    // must load or train
    if(!args.load && !args.train)
        args.train = true;

    // enable hyper training
    if(args.train)
    {
        print_training_params(args);

        train<T>(args, ale, model, device);

        // only save after training
        if(args.save)
            torch::save(model, args.save_file);
    }

    // play game using trained model, random actions if empty
    if(args.game)
    {
        args.train = false;
        train(args, ale, model, device);
    }
}

int main(int argc, char* argv[])
{
    struct args args;

    // parse command line options
    argp_parse (&argp, argc, argv, 0, 0, &args);

    // output date and time
    auto const now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::cout << std::ctime(&time);

    // random seed
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> rand_seed(100, 1000);

    // initialize Arcade Learning Environment
    ale::ALEInterface ale;

    // initialize game
    ale.setInt("random_seed", rand_seed(gen));
    ale.setBool("display_screen", args.display);
    ale.setBool("sound", args.sound);
    ale.loadROM("./rom/space_invaders.bin");

    // default to CPU
    auto device = torch::Device(torch::kCPU);

    // switch to GPU if available
    if(torch::cuda::is_available())
        device = torch::Device(torch::kCUDA);

    if(args.dueling_dqn)
        select_nn<DuelingDQNImpl>(args, ale, device);
    else
        select_nn<DQNImpl>(args, ale, device);

    return 0;
}
