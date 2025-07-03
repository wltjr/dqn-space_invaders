
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

#include "replay_memory.hpp"
#include "state_history.hpp"

#define STRINGIFY(x) STRINGIFY2(x)
#define STRINGIFY2(x) #x

// default values
#define EPISODES 10
#define NOOP 30
#define SKIP 2
// hyper parameters
#define ALPHA 0.00025            // learning rate
#define GAMMA 0.99               // discount factor
#define EPSILON 1.0              // exploration rate (starting value)
#define EPSILON_MIN 0.1          // minimum exploration rate
#define EPSILON_DECAY 0.999999   // decay rate for exploration
#define MEMORY 50000             // replay memory buffer size
#define MEMORY_MIN 10000         // minimum replay memory buffer size
#define UPDATE_FREQ 1000         // target network update frequency
#define BATCH_SIZE 32            // minibatch sample size
#define HISTORY_SIZE 4           // agent history size
#define LIVES 1                  // default lives

const char *argp_program_version = "Version 0.1";
const char *argp_program_bug_address = "w@wltjr.com";

const char *PT_FILE = "dqn_space_invaders.pt";

const int ACTIONS = 6;
const int HEIGHT = 210;
const int WIDTH = 160;
const int CROP_X = 13; // (110 - 84) / 2
const int CROP_HEIGHT = 84;
const int CROP_WIDTH = 110;

struct NetImpl : torch::nn::Module
{
    NetImpl(int64_t frames, int64_t actions)
        : conv1(torch::nn::Conv2dOptions(frames, 32, 8).stride(4)), //  N , 4 x 8
          conv2(torch::nn::Conv2dOptions(32, 64, 4).stride(2)), // 32 , 8 x 8
          conv3(torch::nn::Conv2dOptions(64, 64, 3).stride(1)), // 64 , 4 x 4
          fc1(3136, 512), // 64 x 7 x 7
          fc2(512, actions)
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(conv1(x));
        x = torch::relu(conv2(x));
        x = torch::relu(conv3(x));
        x = torch::flatten(x,1);
        x = torch::relu(fc1(x));
        return fc2(x);
    }

    torch::Tensor act(torch::Tensor state)
    {
        torch::Tensor q_value = forward(state);
        torch::Tensor action = std::get<1>(q_value.max(1));
        return action;
    }

    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::Conv2d conv3;
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
};

TORCH_MODULE(Net);

// command line arguments
struct args
{
    bool display = false;
    bool game = false;
    bool load = false;
    bool png = false;
    bool save = false;
    bool sound = false;
    bool train = false;
    int batch_size = BATCH_SIZE;
    int episodes = EPISODES;
    int history_size = HISTORY_SIZE;
    int lives = LIVES;
    int memory = MEMORY;
    int memory_min = MEMORY_MIN;
    int noop = NOOP;
    int skip = SKIP;
    int update_freq = UPDATE_FREQ;
    float alpha = ALPHA;
    float gamma = GAMMA;
    float epsilon = EPSILON;
    float epsilon_min = EPSILON_MIN;
    float epsilon_decay = EPSILON_DECAY;
    std::string load_file = PT_FILE;
    std::string save_file = PT_FILE;
};

// help menu
static struct argp_option options[] = {
    {0,0,0,0,"Optional arguments:",1},
    {"audio",'a',0,0," Enable audio/sound ",1},
    {"display",'d',0,0," Enable display on screen ",1},
    {"episodes",'e',STRINGIFY(EPISODES),0," Number of episodes ",1},
    {"game",'g',0,0," Play game using model ",1},
    {"load",'l',PT_FILE,OPTION_ARG_OPTIONAL," Load the model from file ",1},
    {"png",'p',0,0," Enable saving a PNG image per episode ",1},
    {"save",'s',PT_FILE,OPTION_ARG_OPTIONAL," Save the model to file ",1},
    {"train",'t',0,0," Train the agent using hyper ",1},
    {0,0,0,0,"Hyper parameters:",2},
    {"alpha",'A',STRINGIFY(ALPHA),0," Alpha learning rate",2},
    {"gamma",'G',STRINGIFY(GAMMA),0," Gamma learning rate discount factor",2},
    {"epsilon",'E',STRINGIFY(EPSILON),0," Epsilon exploration rate (starting value)",2},
    {"final",'F',STRINGIFY(EPSILON_MIN),0," Final/minimum exploration rate (final value)",2},
    {"decay",'D',STRINGIFY(EPSILON_DECAY),0," Decay rate for exploration",2},
    {"knowledge",'K',STRINGIFY(MEMORY_MIN),0," Replay memory buffer minimum knowledge/size",2},
    {"memory",'M',STRINGIFY(MEMORY),0," Replay memory buffer size",2},
    {"noop",'N',STRINGIFY(NOOP),0," Skip initial frames using noop action",2},
    {"skip",'S',STRINGIFY(SKIP),0," Skip frames and repeat actions",2},
    {"update_freq",'U',STRINGIFY(UPDATE_FREQ),0," Target network update frequency",2},
    {"batch_size",'B',STRINGIFY(BATCH_SIZE),0," Minibatch sample size for SGD update",2},
    {"history",'H',STRINGIFY(HISTORY_SIZE),0," Number of frames used as network input",2},
    {"lives",'L',STRINGIFY(LIVES),0," Default lives 1 up to game max of 3",2},
    {0,0,0,0,"GNU Options:", 3},
    {0,0,0,0,0,0}
};

/**
 * @brief Parse command line options
 *
 * @param key the integer key provided by the current option to the option parser.
 * @param arg the name of an argument associated with this option
 * @param state points to a struct argp_state
 *
 * @return ARGP_ERR_UNKNOWN for any key value not recognize
 */
static error_t parse_opt(int key, char *arg, struct argp_state *state) {

    struct args *args = (struct args*)state->input;

    switch(key) {
        case 'a':
            args->sound = true;
            break;
        case 'd':
            args->display = true;
            break;
        case 'e':
            args->episodes = arg ? atoi (arg) : EPISODES;
            break;
        case 'g':
            args->game = true;
            break;
        case 'l':
            args->load = true;
            args->load_file = arg ? arg : PT_FILE;
            break;
        case 'p':
            args->png = true;
            break;
        case 's':
            args->save = true;
            args->save_file = arg ? arg : PT_FILE;
            break;
        case 't':
            args->train = true;
            break;
        case 'A':
            args->alpha = arg ? atof (arg) : ALPHA;
            break;
        case 'B':
            args->batch_size = arg ? atoi (arg) : BATCH_SIZE;
            break;
        case 'G':
            args->gamma = arg ? atof (arg) : GAMMA;
            break;
        case 'E':
            args->epsilon = arg ? atof (arg) : EPSILON;
            break;
        case 'F':
            args->epsilon_min = arg ? atof (arg) : EPSILON_MIN;
            break;
        case 'H':
            args->history_size = arg ? atoi (arg) : HISTORY_SIZE;
            break;
        case 'K':
            args->memory_min = arg ? atoi (arg) : MEMORY_MIN;
            break;
        case 'L':
            args->lives = arg ? atoi (arg) : LIVES;
            break;
        case 'M':
            args->memory = arg ? atoi (arg) : MEMORY;
            break;
        case 'N':
            args->noop = arg ? atoi (arg) : NOOP;
            break;
        case 'D':
            args->epsilon_decay = arg ? atof (arg) : EPSILON_DECAY;
            break;
        case 'S':
            args->skip = arg ? atoi (arg) : SKIP;
            break;
        case 'U':
            args->update_freq = arg ? atoi (arg) : UPDATE_FREQ;
            break;
        default:
            return ARGP_ERR_UNKNOWN;
    }
    return(0);
}

#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
static struct argp argp	 =  { options, parse_opt };


/**
 * @brief Copy policy model network to target model network
 * 
 * @param policy model network
 * @param target model network
 */
void clone_network(torch::nn::Module &policy,
                   torch::nn::Module &target)
{
    // enable parameters copying
    torch::autograd::GradMode::set_enabled(false);
    auto params = policy.named_parameters(true);
    auto buffers = policy.named_buffers(true);
    auto new_params = target.named_parameters();

    for (auto &param : new_params)
    {
        auto name = param.key();
        auto *t = params.find(name);
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
torch::Tensor stack_state_tensors(int &history_size,
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
 * @brief Initialize weights in a neural network
 * 
 * @param model reference to neural network module
 */
void init_weights(torch::nn::Module& model)
{
    torch::NoGradGuard no_grad;

    for (auto &p : model.named_parameters()) {
        std::string y = p.key();
        auto z = p.value();
        auto s = y.find(".",0) + 1;

        if (y.compare(s, 6, "weight") == 0)
            z.uniform_(0.0, 1.0);
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
void train(args &args, 
           ale::ALEInterface &ale,
           std::shared_ptr<NetImpl>  model,
           torch::Device &device)
{
    int update;
    int max_episode;
    ale::reward_t max_score;
    ReplayMemory memory(args.memory);
    NetImpl policy(args.history_size, ACTIONS);
    torch::optim::Adam optimizer(policy.parameters(),
                                 torch::optim::AdamOptions(args.alpha));

    // initialize random device
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> rand_action(0, ACTIONS-1);
    std::uniform_real_distribution<> rand_epsilon(0.0, args.epsilon);

    auto start = std::chrono::high_resolution_clock::now();

    // limit max lives to game limit
    if(args.lives > ale.lives())
        args.lives = ale.lives();

    max_episode = -1;
    max_score = -1;
    update = args.update_freq - 1;

    if(args.train)
    {
        // init local policy and set device
        init_weights(policy);
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
        for(int i = 1; i < args.history_size; i++, steps++)
        {
            cv::Mat state;

            ale.act(ale::Action::PLAYER_A_NOOP);
            state = scale_crop_screen(ale, state);
            state_history.add(state);
        }

        for(; !ale.game_over() && lives > 0; steps++)
        {
            ale::reward_t reward;
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
            total_reward += reward;

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
                std::vector<int64_t> rewards;
                std::vector<int64_t> dones;
                std::vector<torch::Tensor> state_nexts;
                std::vector<ReplayMemory::replay_t> batch;

                // normalize reward -1, 0, or 1
                if(reward > 0)
                    reward = 1;

                // skip k frames, repeat action
                for(int k = 0; k < args.skip; steps++, k++)
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
                reward_tensor = torch::tensor(reward, options).to(device);
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
                        rewards.emplace_back(b.reward.item().to<int64_t>());
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
                                                  options).to(device);
                dones_tensor = torch::from_blob(dones.data(),
                                                { static_cast<int64_t>(dones.size()), 1 },
                                                options).to(device);

                // get q-values from policy and target
                q_values = policy.forward(states_tensor).to(device);
                next_target_q_values = model->forward(state_nexts_tensor).to(device);
                next_q_values = policy.forward(state_nexts_tensor).to(device);

                // calculate targets for q-learning update 
                q_value = q_values.gather(0, actions_tensor).to(device);
                maximum = std::get<1>(next_q_values.max(1)).unsqueeze(1).to(device);
                next_q_value = next_target_q_values.gather(1, maximum).to(device);
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
                if (i == update)
                {
                    update += args.update_freq;
                    clone_network(policy, *model);
                }

                // decay epsilon
                args.epsilon = std::max(args.epsilon_min, 
                                        args.epsilon * args.epsilon_decay);
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

        std::cout << std::format("Episode {} score: {} steps: {}",
                                 i, total_reward, steps);
        // output only when training
        if(args.train)
            std::cout << std::format(" ai: {} random: {} epsilon: {} avg loss: {}",
                                     ai, random, args.epsilon, (loss_episode / trained));
        std::cout << std::endl;
        ale.reset_game();
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);

    std::cout << std::endl
              << std::format("Elapsed Time: {}s - Episode {} Max Score: {}",
                             duration.count(), max_episode, max_score)
              << std::endl;
}


int main(int argc, char* argv[])
{
    struct args args;
    std::shared_ptr<NetImpl> model;

    // parse command line options
    argp_parse (&argp, argc, argv, 0, 0, &args);

    // output date and time
    auto const now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::cout << std::ctime(&time);

    // initialize Arcade Learning Environment
    ale::ALEInterface ale;

    // initialize game
    ale.setInt("random_seed", 123);
    ale.setBool("display_screen", args.display);
    ale.setBool("sound", args.sound);
    ale.loadROM("./rom/space_invaders.bin");

    // default to CPU
    torch::Device device = torch::Device(torch::kCPU);

    // switch to GPU if available
    if(torch::cuda::is_available())
        device = torch::Device(torch::kCUDA);

    model = std::make_shared<NetImpl>(args.history_size, ACTIONS);

    // load model
    if(args.load)
        torch::load(model, args.load_file);
    else
        init_weights(*model);

    // set model device
    model->to(device);

    // must load or train
    if(!args.load && !args.train)
        args.train = true;

    // enable hyper training
    if(args.train)
    {
        std::cout << "Training Parameters:" << std::endl
                  << "Lives:         " << args.lives << std::endl
                  << "Episodes:      " << args.episodes << std::endl
                  << "Alpha:         " << args.alpha << std::endl
                  << "Gamma:         " << args.gamma << std::endl
                  << "Epsilon:       " << args.epsilon << std::endl
                  << "Epsilon Min:   " << args.epsilon_min << std::endl
                  << "Epsilon Decay: " << args.epsilon_decay << std::endl
                  << "Replay:        " << args.memory << std::endl
                  << "Replay Min:    " << args.memory_min << std::endl
                  << "Noop:          " << args.noop << std::endl
                  << "Frame Skip:    " << args.skip << std::endl
                  << "Update Freq.:  " << args.update_freq << std::endl
                  << "Batch Size:    " << args.batch_size << std::endl
                  << "History Size:  " << args.history_size << std::endl;

        train(args, ale, model, device);

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

    return 0;
}
