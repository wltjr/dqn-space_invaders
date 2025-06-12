
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
    NetImpl()
        : conv1(torch::nn::Conv2dOptions(32, 64, 8).stride(4)), // 84 x 84 x 4, 32
          conv2(torch::nn::Conv2dOptions(32, 64, 4).stride(2)), // 32 , 8 x 8
          conv3(torch::nn::Conv2dOptions(64, 9, 3).stride(1)), // 64 , 4 x 4
          fc1(3136, 512), // 64 x 7 x 7
          fc2(512, ACTIONS)
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
    int episodes = EPISODES;
    int noop = NOOP;
    int skip = SKIP;
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
    {"min",'M',STRINGIFY(EPSILON_MIN),0," Minimum exploration rate",2},
    {"decay",'D',STRINGIFY(EPSILON_DECAY),0," Decay rate for exploration",2},
    {"noop",'N',STRINGIFY(NOOP),0," Skip initial frames using noop action",2},
    {"skip",'S',STRINGIFY(SKIP),0," Skip frames and repeat actions",2},
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
        case 'G':
            args->gamma = arg ? atof (arg) : GAMMA;
            break;
        case 'E':
            args->epsilon = arg ? atof (arg) : EPSILON;
            break;
        case 'M':
            args->epsilon_min = arg ? atof (arg) : EPSILON_MIN;
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
        default:
            return ARGP_ERR_UNKNOWN;
    }
    return(0);
}

#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
static struct argp argp	 =  { options, parse_opt };


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
 * @brief Train agent using deep q-network
 * 
 * @param args reference to args structure
 * @param ale reference to arcade learning environment
 * @param model reference to libtorch model
 */
void train(args &args, 
           ale::ALEInterface &ale,
           std::shared_ptr<NetImpl>  model)
{
    int max_episode;
    ale::reward_t max_score;

    // initialize random device
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> rand_action(0, ACTIONS-1);
    std::uniform_real_distribution<> rand_epsilon(0.0, args.epsilon);

    auto start = std::chrono::high_resolution_clock::now();


    max_episode = -1;
    max_score = -1;

    for(int i = 0; i < args.episodes ;i++)
    {
        ale::reward_t total_reward;
        int steps;
        int lives;

        lives = ale.lives();
        steps = 0;
        total_reward = 0;

        if(args.train)
        {
            // skip initial frames with noop action
            for(; steps < args.noop; steps++)
                ale.act(ale::Action::PLAYER_A_NOOP);
        }

        for(; !ale.game_over(); steps++)
        {
            ale::reward_t reward;
            ale::Action action;
            cv::Mat state;

            state = scale_crop_screen(ale, state);

            // take action & collect reward
            reward = ale.act(action);
            total_reward += reward;

            if(args.train)
            {
                // normalize reward -1, 0, or 1
                if(reward > 0)
                    reward = 1;

                // skip k frames, repeat action
                for(int k = 0; k < args.skip; steps++, k++)
                    total_reward += ale.act(action);

                // penalty for dying
                if(ale.lives() < lives)
                {
                    reward = -10;
                    lives = ale.lives();
                }
                // penalty for noop
                else if(action == ale::Action::PLAYER_A_NOOP)
                    reward = -1;


                // decay epsilon
                args.epsilon = std::max(args.epsilon_min, 
                                        args.epsilon * args.epsilon_decay);
            }
        }

        // track max episode & score
        if(total_reward > max_score)
        {
            max_episode = i;
            max_score = total_reward;
        }

        // save final episode results to file
        if(args.png)
            ale.saveScreenPNG(std::format("episode-{}.png", i));

        std::cout << std::format("Episode {} score: {} steps: {} epsilon: {}",
                                 i, total_reward, steps, args.epsilon)
                  << std::endl;
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

    // initialize Arcade Learning Environment
    ale::ALEInterface ale;

    // initialize game
    ale.setInt("random_seed", 123);
    ale.setBool("display_screen", args.display);
    ale.setBool("sound", args.sound);
    ale.loadROM("./rom/space_invaders.bin");

    // load model
    if(args.load)
        torch::load(model, args.load_file);
    else
        model = std::make_shared<NetImpl>();

    // must load or train
    if(!args.load && !args.train)
        args.train = true;

    // enable hyper training
    if(args.train)
    {
        std::cout << "Training Parameters:" << std::endl
                  << "Episodes:      " << args.episodes << std::endl
                  << "Alpha:         " << args.alpha << std::endl
                  << "Gamma:         " << args.gamma << std::endl
                  << "Epsilon:       " << args.epsilon << std::endl
                  << "Epsilon Min:   " << args.epsilon_min << std::endl
                  << "Epsilon Decay: " << args.epsilon_decay << std::endl
                  << "Noop:          " << args.noop << std::endl
                  << "Frame Skip:    " << args.skip << std::endl;

        train(args, ale, model);

        // only save after training
        if(args.save)
            torch::save(model, args.save_file);
    }

    // play game using trained model, random actions if empty
    if(args.game)
    {
        args.train = false;
        train(args, ale, model);
    }

    return 0;
}
