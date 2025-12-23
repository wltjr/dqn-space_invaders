#pragma once

#include <argp.h>

#include <string>

#define STRINGIFY(x) STRINGIFY2(x)
#define STRINGIFY2(x) #x

// default values
#define PT_FILE "dqn_space_invaders.pt"
#define EPISODES 10
#define NOOP 30
#define SKIP 2

// hyper parameters
#define ALPHA 0.00025            // learning rate
#define GAMMA 0.99               // discount factor
#define EPSILON 1.0              // exploration rate (starting value)
#define EPSILON_MIN 0.1          // minimum exploration rate
#define EPSILON_DECAY 0.99999    // decay rate for exploration
#define MEMORY 50000             // replay memory buffer size
#define MEMORY_MIN 10000         // minimum replay memory buffer size
#define UPDATE_FREQ 10000        // target network update frequency
#define BATCH_SIZE 32            // minibatch sample size
#define HISTORY_SIZE 4           // agent history size
#define LIVES 3                  // default lives
#define INIT_WEIGHTS 1           // default init weights method Kaiming Uniform

// init weights method enum
enum class InitWeightMethod
{
    kaiming_normal,
    kaiming_uniform,
    xavier_normal,
    xavier_uniform
};

// command line arguments
struct args
{
    bool clip = false;
    bool double_dqn = false;
    bool dueling_dqn = false;
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
    int init_weights = INIT_WEIGHTS;
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
} typedef args_t;

// help menu
constexpr static struct argp_option options[] = {
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
    {"clip",'C',0,0," Clip/limit rewards to [-1,1]",2},
    {"weight",'W',STRINGIFY(INIT_WEIGHTS),0," Init weights, 0/1 Kaiming Norm/Uniform, 2/3 Xavier Norm/Uniform",2},
    {"rival",'R',0,0," Enable Dueling DQN",2},
    {"twice",'T',0,0," Enable Double DQN",2},
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
error_t parse_opt(int key, char *arg, struct argp_state *state);

/**
 * @brief Print training information to stdout
 *
 * @param args reference to an argp argument structure
 */
void print_training_params(args_t &args);

#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
constexpr static struct argp argp	 =  { options, parse_opt };
