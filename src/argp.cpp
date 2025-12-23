
#include <iostream>

#include "argp.hpp"

error_t parse_opt(int key, char *arg, struct argp_state *state)
{
    auto args = (struct args*)state->input;

    switch(key)
    {
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
        case 'C':
            args->dueling_dqn = true;
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
        case 'R':
            args->dueling_dqn = true;
            break;
        case 'S':
            args->skip = arg ? atoi (arg) : SKIP;
            break;
        case 'T':
            args->double_dqn = true;
            break;
        case 'U':
            args->update_freq = arg ? atoi (arg) : UPDATE_FREQ;
            break;
        case 'W':
            args->init_weights = arg ? atoi (arg) : INIT_WEIGHTS;
            break;
        default:
            return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

void print_training_params(args_t &args)
{
    std::string init_weights_method;
    std::string network;
    std::string type;
    auto iwm = static_cast<InitWeightMethod>(args.init_weights);

    if(iwm == InitWeightMethod::kaiming_normal)
        init_weights_method = "Kaiming normal";
    else if(iwm == InitWeightMethod::xavier_normal)
        init_weights_method = "Xavier normal";
    else if(iwm == InitWeightMethod::xavier_uniform)
        init_weights_method = "Xavier uniform";
    else
        init_weights_method = "Kaiming uniform";

    if(args.double_dqn)
        type = "Double";
    else
        type = "Regular";

    if(args.dueling_dqn)
        network = "Dueling DQN";
    else
        network = "DQN";

    std::cout << std::endl
                << "Training Parameters:" << std::endl
                << "Lives:         " << args.lives << std::endl
                << "Episodes:      " << args.episodes << std::endl
                << "Clip Rewards:  " << args.clip << std::endl
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
                << "History Size:  " << args.history_size << std::endl
                << "Init Weights:  " << init_weights_method << std::endl
                << "DQN:           " << type << std::endl
                << "Network:       " << network << std::endl
                << std::endl;
}
