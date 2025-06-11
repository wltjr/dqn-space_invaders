# DQN-Space Invaders

Deep Q-Network Space Invaders using the [Arcade Learning Environment (ALE)](https://ale.farama.org/)

## State Space
The state space consists of frame of the grayscale game screen resized 50% and
cropped to 84 x 84.

### Action Space
The
[ALE Space Invaders Action Space](https://ale.farama.org/environments/space_invaders/#actions)
has been reduced from 6 to the following 4 actions.

| Value | Meaning |
|-------|---------|
| 0 | NOOP |
| 1 | FIRE |
| 2 | RIGHT |
| 3 | LEFT |
| 4 | RIGHT-FIRE |
| 5 | LEFT-FIRE |

## System Requirements

The following software is required for proper operation

- [GCC >= 14.1](https://gcc.gnu.org/releases.html)
- [CMake >= 3.20](https://cmake.org/download/)
- [libsdl >= 2.30](https://www.libsdl.org/)
- [ALE >= 0.11.0](https://ale.farama.org/)
- [OpenCV >= 4.11.0](https://opencv.org/releases/)
- [PyTorch >= 2.7.0](https://pytorch.org/get-started/locally/)

Atari ROM `space_invaders.bin` was obtained from
[Atari Mania](https://www.atarimania.com/game-atari-2600-vcs-space-invaders_s6947.html)

## Build

Run the following commands in the root directory of the repository to compile
all executables. The base project uses cmake build system with default of make.

```bash
cmake ./
make
```

## Run

The primary executable is `dqnsi` multi-agent hedonic simulation environment.
The program is implemented using GNU Argp, and has available `--help` menu for
information on the arguments that each program accepts, which are required and
are optional.
