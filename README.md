# AlphaZero

Implementation of AlphaZero ([link to the paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf)). To create your own game rules just subclass `Action`, `State`, `PI`, `Model`, `Game`. If you actually want to play against the trained version of your model subclass `Agent` or `CLIAgent` and use `GameRunner` to play though your game. 

Differences from the original paper that are still yet to implement:
- [ ] Dirichlet noise not currently added to neural network predictions during training
- [ ] PUCT algorithm only uses constant c instead of a dynamic c that gets smaller as games progress
- [ ] Current simulations per move is much lower than the 800 used in the paper due to computational constraints
- [ ] Model predictions not fully optimized for parallelization and could encounter a race condition upon training