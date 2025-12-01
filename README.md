# SF paper Reproduction

Paper: "Successor Features for Transfer in Reinforcement Learning" - Barreto et al.

- [x] Simple Q-learning implementation
- [x] SF Q-learning implementation

### Assumptions
- full observability (`feature_extractor.py`)

Note: each task is specified by its index and used by `env.reset(seed=task)`

TODOS:

- [ ] Currently, all RewardObjects are the same in the representation -> FIX!
- [ ] QL and SFQL on bigger ENVs?
- [ ] add plots / tensorboard
- [ ] Black code formater?