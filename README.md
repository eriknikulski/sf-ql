# SF paper Reproduction

Paper: "Successor Features for Transfer in Reinforcement Learning" - Barreto et al.

- [x] Simple Q-learning implementation
- [x] SF Q-learning implementation

### Assumptions
- full observability (`feature_extractor.py`)

Note: each task is specified by its index and used by `env.reset(seed=task)`

### Tensorboard

Start tensorboard with: `python -m tensorboard.main --logdir=runs`

TODOS:

- [ ] QL and SFQL on bigger ENVs?
- [ ] Black code formater?