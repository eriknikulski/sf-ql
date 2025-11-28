from typing import Optional, Tuple

import minigrid.core.constants as minigrid_constants
import numpy as np

from config import Config
from feature_extractor import MinigridFeaturesExtractor
from q_learning import QL, QFunction


class Phi:
    """
    Phi encoder of (state, action, next_state) tuple.

    Note: This implementation relies on a fully-observable environment,
        since the agent is detected to be on the goal when the goal is not visible.
        (The agent occludes the goal)
    """
    GOAL_VALUE = minigrid_constants.OBJECT_TO_IDX['goal']

    def __init__(self) -> None:
        self.embedding_size = 1

    def __call__(self, state: dict, action: int, next_state: dict) -> np.ndarray:
        matches = np.argwhere(state['image'] == self.GOAL_VALUE)
        agent_on_goal = len(matches) == 0

        return np.array([int(agent_on_goal)])


class SFQFunction(QFunction):
    def __init__(
            self,
            feature_extractor: MinigridFeaturesExtractor,
            phi: Phi,
            tasks: int,
            action_space_size: int,
            initial_q_value: float,
            initial_w_value: float,
            alpha: float,
            alpha_w: float,
            gamma: float,
    ) -> None:
        super(SFQFunction, self).__init__(feature_extractor, action_space_size, initial_q_value, alpha, gamma)
        self.phi = phi
        self.tasks = tasks
        self.embedding_size = phi.embedding_size
        self.initial_w_value = initial_w_value
        self.alpha_w = alpha_w

        self.task = None
        self.current_best_task = None

        self.z = None   # important to not accidentally rely on it
        self.Z = None
        self._init_Z()

        self.w = None
        self._init_w()

    def __getitem__(self, key: Tuple[dict, int]) -> None:
        state, action = key
        return self.feature_extractor(state) @ self.Z[self.task, action] @ self.w

    def _init_Z(self, task: Optional[int] = None) -> None:
        assert task is None or task < self.tasks

        if task is None:
            self.Z = np.full(
                shape=(self.tasks, self.action_space_size, self.feature_extractor.features_dim, self.embedding_size),
                fill_value=self.initial_q_value,
            )
        elif task > 0:
            self.Z[task] = self.Z[task - 1]

    def _init_w(self) -> None:
        self.w = np.full(
            shape=(self.tasks, self.embedding_size,),
            fill_value=self.initial_w_value,
        )

    def task_init(self, task: int) -> None:
        self.task = task
        self._init_Z(task)

    def step_init(self, state: dict, step: int) -> None:
        self.current_best_task = self.get_best_task(state)

    def psi(self, state: dict, action: int, task: Optional[int] = None) -> float:
        if task is None:
            task = self.task

        return self.feature_extractor(state) @ self.Z[task, action]

    def _td_update_Z(
            self,
            task: int,
            state: dict,
            action: int,
            next_state: dict,
            next_action: int,
            gamma: float,
    ) -> None:
        state_features = self.feature_extractor(state)
        phi_vec = self.phi(state, action, next_state)

        # update Z[t, a]
        td_error = phi_vec + gamma * self.psi(next_state, next_action, task) - self.psi(state, action, task)
        for k in range(self.embedding_size):
            self.Z[task, action, :, k] += self.alpha * td_error[k] * state_features

    def update(
            self,
            state: dict,
            action: int,
            reward: float,
            next_state: dict,
            gamma: Optional[float] = None,
    ) -> None:
        if gamma is None:
            gamma = self.gamma

        # update task vector w
        phi_vec = self.phi(state, action, next_state)
        self.w[self.task] += self.alpha_w * (reward - phi_vec @ self.w[self.task]) * phi_vec

        # update Z[t, a]
        next_action, _ = self.get_best_action(next_state)
        self._td_update_Z(self.task, state, action, next_state, next_action, gamma)

        if self.current_best_task != self.task:
            # TD update for the task which was responsible for taking the action
            # # update Z[c, a]
            next_action = self.get_action(
                next_state,
                task_psi=self.current_best_task,
                task_weight=self.current_best_task,
            )
            self._td_update_Z(self.current_best_task, state, action, next_state, next_action, gamma)

    def get_best_task(self, state: dict) -> int:
        """
        Get the task that has the highest Q-value for the state and the current task description.
        Considered tasks are from 0 to (including) {max_task}.

        :param state: Current state
        :return: task with highest Q-value
        """
        if self.tasks == 1:
            return 0
        best_value = None
        best_tasks = []

        state_features = self.feature_extractor(state)

        # find best tasks
        for task in range(min(self.task + 1, self.tasks)):
            state_q_values = self.Z[task] @ self.w[self.task] @ state_features
            value = np.max(state_q_values)

            if best_value is None or value > best_value:
                best_value = value
                best_tasks = [task]
            elif value == best_value:
                best_tasks.append(task)

        # break ties arbitrarily
        if len(best_tasks) > 1:
            idx = np.random.randint(len(best_tasks))
        else:
            idx = 0

        return best_tasks[idx]

    def get_action(self, state: dict, task_psi: Optional[int] = None, task_weight: Optional[int] = None) -> int:
        """
        Gets the best action for the provided task or (fallback) for the current best task,
        calculated at the beginning of the step.

        :param state: state where the action should be chosen
        :param task_psi: task from which psi should be used (default: current best task)
        :param task_weight: task from which weight should be used (default: current task)
        :return: action
        """
        if task_psi is None:
            task_psi = self.current_best_task
        if task_weight is None:
            task_weight = self.task

        state_features = self.feature_extractor(state)
        state_q_table = self.Z[task_psi] @ self.w[task_weight] @ state_features

        # argmax tie-breaking
        action = int(np.argmax(state_q_table + 1e-8 * np.random.randn(len(state_q_table))))
        return action

    def get_best_action(self, state: dict) -> Tuple[int, int]:
        """
        Get the best action over all tasks given a state and the current task description w

        :param state: current state
        :return: action, task in which the action was chosen
        """
        best_value = None
        best = []

        state_features = self.feature_extractor(state)

        # find best actions and tasks
        for task in range(min(self.task + 1, self.tasks)):
            state_q_values = self.Z[task] @ self.w[self.task] @ state_features
            actions = np.argwhere(state_q_values == np.amax(state_q_values))
            if actions.size > 0:
                value = state_q_values[actions[0][0]]
                if best_value is None or value > best_value:
                    best = list(zip(actions.flatten(), [task] * actions.size))
                    best_value = value
                elif value == best_value:
                    best.extend(list(zip(actions.flatten(), [task] * actions.size)))

        # break ties arbitrarily
        if len(best) > 1:
            idx = np.random.randint(len(best))
        else:
            idx = 0

        return best[idx]


class SFQL(QL):
    def __init__(
            self,
            *args,
            alpha_w: Optional[float] = None,
            initial_w_value: Optional[float] = None,
            **kwargs,
    ) -> None:
        super(SFQL, self).__init__(*args, **kwargs)

        self.config = Config()

        self.alpha_w = self.config.get_or_raise(alpha_w, 'sfq_learning', 'alpha_w')
        self.initial_w_value = self.config.get_or_raise(initial_w_value, 'sfq_learning', 'initial_w_value')

        self.Q = SFQFunction(
            feature_extractor=MinigridFeaturesExtractor(self.observation_space),
            phi=Phi(),
            tasks=self.tasks,
            action_space_size=self.action_space.n,
            initial_q_value=self.initial_q_value,
            initial_w_value=self.initial_w_value,
            alpha=self.alpha,
            alpha_w=self.alpha_w,
            gamma=self.gamma,
        )
