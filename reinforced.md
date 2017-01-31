##### RL Course by David Silver - Lecture 1: Introduction to Reinforcement Learning
[video](https://www.youtube.com/watch?v=2pWv7GOvuf0)

**reward** hypothesis: maximisation of expected accumulative reward

sequential decision making: select actions to maximise total future reward

Based on the observation and reward signal, the "brain" decides the action.

At each step, the (learning) agent

1. execute action
2. receive observation
3. receive scalar reward

The **history** is the sequence of observations, actions, rewards (markov state)

**State** is the information used to determine what happens next.
 
1. environment state: the env 's private representation (markov state)
 * not visible to the agent
 * contain irrelevant information
2. agent state: the agent's internal representation
3. information state (markov state): contains all the useful information from the history
 * the future is independent of the past given the present

full observability: agent directly observed env state, markov decision process (MDP)

partial observability: agent indirectly observes env
 * texus hold'em
 * agent state $\neq$ env state
  





