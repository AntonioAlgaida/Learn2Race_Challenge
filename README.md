![Learn to Race Banner](docs/l2r_banner.jpg)

# [Learn to Race Challenge](https://www.aicrowd.com/challenges/learn-to-race-autonomous-racing-virtual-challenge) | Starter Kit 
[![Discord](https://img.shields.io/discord/565639094860775436.svg)](https://discord.gg/fNRrSvZkry)

This repository is the Learn to Race Challenge **Submission template and Starter kit**! Clone the repository to compete now!

**This repository contains**:
*  **Documentation** on how to submit your models to the leaderboard
*  **The procedure** for best practices and information on how we evaluate your agent, etc.
*  **Starter code** for you to get started!

# Table of Contents

- [Competition Overview](#competition-overview)
    + [Competition Stages](#competition-stages)
- [Getting Started](#getting-started)
- [How to write your own agent?](#how-to-write-your-own-agent)
- [How to start participating?](#how-to-start-participating)
  * [Setup](#setup)
  * [How do I specify my software runtime / dependencies?](#how-do-i-specify-my-software-runtime-dependencies)
  * [What should my code structure be like?](#what-should-my-code-structure-be-like)
  * [How to make a submission?](#how-to-make-a-submission)
- [Other Concepts](#other-concepts)
    + [Evaluation Metrics](#evaluation-metrics)
    + [Ranking Criteria](#ranking-criteria)
    + [Time constraints](#time-constraints)
  * [Local Evaluation](#local-evaluation)
  * [Contributing](#contributing)
  * [Contributors](#contributors)
- [Important links](#-important-links)


#  Competition Overview
The Learn to Race Challenge is an opportunity for researchers and machine learning enthusiasts to test their skills by developing autonomous agents that can adhere to safety specifications in high-speed racing. Racing demands each vehicle to drive at its physical limits with barely any margin for safety, when any infraction could lead to catastrophic failures. Given this inherent tension, we envision autonomous racing to serve as a particularly challenging proving ground for safe learning algorithms.

### Competition Stages
The challenge consists of two stages: 
- In **Stage 1**, participants will train their models locally, and then upload submit model checkpoints to AIcrowd for evaluation on *Thruxton Circuit*, which is included in the Learn-to-Race environment. Each team will be able to submit agents to the evaluation service with a limit of 1 successful submission every 24 hours. The top 10 teams on the leader board will enter **Stage 2**.

![](https://images.aicrowd.com/uploads/ckeditor/pictures/633/content_final_gif_white.gif)

- In **Stage 2**, participants will submit their models (with checkpoints) to AIcrowd for training on an unseen track for a time budget of one hour, during which the number of safety infractions will be accumulated as one of the evaluation metrics. After the one-hour ‘practice’ period, the agent will be evaluated on the unseen track. Each team may submit up to three times for this stage, and the best results will be used for the final ranking. This is intended to give participants a chance to deal with bugs or submission errors.



#  Getting Started
1. **Sign up** to join the competition [on the AIcrowd website](https://www.aicrowd.com/challenges/learn-to-race-autonomous-racing-virtual-challenge).
2. **Download** the Arrival Autonomous Racing Simulator [from this link](https://www.aicrowd.com/clef_tasks/82/task_dataset_files?challenge_id=954).
3. **Fork** this starter kit repository. You can use [this link](https://gitlab.aicrowd.com/learn-to-race/l2r-starter-kit/-/forks/new) to create a fork.
4. **Clone** your forked repo and start developing your autonomous racing agent.
5. **Develop** your autonomous racing agents following the template in [how to write your own agent](#how-to-write-your-own-agent) section.
6. [**Submit**](#how-to-make-a-submission) your trained models to [AIcrowd Gitlab](https://gitlab.aicrowd.com) for evaluation [(full instructions below)](#how-to-make-a-submission). The automated evaluation setup will evaluate the submissions on the racetrack and report the metrics on the leaderboard of the competition.


# How to write your own agent?

We recommend that you place the code for all your agents in the `agents` directory (though it is not mandatory). You should implement the

- `select_action`
- `register_reset`
- `training` (Needed only in stage 2)
- `load_model` (Needed only in stage 2)
- `save_model` (Needed only in stage 2)
  
methods as specified in the [`BaseAgent`](agents/base.py) class. We recommend that you write your code in such a way that it implements `training`, `load_model`, and `save_model` methods as expected. This will ensure that your code is ready for stage 2 evaluations. 

Please refer the [`BaseAgent`](agents/base.py) class for the input/output interfaces.

Update the `SubmissionConfig` in [config.py](config.py#L5) to use your new agent class instead of the `SACAgent`.

# How to start participating?

## Setup

1. **Add your SSH key** to AIcrowd GitLab

You can add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/profile/keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair).

2. **Fork the repository**. You can use [this link](https://gitlab.aicrowd.com/learn-to-race/l2r-starter-kit/-/forks/new) to create a fork.

2.  **Clone the repository**

    ```
    git clone git@gitlab.aicrowd.com:<YOUR_AICROWD_USERNAME>/l2r-starter-kit.git
    ```

3. **Install** competition specific dependencies!
    ```
    cd l2r-starter-kit
    pip install -r requirements.txt
    ```

4. Try out the SAC agent by running `python rollout.py`. You should start the simulator first, by running `bash <simulator_path>/ArrivalSim-linux-0.7.1.188691/LinuxNoEditor/ArrivalSim.sh -openGL`. You can also checkout the [random agent](agents/random_agent.py) implementation for a minimal reference code.

5. Write your own agent as described in [How to write your own agent](#how-to-write-your-own-agent) section.

6. Make a submission as described in [How to make a submission](#how-to-make-a-submission) section.

## How do I specify my software runtime / dependencies?

We accept submissions with custom runtime, so you don't need to worry about which libraries or framework to pick from.

The configuration files typically include `requirements.txt` (pypi packages), `apt.txt` (apt packages) or even your own `Dockerfile`.

You can check detailed information about the same in the 👉 [runtime.md](docs/runtime.md) file.

## What should my code structure be like?

Please follow the example structure as it is in the starter kit for the code structure.
The different files and directories have following meaning:

```
.
├── aicrowd.json           # Submission meta information - like your username
├── apt.txt                # Packages to be installed inside docker image
├── requirements.txt       # Python packages to be installed
├── rollout.py             # Entrypoint to test your code locally (DO NOT EDIT, will be replaced during evaluation)
├── config.py              # File containing env, simulator and submission configuration
├── l2r/                   # Directory containing L2R env specific scripts
├── evaluator/             # Helper scripts for local evaluation (will be ignored during evaluation)
├── racetracks/            # L2R racetrack data (DO NOT EDIT, will be replaced during evaluation)
├── utility/               # Helper scripts to simplify submission flow
└── agents                 # Place your agents related code here
    ├── base.py            # Code for base agent
    └── <my_agent>.py      # IMPORTANT: Your agent code
```

Finally, **you must specify an AIcrowd submission JSON in `aicrowd.json` to be scored!** 

The `aicrowd.json` of each submission should contain the following content:

```json
{
  "challenge_id": "learn-to-race-autonomous-racing-virtual-challenge",
  "authors": ["your-aicrowd-username"],
  "description": "(optional) description about your awesome agent",
  "external_dataset_used": false
}
```

This JSON is used to map your submission to the challenge - so please remember to use the correct `challenge_id` as specified above.

## How to make a submission?

👉 [submission.md](/docs/submission.md)

**Best of Luck** :tada: :tada:

# Other Concepts
### Evaluation Metrics
- **Success Rate**: Each race track is partitioned into a fixed number of segments and the success rate is calculated as the number of successfully completed segments over the total number of segments. If the agent fails at a certain segment, it will respawn stationarily at the beginning of the next segment. If the agent successfully completes a segment, it will continue on to the next segment carrying over the current speed.
- **Average Speed**: Average speed is defined as the total distance traveled divided by total tome.
- **Number of Safety Infractions** (Stage 2 ONLY): The number of safety infractions is accumulated during the 1-hour ‘practice’ period in Stage 2 of the competition. The agent is considered to have incurred a safety infraction if 2 wheels of the vehicle leave the drivable area, the vehicle collides with an object, or does not progress for a number of steps (e.g. stuck). In Learn-to-Race, the agent is considered having failed upon any safety infraction. 

### Ranking Criteria
- In Stage 1, the submissions will first be ranked on success rate, and then submissions with the same success rate will be ranked on average speed.
- In Stage 2, the submissions will first be ranked on success rate, and then submissions with the same success rate will be ranked on a weighted sum of the total number of safety infractions and the average speed. 

### Time constraints
- To prevent the participants from achieving a high success rate by driving very slowly, the maximum episode length will be set based on an average speed of 30km/h. The evaluation will terminate if the maximum episode length is reached and metrics will be computed based on performance up till that point.   


## Local Evaluation
- Participants can run the evaluation protocol for their agent locally with or without any constraint posed by the Challenge to benchmark their agents privately.
- Remember to start the simulator first, by executing `bash <simulator_path>/ArrivalSim-linux-0.7.1.188691/LinuxNoEditor/ArrivalSim.sh -openGL`.
- Participants can familiarize themselves with the code base by trying out the random agent, as a minimal example, by running `python rollout.py`. 
- Upon finishing the `select_action` method in the agent class, one should be able to execute the `evaluation_routine` method in `rollout.py`.
- One should write the training procedures in the `training` method in the agent class, and then one can execute the `training_routine` method in `rollout.py`.

## Contributing

🙏 You can share your solutions or any other baselines by contributing directly to this repository by opening merge request.

- Add your implemntation as `agents/<your_agent>.py`.
- Test it out using `python rollout.py`.
- Add any documentation for your approach at top of your file.
- Import it in `config.py`
- Create merge request! 🎉🎉🎉 

## Contributors

- [Jon Francis](https://www.aicrowd.com/participants/jon_francis)
- [Shravya Bhat](https://www.aicrowd.com/participants/shravyab)
- [Jyotish](https://www.aicrowd.com/participants/jyotish)

# 📎 Important links

💪 &nbsp;Challenge Page: https://www.aicrowd.com/challenges/learn-to-race-autonomous-racing-virtual-challenge

🗣️ &nbsp;Discussion Forum: https://www.aicrowd.com/challenges/learn-to-race-autonomous-racing-virtual-challenge/discussion

🏆 &nbsp;Leaderboard: https://www.aicrowd.com/challenges/learn-to-race-autonomous-racing-virtual-challenge/leaderboards
