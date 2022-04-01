"""This is OpenAI' Spinning Up PyTorch implementation of Soft-Actor-Critic with
minor adjustments.
For the official documentation, see below:
https://spinningup.openai.com/en/latest/algorithms/sac.html#documentation-pytorch-version
Source:
https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py
"""
import itertools
import queue, threading
from copy import deepcopy

import torch
import numpy as np
from gym.spaces import Box
from torch.optim import Adam

from agents.base import BaseAgent
from l2r.common.models.network import ActorCritic
from l2r.common.models.vae import VAE
from l2r.common.utils import RecordExperience
from l2r.common.utils import setup_logging

from ruamel.yaml import YAML

from agents.replay_buffer import ReplayBuffer

from IL.IL_NETs import IL_Net_small
from pickle import load
from PIL import Image
from torchvision.transforms.functional import crop
from torchvision import transforms
from collections import deque

DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# seed = np.random.randint(255)
# torch.manual_seed(seed)
# np.random.seed(seed)


class SACAgent(BaseAgent):
    """Adopted from https://github.com/learn-to-race/l2r/blob/main/l2r/baselines/rl/sac.py"""

    def __init__(self):
        super(SACAgent, self).__init__()

        self.cfg = self.load_model_config("models/sac/params-sac.yaml")
        self.file_logger, self.tb_logger = self.setup_loggers()

        if self.cfg["record_experience"]:
            self.setup_experience_recorder()

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        # self.act_limit = self.action_space.high[0]

        self.setup_vision_encoder()
        self.set_params()
        
        self.prev_act_ste = np.zeros(1)
        self.prev_err_ste = 0
        self.q_err_ste = deque(np.zeros(10), maxlen=10)
        self.pre_q_ste = deque(np.zeros(4), maxlen=4)
        
        self.prev_act_acc = np.zeros(1)
        self.prev_err_acc = 0
        self.q_err_acc = deque(np.zeros(10), maxlen=10)
        
        self.q_feats = deque(np.zeros((4, 33)), maxlen=4)

        # self.factor = 0.99
        
        self.oracle = IL_Net_small().to(DEVICE)
        self.oracle.load_state_dict(torch.load('IL/ckpts/best_IL.pth'))
        self.oracle.eval()
        
        self.ste_q_transformer = load(open('IL/scalers/ste_scaler.pkl', 'rb'))
        self.acc_q_transformer = load(open('IL/scalers/acc_scaler.pkl', 'rb'))
        
        self.transformations = transforms.Compose([transforms.ToTensor(),
                                                   transforms.RandomInvert(),
                                                   transforms.Grayscale(),
                                                   transforms.Resize((96, 96*2)),
                                                   transforms.Normalize((0.5), (0.5))])
        self.my_neps = 0

    def select_action(self, obs, encode=True):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        # print('H')
        obs = []
        for a in list(self.q_feats):
            obs.append(a)
        obs = np.reshape(obs, (-1, 1))
        obs = torch.tensor(obs)
        if encode:
            obs = self._encode(obs)
        if self.t > self.cfg["start_steps"]:
            a = self.actor_critic.act(obs.to(DEVICE), self.deterministic)
            a = a  # numpy array...
            self.record["transition_actor"] = "learner"
        else:
            a = self.action_space.sample()
            self.record["transition_actor"] = "random"
        # self.t = self.t + 1
        # action = a*self.factor + (1-self.factor)*self.prev_acts
        
        # https://medium.com/@techreigns/pid-controller-for-autonomous-vehicles-ed2c2ccbbb86
        ku = 1
        kp = 1#-0.6*ku
        kd = -0.05#kp*12/8
        ki = -0.01#2*kp/1250
        
        et_ste = a[0]#-self.prev_act_ste
        self.q_err_ste.append(et_ste)
        self.pre_q_ste.append(a[0])
        
        action_ste = kp*et_ste + kd*(et_ste - self.prev_err_ste) + ki*np.array(self.q_err_ste).sum() #v1
        # action_ste = a[0] + kd*(et_ste - self.prev_err_ste) + ki*np.array(self.q_err_ste).sum() # v2 Funciona muy bien, muy prometedor pero hay que ajustar lso parámetros para que no bounce mucho, mucho mejor que lo normal
        # action_ste = np.array(self.pre_q_ste).sum()/4

        
        # print(f'Action steering: {action_ste}; {et_ste}; {self.prev_err_ste}; {np.array(self.q_err_ste).sum()}')
        self.prev_err_ste = et_ste
        self.prev_act_ste = action_ste
        
        et_acc = a[1]-self.prev_act_acc
        self.q_err_acc.append(et_acc)
        
        # action_acc = kp*et_acc + kd*(et_acc - self.prev_err_acc) + ki*np.array(self.q_err_acc).sum()
        
        # self.prev_err_acc = et_acc
        # self.prev_act_acc = action_acc
        
        action_ste = np.clip(action_ste, -1., 1.)
        # action_acc = np.clip(action_acc, -1., 1.)

        return [a[0], a[1]]
        # return action


    def change_range(self, OldValue, OldMax, OldMin, NewMax, NewMin): 
        NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
        return NewValue
        
    def oracle_action(self, img):
        img = Image.fromarray(img)
        img = crop(img, 150, 0, 125, 500)
        img = self.transformations(img).reshape(1, 1, 96, 192)
        ste, acc = self.oracle(img.to(DEVICE)).detach().cpu()[0]
        trans_ste = self.ste_q_transformer.inverse_transform(ste.reshape(-1, 1))[0]
        trans_acc = self.acc_q_transformer.inverse_transform(acc.reshape(-1, 1))[0]
        
        # [steering, acceleration], expected to be in the range (-1., 1.)
        # trans_ste = self.change_range(trans_ste[0], .2, -.2, 1., --1.) # Cambio de rango, primero del range entrenado al range del imitation dataset
        # trans_acc = self.change_range(trans_acc, 4.0, -6.0, 6.0, -.1) # luego del range del imitation dataset al range que quiero entre 6.0 y -0.1
        # trans_acc = self.change_range(trans_acc,  6.0, -0.1, 1.0, -1.0) # finalmente lo cambio al rango en que estoy trabajando. No estoy muy seguro de esto

        return trans_ste[0], trans_acc
        
    def register_reset(self, obs) -> np.array:
        """
        Same input/output as select_action, except this method is called at episodal reset.
        """
        # camera, features, state = obs
        self.deterministic = True
        self.t = 1e6

    def load_model(self, path):
        self.actor_critic.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def setup_experience_recorder(self):
        self.save_queue = queue.Queue()
        self.save_batch_size = 1024
        self.record_experience = RecordExperience(
            self.cfg["record_dir"],
            self.cfg["track_name"],
            self.cfg["experiment_name"],
            self.file_logger,
            self,
        )
        self.save_thread = threading.Thread(target=self.record_experience.save_thread)
        self.save_thread.start()

    def setup_vision_encoder(self):
        assert self.cfg["use_encoder_type"] in [
            "vae"
        ], "Specified encoder type must be in ['vae']"
        speed_hiddens = self.cfg[self.cfg["use_encoder_type"]]["speed_hiddens"]
        self.feat_dim = self.cfg[self.cfg["use_encoder_type"]]["latent_dims"] + 1
        self.obs_dim = (
            self.cfg[self.cfg["use_encoder_type"]]["latent_dims"] + speed_hiddens[-1]
            if self.cfg["encoder_switch"]
            else None
        )

        if self.cfg["use_encoder_type"] == "vae":
            self.backbone = VAE(
                im_c=self.cfg["vae"]["im_c"],
                im_h=self.cfg["vae"]["im_h"],
                im_w=self.cfg["vae"]["im_w"],
                z_dim=self.cfg["vae"]["latent_dims"],
            )
            self.backbone.load_state_dict(
                torch.load(self.cfg["vae"]["vae_chkpt_statedict"], map_location=DEVICE)
            )
        else:
            raise NotImplementedError

        self.backbone.to(DEVICE)

    def set_params(self):
        self.save_episodes = True
        self.episode_num = 0
        self.best_ret = 0
        self.t = 0
        self.deterministic = False
        self.atol = 1e-3
        self.store_from_safe = False
        self.pi_scheduler = None
        self.t_start = 0
        self.best_pct = 0

        # This is important: it allows child classes (that extend this one) to "push up" information
        # that this parent class should log
        self.metadata = {}
        self.record = {"transition_actor": ""}

        self.action_space = Box(-1, 1, (2,))
        self.act_dim = self.action_space.shape[0]

        # Experience buffer
        self.replay_buffer = ReplayBuffer(
            obs_dim=self.feat_dim, act_dim=self.act_dim, size=self.cfg["replay_size"]
        )

        self.actor_critic = ActorCritic(
            self.obs_dim*4,
            self.action_space,
            self.cfg,
            latent_dims=self.obs_dim,
            device=DEVICE,
        )

        if self.cfg["checkpoint"] and self.cfg["load_checkpoint"]:
            self.load_model(self.cfg["checkpoint"])

        self.actor_critic_target = deepcopy(self.actor_critic)

    @staticmethod
    def load_model_config(path):
        yaml = YAML()
        params = yaml.load(open(path))
        sac_kwargs = params["agent_kwargs"]
        return sac_kwargs

    def setup_loggers(self):
        save_path = self.cfg["model_save_path"]
        loggers = setup_logging(save_path, self.cfg["experiment_name"], True)
        loggers[0]("Using random seed: {}".format(0))
        return loggers

    def compute_loss_q(self, data):
        """Set up function for computing SAC Q-losses."""
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        q1 = self.actor_critic.q1(o, a)
        q2 = self.actor_critic.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.actor_critic.pi(o2)

            # Target Q-values
            q1_pi_targ = self.actor_critic_target.q1(o2, a2)
            q2_pi_targ = self.actor_critic_target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.cfg["gamma"] * (1 - d) * (
                q_pi_targ - self.cfg["alpha"] * logp_a2
            )

        # MSE loss against Bellman backup
        loss_q1 = (self.replay_buffer.weights * (q1 - backup) ** 2).mean()
        loss_q2 = (self.replay_buffer.weights * (q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(
            Q1Vals=q1.detach().cpu().numpy(), Q2Vals=q2.detach().cpu().numpy()
        )

        return loss_q, q_info

    def compute_loss_pi(self, data):
        """Set up function for computing SAC pi loss."""
        o = data["obs"]
        pi, logp_pi = self.actor_critic.pi(o)
        q1_pi = self.actor_critic.q1(o, pi)
        q2_pi = self.actor_critic.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.cfg["alpha"] * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info

    def update(self, data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(
                self.actor_critic.parameters(), self.actor_critic_target.parameters()
            ):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.cfg["polyak"])
                p_targ.data.add_((1 - self.cfg["polyak"]) * p.data)

    def _step(self, env, action):
        obs, reward, done, info = env.step(action)
        return obs[1], self._encode(obs), obs[0], reward, done, info

    def _reset(self, env, random_pos=False):
        camera = 0
        while (np.mean(camera) == 0) | (np.mean(camera) == 255):
            obs = env.reset(random_pos=random_pos)
            (state, camera), _ = obs
        return camera, self._encode((state, camera)), state

    def _encode(self, o):
        state, img = o

        if self.cfg["use_encoder_type"] == "vae":
            img_embed = self.backbone.encode_raw(np.array(img), DEVICE)[0][0]
            speed = (
                torch.tensor((state[4] ** 2 + state[3] ** 2 + state[5] ** 2) ** 0.5)
                .float()
                .reshape(1, -1)
                .to(DEVICE)
            )
            out = torch.cat([img_embed.unsqueeze(0), speed], dim=-1).squeeze(
                0
            )  # torch.Size([33])
            self.using_speed = 1
        else:
            raise NotImplementedError

        assert not torch.sum(torch.isnan(out)), "found a nan value"
        out[torch.isnan(out)] = 0

        return out

    def eval(self, n_eps, env):
        print("Evaluation:")
        val_ep_rets = []

        # Not implemented for logging multiple test episodes
        assert self.cfg["num_test_episodes"] == 1

        for j in range(self.cfg["num_test_episodes"]):
            camera, features, state = self._reset(env, random_pos=False)
            d, ep_ret, ep_len, n_val_steps, self.metadata = False, 0, 0, 0, {}
            camera, features, state2, r, d, info = self._step(env, [0, 1])
            experience, t = [], 0

            while (not d) & (ep_len <= self.cfg["max_ep_len"]):
                # Take deterministic actions at test time
                self.deterministic = True
                self.t = 1e6
                a = self.select_action(features, encode=False)
                camera2, features2, state2, r, d, info = self._step(env, a)

                # Check that the camera is turned on
                assert (np.mean(camera2) > 0) & (np.mean(camera2) < 255)

                ep_ret += r
                ep_len += 1
                n_val_steps += 1

                # Prevent the agent from being stuck
                if np.allclose(state2[15:16], state[15:16], atol=self.atol, rtol=0):
                    # self.file_logger("Sampling random action to get unstuck")
                    a = env.action_space.sample()
                    # Step the env
                    camera2, features2, state2, r, d, info = self._step(env, a)
                    ep_len += 1

                if self.cfg["record_experience"]:
                    recording = self.add_experience(
                        action=a,
                        camera=camera,
                        next_camera=camera2,
                        done=d,
                        env=env,
                        feature=features,
                        next_feature=features2,
                        info=info,
                        state=state,
                        next_state=state2,
                        step=t,
                    )
                    experience.append(recording)

                features = features2
                camera = camera2
                state = state2
                t += 1

            self.file_logger(f"[eval episode] {info}")

            val_ep_rets.append(ep_ret)
            self.metadata["info"] = info
            self.log_val_metrics_to_tensorboard(info, ep_ret, n_eps, n_val_steps)

            # Quickly dump recently-completed episode's experience to the multithread queue,
            # as long as the episode resulted in "success"
            if self.cfg["record_experience"]:  # and self.metadata['info']['success']:
                self.file_logger("writing experience")
                self.save_queue.put(experience)

        self.checkpoint_model(ep_ret, n_eps)
        self.update_best_pct_complete(info)

        return val_ep_rets

    def update_best_pct_complete(self, info):
        if self.best_pct < info["metrics"]["pct_complete"]:
            for cutoff in [93, 100]:
                if (self.best_pct < cutoff) & (
                    info["metrics"]["pct_complete"] >= cutoff
                ):
                    self.pi_scheduler.step()
            self.best_pct = info["metrics"]["pct_complete"]

    def checkpoint_model(self, ep_ret, n_eps):
        # Save if best (or periodically)
        # if ep_ret > self.best_ret:  # and ep_ret > 100):
        path_name = f"{self.cfg['model_save_path']}/best_{self.cfg['experiment_name']}_episode_{n_eps}.statedict"
        self.file_logger(
            f"New best episode reward of {round(ep_ret, 1)}! Saving: {path_name}"
        )
        self.best_ret = ep_ret
        torch.save(self.actor_critic.state_dict(), path_name)
        path_name = f"{self.cfg['model_save_path']}/best_{self.cfg['experiment_name']}_episode_{n_eps}.statedict"
        try:
            # Try to save Safety Actor-Critic, if present
            torch.save(self.safety_actor_critic.state_dict(), path_name)
        except:
            pass

        # elif self.save_episodes and (n_eps + 1 % self.cfg["save_freq"] == 0):
        #     path_name = f"{self.cfg['model_save_path']}/{self.cfg['experiment_name']}_episode_{n_eps}.statedict"
        #     self.file_logger(
        #         f"Periodic save (save_freq of {self.cfg['save_freq']}) to {path_name}"
        #     )
        #     torch.save(self.actor_critic.state_dict(), path_name)
        #     path_name = f"{self.cfg['model_save_path']}/{self.cfg['experiment_name']}_episode_{n_eps}.statedict"
        #     try:
        #         # Try to save Safety Actor-Critic, if present
        #         torch.save(self.safety_actor_critic.state_dict(), path_name)
        #     except:
        #         pass

    def training(self, env):
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(
            self.actor_critic.q1.parameters(), self.actor_critic.q2.parameters()
        )

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(
            self.actor_critic.policy.parameters(), lr=self.cfg["lr"]
        )
        self.q_optimizer = Adam(self.q_params, lr=self.cfg["lr"])
        self.pi_scheduler = torch.optim.lr_scheduler.StepLR(
            self.pi_optimizer, 1, gamma=0.5
        )

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.actor_critic_target.parameters():
            p.requires_grad = False

        # Count variables (protip: try to get a feel for how different size networks behave!)
        # var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])

        # Prepare for interaction with environment
        # start_time = time.time()
        best_ret, ep_ret, ep_len = 0, 0, 0

        self._reset(env, random_pos=True)
        camera, feat, state, r, d, info = self._step(env, [0, 1])
        
        self.q_feats.append(feat.cpu().detach().numpy())

        experience = []
        speed_dim = 1 if self.using_speed else 0
        assert (
            len(feat)
            == self.cfg[self.cfg["use_encoder_type"]]["latent_dims"] + speed_dim
        ), "'o' has unexpected dimension or is a tuple"

        t_start = self.t_start
        # Main loop: collect experience in env and update/log each epoch
        for t in range(self.t_start, self.cfg["total_steps"]):
            # oracle_ste, oracle_acc = self.oracle_action(camera[0])
            # a = self.select_action(feat, encode=False)
            
            # factor_oracle = -0.006442685 + (0.9998196 + 0.006442685)/(1 + (self.my_neps/100.3081)**5.476169) # Especie de sigmoid que tiene los siguiente valores (0,1),(100,0.5),(200,0) # Le va restando importancia al oracle a lo largo de las epochs
            # factor_actor = 1.0171 + (0.02520802 - 1.0171)/(1 + (self.my_neps/100.9811)**4.359984) # Especie de sigmoid que tiene los siguiente valores (0,0),(100,0.5),(200,1) # Le va dando importancia al actor a lo largo de las epochs

            # a = factor_actor*a + factor_oracle*np.array((oracle_ste, oracle_acc[0]))
            
            # a = [oracle_ste, oracle_acc]
            
            # if np.random.rand() > 0.01*self.my_neps: # Oracle action # 50% of odds in 100 eps
                # oracle_ste, oracle_acc = self.oracle_action(camera[0])
                # a = [oracle_ste, oracle_acc]
            
            # else: # Actor action:
            a = self.select_action(feat, encode=False)
            # Step the env
            camera2, feat2, state2, r, d, info = self._step(env, a)

            # Check that the camera is turned on
            assert (np.mean(camera2) > 0) & (np.mean(camera2) < 255)

            # Prevents the agent from getting stuck by sampling random actions
            # self.atol for SafeRandom and SPAR are set to -1 so that this condition does not activate
            if np.allclose(state2[15:16], state[15:16], atol=self.atol, rtol=0):
                # self.file_logger("Sampling random action to get unstuck")
                a = env.action_space.sample()

                # Step the env
                camera2, feat2, state2, r, d, info = self._step(env, a)
                ep_len += 1

            state = state2
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == self.cfg["max_ep_len"] else d
            prev_q_feats = self.q_feats.copy()
            
            self.q_feats.append(feat.cpu().detach().numpy())

            # Store experience to replay buffer
            if (not np.allclose(state2[15:16], state[15:16], atol=3e-1, rtol=0)) | (
                r != 0
            ):
                # self.replay_buffer.store(feat, a, r, feat2, d)
                self.replay_buffer.store(prev_q_feats, a, r, self.q_feats, d)

            else:
                # print('Skip')
                skip = True

            if self.cfg["record_experience"]:
                recording = self.add_experience(
                    action=a,
                    camera=camera,
                    next_camera=camera2,
                    done=d,
                    env=env,
                    feature=feat,
                    next_feature=feat2,
                    info=info,
                    reward=r,
                    state=state,
                    next_state=state2,
                    step=t,
                )
                experience.append(recording)

                # quickly pass data to save thread
                # if len(experience) == self.save_batch_size:
                #    self.save_queue.put(experience)
                #    experience = []

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            feat = feat2
            state = state2  # in case we, later, wish to store the state in the replay as well
            camera = camera2  # in case we, later, wish to store the state in the replay as well
            # self.q_feats.append(feat.cpu().detach().numpy())

            # Update handling
            if (t >= self.cfg["update_after"]) & (t % self.cfg["update_every"] == 0):
                for j in range(self.cfg["update_every"]):
                    batch = self.replay_buffer.sample_batch(self.cfg["batch_size"])
                    self.update(data=batch)

            if (t + 1) % self.cfg["eval_every"] == 0:
                # eval on test environment
                val_returns = self.eval(t // self.cfg["eval_every"], env)
                self.checkpoint_model(ep_ret, self.episode_num)

                # Reset
                (
                    camera,
                    ep_len,
                    ep_ret,
                    experience,
                    feat,
                    state,
                    t_start,
                ) = self.reset_episode(env, t)

            # End of trajectory handling
            if d or (ep_len == self.cfg["max_ep_len"]):
                self.checkpoint_model(ep_ret, self.episode_num)

                self.metadata["info"] = info
                self.episode_num += 1
                msg = f"[Ep {self.episode_num }] {self.metadata}"
                self.file_logger(msg)
                self.log_train_metrics_to_tensorboard(ep_ret, t, t_start)

                # Quickly dump recently-completed episode's experience to the multithread queue,
                # as long as the episode resulted in "success"
                if self.cfg[
                    "record_experience"
                ]:  # and self.metadata['info']['success']:
                    self.file_logger("Writing experience")
                    self.save_queue.put(experience)
                
                self.my_neps += 1
                # Reset
                (
                    camera,
                    ep_len,
                    ep_ret,
                    experience,
                    feat,
                    state,
                    t_start,
                ) = self.reset_episode(env, t)

    def reset_episode(self, env, t):
        camera, feat, state = self._reset(env, random_pos=True)
        ep_ret, ep_len, self.metadata, experience = 0, 0, {}, []
        t_start = t + 1
        camera, feat, state2, r, d, info = self._step(env, [0, 1])
        return camera, ep_len, ep_ret, experience, feat, state, t_start

    def add_experience(
        self,
        action,
        camera,
        next_camera,
        done,
        env,
        feature,
        next_feature,
        info,
        reward,
        state,
        next_state,
        step,
    ):
        self.recording = {
            "step": step,
            "nearest_idx": env.nearest_idx,
            "camera": camera,
            "feature": feature.detach().cpu().numpy(),
            "state": state,
            "action_taken": action,
            "next_camera": next_camera,
            "next_feature": next_feature.detach().cpu().numpy(),
            "next_state": next_state,
            "reward": reward,
            "episode": self.episode_num,
            "stage": "training",
            "done": done,
            "transition_actor": self.record["transition_actor"],
            "metadata": info,
        }
        return self.recording

    def log_val_metrics_to_tensorboard(self, info, ep_ret, n_eps, n_val_steps):
        self.tb_logger.add_scalar("val/episodic_return", ep_ret, n_eps)
        self.tb_logger.add_scalar("val/ep_n_steps", n_val_steps, n_eps)

        try:
            self.tb_logger.add_scalar(
                "val/ep_pct_complete", info["metrics"]["pct_complete"], n_eps
            )
            self.tb_logger.add_scalar(
                "val/ep_total_time", info["metrics"]["total_time"], n_eps
            )
            self.tb_logger.add_scalar(
                "val/ep_total_distance", info["metrics"]["total_distance"], n_eps
            )
            self.tb_logger.add_scalar(
                "val/ep_avg_speed", info["metrics"]["average_speed_kph"], n_eps
            )
            self.tb_logger.add_scalar(
                "val/ep_avg_disp_err",
                info["metrics"]["average_displacement_error"],
                n_eps,
            )
            self.tb_logger.add_scalar(
                "val/ep_traj_efficiency",
                info["metrics"]["trajectory_efficiency"],
                n_eps,
            )
            self.tb_logger.add_scalar(
                "val/ep_traj_admissibility",
                info["metrics"]["trajectory_admissibility"],
                n_eps,
            )
            self.tb_logger.add_scalar(
                "val/movement_smoothness",
                info["metrics"]["movement_smoothness"],
                n_eps,
            )
        except:
            pass

        # TODO: Find a better way: requires knowledge of child class API :(
        if "safety_info" in self.metadata:
            self.tb_logger.add_scalar(
                "val/ep_interventions",
                self.metadata["safety_info"]["ep_interventions"],
                n_eps,
            )

    def log_train_metrics_to_tensorboard(self, ep_ret, t, t_start):
        self.tb_logger.add_scalar("train/episodic_return", ep_ret, self.episode_num)
        self.tb_logger.add_scalar(
            "train/ep_total_time",
            self.metadata["info"]["metrics"]["total_time"],
            self.episode_num,
        )
        self.tb_logger.add_scalar(
            "train/ep_total_distance",
            self.metadata["info"]["metrics"]["total_distance"],
            self.episode_num,
        )
        self.tb_logger.add_scalar(
            "train/ep_avg_speed",
            self.metadata["info"]["metrics"]["average_speed_kph"],
            self.episode_num,
        )
        self.tb_logger.add_scalar(
            "train/ep_avg_disp_err",
            self.metadata["info"]["metrics"]["average_displacement_error"],
            self.episode_num,
        )
        self.tb_logger.add_scalar(
            "train/ep_traj_efficiency",
            self.metadata["info"]["metrics"]["trajectory_efficiency"],
            self.episode_num,
        )
        self.tb_logger.add_scalar(
            "train/ep_traj_admissibility",
            self.metadata["info"]["metrics"]["trajectory_admissibility"],
            self.episode_num,
        )
        self.tb_logger.add_scalar(
            "train/movement_smoothness",
            self.metadata["info"]["metrics"]["movement_smoothness"],
            self.episode_num,
        )
        self.tb_logger.add_scalar("train/ep_n_steps", t - t_start, self.episode_num)
