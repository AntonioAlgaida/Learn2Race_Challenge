import torch
import torch.nn as nn
from l2r.baselines.core import mlp, SquashedGaussianMLPActor


def resnet18(pretrained=True):
    model = torch.hub.load("pytorch/vision:v0.6.0", "resnet18", pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Identity()
    return model


class Qfunction(nn.Module):
    """
    Modified from the core MLPQFunction and MLPActorCritic to include a speed encoder
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # pdb.set_trace()
        self.speed_encoder = mlp(
            [1] + self.cfg[self.cfg["use_encoder_type"]]["speed_hiddens"]
        )
        # self.regressor = mlp(
        #     [
        #         self.cfg[self.cfg["use_encoder_type"]]["latent_dims"]
        #         + self.cfg[self.cfg["use_encoder_type"]]["speed_hiddens"][-1]
        #         + 2
        #     ]
        #     + self.cfg[self.cfg["use_encoder_type"]]["hiddens"]
        #     + [1]
        # )
        self.regressor = mlp( [162] + self.cfg[self.cfg["use_encoder_type"]]["hiddens"] + [1])
        # self.lr = cfg['resnet']['LR']

    def forward(self, obs_feat, action):
        # if obs_feat.ndimension() == 1:
        #    obs_feat = obs_feat.unsqueeze(0)
        img_embed = obs_feat[
            ..., : self.cfg[self.cfg["use_encoder_type"]]["latent_dims"]
        ]  # n x latent_dims
        bs = img_embed.shape[0]

        img_embed = torch.reshape(img_embed, (bs, -1))
        speed = obs_feat[
            ..., self.cfg[self.cfg["use_encoder_type"]]["latent_dims"] :
        ]  # n x 1
            
        # speed = torch.reshape(speed, (bs, -1))

        spd_embed = self.speed_encoder(speed)  # n x 16
        spd_embed = torch.reshape(spd_embed, (bs, -1))

        out = self.regressor(torch.cat([img_embed, spd_embed, action], dim=1))  # n x 1
        # pdb.set_trace()
        return out.view(-1)


class DuelingNetwork(nn.Module):
    """
    Further modify from Qfunction to
        - Add an action_encoder
        - Separate state-dependent value and advantage
            Q(s, a) = V(s) + A(s, a)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.speed_encoder = mlp(
            [1] + self.cfg[self.cfg["use_encoder_type"]]["speed_hiddens"]
        )
        self.action_encoder = mlp(
            [2] + self.cfg[self.cfg["use_encoder_type"]]["action_hiddens"]
        )

        n_obs = (
            self.cfg[self.cfg["use_encoder_type"]]["latent_dims"]
            + self.cfg[self.cfg["use_encoder_type"]]["speed_hiddens"][-1]
        )
        # self.V_network = mlp([n_obs] + self.cfg[self.cfg['use_encoder_type']]['hiddens'] + [1])
        self.A_network = mlp(
            [n_obs + self.cfg[self.cfg["use_encoder_type"]]["action_hiddens"][-1]]
            + self.cfg[self.cfg["use_encoder_type"]]["hiddens"]
            + [1]
        )
        # self.lr = cfg['resnet']['LR']

    def forward(self, obs_feat, action, advantage_only=False):
        # if obs_feat.ndimension() == 1:
        #    obs_feat = obs_feat.unsqueeze(0)
        img_embed = obs_feat[
            ..., : self.cfg[self.cfg["use_encoder_type"]]["latent_dims"]
        ]  # n x latent_dims
        speed = obs_feat[
            ..., self.cfg[self.cfg["use_encoder_type"]]["latent_dims"] :
        ]  # n x 1
        spd_embed = self.speed_encoder(speed)  # n x 16
        action_embed = self.action_encoder(action)

        out = self.A_network(torch.cat([img_embed, spd_embed, action_embed], dim=-1))
        """
        if advantage_only == False:
            V = self.V_network(torch.cat([img_embed, spd_embed], dim = -1)) # n x 1
            out += V
        """
        return out.view(-1)


class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        cfg,
        activation=nn.ReLU,
        latent_dims=None,
        device="cpu",
        safety=False,  ## Flag to indicate architecture for Safety_actor_critic
    ):
        super().__init__()
        self.cfg = cfg
        obs_dim = observation_space.shape[0] if latent_dims is None else latent_dims
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.speed_encoder = mlp(
            [4] + self.cfg[self.cfg["use_encoder_type"]]["speed_hiddens"]
        )
        self.policy = SquashedGaussianMLPActor(
            136,
            act_dim,
            cfg[cfg["use_encoder_type"]]["actor_hiddens"],
            activation,
            act_limit,
        )
        if safety:
            self.q1 = DuelingNetwork(cfg)
        else:
            self.q1 = Qfunction(cfg)
            self.q2 = Qfunction(cfg)
        self.device = device
        self.to(device)

    def pi(self, obs_feat, deterministic=False):
        # if obs_feat.ndimension() == 1:
        #    obs_feat = obs_feat.unsqueeze(0)
        # img_embed = obs_feat[
        #     ..., : self.cfg[self.cfg["use_encoder_type"]]["latent_dims"]
        # ]  # n x latent_dims
        # speed = obs_feat[
        #     ..., self.cfg[self.cfg["use_encoder_type"]]["latent_dims"] :
        # ]  # n x 1
        
        img_embed = obs_feat[:,:,:32]
        
        speed = obs_feat[:,:,-1]
        
        spd_embed = self.speed_encoder(speed.float())  # n x 8
        
        bs = img_embed.shape[0]
        feat = torch.cat([img_embed.reshape(bs, -1), spd_embed], dim=1).unsqueeze(dim=-1)[:, :, 0]
        return self.policy(feat, deterministic, True)

    def act(self, obs_feat, deterministic=False):
        # if obs_feat.ndimension() == 1:
        #    obs_feat = obs_feat.unsqueeze(0)
        
        
        img_embed = torch.concat((obs_feat[:32],
                                  obs_feat[33:33+32],
                                  obs_feat[2*33:2*33+32],
                                  obs_feat[3*33:3*33+32]))
        
        speed = torch.concat((obs_feat[33],
                                  obs_feat[33*2],
                                  obs_feat[33*3],
                                  obs_feat[-1]))
        
        with torch.no_grad():
            # img_embed = obs_feat[:self.cfg[self.cfg["use_encoder_type"]]["latent_dims"]]  # n x latent_dims
            # speed = obs_feat[self.cfg[self.cfg["use_encoder_type"]]["latent_dims"] :]  # n x 1
            # pdb.set_trace()
            spd_embed = self.speed_encoder(speed.float())  # n x 8
            feat = torch.cat([img_embed[:,0], spd_embed], dim=0)
            a, _ = self.policy(feat, deterministic, False)
            a = a.squeeze(0)
        return a.numpy() if self.device == "cpu" else a.cpu().numpy()
