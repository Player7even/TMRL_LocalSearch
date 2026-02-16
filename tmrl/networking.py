import datetime
import os
import socket
import time
import atexit
import json
import shutil
import tempfile
import itertools
from os.path import exists

import numpy as np
from requests import get
from tlspyo import Relay, Endpoint
import random as rd, csv

from tmrl.actor import ActorModule
from tmrl.util import dump, load, partial_to_dict
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj

import logging


__docformat__ = "google"

def print_with_timestamp(s):
    x = datetime.datetime.now()
    sx = x.strftime("%x %X ")
    logging.info(sx + str(s))


def print_ip():
    public_ip = get('http://api.ipify.org').text
    local_ip = socket.gethostbyname(socket.gethostname())
    print_with_timestamp(f"public IP: {public_ip}, local IP: {local_ip}")

class Buffer:
    """
    Buffer of training samples.

    `Server`, `RolloutWorker` and `Trainer` all have their own `Buffer` to store and send training samples.

    Samples are tuples of the form (`act`, `new_obs`, `rew`, `terminated`, `truncated`, `info`)
    """
    def __init__(self, maxlen=cfg.BUFFERS_MAXLEN):
        """
        Args:
            maxlen (int): buffer length
        """
        self.memory = []
        self.stat_train_return = 0.0  
        self.stat_test_return = 0.0  
        self.stat_train_steps = 0  
        self.stat_test_steps = 0 
        self.maxlen = maxlen

    def clip_to_maxlen(self):
        lenmem = len(self.memory)
        if lenmem > self.maxlen:
            print_with_timestamp("buffer overflow. Discarding old samples.")
            self.memory = self.memory[(lenmem - self.maxlen):]

    def append_sample(self, sample):
        """
        Appends `sample` to the buffer.

        Args:
            sample (Tuple): a training sample of the form (`act`, `new_obs`, `rew`, `terminated`, `truncated`, `info`)
        """
        self.memory.append(sample)
        self.clip_to_maxlen()

    def clear(self):
        """
        Clears the buffer but keeps train and test returns.
        """
        self.memory = []

    def __len__(self):
        return len(self.memory)

    def __iadd__(self, other):
        self.memory += other.memory
        self.clip_to_maxlen()
        self.stat_train_return = other.stat_train_return
        self.stat_test_return = other.stat_test_return
        self.stat_train_steps = other.stat_train_steps
        self.stat_test_steps = other.stat_test_steps
        return self

class Server:
    """
    Central server.

    The `Server` lets 1 `Trainer` and n `RolloutWorkers` connect.
    It buffers experiences sent by workers and periodically sends these to the trainer.
    It also receives the weights from the trainer and broadcasts these to the connected workers.
    """
    def __init__(self,
                 port=cfg.PORT,
                 password=cfg.PASSWORD,
                 local_port=cfg.LOCAL_PORT_SERVER,
                 header_size=cfg.HEADER_SIZE,
                 security=cfg.SECURITY,
                 keys_dir=cfg.CREDENTIALS_DIRECTORY,
                 max_workers=cfg.NB_WORKERS):
        """
        Args:
            port (int): tlspyo public port
            password (str): tlspyo password
            local_port (int): tlspyo local communication port
            header_size (int): tlspyo header size (bytes)
            security (Union[str, None]): tlspyo security type (None or "TLS")
            keys_dir (str): tlspyo credentials directory
            max_workers (int): max number of accepted workers
        """
        self.__relay = Relay(port=port,
                             password=password,
                             accepted_groups={
                                 'trainers': {
                                     'max_count': 1,
                                     'max_consumables': None},
                                 'workers': {
                                     'max_count': max_workers,
                                     'max_consumables': None}},
                             local_com_port=local_port,
                             header_size=header_size,
                             security=security,
                             keys_dir=keys_dir)

class TrainerInterface:
    """
    This is the trainer's network interface
    This connects to the server
    This receives samples batches and sends new weights
    """
    def __init__(self,
                 server_ip=None,
                 server_port=cfg.PORT,
                 password=cfg.PASSWORD,
                 local_com_port=cfg.LOCAL_PORT_TRAINER,
                 header_size=cfg.HEADER_SIZE,
                 max_buf_len=cfg.BUFFER_SIZE,
                 security=cfg.SECURITY,
                 keys_dir=cfg.CREDENTIALS_DIRECTORY,
                 hostname=cfg.HOSTNAME,
                 model_path=cfg.MODEL_PATH_TRAINER):

        self.model_path = model_path
        self.server_ip = server_ip if server_ip is not None else '127.0.0.1'
        self.__endpoint = Endpoint(ip_server=self.server_ip,
                                   port=server_port,
                                   password=password,
                                   groups="trainers",
                                   local_com_port=local_com_port,
                                   header_size=header_size,
                                   max_buf_len=max_buf_len,
                                   security=security,
                                   keys_dir=keys_dir,
                                   hostname=hostname)

        print_with_timestamp(f"server IP: {self.server_ip}")

        self.__endpoint.notify(groups={'trainers': -1})

    def broadcast_model(self, model: ActorModule):
        """
        model must be an ActorModule
        broadcasts the model's weights to all connected RolloutWorkers
        """
        model.save(self.model_path)
        with open(self.model_path, 'rb') as f:
            weights = f.read()
        self.__endpoint.broadcast(weights, "workers")

    def retrieve_buffer(self):
        """
        returns the TrainerInterface's buffer of training samples
        """
        buffers = self.__endpoint.receive_all()
        res = Buffer()
        for buf in buffers:
            res += buf
        self.__endpoint.notify(groups={'trainers': -1})
        return res


def log_environment_variables():
    """
    add certain relevant environment variables to our config
    usage: `LOG_VARIABLES='HOME JOBID' python ...`
    """
    return {k: os.environ.get(k, '') for k in os.environ.get('LOG_VARIABLES', '').strip().split()}


def load_run_instance(checkpoint_path):
    """
    Default function used to load trainers from checkpoint path
    Args:
        checkpoint_path: the path where instances of run_cls are checkpointed
    Returns:
        An instance of run_cls loaded from checkpoint_path
    """
    return load(checkpoint_path)


def dump_run_instance(run_instance, checkpoint_path):
    """
    Default function used to dump trainers to checkpoint path
    Args:
        run_instance: the instance of run_cls to checkpoint
        checkpoint_path: the path where instances of run_cls are checkpointed
    """
    dump(run_instance, checkpoint_path)


def iterate_epochs(run_cls,
                   interface: TrainerInterface,
                   checkpoint_path: str,
                   dump_run_instance_fn=dump_run_instance,
                   load_run_instance_fn=load_run_instance,
                   epochs_between_checkpoints=1,
                   updater_fn=None):
    """
    Main training loop (remote)
    The run_cls instance is saved in checkpoint_path at the end of each epoch
    The model weights are sent to the RolloutWorker every model_checkpoint_interval epochs
    Generator yielding episode statistics (list of pd.Series) while running and checkpointing
    """
    checkpoint_path = checkpoint_path or tempfile.mktemp("_remove_on_exit")

    try:
        logging.debug(f"checkpoint_path: {checkpoint_path}")
        if not exists(checkpoint_path):
            logging.info(f"=== specification ".ljust(70, "="))
            run_instance = run_cls()
            dump_run_instance_fn(run_instance, checkpoint_path)
            logging.info(f"")
        else:
            logging.info(f"Loading checkpoint...")
            t1 = time.time()
            run_instance = load_run_instance_fn(checkpoint_path)
            logging.info(f" Loaded checkpoint in {time.time() - t1} seconds.")
            if updater_fn is not None:
                logging.info(f"Updating checkpoint...")
                t1 = time.time()
                run_instance = updater_fn(run_instance, run_cls)
                logging.info(f"Checkpoint updated in {time.time() - t1} seconds.")

        while run_instance.epoch < run_instance.epochs:
            yield run_instance.run_epoch(interface=interface)
            if run_instance.epoch % epochs_between_checkpoints == 0:
                logging.info(f" saving checkpoint...")
                t1 = time.time()
                dump_run_instance_fn(run_instance, checkpoint_path)
                logging.info(f" saved checkpoint in {time.time() - t1} seconds.")

    finally:
        if checkpoint_path.endswith("_remove_on_exit") and exists(checkpoint_path):
            os.remove(checkpoint_path)


def run_with_wandb(entity, project, run_id, interface, run_cls, checkpoint_path: str = None, dump_run_instance_fn=None, load_run_instance_fn=None, updater_fn=None):
    """
    Main training loop (remote).

    saves config and stats to https://wandb.com
    """
    dump_run_instance_fn = dump_run_instance_fn or dump_run_instance
    load_run_instance_fn = load_run_instance_fn or load_run_instance
    wandb_dir = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, wandb_dir, ignore_errors=True)
    import wandb
    logging.debug(f" run_cls: {run_cls}")
    config = partial_to_dict(run_cls)
    config['environ'] = log_environment_variables()
    resume = checkpoint_path and exists(checkpoint_path)
    wandb_initialized = False
    err_cpt = 0
    while not wandb_initialized:
        try:
            wandb.init(dir=wandb_dir, entity=entity, project=project, id=run_id, resume=resume, config=config)
            wandb_initialized = True
        except Exception as e:
            err_cpt += 1
            logging.warning(f"wandb error {err_cpt}: {e}")
            if err_cpt > 10:
                logging.warning(f"Could not connect to wandb, aborting.")
                exit()
            else:
                time.sleep(10.0)
    for stats in iterate_epochs(run_cls, interface, checkpoint_path, dump_run_instance_fn, load_run_instance_fn, 1, updater_fn):
        [wandb.log(json.loads(s.to_json())) for s in stats]


def run(interface, run_cls, checkpoint_path: str = None, dump_run_instance_fn=None, load_run_instance_fn=None, updater_fn=None):
    """
    Main training loop (remote).
    """
    dump_run_instance_fn = dump_run_instance_fn or dump_run_instance
    load_run_instance_fn = load_run_instance_fn or load_run_instance
    for stats in iterate_epochs(run_cls, interface, checkpoint_path, dump_run_instance_fn, load_run_instance_fn, 1, updater_fn):
        pass


class Trainer:
    """
    Training entity.

    The `Trainer` object is where RL training happens.
    Typically, it can be located on a HPC cluster.
    """
    def __init__(self,
                 training_cls=cfg_obj.TRAINER,
                 server_ip=cfg.SERVER_IP_FOR_TRAINER,
                 server_port=cfg.PORT,
                 password=cfg.PASSWORD,
                 local_com_port=cfg.LOCAL_PORT_TRAINER,
                 header_size=cfg.HEADER_SIZE,
                 max_buf_len=cfg.BUFFER_SIZE,
                 security=cfg.SECURITY,
                 keys_dir=cfg.CREDENTIALS_DIRECTORY,
                 hostname=cfg.HOSTNAME,
                 model_path=cfg.MODEL_PATH_TRAINER,
                 checkpoint_path=cfg.CHECKPOINT_PATH,
                 dump_run_instance_fn: callable = None,
                 load_run_instance_fn: callable = None,
                 updater_fn: callable = None):
        """
        Args:
            training_cls (type): training class (subclass of tmrl.training_offline.TrainingOffline)
            server_ip (str): ip of the central `Server`
            server_port (int): public port of the central `Server`
            password (str): password of the central `Server`
            local_com_port (int): port used by `tlspyo` for local communication
            header_size (int): number of bytes used for `tlspyo` headers
            max_buf_len (int): maximum number of messages queued by `tlspyo`
            security (str): `tlspyo security type` (None or "TLS")
            keys_dir (str): custom credentials directory for `tlspyo` TLS security
            hostname (str): custom TLS hostname
            model_path (str): path where a local copy of the model will be saved
            checkpoint_path: path where the `Trainer` will be checkpointed (`None` = no checkpointing)
            dump_run_instance_fn (callable): custom serializer (`None` = pickle.dump)
            load_run_instance_fn (callable): custom deserializer (`None` = pickle.load)
            updater_fn (callable): custom updater (`None` = no updater). If provided, this must be a function \
            that takes a checkpoint and training_cls as argument and returns an updated checkpoint. \
            The updater is called after a checkpoint is loaded, e.g., to update your checkpoint with new arguments.
        """
        self.checkpoint_path = checkpoint_path
        self.dump_run_instance_fn = dump_run_instance_fn
        self.load_run_instance_fn = load_run_instance_fn
        self.updater_fn = updater_fn
        self.training_cls = training_cls
        self.interface = TrainerInterface(server_ip=server_ip,
                                          server_port=server_port,
                                          password=password,
                                          local_com_port=local_com_port,
                                          header_size=header_size,
                                          max_buf_len=max_buf_len,
                                          security=security,
                                          keys_dir=keys_dir,
                                          hostname=hostname,
                                          model_path=model_path)

    def run(self):
        """
        Runs training.
        """
        run(interface=self.interface,
            run_cls=self.training_cls,
            checkpoint_path=self.checkpoint_path,
            dump_run_instance_fn=self.dump_run_instance_fn,
            load_run_instance_fn=self.load_run_instance_fn,
            updater_fn=self.updater_fn)

    def run_with_wandb(self,
                       entity=cfg.WANDB_ENTITY,
                       project=cfg.WANDB_PROJECT,
                       run_id=cfg.WANDB_RUN_ID,
                       key=None):
        """
        Runs training while logging metrics to wandb_.

        .. _wandb: https://wandb.ai

        Args:
            entity (str): wandb entity
            project (str): wandb project
            run_id (str): name of the run
            key (str): wandb API key
        """
        if key is not None:
            os.environ['WANDB_API_KEY'] = key
        run_with_wandb(entity=entity,
                       project=project,
                       run_id=run_id,
                       interface=self.interface,
                       run_cls=self.training_cls,
                       checkpoint_path=self.checkpoint_path,
                       dump_run_instance_fn=self.dump_run_instance_fn,
                       load_run_instance_fn=self.load_run_instance_fn,
                       updater_fn=self.updater_fn)

class RolloutWorker:
    """Actor.

    A `RolloutWorker` deploys the current policy in the environment.
    A `RolloutWorker` may connect to a `Server` to which it sends buffered experience.
    Alternatively, it may exist in standalone mode for deployment.
    """
    def __init__(
            self,
            env_cls,
            actor_module_cls,
            sample_compressor: callable = None,
            device="cpu",
            max_samples_per_episode=np.inf,
            model_path=cfg.MODEL_PATH_WORKER,
            obs_preprocessor: callable = None,
            crc_debug=False,
            model_path_history=cfg.MODEL_PATH_SAVE_HISTORY,
            model_history=cfg.MODEL_HISTORY,
            standalone=False,
            server_ip=None,
            server_port=cfg.PORT,
            password=cfg.PASSWORD,
            local_port=cfg.LOCAL_PORT_WORKER,
            header_size=cfg.HEADER_SIZE,
            max_buf_len=cfg.BUFFER_SIZE,
            security=cfg.SECURITY,
            keys_dir=cfg.CREDENTIALS_DIRECTORY,
            hostname=cfg.HOSTNAME
    ):
        """
        Args:
            env_cls (type): class of the Gymnasium environment (subclass of tmrl.envs.GenericGymEnv)
            actor_module_cls (type): class of the module containing the policy (subclass of tmrl.actor.ActorModule)
            sample_compressor (callable): compressor for sending samples over the Internet; \
            when not `None`, `sample_compressor` must be a function that takes the following arguments: \
            (prev_act, obs, rew, terminated, truncated, info), and that returns them (modified) in the same order: \
            when not `None`, a `sample_compressor` works with a corresponding decompression scheme in the `Memory` class
            device (str): device on which the policy is running
            max_samples_per_episode (int): if an episode gets longer than this, it is reset
            model_path (str): path where a local copy of the policy will be stored
            obs_preprocessor (callable): utility for modifying observations retrieved from the environment; \
            when not `None`, `obs_preprocessor` must be a function that takes an observation as input and outputs the \
            modified observation
            crc_debug (bool): useful for debugging custom pipelines; leave to False otherwise
            model_path_history (str): (include the filename but omit .tmod) path to the saved history of policies; \
            we recommend you leave this to the default
            model_history (int): policies are saved every `model_history` new policies (0: not saved)
            standalone (bool): if True, the worker will not try to connect to a server
            server_ip (str): ip of the central server
            server_port (int): public port of the central server
            password (str): tlspyo password
            local_port (int): tlspyo local communication port; usually, leave this to the default
            header_size (int): tlspyo header size (bytes)
            max_buf_len (int): tlspyo max number of messages in buffer
            security (str): tlspyo security type (None or "TLS")
            keys_dir (str): tlspyo credentials directory; usually, leave this to the default
            hostname (str): tlspyo hostname; usually, leave this to the default
        """
        self.obs_preprocessor = obs_preprocessor
        self.get_local_buffer_sample = sample_compressor
        self.env = env_cls()
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        self.model_path = model_path
        self.model_path_history = model_path_history
        self.device = device
        self.actor = actor_module_cls(observation_space=obs_space, action_space=act_space).to_device(self.device)
        self.standalone = standalone
        if os.path.isfile(self.model_path):
            logging.debug(f"Loading model from {self.model_path}")
            self.actor = self.actor.load(self.model_path, device=self.device)
        else:
            logging.debug(f"No model found at {self.model_path}")
        self.buffer = Buffer()
        self.max_samples_per_episode = max_samples_per_episode
        self.crc_debug = crc_debug
        self.model_history = model_history
        self._cur_hist_cpt = 0
        self.model_cpt = 0

        self.debug_ts_cpt = 0
        self.debug_ts_res_cpt = 0

        self.server_ip = server_ip if server_ip is not None else '127.0.0.1'

        print_with_timestamp(f"server IP: {self.server_ip}")

        if not self.standalone:
            self.__endpoint = Endpoint(ip_server=self.server_ip,
                                       port=server_port,
                                       password=password,
                                       groups="workers",
                                       local_com_port=local_port,
                                       header_size=header_size,
                                       max_buf_len=max_buf_len,
                                       security=security,
                                       keys_dir=keys_dir,
                                       hostname=hostname,
                                       deserializer_mode="synchronous")
        else:
            self.__endpoint = None

    def act(self, obs, test=False):
        """
        Select an action based on observation `obs`

        Args:
            obs (nested structure): observation
            test (bool): directly passed to the `act()` method of the `ActorModule`

        Returns:
            numpy.array: action computed by the `ActorModule`
        """
        
        action = self.actor.act_(obs, test=test)
        return action

    def reset(self, collect_samples):
        """
        Starts a new episode.

        Args:
            collect_samples (bool): if True, samples are buffered and sent to the `Server`

        Returns:
            Tuple:
            (nested structure: observation retrieved from the environment,
            dict: information retrieved from the environment)
        """
        obs = None
        try:
            act = self.env.unwrapped.default_action  
        except AttributeError:
            act = None
        new_obs, info = self.env.reset()
        if self.obs_preprocessor is not None:
            new_obs = self.obs_preprocessor(new_obs)
        rew = 0.0
        terminated, truncated = False, False
        if collect_samples:
            if self.crc_debug:
                self.debug_ts_cpt += 1
                self.debug_ts_res_cpt = 0
                info['crc_sample'] = (obs, act, new_obs, rew, terminated, truncated)
                info['crc_sample_ts'] = (self.debug_ts_cpt, self.debug_ts_res_cpt)
            if self.get_local_buffer_sample:
                sample = self.get_local_buffer_sample(act, new_obs, rew, terminated, truncated, info)
            else:
                sample = act, new_obs, rew, terminated, truncated, info
            self.buffer.append_sample(sample)
        return new_obs, info

    def step(self, obs, test, collect_samples, last_step=False):
        """
        Performs a full RL transition.

        A full RL transition is `obs` -> `act` -> `new_obs`, `rew`, `terminated`, `truncated`, `info`.
        Note that, in the Real-Time RL setting, `act` is appended to a buffer which is part of `new_obs`.
        This is because is does not directly affect the new observation, due to real-time delays.

        Args:
            obs (nested structure): previous observation
            test (bool): passed to the `act()` method of the `ActorModule`
            collect_samples (bool): if True, samples are buffered and sent to the `Server`
            last_step (bool): if True and `terminated` is False, `truncated` will be set to True

        Returns:
            Tuple:
            (nested structure: new observation,
            float: new reward,
            bool: episode termination signal,
            bool: episode truncation signal,
            dict: information dictionary)
        """
        act = self.act(obs, test=test)
        new_obs, rew, terminated, truncated, info = self.env.step(act)
        if self.obs_preprocessor is not None:
            new_obs = self.obs_preprocessor(new_obs)
        if collect_samples:
            if last_step and not terminated:
                truncated = True
            if self.crc_debug:
                self.debug_ts_cpt += 1
                self.debug_ts_res_cpt += 1
                info['crc_sample'] = (obs, act, new_obs, rew, terminated, truncated)
                info['crc_sample_ts'] = (self.debug_ts_cpt, self.debug_ts_res_cpt)
            if self.get_local_buffer_sample:
                sample = self.get_local_buffer_sample(act, new_obs, rew, terminated, truncated, info)
            else:
                sample = act, new_obs, rew, terminated, truncated, info
            self.buffer.append_sample(sample)
        return new_obs, rew, terminated, truncated, info

    def collect_train_episode(self, max_samples=None):
        """
        Collects a maximum of `max_samples` training transitions (from reset to terminated or truncated)

        This method stores the episode and the training return in the local `Buffer` of the worker
        for sending to the `Server`.

        Args:
            max_samples (int): if the environment is not `terminated` after `max_samples` time steps,
                it is forcefully reset and `truncated` is set to True.
        """
        if max_samples is None:
            max_samples = self.max_samples_per_episode

        iterator = range(max_samples) if max_samples != np.inf else itertools.count()

        ret = 0.0
        steps = 0
        obs, info = self.reset(collect_samples=True)
        for i in iterator:
            obs, rew, terminated, truncated, info = self.step(obs=obs, test=False, collect_samples=True, last_step=i == max_samples - 1)
            ret += rew
            steps += 1
            if terminated or truncated:
                break
        self.buffer.stat_train_return = ret
        self.buffer.stat_train_steps = steps

    def run_episodes(self, max_samples_per_episode=None, nb_episodes=np.inf, train=False):
        """
        Runs `nb_episodes` episodes.

        Args:
            max_samples_per_episode (int): same as run_episode
            nb_episodes (int): total number of episodes to collect
            train (bool): same as run_episode
        """
        if max_samples_per_episode is None:
            max_samples_per_episode = self.max_samples_per_episode

        iterator = range(nb_episodes) if nb_episodes != np.inf else itertools.count()

        for _ in iterator:
            self.run_episode(max_samples_per_episode, train=train)

    def run_episode(self, max_samples=None, train=False):
        """
        Collects a maximum of `max_samples` test transitions (from reset to terminated or truncated).

        Args:
            max_samples (int): At most `max_samples` samples are collected per episode.
                If the episode is longer, it is forcefully reset and `truncated` is set to True.
            train (bool): whether the episode is a training or a test episode.
                `step` is called with `test=not train`.
        """
        if max_samples is None:
            max_samples = self.max_samples_per_episode

        iterator = range(max_samples) if max_samples != np.inf else itertools.count()

        ret = 0.0
        steps = 0
        obs, info = self.reset(collect_samples=False)
        for _ in iterator:
            obs, rew, terminated, truncated, info = self.step(obs=obs, test=not train, collect_samples=False)
            ret += rew
            steps += 1
            if terminated or truncated:
                break
        self.buffer.stat_test_return = ret
        self.buffer.stat_test_steps = steps

    def run(self, test_episode_interval=0, nb_episodes=np.inf, verbose=True, expert=False):
        """
        Runs the worker for `nb_episodes` episodes.

        This method sends episodes continuously to the Server, and checks for new weights between episodes.
        For synchronous or more fine-grained sampling, use synchronous or lower-level APIs.
        For deployment, use `run_episodes` rather than `run`.

        Args:
            test_episode_interval (int): a test episode is collected for every `test_episode_interval` train episodes;
                set to 0 to not collect test episodes.
            nb_episodes (int): maximum number of train episodes to collect.
            verbose (bool): whether to log INFO messages.
            expert (bool): experts send training samples without updating their model nor running test episodes.
        """

        iterator = range(nb_episodes) if nb_episodes != np.inf else itertools.count()

        if expert:
            if not verbose:
                for _ in iterator:
                    self.collect_train_episode(self.max_samples_per_episode)
                    self.send_and_clear_buffer()
                    self.ignore_actor_weights()
            else:
                for _ in iterator:
                    print_with_timestamp("collecting expert episode")
                    self.collect_train_episode(self.max_samples_per_episode)
                    print_with_timestamp("copying buffer for sending")
                    self.send_and_clear_buffer()
                    self.ignore_actor_weights()
        elif not verbose:
            if not test_episode_interval:
                for _ in iterator:
                    self.collect_train_episode(self.max_samples_per_episode)
                    self.send_and_clear_buffer()
                    self.update_actor_weights(verbose=False)
            else:
                for episode in iterator:
                    if episode % test_episode_interval == 0 and not self.crc_debug:
                        self.run_episode(self.max_samples_per_episode, train=False)
                    self.collect_train_episode(self.max_samples_per_episode)
                    self.send_and_clear_buffer()
                    self.update_actor_weights(verbose=False)
        else:
            for episode in iterator:
                if test_episode_interval and episode % test_episode_interval == 0 and not self.crc_debug:
                    print_with_timestamp("running test episode")
                    self.run_episode(self.max_samples_per_episode, train=False)
                print_with_timestamp("collecting train episode")
                self.collect_train_episode(self.max_samples_per_episode)
                print_with_timestamp("copying buffer for sending")
                self.send_and_clear_buffer()
                print_with_timestamp("checking for new weights")
                self.update_actor_weights(verbose=True)

    def run_synchronous(self,
                        test_episode_interval=0,
                        nb_steps=np.inf,
                        initial_steps=1,
                        max_steps_per_update=np.inf,
                        end_episodes=True,
                        verbose=False):
        """
        Collects `nb_steps` steps while synchronizing with the Trainer.

        This method is useful for traditional (non-real-time) environments that can be stepped fast.
        It also works for rtgym environments with `wait_on_done` enabled, just set `end_episodes` to `True`.

        Note: This method does not collect test episodes. Periodically use `run_episode(train=False)` if you wish to.

        Args:
            test_episode_interval (int): a test episode is collected for every `test_episode_interval` train episodes;
                set to 0 to not collect test episodes. NB: `end_episodes` must be `True` to collect test episodes.
            nb_steps (int): total number of steps to collect (after `initial_steps`).
            initial_steps (int): initial number of steps to collect before waiting for the first model update.
            max_steps_per_update (float): maximum number of steps to collect per model received from the Server
                (this can be a non-integer ratio).
            end_episodes (bool): when True, waits for episodes to end before sending samples and waiting for updates.
                When False (default), pauses whenever the max_steps_per_update ratio is exceeded.
            verbose (bool): whether to log INFO messages.
        """

        if verbose:
            logging.info(f"Collecting {initial_steps} initial steps")

        iteration = 0
        done = False
        while iteration < initial_steps:
            steps = 0
            ret = 0.0
            obs, info = self.reset(collect_samples=True)
            done = False
            iteration += 1
            while not done and (end_episodes or iteration < initial_steps):
                #step
                obs, rew, terminated, truncated, info = self.step(obs=obs,
                                                                  test=False,
                                                                  collect_samples=True,
                                                                  last_step=steps == self.max_samples_per_episode - 1)
                iteration += 1
                steps += 1
                ret += rew
                done = terminated or truncated
            #send the collected samples to the Server
            self.buffer.stat_train_return = ret
            self.buffer.stat_train_steps = steps
            if verbose:
                logging.info(f"Sending buffer (initial steps)")
            self.send_and_clear_buffer()

        i_model = 1

        #wait for the first updated model if required here
        ratio = (iteration + 1) / i_model
        while ratio > max_steps_per_update:
            if verbose:
                logging.info(f"Ratio {ratio} > {max_steps_per_update}, sending buffer checking updates")
            self.send_and_clear_buffer()
            i_model += self.update_actor_weights(verbose=verbose, blocking=True)
            ratio = (iteration + 1) / i_model

        #collect further samples while synchronizing with the Trainer
        iteration = 0
        episode = 0
        steps = 0
        ret = 0.0

        while iteration < nb_steps:

            if done:
                #test episode
                if test_episode_interval > 0 and episode % test_episode_interval == 0 and end_episodes:
                    if verbose:
                        print_with_timestamp("running test episode")
                    self.run_episode(self.max_samples_per_episode, train=False)
                #reset
                obs, info = self.reset(collect_samples=True)
                done = False
                iteration += 1
                steps = 0
                ret = 0.0
                episode += 1

            while not done and (end_episodes or ratio <= max_steps_per_update):

                #step
                obs, rew, terminated, truncated, info = self.step(obs=obs,
                                                                  test=False,
                                                                  collect_samples=True,
                                                                  last_step=steps == self.max_samples_per_episode - 1)
                iteration += 1
                steps += 1
                ret += rew

                done = terminated or truncated

                if not end_episodes:
                    #check model and send samples after each step
                    ratio = (iteration + 1) / i_model
                    while ratio > max_steps_per_update:
                        if verbose:
                            logging.info(f"Ratio {ratio} > {max_steps_per_update}, sending buffer checking updates (no eoe)")
                        if not done:
                            if verbose:
                                logging.info(f"Sending buffer (no eoe)")
                            self.send_and_clear_buffer()
                        i_model += self.update_actor_weights(verbose=verbose, blocking=True)
                        ratio = (iteration + 1) / i_model

            if end_episodes:
                #check model and send samples only after episodes end
                ratio = (iteration + 1) / i_model
                while ratio > max_steps_per_update:
                    if verbose:
                        logging.info(
                            f"Ratio {ratio} > {max_steps_per_update}, sending buffer checking updates (eoe)")
                    if not done:
                        if verbose:
                            logging.info(f"Sending buffer (eoe)")
                        self.send_and_clear_buffer()
                    i_model += self.update_actor_weights(verbose=verbose, blocking=True)
                    ratio = (iteration + 1) / i_model

            self.buffer.stat_train_return = ret
            self.buffer.stat_train_steps = steps
            if verbose:
                logging.info(f"Sending buffer - DEBUG ratio {ratio} iteration {iteration} i_model {i_model}")
            self.send_and_clear_buffer()

    def run_env_benchmark(self, nb_steps, test=False, verbose=True):
        """
        Benchmarks the environment.

        This method is only compatible with rtgym_ environments.
        Furthermore, the `"benchmark"` option of the rtgym configuration dictionary must be set to `True`.

        .. _rtgym: https://github.com/yannbouteiller/rtgym

        Args:
            nb_steps (int): number of steps to perform to compute the benchmark
            test (int): whether the actor is called in test or train mode
            verbose (bool): whether to log INFO messages
        """
        if nb_steps == np.inf or nb_steps < 0:
            raise RuntimeError(f"Invalid number of steps: {nb_steps}")

        obs, info = self.reset(collect_samples=False)
        for _ in range(nb_steps):
            obs, rew, terminated, truncated, info = self.step(obs=obs, test=test, collect_samples=False)
            if terminated or truncated:
                break
        res = self.env.unwrapped.benchmarks()
        if verbose:
            print_with_timestamp(f"Benchmark results:\n{res}")
        return res

    def send_and_clear_buffer(self):
        """
        Sends the buffered samples to the `Server`.
        """
        self.__endpoint.produce(self.buffer, "trainers")
        self.buffer.clear()

    def update_actor_weights(self, verbose=True, blocking=False):
        """
        Updates the actor with new weights received from the `Server` when available.

        Args:
            verbose (bool): whether to log INFO messages.
            blocking (bool): if True, blocks until a model is received; otherwise, can be a no-op.

        Returns:
            int: number of new actor models received from the Server (the latest is used).
        """
        weights_list = self.__endpoint.receive_all(blocking=blocking)
        nb_received = len(weights_list)
        if nb_received > 0:
            weights = weights_list[-1]
            with open(self.model_path, 'wb') as f:
                f.write(weights)
            if self.model_history:
                self._cur_hist_cpt += 1
                if self._cur_hist_cpt == self.model_history:
                    x = datetime.datetime.now()
                    with open(self.model_path_history + str(x.strftime("%d_%m_%Y_%H_%M_%S")) + ".tmod", 'wb') as f:
                        f.write(weights)
                    self._cur_hist_cpt = 0
                    if verbose:
                        print_with_timestamp("model weights saved in history")
            self.actor = self.actor.load(self.model_path, device=self.device)
            if verbose:
                print_with_timestamp("model weights have been updated")
        return nb_received

    def ignore_actor_weights(self):
        """
        Clears the buffer of weights received from the `Server`.

        This is useful for expert RolloutWorkers, because all RolloutWorkers receive weights.

        Returns:
            int: number of new (ignored) actor models received from the Server.
        """
        weights_list = self.__endpoint.receive_all(blocking=False)
        nb_received = len(weights_list)
        return nb_received
    
    def get_best_episode_array(self, path=r"C:\Users\[Username]\TmrlData\tm_state_log.csv"):
        """
        Ermittelt und gibt die rewardtechnisch beste Runde aus der csv als Array der Eingabewerte zurück

        Args:
            path (String): Pfad der csv-Datei.

        Returns:
            array: Array mit den Eingabewerten.
            float: Gesamtreward der Runde (nicht weiterverwendet).
        """
        def is_true(val):
            return str(val).strip().lower() == "true"
        
        def safe_float(val):
            try:
                if val is None:
                    return 0.0
                s = str(val).strip()
                if s.lower() in ("", "none", "nan", "null"):
                    return 0.0
                return float(s)
            except (ValueError, TypeError):
                return 0.0
        
        if not os.path.exists(path):
            print(f"csv not found")
            return None, 0.0
        
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        rounds = []
        current = []

        for row in rows:
            terminated = is_true(row.get("terminated"))
            truncated = is_true(row.get("truncated"))

            current.append(row)

            if terminated or truncated:
                rounds.append(current)
                current = []

        if current:
            rounds.append(current)

        def calc_total_reward(round_data):
            total = 0.0
            for r in round_data:
                val = safe_float(r.get("reward"))
                total += val
            return total

        scored = [(i, calc_total_reward(r)) for i, r in enumerate(rounds)]
        best_idx, best_reward = max(scored, key=lambda x: x[1])
        best_round = rounds[best_idx]

        best_array = []
        for row in best_round:
            throttle = safe_float(row.get("throttle"))
            brake    = safe_float(row.get("brake"))
            steer    = safe_float(row.get("steer"))
            reward   = safe_float(row.get("reward"))
            best_array.append([throttle, brake, steer, reward])

        return best_array, best_reward


    def replay(self, path=r"C:\Users\[Username]\TmrlData\tm_state_log.csv", loop=True):
        """
        Replay der besten Runde.

        Args:
            path (String): Pfad der verwendeten csv-Datei.
            loop (bool): Erlauben eines Loops des Replays.

        Returns:
            keine Rückgabe.
        """
        best_episode, _ = self.get_best_episode_array(path)

        actions = np.array([x[:3] for x in best_episode], dtype=float)


        print(f"Loaded {len(actions)} frames.")

        try:
            while True:
                obs = self.env.reset()

                for i, a in enumerate(actions):
                    _, _, terminated, truncated, _ = self.env.step(a)

                    if terminated or truncated:
                        break

                if not loop:
                    break
        except KeyboardInterrupt:
            print("Replay aborted by user.")
        finally:
            self.env.close()
            print("Environment closed.")

    def local_search(self, log_path=r"C:\Users\[Username]\TmrlData\local_search_log.csv", log_path_ls=r"C:\Users\[Username]\TmrlData\local_search_best.csv", delta=0.2, mode='tmrl'):
        """
        Ausführung der Lokalen Suche

        Args:
            log_path (String): Pfad der csv-Datei vom Reinforcement Learning
            log_path_ls (String): Pfad der csv-Datei von der Lokalen Suche; speichert während Durchlauf die Runden im gleichen Format wie im RL ab; erlaubt Unterbrechung der Lokalen Suche, da man später aus dieser Datei die bis dahin beste Runde auslesen und von dort fortsetzen kann.

        Returns:
            keine Rückgabe.
        """
        import numpy as np, random as rd, csv, os

        if not os.path.exists(log_path_ls):
            best_episode, _ = self.get_best_episode_array()
        else:
            best_episode, _ = self.get_best_episode_array(log_path_ls)

        actions = np.array([x[:3] for x in best_episode], dtype=float)

        def evaluate(actions, runs=2):
            rewards = []
            pos = None
            for _ in range(runs):
                total = 0.0
                obs = self.env.reset()
                last_pos = [0.0, 0.0, 0.0]
                for a in actions:
                    obs, reward, terminated, truncated, _ = self.env.step(a)
                    total += reward
                    try:
                        last_pos = [float(obs[2]), float(obs[3]), float(obs[4])]
                    except Exception:
                        pass
                    if terminated or truncated:
                        break    
                rewards.append(total)
                pos = obs
            return sum(rewards) / len(rewards), last_pos
        
        self.env.log_path_override = r"C:\Users\[Username]\TmrlData\local_search_best.csv"

        best_reward, best_pos = evaluate(actions)
        print(f"Initial reward: {best_reward}")
        frames = len(actions)

        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame_index",
                "parameter",
                "old_value",
                "new_value",
                "direction",
                "old_reward",
                "new_reward",
                "old_pos_x", "old_pos_y", "old_pos_z",
                "new_pos_x", "new_pos_y", "new_pos_z"
            ])

        def log_change(frame, param, old_val, new_val, direction, old_rew, new_rew, old_x, old_y, old_z, new_x, new_y, new_z):
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    int(frame), param, float(old_val), float(new_val), direction, 
                    float(old_rew), float(new_rew), float(old_x), float(old_y), float(old_z), 
                    float(new_x), float(new_y), float(new_z)])
                print(f"old_val = {type(old_val)} - value = {old_val} \n"
                      f"old_rew = {type(old_rew)} - value = {old_rew}"
                      f"old_x = {type(old_x)} - value = {old_x}"
                      f"old_y = {type(old_y)} - value = {old_y}"
                      f"old_z = {type(old_z)} - value = {old_z}"
                      )
        

        block_size = 256
        init_block_size = block_size
        base_delta = delta

        while block_size >= 1:
            new_delta = base_delta * (block_size / init_block_size)
            print(f"Blocksize = {block_size}")

            for start_frame in range(0, frames, block_size):
                end_frame = min(start_frame + block_size, frames)
                local_improved = True
                improved = False

                while local_improved:
                    local_improved = False

                    #throttle
                    for sign, label in [(1, "+")]:
                        candidate = actions.copy()
                        old_val = candidate[start_frame, 0]
                        _, old_pos = best_reward, best_pos
                        candidate[start_frame:end_frame, 0] = np.clip(old_val + sign * new_delta, 0, 1)
                        total, pos = evaluate(candidate)
                        log_change(start_frame, "throttle", old_val, candidate[start_frame, 0], label, best_reward, total, old_pos[0], old_pos[1], old_pos[2], pos[0], pos[1], pos[2])
                        
                        if total > best_reward:
                            best_reward, best_pos = total, pos
                            print(f"New best reward: {best_reward}")
                            actions = candidate.copy()
                            local_improved = True
                            improved = True
                            break

                    #brake
                    for sign, label in [(-1, "-")]:
                        candidate = actions.copy()
                        old_val = candidate[start_frame, 1]
                        _, old_pos = best_reward, best_pos
                        candidate[start_frame:end_frame, 1] = np.clip(old_val + sign * new_delta, 0, 1)
                        total, pos = evaluate(candidate)
                        log_change(start_frame, "brake", old_val, candidate[start_frame, 1], label, best_reward, total, old_pos[0], old_pos[1], old_pos[2], pos[0], pos[1], pos[2])
                        if total > best_reward:
                            best_reward, best_pos = total, pos
                            print(f"New best reward: {best_reward}")
                            actions = candidate.copy()
                            local_improved = True
                            improved = True
                            break

                    #steering
                    for sign, label in [(1, "+"), (-1, "-")]:
                        candidate = actions.copy()
                        old_val = candidate[start_frame, 2]
                        _, old_pos = best_reward, best_pos
                        candidate[start_frame:end_frame, 2] = np.clip(old_val + sign * new_delta * 0.25, -1, 1)
                        total, pos = evaluate(candidate)
                        log_change(start_frame, "steering", old_val, candidate[start_frame, 2], label, best_reward, total, old_pos[0], old_pos[1], old_pos[2], pos[0], pos[1], pos[2])
                        if total > best_reward:
                            best_reward, best_pos = total, pos
                            print(f"New best reward: {best_reward}")
                            actions = candidate.copy()
                            local_improved = True
                            improved = True
                            break
            
            if not improved:
                block_size //= 2

        if hasattr(self.env, "log_path_override"):
            del self.env.log_path_override
        if hasattr(self, "env"):
                self.env.close()
                del self.env
        print(f"Maximum reached: {best_reward}")
        print("Ending Hill Climbing...")
