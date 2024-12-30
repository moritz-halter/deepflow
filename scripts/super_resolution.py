import os.path
import sys

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from pathlib import Path
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from losses import LpLoss
from data.datasets.aneurisk import load_aneurisk_dataset
from utils import get_wandb_api_key, count_model_params, get_project_root
from training import setup, Trainer
from models.base_model import get_model

os.environ["WANDB__SERVICE_WAIT"] = "300"
# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig("./aneurisk_config.yaml", config_name="default", config_folder="../config"),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None)
    ]
)
config = pipe.read_conf()

# Set-up distributed communication, if using
device, is_logger = setup(config)

# Set up WandB logging
wandb_init_args = None
if config.wandb.log and is_logger:
    print(config.wandb.log)
    print(config)
    wandb.login(key=get_wandb_api_key()) # use api_key_file parameter to set custom location for key file
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = "_".join(
            f"{var}"
            for var in [
                config_name,
                config.fno.n_layers,
                config.fno.n_modes,
                config.fno.hidden_channels,
                config.fno.factorization,
                config.fno.rank,
            ]
        )
    wandb_init_args = dict(
        config=config,
        name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )
    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]
    wandb.init(**wandb_init_args)

# Make sure we only print information when needed
config.verbose = config.verbose and is_logger

# Print config to screen
if config.verbose:
    pipe.log()
    sys.stdout.flush()

data_dir = get_project_root().joinpath(config.data.folder)

arch_config = getattr(config, config.arch)
try:
    domain_flag = getattr(arch_config, 'domain_agnostic', False)
except KeyError:
    domain_flag = False

train_loader, test_loaders, data_processor, t_out = load_aneurisk_dataset(
    data_root=data_dir,
    train_resolution=config.data.train_resolution,
    train_test_split=config.data.train_test_split,
    batch_size=config.data.batch_size,
    geometric_prior_mode=config.data.geometric_prior_mode,
    geometric_prior=config.data.geometric_prior,
    test_resolutions=config.data.test_resolutions,
    test_batch_sizes=config.data.test_batch_sizes,
    domain_flag=domain_flag,
    n_t=config.data.n_t,
    super_resolution_rate=config.data.super_resolution_rate,
    super_resolution_dim=config.data.super_resolution_dim
)

arch_config.n_modes = [config.fno4d.n_modes] * 3 + [1] if config.data.super_resolution_dim == 'space' else [config.fno4d.n_modes] * 4
geo_channels = 32 if config.data.geometric_prior == 'eig' else 1
arch_config.geo_channels = geo_channels
arch_config.geo_feature = config.data.geometric_prior_mode in [2, 3]
arch_config.geo_embedding = config.data.geometric_prior_mode == 2
if config.data.geometric_prior_mode == 1:
    arch_config.data_channels = arch_config.data_channels + geo_channels
arch_config.super_resolution_dim = config.data.super_resolution_dim
arch_config.super_resolution_rate = config.data.super_resolution_rate
model = get_model(config)
model = model.to(device)

data_processor = data_processor.to(device)

if len(list(model.parameters())) > 0:
    # Create the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.opt.learning_rate,
        weight_decay=config.opt.weight_decay,
    )

    if config.opt.scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.opt.gamma,
            patience=config.opt.scheduler_patience,
            mode="min",
        )
    elif config.opt.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.opt.scheduler_T_max
        )
    elif config.opt.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
        )
    else:
        raise ValueError(f"Got scheduler={config.opt.scheduler}")
else:
    optimizer = None
    scheduler = None

# Creating the losses
l2loss = LpLoss(d=4, p=2, reduce_dims=[0, 1], reductions='mean')
l2loss_per_channel = LpLoss(d=4, p=2, reduce_dims=0, reductions='mean')
if config.opt.training_loss == "l2":
    train_loss = l2loss
else:
    raise ValueError(
        f'Got training_loss={config.opt.training_loss} '
        f'but expected one of ["l2"]'
    )
eval_losses = {"l2": l2loss, "l2_per_channel": l2loss_per_channel}

if config.verbose:
    print("\n### MODEL ###\n", model)
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULER ###\n", scheduler)
    print("\n### LOSSES ###")
    print(f"\n * Train: {train_loss}")
    print(f"\n * Test: {eval_losses}")
    print(f"\n### Beginning Training...\n")
    sys.stdout.flush()


trainer = Trainer(
    model=model,
    n_epochs=config.opt.n_epochs,
    data_processor=data_processor,
    device=device,
    eval_interval=config.wandb.eval_interval,
    log_output=config.wandb.log_output,
    verbose=config.verbose,
    wandb_log=config.wandb.log
)

# Log parameter count
if is_logger:
    n_params = count_model_params(model)

    if config.verbose:
        print(f"\nn_params: {n_params}")
        sys.stdout.flush()

    if config.wandb.log:
        to_log = {"n_params": n_params}
        if config.n_params_baseline is not None:
            to_log["n_params_baseline"] = (config.n_params_baseline,)
            to_log["compression_ratio"] = (config.n_params_baseline / n_params,)
            to_log["space_savings"] = 1 - (n_params / config.n_params_baseline)
        wandb.log(to_log)
        wandb.watch(model)

save_dir = Path('./ckpt') if config.save_dir is None else Path(config.save_dir)
assert save_dir.is_dir(), "%s is not a directory" % save_dir.as_posix()
if config.wandb.name is not None:
    save_dir = save_dir.joinpath(config.wandb.name)
    if not save_dir.exists():
        save_dir.mkdir()
trainer.train(
    train_loader,
    test_loaders,
    optimizer,
    scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
    save_best=None,
    save_every=10,
    save_dir=save_dir
)

if config.wandb.log and is_logger:
    wandb.finish()

if config.verbose and is_logger and torch.cuda.is_available():
    print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
    print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
    print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))
