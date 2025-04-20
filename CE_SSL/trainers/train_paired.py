from CE_SSL.utils.misc import (
    LARS,
    MomentumPairedUpdate,
    UpdateTau,
    LogLR,
    ShufflePairs
)
from CE_SSL.losses.loss_mmcr_momentum_paired import MMCR_Momentum_Loss as MMCR_Momentum_Loss_Paired
from CE_SSL.losses.loss_barlow_paired import Barlow_Loss as Barlow_Loss_Paired
from CE_SSL.losses.loss_simclr_paired import SimCLR_Loss as SimCLR_Loss_Paired
from CE_SSL.data.datasets_paired import get_datasets
from CE_SSL.utils.online_eval import OnlineEval
from CE_SSL.models.models_paired import Model, ComposerWrapper
from CE_SSL.models.models_momentum_paired import MomentumModel, MomentumComposerWrapper

import torch
import composer
from composer.optim.scheduler import CosineAnnealingWithWarmupScheduler
import submitit

import os


def train(gpu, args, **kwargs):
    # composer doesn't require init_dist_gpu() function call
    job_env = submitit.JobEnvironment()
    args.gpu = job_env.local_rank
    args.rank = job_env.global_rank

    # better port
    tmp_port = os.environ["SLURM_JOB_ID"]
    tmp_port = int(tmp_port[-4:]) + 50000
    args.port = tmp_port

    os.environ["RANK"] = str(job_env.global_rank)
    os.environ["WORLD_SIZE"] = str(args.n_gpus * args.n_nodes)
    os.environ["LOCAL_RANK"] = str(job_env.local_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(args.n_gpus)
    os.environ["NODE_RANK"] = str(int(os.getenv("SLURM_NODEID")))
    os.environ["MASTER_ADDR"] = args.host_name_
    os.environ["MASTER_PORT"] = str(args.port)
    os.environ["PYTHONUNBUFFERED"] = "1"

    args.torch_cuda_device_count = torch.cuda.device_count()
    args.slurm_nodeid = int(os.getenv("SLURM_NODEID"))
    args.slurm_nnodes = int(os.getenv("SLURM_NNODES"))

    print(args)

    # datasets
    train_data, memory_data, test_data = get_datasets(dataset=args.dataset, use_zip=args.use_zip, weak_transform=args.weak_transform)

    # samplers
    train_sampler = torch.utils.data.DistributedSampler(
        train_data,
        num_replicas=args.world_size,
        rank=args.rank,
    )
    memory_sampler = torch.utils.data.DistributedSampler(
        memory_data,
        num_replicas=args.world_size,
        rank=args.rank,
    )
    test_sampler = torch.utils.data.DistributedSampler(
        test_data,
        num_replicas=args.world_size,
        rank=args.rank,
    )

    # dataloaders
    batch_size = int(args.batch_size / args.n_gpus / args.n_nodes)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
    )

    memory_loader = torch.utils.data.DataLoader(
        dataset=memory_data,
        batch_size=512,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=memory_sampler,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=128,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=test_sampler,
    )

    # objective/model
    args.distributed = args.n_gpus * args.n_nodes > 1
    projector_dims = [8192, 8192, 512]
    if args.objective == "MMCR":
        if args.dataset == "imagenet_100":
            projector_dims = [4096, 512]
        objective = MMCR_Momentum_Loss_Paired(equi_lmbda=args.lmbda, lmbda=0.0, distributed=args.distributed)
        model = MomentumModel(projector_dims_1=projector_dims, projector_dims_2=projector_dims, resnet_size=args.resnet_size)
    elif args.objective == "Barlow":
        projector_dims = [8192, 8192, 8192]
        objective = Barlow_Loss_Paired(equi_lmbda=args.lmbda, lmbda=5e-3, distributed=args.distributed, out_dim=8192)
        model = Model(projector_dims_1=projector_dims, projector_dims_2=projector_dims, resnet_size=args.resnet_size)
    elif args.objective == "SimCLR":
        objective = SimCLR_Loss_Paired(equi_lmbda=args.lmbda, lmbda=0.15, distributed=args.distributed)
        model = Model(projector_dims_1=projector_dims, projector_dims_2=projector_dims, bias_proj=False)
    elif args.objective == "BYOL":
        objective = BYOL_Loss_Paired(equi_lmbda=args.lmbda)
        model = BYOLModel()
    

    if "MMCR" in args.objective:
        objective = torch.nn.SyncBatchNorm.convert_sync_batchnorm(objective)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        wrapped_model = MomentumComposerWrapper(module=model, objective=objective) 
    else:
        objective = torch.nn.SyncBatchNorm.convert_sync_batchnorm(objective)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        wrapped_model = ComposerWrapper(module=model, objective=objective)


    # optimizer
    lr = args.lr * args.batch_size / 256
    optimizer = LARS(
        model.parameters(),
        lr=lr,
        weight_decay=1e-6,
        momentum=0.9,
        weight_decay_filter=True,
        lars_adaptation_filter=True,
    )

    # scheduler
    scheduler = CosineAnnealingWithWarmupScheduler(t_warmup="10ep", alpha_f=0.001)

    # callbacks
    callback_list = [ShufflePairs(), LogLR(), OnlineEval(test_loader)]
    if args.objective == "MMCR":
        callback_list.append(MomentumPairedUpdate(tau=args.tau))

    print(model)

    # trainer
    trainer = composer.Trainer(
        train_dataloader=train_loader,
        optimizers=optimizer,
        model=wrapped_model,
        max_duration=args.epochs,
        precision="amp",
        algorithms=[
            composer.algorithms.ChannelsLast(),
        ],
        device="gpu",
        callbacks=callback_list,
        schedulers=(scheduler),
        save_interval=args.save_freq,
        save_overwrite=True,
        save_folder=args.save_folder,
    )

    trainer.fit()

