from mcli import RunConfig
from ygong.mosaic.scaling_config import ScalingConfig
from typing import Optional
import os
# import databricks_genai.api.config as cfg


class MPT125MConfig:
    def __init__(self, experimentName: str, data: str):
        # TODO: validate the inputs and remove the yu.gong hardcode
        self.mlflow_experimentName = f"/Users/yu.gong@databricks.com/{experimentName}"
        workspace_url = os.environ.get('WORKSPACE_URL')
        self.mlflow_trackingUri = "databricks"
        # self.mlflow_trackingUri = "databricks" if workspace_url is None else workspace_url

        self.data = data
        
        # the run name is pre-configured for all config-driven pretrain runs
        self.name = "mpt125m-config-driven-pretrain"

        ########################################
        # model parameters
        ########################################
        self.max_seq_len = 2048
        self.global_seed = 17
        # TODO: hardcode, need to respect self.data
        self.data_remote = "s3://aaron-mlflow-demo/ygong-c4-process/"
        self.data_local = "./my-copy-c4"
        self.commands = [
            "cd llm-foundry/scripts",
            "train/launcher.py train/train.py /mnt/config/parameters.yaml train_loader.dataset.split=train_small eval_loader.dataset.split=val"
        ]
        # api_token, endpoint = cfg.get_config_from_env()
        # self.workspace_url = "https://dbc-04ac0685-8857.staging.cloud.databricks.com/"

       
    def toRunConfig(self, scalingConfig: ScalingConfig):
        return RunConfig(
            name=self.name,
            image='mosaicml/llm-foundry:2.2.1_cu121_flash2-latest',
            command="\n".join(self.commands),
            compute=scalingConfig.toCompute,
            scheduling={},
            integrations=[
                {
                   'integration_type': 'git_repo',
                   'git_repo': 'ygong1/llm-foundry',
                   'git_branch': 'prototype',
                   'pip_install': '-e .[gpu]',
                   'ssh_clone': False
                },
                {
                   'integration_type': 'pip_packages',
                   'packages': ['pynvml'],
                },
            ],
            parameters=self.parameters(),
            env_variables={},
        )
    
    def parameters(self):
       return {
           "data_local": "./my-copy-c4",
           "data_remote": "s3://aaron-mlflow-demo/ygong-c4-process/",
           "max_seq_len": self.max_seq_len,
           "global_seed": self.global_seed,
           "run_name": None,
            "model": {
                "name": "mpt_causal_lm",
                "init_device": "meta",
                "d_model": 768,
                "n_heads": 12,
                "n_layers": 12,
                "expansion_ratio": 4,
                "max_seq_len": self.max_seq_len,
                "vocab_size": 50368,
                "attn_config": {
                    "attn_impl": "triton"
                }
            },
            "tokenizer": {
                "name": "EleutherAI/gpt-neox-20b",
                "kwargs": {
                    "model_max_length": self.max_seq_len
                }
            },
            "train_loader": {
                "name": "text",
                "dataset": {
                    "local": f"{self.data_local}",
                    "remote": f"{self.data_remote}",
                    "split": "train",
                    "shuffle": True,
                    "max_seq_len": self.max_seq_len,
                    "shuffle_seed": self.global_seed
                },
                "drop_last": True,
                "num_workers": 8
            },
            "eval_loader": {
                "name": "text",
                "dataset": {
                    "local": f"{self.data_local}",
                    "remote": f"{self.data_remote}",
                    "split": "val",
                    "shuffle": False,
                    "max_seq_len": self.max_seq_len,
                    "shuffle_seed": self.global_seed
                },
                "drop_last": False,
                "num_workers": 8
            },
            "scheduler": {
                "name": "cosine_with_warmup",
                "t_warmup": "100ba",
                "alpha_f": 0.1
            },
            "optimizer": {
                "name": "decoupled_adamw",
                "lr": 6.0e-4,
                "betas": [0.9, 0.95],
                "eps": 1.0e-08,
                "weight_decay": 0.0
            },
            "algorithms": {
                "gradient_clipping": {
                    "clipping_type": "norm",
                    "clipping_threshold": 1.0
                }
            },
            "max_duration": "480ba",  # ~ 2.5B tokens, original
            "eval_interval": "50ba",  # original 500
            "eval_first": False,
            "eval_subset_num_batches": -1,
            "global_train_batch_size": 256,
            "seed": self.global_seed,
            "device_eval_batch_size": 16,
            "device_train_microbatch_size": 16,
            "precision": "amp_bf16",
            "fsdp_config": {
                "sharding_strategy": "FULL_SHARD",
                "mixed_precision": "PURE",
                "activation_checkpointing": False,
                "activation_checkpointing_reentrant": False,
                "activation_cpu_offload": False,
                "limit_all_gathers": True
            },
            "progress_bar": False,
            "log_to_console": True,
            "console_log_interval": "10ba",
            "callbacks": {
                "speed_monitor": {
                    "window_size": 10
                },
                "lr_monitor": {},
                "memory_monitor": {},
                "runtime_estimator": {}
            },
            "loggers": {
                "mlflow": {
                    "experiment_name": self.mlflow_experimentName,
                    "tracking_uri": "databricks",
                    "synchronous": False,
                    "log_system_metrics": True
                }
            }
       }