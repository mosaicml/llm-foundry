from mcli import RunConfig
from ygong.mosaic.scaling_config import ScalingConfig
from typing import Dict, List, Optional
import os
import shlex
# import databricks_genai.api.config as cfg

class WSFSIntegration:
    def __init__(
            self, 
            wsfs_path: str, 
            entry_point: Optional[str] = None,
            args: Optional[List[str]] = None):
        """
        Class to represent the integration with Databricks WSFS.

        :params: wsfs_path: str Absolute path 
        :params: entry_point: str Required if the wsfs_path is a directory
        """
        self.wsfs_path = wsfs_path
        self.entry_point = entry_point
        self.args = args

    def get_entry_command(self):
        entry_file_path = ""
        if self.entry_point is not None:
            if self.entry_point.startswith("/Workspace"):
                entry_file_path = self.entry_point
            else:
                entry_file_path = os.path.join(self.wsfs_path, self.entry_point)
        else:
            entry_file_path = self.wsfs_path
        if self.args is None:
            return f"python3 {shlex.quote(entry_file_path)}"
        return f"python3 {shlex.quote(entry_file_path)} {' '.join(self.args)}"

    def toDict(self):
        return {
            "integration_type": "wsfs",
            "wsfs_path": self.wsfs_path,
            "entrypoint": self.entry_point,
            "args": self.args,
        }


class MPT125MConfig:
    def __init__(
            self,
            experimentName: str,
            data: str, 
            priority: str = 'high',
            preemptible: bool = False, 
            retry_on_system_failure: bool = False,
            wsfs_integration: Optional[WSFSIntegration] = None):
        # TODO: validate the inputs and remove the yu.gong hardcode
        self.mlflow_experimentName = f"/Users/yu.gong@databricks.com/{experimentName}"
        self.mlflow_trackingUri = "databricks"
        # self.mlflow_trackingUri = "databricks" if workspace_url is None else workspace_url

        self.data = data

        # Scheudling parameters
        self.priority = priority
        self.preemptible = preemptible
        self.retry_on_system_failure = retry_on_system_failure
        
        # the run name is pre-configured for all config-driven pretrain runs
        self.name = "mpt125m-config-driven-pretrain"

        ########################################
        # model parameters
        ########################################
        self.max_seq_len = 2048
        self.global_seed = 17
        self.data_remote = self.data
        self.data_local = ""
        self.commands = []
        if wsfs_integration is not None:
            # The first group of commands are to download the object(file or directory) from
            # databricks WSFS using PAT token and url.
            # The second command try to unzip if the object from WSFS is directory.
            # TODO: Read the token and host name from env vars or /mnt/jwt-secret/.databrickscfg
            self.commands = [
                f"""
                DATABRICKS_HOST="https://oregon.staging.cloud.databricks.com"
                DATABRICKS_TOKEN="dapid5af61ff89674be90c3e86ae9fc10c2e"
                WSFS_PATH="{wsfs_integration.wsfs_path}"
                DIR_NAME=$(dirname "$WSFS_PATH")
                ENCODED_WSFS_PATH=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$WSFS_PATH'))")
                mkdir -p "$DIR_NAME"
                curl -X GET -o "$WSFS_PATH" "${{DATABRICKS_HOST}}/api/2.0/workspace/export?path=${{ENCODED_WSFS_PATH}}&direct_download=true" \
                -H "Authorization: Bearer $DATABRICKS_TOKEN" 

                if file "$WSFS_PATH" | grep -q "Zip archive data"; then
                    mv "$WSFS_PATH" "${{WSFS_PATH}}.zip"
                    apt update && apt install unzip
                    unzip -d "$DIR_NAME" "${{WSFS_PATH}}.zip"
                    rm -f "${{WSFS_PATH}}.zip"
                else
                    echo "$WSFS_PATH is not a ZIP file."
                fi
                """
            ]
            self.commands.append(wsfs_integration.get_entry_command())
        else:
            self.commands = [
                "cd llm-foundry/scripts",
                "train/launcher.py train/train.py /mnt/config/parameters.yaml train_loader.dataset.split=train eval_loader.dataset.split=val"
            ]
    
       
    def toRunConfig(self, scalingConfig: ScalingConfig):
        return RunConfig(
            name=self.name,
            image='mosaicml/llm-foundry:2.2.1_cu121_flash2-latest',
            command="\n".join(self.commands),
            compute=scalingConfig.toCompute,
            scheduling={
                'priority': self.priority,
                'preemptible': self.preemptible,
                'retry_on_system_failure': self.retry_on_system_failure
            },
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
                   'packages': ['pynvml', 'mosaicml-streaming[databricks]'],
                },
            ],
            parameters=self.parameters(),
            env_variables={},
        )
    
    def parameters(self):
       return {
           "data_local": self.data_local,
           "data_remote": self.data,
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