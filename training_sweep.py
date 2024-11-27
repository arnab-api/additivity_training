# Imports
import torch as t
from nnsight import LanguageModel
import argparse
import itertools
import gc
import os
from datetime import datetime

from dictionary_learning.training import trainSAE
from dictionary_learning.trainers.standard import StandardTrainer
from dictionary_learning.trainers.top_k import TrainerTopK, AutoEncoderTopK
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.buffer import ActivationBuffer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True, help="where to store sweep")
    parser.add_argument("--no_wandb_logging", action="store_true", help="omit wandb logging")
    parser.add_argument("--dry_run", action="store_true", help="dry run sweep")
    parser.add_argument("--num_tokens", type=int, required=True, help="total number of training tokens")
    parser.add_argument(
        "--layers", type=int, nargs="+", required=True, help="layers to train SAE on"
    )
    parser.add_argument(
        "--width_exponents", type=int, nargs="+", required=True, help="power of 2 for total number of SAE latents"
    )
    parser.add_argument("--device", type=str, help="device to train on")
    args = parser.parse_args()
    return args


def run_sae_training(
    layer: int,
    width_exponents: list[int],
    num_tokens: int,
    save_dir: str,
    device: str,
    dry_run: bool = False,
    no_wandb_logging: bool = False,
):
    # model parameters

    model_name = "EleutherAI/pythia-70m-deduped"
    model = LanguageModel(
        model_name,
        device_map=device,
    )

    submodule = model.gpt_neox.layers[layer]
    submodule_name = f"resid_post_layer_{layer}"
    io = "out"
    activation_dim = model.config.hidden_size

    # model_name = "google/gemma-2-2b"
    # model = LanguageModel(
    #     model_name,
    #     device_map=device,
    #     low_cpu_mem_usage=True,
    #     attn_implementation="eager",
    #     torch_dtype=t.bfloat16,
    # )

    # submodule = model.model.layers[layer]
    # submodule_name = f"resid_post_layer_{layer}"
    # io = "out"
    # activation_dim = model.config.hidden_size

    # data processing parameters
    context_length = 128
    buffer_size = int(2048)
    llm_batch_size = 32  # 32 on a 24GB RTX 3090
    sae_batch_size = 2048  # 2048 on a 24GB RTX 3090

    # sae training parameters
    random_seeds = [0]
    ks = [20, 40, 80, 160, 320, 640]
    dict_sizes = [int(2**i) for i in width_exponents]

    steps = int(num_tokens / sae_batch_size)  # Total number of batches to train

    # topk sae training parameters
    decay_start = 24000
    auxk_alpha = 1 / 32

    # saving intermediate checkpoints
    desired_checkpoints = t.logspace(-3, 0, 7).tolist()
    desired_checkpoints = [0.0] + desired_checkpoints[:-1]
    desired_checkpoints.sort()
    save_steps = [int(steps * step) for step in desired_checkpoints]
    save_steps.sort()

    # wandb logging parameters
    wandb_entity = "canrager"
    wandb_project = "checkpoint_sae_sweep"
    log_steps = 100  # Log the training on wandb
    if no_wandb_logging:
        log_steps = None

    # Initializing the activation buffer (training data for SAE)
    generator = hf_dataset_to_generator("monology/pile-uncopyrighted")

    activation_buffer = ActivationBuffer(
        generator,
        model,
        submodule,
        n_ctxs=buffer_size,
        ctx_len=context_length,
        refresh_batch_size=llm_batch_size,
        out_batch_size=sae_batch_size,
        io=io,
        d_submodule=activation_dim,
        device=device,
    )

    # create the list of configs
    trainer_configs = []
    for seed, k, dict_size in itertools.product(
        random_seeds, ks, dict_sizes
    ):
        trainer_configs.append([{
                "trainer": TrainerTopK,
                "dict_class": AutoEncoderTopK,
                "activation_dim": activation_dim,
                "dict_size": dict_size,
                "k": k,
                "auxk_alpha": auxk_alpha,  # see Appendix A.2
                "decay_start": decay_start,  # when does the lr decay start
                "steps": steps,  # when when does training end
                "seed": seed,
                "wandb_name": f"TopKTrainer-{model_name}-{submodule_name}",
                "device": device,
                "layer": layer,
                "lm_name": model_name,
                "submodule_name": submodule_name,
            },
            { # This will be our trainer TopK with additivity loss.
                "trainer": TrainerTopK, # change to our new trainer implementation
                "dict_class": AutoEncoderTopK,
                "activation_dim": activation_dim,
                "dict_size": dict_size,
                "k": k,
                "auxk_alpha": auxk_alpha,  # see Appendix A.2
                "decay_start": decay_start,  # when does the lr decay start
                "steps": steps,  # when when does training end
                "seed": seed,
                "wandb_name": f"TopKTrainer-{model_name}-{submodule_name}",
                "device": device,
                "layer": layer,
                "lm_name": model_name,
                "submodule_name": submodule_name,
            }
        ])

    # Determine output filename, log to stdout
    mmdd = datetime.now().strftime('%m%d')
    model_id = model_name.split('/')[1]
    width_str = "_".join([str(i) for i in width_exponents])
    save_name = f"{model_id}_layer-{layer}_width-2pow{width_str}_date-{mmdd}"
    save_dir = os.path.join(save_dir, save_name)

    print(f"save_dir: {save_dir}")
    print(f"desired_checkpoints: {desired_checkpoints}")
    print(f"save_steps: {save_steps}")
    print(f"num_tokens: {num_tokens}")
    print(f"len trainer configs: {len(trainer_configs)}")
    print(f"trainer_configs: {trainer_configs}")
   

    if not dry_run:
        # actually run the sweep
        trainSAE(
            data=activation_buffer,
            trainer_configs=trainer_configs,
            steps=steps,
            save_steps=save_steps,
            save_dir=save_dir,
            log_steps=log_steps,
            use_wandb=not no_wandb_logging,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
        )


if __name__ == "__main__":
    args = get_args()
    for layer in args.layers:
        run_sae_training(
            layer=layer,
            save_dir=args.save_dir,
            num_tokens=args.num_tokens,
            width_exponents=args.width_exponents,
            device=args.device,
            dry_run=args.dry_run,
            no_wandb_logging=args.no_wandb_logging,
        )
