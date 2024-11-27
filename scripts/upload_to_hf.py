from huggingface_hub import HfApi
api = HfApi()

# repo_name ="sensharma/sae-disent_llama-1B_layer-8_wiki-2M"
repo_name = "sensharma/sae_llama-1B_layer-8_wiki-2M"

from huggingface_hub import create_repo
create_repo(repo_name, private=False, repo_type="model", exist_ok=True)

api.upload_folder(
    # folder_path="/share/u/arnab/Codes/Projects/test_sae/results/train_disent_sae/Llama-3.2-1B/wikipedia/2000000/trainer_0",
    folder_path="/share/u/arnab/Codes/Projects/test_sae/results/train_sae/Llama-3.2-1B/wikipedia/2000000/trainer_0",
    path_in_repo="",
    repo_id=repo_name,
    repo_type="model",
)