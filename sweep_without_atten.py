import wandb

# Base data directory and language selection
dataset_path = "/speech/nishanth/dl_assig/dakshina_dataset_v1.0"
lang = "hindi"

# Sweep configuration for no-attention experiments
no_attn_sweep = {
    "method": "bayes",
    "project": "DL_Assignm__03",
    "entity": "ee22s084-indian-institute-of-technology-madras",
    "name": f"{lang}without_attention",
    "metric": {
        "name": "val_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "embedding_size": {"values": [256, 128]},
        "encoder_num_layers": {"values": [2, 3]},
        "decoder_num_layers": {"values": [1, 2]},
        "hidden_size": {"values": [128, 256]},
        "encoder_name": {"values": ["RNN", "GRU"]},
        "decoder_name": {"values": ["RNN", "GRU"]},
        "dropout_p": {"values": [0.2, 0.3]},
        "learning_rate": {"values": [0.001, 0.003]},
        "teacher_forcing_p": {"values": [0.5, 0.8]},
        "apply_attention": {"values": [False]},
        "encoder_bidirectional": {"values": [True]},
        "beam_size": {"values": [1]}
    }
}

# Create sweep on W&B
sweep_identifier = wandb.sweep(sweep=no_attn_sweep, project=no_attn_sweep["project"])
print(f"Initialized sweep: {sweep_identifier}")

# Function to run each individual sweep configuration
def run_no_attn_trial():
    from trainer import TransliterationTrainer, Config

    wandb.init()
    cfg = wandb.config

    wandb.run.name = (
        f"beam_{cfg.beam_size}_lr_{cfg.learning_rate}_hdsz_{cfg.hidden_size}_"
        f"emb_{cfg.embedding_size}_enc_layers_{cfg.encoder_num_layers}_"
        f"dec_layers_{cfg.decoder_num_layers}_enc_{cfg.encoder_name}_"
        f"dec_{cfg.decoder_name}_bidir_{cfg.encoder_bidirectional}_"
        f"drop_{cfg.dropout_p}_tf_{cfg.teacher_forcing_p}_attn_{cfg.apply_attention}"
    )

    run_config = Config(
        language=lang,
        data_root=dataset_path,
        emb_dim=cfg.embedding_size,
        hid_dim=cfg.hidden_size,
        enc_layers=cfg.encoder_num_layers,
        dec_layers=cfg.decoder_num_layers,
        enc_type=cfg.encoder_name,
        dec_type=cfg.decoder_name,
        dropout=cfg.dropout_p,
        use_attention=cfg.apply_attention,
        tf_prob=cfg.teacher_forcing_p,
        use_bidir=cfg.encoder_bidirectional,
        beam_width=cfg.beam_size,
        lr=cfg.learning_rate,
        batch_sz=256,
        n_workers=16,
        wd=0.0005,
        sos_idx=0,
        enable_wandb=True,
    )

    trainer = TransliterationTrainer(run_config)
    trainer.run_training()
    # trainer.evaluate()
    wandb.finish()

# Launch sweep agent
wandb.agent(sweep_identifier, function=run_no_attn_trial, count=50)
