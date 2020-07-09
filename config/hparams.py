from collections import defaultdict
"""
All configurations are set following 
"End-to-End Abstractive Summarization for Meetings" paper.
https://arxiv.org/abs/2004.02016
"""

PARAMS = defaultdict(
    # Environment
    device='cuda',
    # device='cpu',
    workers=24,
    gpu_ids=[0],
    data_dir='data/',
    save_dirpath='checkpoints/with_role_gen_500_fsize_64_pos/',
    use_role=False,
    use_pos=False,
    load_pthpath="",
    vocab_word_path='checkpoints/vocab_word',
    # Training Hyperparemter
    batch_size=1,
    num_epochs=100,
    start_eval_epoch=40,
    fintune_word_embedding=True,
    # Transformer
    embedding_size_word=300,
    embedding_size_role=20,
    embedding_size_pos=12,
    num_heads=2,
    num_hidden_layers=2,
    hidden_size=300,
    min_length=280,
    max_length=800,
    gen_max_length=500,
    attention_key_channels=0,
    attention_value_channels=0,
    filter_size=64,
    # filter_size=32,
    dropout=0.2,
    optimizer_adam_beta1=0.9,
    optimizer_adam_beta2=0.999,
    # Optimizier
    learning_rate=5e-4,
    max_gradient_norm=2,
    # Decoding
    beam_size=12,
    blook_trigram=True
)