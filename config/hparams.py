from collections import defaultdict
"""
All configurations are set following "End-to-End Abstractive Summarization for Meetings" paper.
"""

PARAMS = defaultdict(
    # Environment
    device='cuda',
    workers=1,
    gpu_ids=[0,1,2,3],
    data_dir='data/',
    save_dirpath='checkpoints/',
    # Training Hyperparemter
    batch_size=1,
    num_epochs=60,
    fintune_word_embedding=True,
    # Transformer
    embedding_size_word=300,
    num_heads=2,
    num_hidden_layers=2,
    hidden_size=300,
    max_length=400,
    attention_key_channels=0,
    attention_value_channels=0,
    filter_size=128,
    dropout=0.1,
    optimizer_adam_beta1=0.9,
    optimizer_adam_beta2=0.999,
    # Optimizier
    learning_rate=5e-4,
    max_gradient_norm=2

)