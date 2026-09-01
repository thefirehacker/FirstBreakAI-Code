from dataclasses import dataclass

@dataclass
class DeepSeekV3ModelArgs:

"""
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        scale_fmt (Optional[str]): Format for quantization scale.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """
    max_batch_size int = 8 //maximum batch size for inference
    max_seq_len: int = 4096 * 4 
    vocab_size: int = 102400
    dim int = 2048
    inter_dim int = 10944 //intermediate dimension for MLP layers
    moe_inter_dim int = 1408 //intermediate dimension for MoE layers
    n_layers int = 64 //number of transformer layers
    n_dense_layers int = 1 //number of dense layers in the model
    n_heads int = 16 //number of attention heads


    #MoE layers
    n_routed_experts int = 64 //number of routed experts for MoE layers
    n_shared_experts int = 2 //number of shared experts for MoE layers
    n_activated_experts int = 6 //number of activated experts in MoE layers
    n_expert_groups int = 1 //number of expert groups
    n_limited_groups int = 1 //number of limited groups for MoE routing
    score_func Literal["softmax", "sigmoid"] = "softmax" //scoring function for MoE routing
    route_scale float = 1.0 //scaling factor for routing scores
    q_lora_rank int = 0 //LoRA rank for query projections
    kv_lora_rank int = 512 //LoRA rank for key-value projections
    qk_nope_head_dim int = 128 //dimension for query-key projections without positional embeddings
    qk_rope_head_dim int = 64 //dimension for query-key projections with rotary embeddings
    v_head_dim int = 128 //dimension for value projections

    # Yarn
    original_seq_len int = 4096 //original sequence length
    rope_theta float = 10000.0 //base for rotary positional encoding
    rope_factor float = 40 //scaling factor for extended sequence lengths
    beta_fast int = 32 //fast beta correction factor
    beta_slow int = 1 //slow beta correction factor
    mscale float = 1.0 //scaling factor for extended attention