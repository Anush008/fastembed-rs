#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{linear, linear_no_bias, Activation, Linear, Module, VarBuilder};
use serde::Deserialize;
use std::path::PathBuf;

#[cfg(feature = "hf-hub")]
use hf_hub::api::sync::ApiBuilder;

#[derive(Deserialize, Debug, Clone)]
pub struct Config {
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub head_dim: Option<usize>,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_scaling: Option<f64>,
    pub rope_theta: f64,
    pub sliding_window: Option<usize>,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    pub use_sliding_window: bool,
    pub vocab_size: usize,
    #[serde(default)]
    pub max_window_layers: usize,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

fn scalar_f32(device: &Device, v: f32) -> Result<Tensor> {
    Tensor::from_slice(&[v], (1,), device)?.to_dtype(DType::F32)
}

fn scalar_f64_as_f32(device: &Device, v: f64) -> Result<Tensor> {
    scalar_f32(device, v as f32)
}

pub struct Qwen3RMSNorm {
    weight: Tensor, // [dim]
    eps: f64,
}

impl Qwen3RMSNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((dim,), "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for Qwen3RMSNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let in_dtype = xs.dtype();
        let dev = xs.device();

        let xs_f = xs.to_dtype(DType::F32)?;
        let var = xs_f.powf(2.0)?.mean_keepdim(D::Minus1)?; // [..., 1]

        let eps_t = scalar_f64_as_f32(dev, self.eps)?;
        let var_eps = var.broadcast_add(&eps_t)?;

        // xs * rsqrt(var + eps)  == xs * (1/sqrt(...))
        let inv_rms = var_eps.sqrt()?.recip()?; // [..., 1]
        let normed = xs_f.broadcast_mul(&inv_rms)?; // [..., dim]

        // weight * back_to_input_dtype
        let normed = normed.to_dtype(in_dtype)?;
        let w = self.weight.to_dtype(in_dtype)?;
        normed.broadcast_mul(&w)
    }
}

pub struct Qwen3MLP {
    gate_proj: Linear, // hidden -> intermediate
    up_proj: Linear,   // hidden -> intermediate
    down_proj: Linear, // intermediate -> hidden
    act_fn: Activation,
}

impl Qwen3MLP {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Qwen3MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

pub struct Qwen3RotaryEmbedding {
    inv_freq: Tensor,      // [dim/2] f32
    attention_factor: f32, // HF: attention_scaling (1.0 for default)
}

impl Qwen3RotaryEmbedding {
    pub fn new(cfg: &Config, device: &Device) -> Result<Self> {
        let base = cfg.rope_theta; // f64 for precision
        let dim = cfg.head_dim(); // head_dim
        assert!(dim.is_multiple_of(2), "head_dim must be even, got {dim}");

        // t = [0,2,4,...,dim-2] as f32
        let t = Tensor::arange_step(0u32, dim as u32, 2u32, device)?.to_dtype(DType::F32)?;

        // exponent = t / dim
        let dim_t = scalar_f32(device, dim as f32)?;
        let exponent = t.broadcast_div(&dim_t)?; // [dim/2]

        // inv_freq = 1 / (base ** exponent) = exp(-ln(base) * exponent)
        let ln_base = (base as f32).ln();
        let ln_base_t = scalar_f32(device, ln_base)?;
        let inv_freq = exponent.broadcast_mul(&ln_base_t.neg()?)?.exp()?; // [dim/2]

        Ok(Self {
            inv_freq,
            attention_factor: 1.0,
        })
    }

    /// position_ids: [B,T] (int)
    /// returns (cos, sin): [B,T,dim] in xs.dtype()
    pub fn forward(&self, xs: &Tensor, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, t) = position_ids.dims2()?;
        let d2 = self.inv_freq.dims1()?;
        let dev = xs.device();

        let inv_freq = self.inv_freq.to_device(dev)?.to_dtype(DType::F32)?;
        let pos = position_ids
            .to_device(dev)?
            .to_dtype(DType::F32)?
            .contiguous()?;

        // inv_freq_expanded: [B, d2, 1]
        // Use expand().contiguous() to ensure proper batch handling
        let inv_freq_expanded = inv_freq
            .reshape((1, d2, 1))?
            .expand((b, d2, 1))?
            .contiguous()?;
        // position_ids_expanded: [B, 1, T]
        let pos_expanded = pos.reshape((b, 1, t))?.contiguous()?;

        // freqs = (inv_freq_expanded @ pos_expanded).transpose(1,2) -> [B,T,d2]
        let freqs = inv_freq_expanded
            .matmul(&pos_expanded)?
            .transpose(1, 2)?
            .contiguous()?;

        // emb = cat(freqs, freqs) -> [B,T,dim]
        let emb = Tensor::cat(&[&freqs, &freqs], 2)?;

        let scale = scalar_f32(dev, self.attention_factor)?;
        let cos = emb.cos()?.broadcast_mul(&scale)?;
        let sin = emb.sin()?.broadcast_mul(&scale)?;

        let out_dtype = xs.dtype();
        Ok((cos.to_dtype(out_dtype)?, sin.to_dtype(out_dtype)?))
    }
}

// rotate_half + apply_rotary_pos_emb (HF)
// rotate_half: cat(-x2, x1) along last dim (half-split style)
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x
        .dims()
        .last()
        .copied()
        .ok_or_else(|| candle_core::Error::Msg("empty dims".into()))?;
    assert!(d % 2 == 0, "rotate_half requires even last dim, got {d}");
    let half = d / 2;

    // x1 = x[..., :half], x2 = x[..., half:]
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let nx2 = x2.neg()?;
    Tensor::cat(&[&nx2, &x1], x.rank() - 1)
}

fn apply_rotary_pos_emb(
    q: &Tensor,   // [B, Hq, T, D]
    k: &Tensor,   // [B, Hk, T, D]
    cos: &Tensor, // [B, T, D]
    sin: &Tensor, // [B, T, D]
) -> Result<(Tensor, Tensor)> {
    // unsqueeze_dim=1 -> [B,1,T,D] then broadcast
    let cos_u = cos.unsqueeze(1)?;
    let sin_u = sin.unsqueeze(1)?;

    let q_embed = (q.broadcast_mul(&cos_u)? + rotate_half(q)?.broadcast_mul(&sin_u)?)?;
    let k_embed = (k.broadcast_mul(&cos_u)? + rotate_half(k)?.broadcast_mul(&sin_u)?)?;
    Ok((q_embed, k_embed))
}

// repeat_kv (HF): [B, Nkv, T, D] -> [B, Nh, T, D]
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, n_kv, t, d) = x.dims4()?;
    // [B, Nkv, 1, T, D]
    let x = x.unsqueeze(2)?;
    // broadcast to [B, Nkv, n_rep, T, D]
    let x = x.broadcast_as((b, n_kv, n_rep, t, d))?;
    // reshape to [B, Nkv*n_rep, T, D]
    x.reshape((b, n_kv * n_rep, t, d))
}

// - q_norm/k_norm on head_dim after reshape
// - RoPE on q,k
// - repeat_kv for key/value (GQA)
pub struct Qwen3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Qwen3RMSNorm, // dim = head_dim
    k_norm: Qwen3RMSNorm, // dim = head_dim

    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f32,
}

impl Qwen3Attention {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = cfg.num_kv_groups();
        assert!(
            num_heads.is_multiple_of(num_kv_heads),
            "num_heads must be multiple of num_kv_heads"
        );

        let q_out = num_heads * head_dim;
        let kv_out = num_kv_heads * head_dim;

        let q_proj = if cfg.attention_bias {
            linear(cfg.hidden_size, q_out, vb.pp("q_proj"))?
        } else {
            linear_no_bias(cfg.hidden_size, q_out, vb.pp("q_proj"))?
        };
        let k_proj = if cfg.attention_bias {
            linear(cfg.hidden_size, kv_out, vb.pp("k_proj"))?
        } else {
            linear_no_bias(cfg.hidden_size, kv_out, vb.pp("k_proj"))?
        };
        let v_proj = if cfg.attention_bias {
            linear(cfg.hidden_size, kv_out, vb.pp("v_proj"))?
        } else {
            linear_no_bias(cfg.hidden_size, kv_out, vb.pp("v_proj"))?
        };
        let o_proj = if cfg.attention_bias {
            linear(q_out, cfg.hidden_size, vb.pp("o_proj"))?
        } else {
            linear_no_bias(q_out, cfg.hidden_size, vb.pp("o_proj"))?
        };

        // q_norm/k_norm are RMSNorm(head_dim)
        let q_norm = Qwen3RMSNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = Qwen3RMSNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            scaling: (head_dim as f32).powf(-0.5),
        })
    }

    /// hidden_states: [B,T,H]
    /// position_embeddings: (cos,sin) both [B,T,D]
    /// attention_mask: additive mask [B,1,T,T] (0 or -inf)
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, t, _h) = hidden_states.dims3()?;
        let d = self.head_dim;

        // hidden_shape = [B,T,-1,D]
        // q: [B,T,Nh*D] -> [B,T,Nh,D] -> q_norm -> transpose -> [B,Nh,T,D]
        let q = hidden_states
            .apply(&self.q_proj)?
            .reshape((b, t, self.num_heads, d))?;
        let q = q.apply(&self.q_norm)?.transpose(1, 2)?;

        // k: [B,T,Nkv*D] -> [B,T,Nkv,D] -> k_norm -> [B,Nkv,T,D]
        let k = hidden_states
            .apply(&self.k_proj)?
            .reshape((b, t, self.num_kv_heads, d))?;
        let k = k.apply(&self.k_norm)?.transpose(1, 2)?;

        // v: [B,Nkv,T,D]
        let v = hidden_states
            .apply(&self.v_proj)?
            .reshape((b, t, self.num_kv_heads, d))?
            .transpose(1, 2)?;

        let (cos, sin) = position_embeddings;
        let (q, k) = apply_rotary_pos_emb(&q, &k, cos, sin)?;

        // GQA expand
        let k = repeat_kv(&k, self.num_kv_groups)?; // [B,Nh,T,D]
        let v = repeat_kv(&v, self.num_kv_groups)?; // [B,Nh,T,D]

        // attn_weights = q @ k^T * scaling
        let kt = k.transpose(2, 3)?; // [B,Nh,D,T]
        let mut attn = q.matmul(&kt)?; // [B,Nh,T,T]

        let scale = scalar_f32(attn.device(), self.scaling)?;
        attn = attn.broadcast_mul(&scale)?;

        if let Some(mask) = attention_mask {
            attn = attn.broadcast_add(mask)?;
        }

        // softmax over last dim
        let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;

        // attn_output = attn @ v -> [B,Nh,T,D]
        let out = attn.matmul(&v)?;
        // transpose back -> [B,T,Nh,D] -> reshape [B,T,Nh*D] -> o_proj -> [B,T,H]
        let out = out.transpose(1, 2)?.reshape((b, t, self.num_heads * d))?;
        out.apply(&self.o_proj)
    }
}

pub struct Qwen3DecoderLayer {
    self_attn: Qwen3Attention,
    mlp: Qwen3MLP,
    input_layernorm: Qwen3RMSNorm,          // dim=hidden_size
    post_attention_layernorm: Qwen3RMSNorm, // dim=hidden_size
}

impl Qwen3DecoderLayer {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: Qwen3Attention::new(cfg, vb.pp("self_attn"))?,
            mlp: Qwen3MLP::new(cfg, vb.pp("mlp"))?,
            input_layernorm: Qwen3RMSNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: Qwen3RMSNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_embeddings: (&Tensor, &Tensor),
    ) -> Result<Tensor> {
        // Pre-norm
        let residual = hidden_states.clone();
        let hs = hidden_states.apply(&self.input_layernorm)?;
        let hs = self
            .self_attn
            .forward(&hs, position_embeddings, attention_mask)?;
        let hs = (residual + hs)?;

        // MLP
        let residual = hs.clone();
        let hs2 = hs.apply(&self.post_attention_layernorm)?;
        let hs2 = hs2.apply(&self.mlp)?;
        residual + hs2
    }
}

pub struct Qwen3Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: Qwen3RMSNorm,
    rotary_emb: Qwen3RotaryEmbedding,
    cfg: Config,
    device: Device,
}

impl Qwen3Model {
    pub fn new(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Qwen3DecoderLayer::new(&cfg, vb.pp(format!("layers.{i}")))?);
        }

        let norm = Qwen3RMSNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let rotary_emb = Qwen3RotaryEmbedding::new(&cfg, vb.device())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            cfg,
            device,
        })
    }

    /// input_ids: [B,T]
    /// attention_mask: optionally [B,T] (1=token,0=pad) OR already expanded additive mask [B,1,T,T]
    /// Here we assume additive mask [B,1,T,T] for simplicity (HF does that).
    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask_4d: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, t) = input_ids.dims2()?;

        // inputs_embeds
        let mut hs = self.embed_tokens.forward(input_ids)?; // [B,T,H]

        // position_ids = [B,T] = 0..T-1 (HF uses cache_position; for no-cache this is fine)
        let pos_1d = Tensor::arange(0u32, t as u32, hs.device())?;
        // Use expand() instead of broadcast_as() and make contiguous to avoid striding issues
        let position_ids = pos_1d.unsqueeze(0)?.expand((b, t))?.contiguous()?;

        // position_embeddings = (cos,sin) once
        let (cos, sin) = self.rotary_emb.forward(&hs, &position_ids)?;

        // layers
        for layer in &self.layers {
            hs = layer.forward(&hs, attention_mask_4d, (&cos, &sin))?;
        }

        // final norm
        hs.apply(&self.norm)
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// High-level embedding wrapper around [`Qwen3Model`] that handles tokenization,
/// batching, mean-pooling, and L2-normalization.
pub struct Qwen3TextEmbedding {
    model: Qwen3Model,
    tokenizer: tokenizers::Tokenizer,
}

impl Qwen3TextEmbedding {
    /// Build from an already loaded model and tokenizer.
    pub fn new(model: Qwen3Model, tokenizer: tokenizers::Tokenizer) -> Self {
        Self { model, tokenizer }
    }

    /// Load from a Hugging Face repo (e.g. "Qwen/Qwen3-Embedding-0.6B").
    #[cfg(feature = "hf-hub")]
    pub fn from_hf(
        repo_id: &str,
        device: &Device,
        dtype: DType,
        max_length: usize,
    ) -> Result<Self> {
        use tokenizers::{PaddingParams, PaddingStrategy, TruncationParams};

        let api = ApiBuilder::new()
            .with_progress(true)
            .build()
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let repo = api.model(repo_id.to_string());

        // Load config
        let cfg_path: PathBuf = repo
            .get("config.json")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let cfg: Config = serde_json::from_slice(
            &std::fs::read(&cfg_path).map_err(|e| candle_core::Error::Msg(e.to_string()))?,
        )
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        // Load weights (single or sharded)
        let weight_files: Vec<PathBuf> = if let Ok(p) = repo.get("model.safetensors") {
            vec![p]
        } else {
            let mut files = Vec::new();
            for i in 1.. {
                let candidates: Vec<_> = (1..=20)
                    .filter_map(|total| {
                        let fname = format!("model-{:05}-of-{:05}.safetensors", i, total);
                        repo.get(&fname).ok()
                    })
                    .collect();
                if candidates.is_empty() {
                    break;
                }
                files.extend(candidates.into_iter().take(1));
            }
            if files.is_empty() {
                return Err(candle_core::Error::Msg(
                    "Could not locate model.safetensors or sharded weight files".into(),
                ));
            }
            files
        };

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, device)? };
        let model = Qwen3Model::new(cfg, vb)?;

        // Load tokenizer
        let tok_path: PathBuf = repo
            .get("tokenizer.json")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let mut tokenizer = tokenizers::Tokenizer::from_file(tok_path)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        // Use LEFT padding so the last position is always the real last token.
        // This simplifies last-token pooling and matches the official implementation.
        let _ = tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: tokenizers::PaddingDirection::Left,
            ..Default::default()
        }));
        let _ = tokenizer.with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }));

        // Note: Unlike some other embedding models, Qwen3-Embedding does NOT
        // add an EOS token. The last_token_pool simply takes the last real token.

        Ok(Self { model, tokenizer })
    }

    pub fn config(&self) -> &Config {
        self.model.config()
    }

    pub fn device(&self) -> &Device {
        self.model.device()
    }

    /// Embed a batch of texts, returning normalized embeddings.
    pub fn embed<S: AsRef<str>>(&self, texts: &[S]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let encodings = self
            .tokenizer
            .encode_batch(texts.iter().map(|s| s.as_ref()).collect::<Vec<_>>(), true)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let batch_size = encodings.len();
        let seq_len = encodings[0].len();

        let mut input_ids_vec: Vec<u32> = Vec::with_capacity(batch_size * seq_len);
        let mut attention_mask_vec: Vec<f32> = Vec::with_capacity(batch_size * seq_len);

        for enc in &encodings {
            input_ids_vec.extend(enc.get_ids().iter().copied());
            attention_mask_vec.extend(enc.get_attention_mask().iter().map(|&m| m as f32));
        }

        let device = self.model.device();
        let input_ids = Tensor::from_vec(input_ids_vec, (batch_size, seq_len), device)?;
        let attention_mask_2d =
            Tensor::from_vec(attention_mask_vec, (batch_size, seq_len), device)?;

        // Build 4D attention mask: causal + padding
        let mask_value = -1e4f32;

        // causal_mask[i,j] = 0 if i >= j else mask_value
        let causal = {
            let mut data = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    data[i * seq_len + j] = mask_value;
                }
            }
            Tensor::from_vec(data, (1, 1, seq_len, seq_len), device)?
        };

        // Create padding mask using HuggingFace's prepare_4d_attention_mask:
        // expanded_mask = mask.unsqueeze(1).unsqueeze(2).expand(bsz, 1, tgt_len, src_len)
        // inverted_mask = (1.0 - expanded_mask) * -1e4
        let pad_mask_expanded = attention_mask_2d.unsqueeze(1)?.unsqueeze(2)?; // [B, 1, 1, T]
        let pad_mask_expanded = pad_mask_expanded.expand((batch_size, 1, seq_len, seq_len))?;
        let pad_mask_f32 = pad_mask_expanded.to_dtype(DType::F32)?;
        let ones = Tensor::ones_like(&pad_mask_f32)?;
        let inverted_mask = ones.sub(&pad_mask_f32)?;
        let mask_val_t = Tensor::new(&[mask_value], device)?;
        let pad_additive = inverted_mask.broadcast_mul(&mask_val_t)?;

        // Combine causal + padding mask
        let causal_broadcast = causal.broadcast_as((batch_size, 1, seq_len, seq_len))?;
        let attention_mask_4d = causal_broadcast.add(&pad_additive)?;

        // Forward pass -> [B, T, H]
        let hidden = self.model.forward(&input_ids, Some(&attention_mask_4d))?;

        // Last token pooling: with left padding, the last position is always the real last token
        let pooled = hidden.i((.., seq_len - 1))?; // [B, H]

        // L2 normalize (add epsilon for numerical stability)
        let sum_sq = pooled.sqr()?.sum_keepdim(1)?;
        let eps_tensor = Tensor::new(&[1e-12f32], device)?.broadcast_as(sum_sq.shape())?;
        let norm = sum_sq.add(&eps_tensor)?.sqrt()?;
        let normalized = pooled.broadcast_div(&norm)?;

        // Convert to Vec<Vec<f32>>
        let normalized = normalized.to_dtype(DType::F32)?;
        let data = normalized.to_vec2::<f32>()?;
        Ok(data)
    }
}
