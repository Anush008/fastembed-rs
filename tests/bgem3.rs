#![cfg(feature = "hf-hub")]

use fastembed::{
    Bgem3Embedding, Bgem3InitOptions, Bgem3Model, InitOptionsUserDefined, TokenizerFiles,
    UserDefinedBgem3Model,
};
use std::collections::HashMap;

#[test]
fn test_bgem3_joint_embeddings_match_python() {
    let mut model = Bgem3Embedding::try_new(Bgem3InitOptions::new(Bgem3Model::BGEM3Q))
        .expect("Failed to initialize BGEM3Q model");

    let sentences = vec![
        "fastembed-rs is licensed under Apache  2.0",
        "Superman is the best superhero of all time",
    ];

    let output = model.embed(sentences, None).expect("Embedding failed");

    // 1. Verify Dense Embeddings
    assert_eq!(output.dense.len(), 2);
    assert_eq!(output.dense[0].len(), 1024);
    assert_eq!(output.dense[1].len(), 1024);

    let expected_dense_0 = [-0.017583456, -0.012429078, -0.001542368, 0.062134236, -0.01704353];
    let expected_dense_1 = [-0.009514477, 0.036382556, -0.022897111, -0.010296857, -0.010794208];

    for (i, val) in expected_dense_0.iter().enumerate() {
        assert!((output.dense[0][i] - val).abs() < 1e-4);
    }
    for (i, val) in expected_dense_1.iter().enumerate() {
        assert!((output.dense[1][i] - val).abs() < 1e-4);
    }

    // 2. Verify Sparse Embeddings
    assert_eq!(output.sparse.len(), 2);
    
    let expected_sparse_0: HashMap<usize, f32> = [
        (9, 0.05852255970239639),
        (71, 0.0870199054479599),
        (83, 0.09317997843027115),
        (195, 0.1629659235477447),
        (1379, 0.10179002583026886),
        (1430, 0.15302222967147827),
        (4271, 0.1709854155778885),
        (4295, 0.24088919162750244),
        (9795, 0.1714186817407608),
        (13482, 0.28737571835517883),
        (16655, 0.22440019249916077),
        (86872, 0.18500655889511108),
    ]
    .into_iter()
    .collect();

    let expected_sparse_1: HashMap<usize, f32> = [
        (70, 0.16206477582454681),
        (83, 0.2035697102546692),
        (111, 0.12182330340147018),
        (756, 0.11760648339986801),
        (1601, 0.17275650799274445),
        (1733, 0.1703031063079834),
        (2965, 0.24623024463653564),
        (90865, 0.25013893842697144),
        (183497, 0.31275802850723267),
    ]
    .into_iter()
    .collect();

    assert_eq!(output.sparse[0].indices.len(), expected_sparse_0.len());
    for (idx, val) in output.sparse[0].indices.iter().zip(output.sparse[0].values.iter()) {
        let expected_val = expected_sparse_0.get(idx).expect("Unexpected index in sparse 0");
        assert!((val - expected_val).abs() < 1e-4, "Sparse 0 index {}: expected {}, got {}", idx, expected_val, val);
    }

    assert_eq!(output.sparse[1].indices.len(), expected_sparse_1.len());
    for (idx, val) in output.sparse[1].indices.iter().zip(output.sparse[1].values.iter()) {
        let expected_val = expected_sparse_1.get(idx).expect("Unexpected index in sparse 1");
        assert!((val - expected_val).abs() < 1e-4, "Sparse 1 index {}: expected {}, got {}", idx, expected_val, val);
    }

    // 3. Verify ColBERT Embeddings
    assert_eq!(output.colbert.len(), 2);
    assert_eq!(output.colbert[0].len(), 13);
    assert_eq!(output.colbert[1].len(), 10);

    let expected_colbert_0_tok1 = [-0.032423463, -0.03486132, -0.054851983];
    let expected_colbert_0_tok2 = [-0.019846972, -0.039908897, -0.03129674];
    let expected_colbert_1_tok1 = [-0.020133337, -0.038063653, -0.019141976];
    let expected_colbert_1_tok2 = [-0.0041074636, -0.039726157, -0.04240649];

    for (i, val) in expected_colbert_0_tok1.iter().enumerate() {
        assert!((output.colbert[0][0][i] - val).abs() < 1e-4);
    }
    for (i, val) in expected_colbert_0_tok2.iter().enumerate() {
        assert!((output.colbert[0][1][i] - val).abs() < 1e-4);
    }
    for (i, val) in expected_colbert_1_tok1.iter().enumerate() {
        assert!((output.colbert[1][0][i] - val).abs() < 1e-4);
    }
    for (i, val) in expected_colbert_1_tok2.iter().enumerate() {
        assert!((output.colbert[1][1][i] - val).abs() < 1e-4);
    }
}

#[test]
fn test_bgem3_user_defined_model() {
    // We will verify the user-defined loader by pulling the files from HF and feeding them manually to simulate a local deployment
    
    let model_repo = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(true)
        .build()
        .expect("Failed to build API client")
        .model(Bgem3Model::BGEM3Q.to_string());

    let onnx_file = std::fs::read(
        model_repo
            .get("model_quantized.onnx")
            .expect("Failed to get model file"),
    )
    .expect("Failed to read model file");

    let tokenizer_files = TokenizerFiles {
        tokenizer_file: std::fs::read(model_repo.get("tokenizer.json").unwrap()).unwrap(),
        config_file: std::fs::read(model_repo.get("config.json").unwrap()).unwrap(),
        special_tokens_map_file: std::fs::read(model_repo.get("special_tokens_map.json").unwrap()).unwrap(),
        tokenizer_config_file: std::fs::read(model_repo.get("tokenizer_config.json").unwrap()).unwrap(),
    };

    let user_model = UserDefinedBgem3Model::new(onnx_file, tokenizer_files);
    let mut model = Bgem3Embedding::try_new_from_user_defined(user_model, InitOptionsUserDefined::default())
        .expect("Failed to build user defined BGEM3 model");

    let sentences = vec![
        "fastembed-rs is licensed under Apache  2.0",
        "Superman is the best superhero of all time",
    ];

    let output = model.embed(sentences, None).expect("Embedding failed");

    assert_eq!(output.dense.len(), 2);
    assert_eq!(output.sparse.len(), 2);
    assert_eq!(output.colbert.len(), 2);

    // Verify a simple weight from the first sentence to ensure it produces identical numbers
    let expected_dense_0 = [-0.017583456, -0.012429078, -0.001542368, 0.062134236, -0.01704353];
    for (i, val) in expected_dense_0.iter().enumerate() {
        assert!((output.dense[0][i] - val).abs() < 1e-4);
    }
}

#[test]
fn test_bgem3_custom_max_length() {
    // Verify that the user can override the max length (e.g. to 5 tokens) and it successfully truncates
    let mut model = Bgem3Embedding::try_new(
        Bgem3InitOptions::new(Bgem3Model::BGEM3Q).with_max_length(5)
    )
    .expect("Failed to initialize BGEM3Q model with custom max length");

    let sentences = vec![
        "fastembed-rs is licensed under Apache 2.0 and is very cool",
    ];

    let output = model.embed(sentences, None).expect("Embedding failed");

    // The sentence "fastembed-rs is licensed under Apache 2.0 and is very cool" has more than 5 tokens.
    // If truncated to 5 tokens, the ColBERT output token count (excluding CLS) should be exactly 4 tokens
    // since the total tokens with CLS/EOS is limited by max_length = 5.
    assert_eq!(output.colbert[0].len(), 4);
}

