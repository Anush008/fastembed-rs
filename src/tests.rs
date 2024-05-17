use std::path::Path;

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{read_file_to_bytes, EmbeddingModel, InitOptions, InitOptionsUserDefined, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel, DEFAULT_CACHE_DIR, TextRerank};
use crate::models::RerankerModel;

#[test]
fn test_embeddings() {
    TextEmbedding::list_supported_models()
        .par_iter()
        .for_each(|supported_model| {
            let model: TextEmbedding = TextEmbedding::try_new(InitOptions {
                model_name: supported_model.model.clone(),
                ..Default::default()
            })
            .unwrap();

            let documents = vec![
                "Hello, World!",
                "This is an example passage.",
                "fastembed-rs is licensed under Apache-2.0",
                "Some other short text here blah blah blah",
            ];

            // Generate embeddings with the default batch size, 256
            let embeddings = model.embed(documents.clone(), None).unwrap();

            assert_eq!(embeddings.len(), documents.len());
            for embedding in embeddings {
                assert_eq!(embedding.len(), supported_model.dim);
            }
        });
}

#[test]

fn test_user_defined_embedding_model() {
    // Constitute the model in order to ensure it's downloaded and cached
    let test_model_info = TextEmbedding::get_model_info(&EmbeddingModel::AllMiniLML6V2);

    TextEmbedding::try_new(InitOptions {
        model_name: test_model_info.model,
        ..Default::default()
    })
    .unwrap();

    // Get the directory of the model
    let model_name = test_model_info.model_code.replace('/', "--");
    let model_dir = Path::new(DEFAULT_CACHE_DIR).join(format!("models--{}", model_name));

    // Find the "snapshots" sub-directory
    let snapshots_dir = model_dir.join("snapshots");

    // Get the first sub-directory in snapshots
    let model_files_dir = snapshots_dir
        .read_dir()
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();

    // FInd the onnx file - it will be any file ending with .onnx
    let onnx_file = read_file_to_bytes(
        &model_files_dir
            .read_dir()
            .unwrap()
            .find(|entry| {
                entry
                    .as_ref()
                    .unwrap()
                    .path()
                    .extension()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    == "onnx"
            })
            .unwrap()
            .unwrap()
            .path(),
    )
    .expect("Could not read onnx file");

    // Load the tokenizer files
    let tokenizer_files = TokenizerFiles {
        tokenizer_file: read_file_to_bytes(&model_files_dir.join("tokenizer.json"))
            .expect("Could not read tokenizer.json"),
        config_file: read_file_to_bytes(&model_files_dir.join("config.json"))
            .expect("Could not read config.json"),
        special_tokens_map_file: read_file_to_bytes(
            &model_files_dir.join("special_tokens_map.json"),
        )
        .expect("Could not read special_tokens_map.json"),
        tokenizer_config_file: read_file_to_bytes(&model_files_dir.join("tokenizer_config.json"))
            .expect("Could not read tokenizer_config.json"),
    };
    // Create a UserDefinedEmbeddingModel
    let user_defined_model = UserDefinedEmbeddingModel {
        onnx_file,
        tokenizer_files,
    };

    // Try creating a TextEmbedding instance from the user-defined model
    let user_defined_text_embedding = TextEmbedding::try_new_from_user_defined(
        user_defined_model,
        InitOptionsUserDefined::default(),
    )
    .unwrap();

    let documents = vec![
        "Hello, World!",
        "This is an example passage.",
        "fastembed-rs is licensed under Apache-2.0",
        "Some other short text here blah blah blah",
    ];

    // Generate embeddings over documents
    let embeddings = user_defined_text_embedding
        .embed(documents.clone(), None)
        .unwrap();
    assert_eq!(embeddings.len(), documents.len());
    for embedding in embeddings {
        assert_eq!(embedding.len(), test_model_info.dim);
    }
}

#[test]
fn test_rerank() {
    let result = TextRerank::try_new(RerankerModel::BGERerankerBase,
                                     InitOptions { show_download_progress: true, ..Default::default() }).unwrap();

    let texts = vec![
        ("what is panda?", "hi"),
        ("what is panda?", "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."),
        ("what is panda?", "panda is animal"),
        ("what is panda?", "i dont know"),
        ("what is panda?", "kind of mammal"),
    ];
    let scores = result.rerank(texts, None);
    println!("{:?}", scores);
}