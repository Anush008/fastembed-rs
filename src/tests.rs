use std::fs;
use std::path::Path;

use hf_hub::Repo;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::common::DEFAULT_CACHE_DIR;
use crate::pooling::Pooling;
use crate::sparse_text_embedding::SparseTextEmbedding;
use crate::{
    read_file_to_bytes, EmbeddingModel, ImageEmbedding, ImageInitOptions, InitOptions,
    InitOptionsUserDefined, RerankInitOptions, RerankInitOptionsUserDefined, RerankerModel,
    SparseInitOptions, TextEmbedding, TextRerank, UserDefinedEmbeddingModel,
    UserDefinedRerankingModel,
};

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

            // Clear the model cache to avoid running out of space on GitHub Actions.
            // clean_cache(supported_model.model_code.clone())
        });
}

#[test]
fn test_sparse_embeddings() {
    SparseTextEmbedding::list_supported_models()
        .par_iter()
        .for_each(|supported_model| {
            let model: SparseTextEmbedding = SparseTextEmbedding::try_new(SparseInitOptions {
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
            embeddings.into_iter().for_each(|embedding| {
                assert!(embedding.values.iter().all(|&v| v > 0.0));
                assert!(embedding.indices.len() < 100);
                assert_eq!(embedding.indices.len(), embedding.values.len());
            });

            // Clear the model cache to avoid running out of space on GitHub Actions.
            clean_cache(supported_model.model_code.clone())
        });
}

#[test]
fn test_user_defined_embedding_model() {
    // Constitute the model in order to ensure it's downloaded and cached
    let test_model_info = TextEmbedding::get_model_info(&EmbeddingModel::AllMiniLML6V2);
    let pooling = Some(Pooling::Mean);

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
    let tokenizer_file = read_file_to_bytes(&model_files_dir.join("tokenizer.json"))
        .expect("Could not read tokenizer.json");
    // Create a UserDefinedEmbeddingModel
    let user_defined_model = UserDefinedEmbeddingModel {
        onnx_file,
        tokenizer_file,
        pooling,
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
    TextRerank::list_supported_models()
        .par_iter()
        .for_each(|supported_model| {

        let result = TextRerank::try_new(RerankInitOptions {
            model_name: supported_model.model.clone(),
            show_download_progress: true,
            ..Default::default()
        })
        .unwrap();

        let documents = vec![
            "hi",
            "The giant panda, sometimes called a panda bear or simply panda, is a bear species endemic to China.",
            "panda is an animal",
            "i dont know",
            "kind of mammal",
        ];

        let results = result
            .rerank("what is panda?", documents.clone(), true, None)
            .unwrap();

        assert_eq!(results.len(), documents.len(), "rerank model {:?} failed", supported_model);
        assert_eq!(results[0].document.as_ref().unwrap(), "panda is an animal");
        assert_eq!(results[1].document.as_ref().unwrap(), "The giant panda, sometimes called a panda bear or simply panda, is a bear species endemic to China.");

        // Clear the model cache to avoid running out of space on GitHub Actions.
        clean_cache(supported_model.model_code.clone())
    });
}

#[test]
fn test_user_defined_reranking_model() {
    // Constitute the model in order to ensure it's downloaded and cached
    let test_model_info: crate::RerankerModelInfo =
        TextRerank::get_model_info(&RerankerModel::JINARerankerV1TurboEn);

    TextRerank::try_new(RerankInitOptions {
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

    // FInd the onnx file - it will be any file in ./onnx ending with .onnx
    let onnx_file = read_file_to_bytes(
        &model_files_dir
            .join("onnx")
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
    let tokenizer_file = read_file_to_bytes(&model_files_dir.join("tokenizer.json"))
        .expect("Could not read tokenizer.json");
    // Create a UserDefinedEmbeddingModel
    let user_defined_model = UserDefinedRerankingModel {
        onnx_file,
        tokenizer_file,
    };

    // Try creating a TextEmbedding instance from the user-defined model
    let user_defined_reranker = TextRerank::try_new_from_user_defined(
        user_defined_model,
        RerankInitOptionsUserDefined::default(),
    )
    .unwrap();

    let documents = vec![
        "Hello, World!",
        "This is an example passage.",
        "fastembed-rs is licensed under Apache-2.0",
        "Some other short text here blah blah blah",
    ];

    // Generate embeddings over documents
    let results = user_defined_reranker
        .rerank("Ciao, Earth!", documents.clone(), false, None)
        .unwrap();

    assert_eq!(results.len(), documents.len());
    assert_eq!(results.first().unwrap().index, 0);
}

#[test]
fn test_image_embedding_model() {
    ImageEmbedding::list_supported_models()
        .par_iter()
        .for_each(|supported_model| {
            let model: ImageEmbedding = ImageEmbedding::try_new(ImageInitOptions {
                model_name: supported_model.model.clone(),
                ..Default::default()
            })
            .unwrap();

            let images = vec!["assets/image_0.png", "assets/image_1.png"];

            // Generate embeddings with the default batch size, 256
            let embeddings = model.embed(images.clone(), None).unwrap();

            assert_eq!(embeddings.len(), images.len());
            for embedding in embeddings {
                assert_eq!(embedding.len(), supported_model.dim);
            }

            // Clear the model cache to avoid running out of space on GitHub Actions.
            clean_cache(supported_model.model_code.clone())
        });
}

fn clean_cache(model_code: String) {
    let repo = Repo::model(model_code);
    let cache_dir = format!("{}/{}", DEFAULT_CACHE_DIR, repo.folder_name());
    let res = fs::remove_dir_all(cache_dir);
    assert!(res.is_ok());
}
// This is item "test-environment-aeghhgwpe-pro02a" of the [Aguana corpus](http://argumentation.bplaced.net/arguana/data)
fn get_sample_text() -> String {
    let t = "Light: FastEmbed is a lightweight library with few external dependencies. We don't require a GPU and don't download GBs of PyTorch dependencies, and instead use the ONNX Runtime. This makes it a great candidate for serverless runtimes like AWS Lambda. Fast: FastEmbed is designed for speed. We use the ONNX Runtime, which is faster than PyTorch. We also use data parallelism for encoding large datasets. Accurate: FastEmbed is better than OpenAI Ada-002. We also support an ever-expanding set of models, including a few multilingual models.";
    t.to_string()
}

#[test]
fn test_bgesmallen1point5_match_python_counterpart() {
    let model = TextEmbedding::try_new(InitOptions {
        model_name: EmbeddingModel::BGESmallENV15,
        show_download_progress: true,
        ..Default::default()
    })
    .expect("Create model succesfully");
    let text = get_sample_text();
    // baseline is generated in python using Xenova/bge-small-en-v1.5.onnx
    // we only take a 10 items to keep the test file polite
    let baseline: Vec<f32> = vec![
        -8.27567950e-02,
        1.47374356e-02,
        -6.30538464e-02,
        3.90174128e-02,
        7.83568323e-02,
        3.55395637e-02,
        -8.25874433e-02,
        -2.60544941e-02,
        6.47016615e-03,
        1.35167930e-02,
    ];

    let embeddings = model.embed(vec![text], None).expect("create successfully");
    let tolerance: f32 = 1e-3;
    for (expected, actual) in embeddings[0]
        .clone()
        .into_iter()
        .take(baseline.len())
        .zip(baseline.into_iter())
    {
        println!("{}: {}", expected, actual);
        assert!((expected - actual).abs() < tolerance);
    }
}

#[test]
fn test_allminilml6v2_match_python_counterpart() {
    let model = TextEmbedding::try_new(InitOptions {
        model_name: EmbeddingModel::AllMiniLML6V2,
        show_download_progress: true,
        ..Default::default()
    })
    .expect("Create model succesfully");

    let text = get_sample_text();
    // baseline is generated in python using qdrant/all-mini-lm-l6-v2.onnx
    // we only take a 10 items to keep the test file polite
    let baseline: Vec<f32> = vec![
        3.51051763e-02,
        1.04604298e-02,
        3.76799852e-02,
        7.07363337e-02,
        9.09777507e-02,
        -2.50771474e-02,
        -2.21438203e-02,
        -1.01643587e-02,
        4.66012731e-02,
        7.43136629e-02,
    ];

    let embeddings = model.embed(vec![text], None).expect("create successfully");
    let tolerance: f32 = 1e-6;
    for (expected, actual) in embeddings[0]
        .clone()
        .into_iter()
        .take(baseline.len())
        .zip(baseline.into_iter())
    {
        assert!((expected - actual).abs() < tolerance);
    }
}
