#![cfg(feature = "online")]

use std::fs;
use std::path::Path;

use hf_hub::Repo;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use fastembed::{
    read_file_to_bytes, Embedding, EmbeddingModel, ImageEmbedding, ImageInitOptions, InitOptions,
    InitOptionsUserDefined, OnnxSource, Pooling, QuantizationMode, RerankInitOptions,
    RerankInitOptionsUserDefined, RerankerModel, SparseInitOptions, SparseTextEmbedding,
    TextEmbedding, TextRerank, TokenizerFiles, UserDefinedEmbeddingModel,
    UserDefinedRerankingModel, DEFAULT_CACHE_DIR,
};

/// A small epsilon value for floating point comparisons.
const EPS: f32 = 1e-2;

/// Precalculated embeddings for the supported models using #99
/// (4f09b6842ce1fcfaf6362678afcad9a176e05304).
///
/// These are the sum of all embedding values for each document. While not
/// perfect, they should be good enough to verify that the embeddings are being
/// generated correctly.
///
/// If you have just inserted a new `EmbeddingModel` variant, please update the
/// expected embeddings.
///
/// # Returns
///
/// If the embeddings are correct, this function returns `Ok(())`. If there are
/// any mismatches, it returns `Err(Vec<usize>)` with the indices of the
/// mismatched embeddings.
#[allow(unreachable_patterns)]
fn verify_embeddings(model: &EmbeddingModel, embeddings: &[Embedding]) -> Result<(), Vec<usize>> {
    let expected = match model {
        EmbeddingModel::AllMiniLML12V2 => [-0.12147753, 0.30144796, -0.06882502, -0.6303331],
        EmbeddingModel::AllMiniLML12V2Q => [-0.07808663, 0.27919534, -0.0770612, -0.75660324],
        EmbeddingModel::AllMiniLML6V2 => [0.59605527, 0.36542925, -0.16450031, -0.40903988],
        EmbeddingModel::AllMiniLML6V2Q => [0.5677276, 0.40180072, -0.15454668, -0.4672576],
        EmbeddingModel::BGEBaseENV15 => [-0.51290065, -0.4844747, -0.53036124, -0.5337459],
        EmbeddingModel::BGEBaseENV15Q => [-0.5130697, -0.48461288, -0.53067875, -0.5337806],
        EmbeddingModel::BGELargeENV15 => [-0.19347441, -0.28394595, -0.1549195, -0.22201893],
        EmbeddingModel::BGELargeENV15Q => [-0.19366685, -0.2842059, -0.15471499, -0.22216901],
        EmbeddingModel::BGESmallENV15 => [0.09881669, 0.15151203, 0.12057499, 0.13641948],
        EmbeddingModel::BGESmallENV15Q => [0.09881936, 0.15154803, 0.12057378, 0.13639033],
        EmbeddingModel::BGESmallZHV15 => [-1.1194772, -1.0928253, -1.0325904, -1.0050416],
        EmbeddingModel::GTEBaseENV15 => [-1.6900877, -1.7148916, -1.7333382, -1.5121834],
        EmbeddingModel::GTEBaseENV15Q => [-1.7032102, -1.7076654, -1.729326, -1.5317788],
        EmbeddingModel::GTELargeENV15 => [-1.6457459, -1.6582386, -1.6809471, -1.6070237],
        EmbeddingModel::GTELargeENV15Q => [-1.6044945, -1.6469251, -1.6828246, -1.6265479],
        EmbeddingModel::MultilingualE5Base => [-0.057211064, -0.14287914, -0.071678676, -0.17549144],
        EmbeddingModel::MultilingualE5Large => [-0.7473163, -0.76040405, -0.7537941, -0.72920954],
        EmbeddingModel::MultilingualE5Small => [-0.2640718, -0.13929011, -0.08091972, -0.12388548],
        EmbeddingModel::MxbaiEmbedLargeV1 => [-0.2032495, -0.29803938, -0.15803768, -0.23155808],
        EmbeddingModel::MxbaiEmbedLargeV1Q => [-0.1811538, -0.2884392, -0.1636593, -0.21548103],
        EmbeddingModel::NomicEmbedTextV1 => [0.13788113, 0.10750078, 0.050809078, 0.09284662],
        EmbeddingModel::NomicEmbedTextV15 => [0.1932303, 0.13795732, 0.14700879, 0.14940643],
        EmbeddingModel::NomicEmbedTextV15Q => [0.20999804, 0.13103808, 0.14427708, 0.13452803],
        EmbeddingModel::ParaphraseMLMiniLML12V2 => [-0.07795018, -0.059113946, -0.043668486, -0.1880083],
        EmbeddingModel::ParaphraseMLMiniLML12V2Q => [-0.07749095, -0.058981877, -0.043487836, -0.18775631],
        EmbeddingModel::ParaphraseMLMpnetBaseV2 => [0.39132136, 0.49490625, 0.65497226, 0.34237382],
        EmbeddingModel::ClipVitB32 => [0.7057363, 1.3549932, 0.46823958, 0.52351093],
        _ => panic!("Model {model} not found. If you have just inserted this `EmbeddingModel` variant, please update the expected embeddings."),
    };

    let mismatched_indices = embeddings
        .iter()
        .map(|embedding| embedding.iter().sum::<f32>())
        .zip(expected.iter())
        .enumerate()
        .filter_map(|(i, (sum, &expected))| {
            if (sum - expected).abs() > EPS {
                eprintln!(
                    "Mismatched embeddings for model {model} at index {i}: {sum} != {expected}",
                    model = model,
                    i = i,
                    sum = sum,
                    expected = expected
                );
                Some(i)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    if mismatched_indices.is_empty() {
        Ok(())
    } else {
        Err(mismatched_indices)
    }
}

macro_rules! create_embeddings_test {
    (
        name: $name:ident,
        batch_size: $batch_size:expr,
    ) => {
        #[test]
        fn $name() {
            TextEmbedding::list_supported_models()
                .par_iter()
                .for_each(|supported_model| {
                    let model: TextEmbedding = TextEmbedding::try_new(InitOptions::new(supported_model.model.clone()))
                    .unwrap();

                    let documents = vec![
                        "Hello, World!",
                        "This is an example passage.",
                        "fastembed-rs is licensed under Apache-2.0",
                        "Some other short text here blah blah blah",
                    ];

                    // Generate embeddings with the default batch size, 256
                    let batch_size = $batch_size;
                    let embeddings = model.embed(documents.clone(), batch_size);

                    if matches!(
                        (batch_size, supported_model.model.get_quantization_mode()),
                        (Some(n), QuantizationMode::Dynamic) if n < documents.len()
                    ) {
                        // For Dynamic quantization, the batch size must be greater than or equal to the number of documents
                        // Otherwise, an error is expected
                        assert!(embeddings.is_err(), "Expected error for batch size < document count for {model} using dynamic quantization.", model=supported_model.model);
                    } else {
                        let embeddings = embeddings.unwrap_or_else(
                            |exc| panic!("Expected embeddings for {model} to be generated successfully: {exc}", model=supported_model.model, exc=exc),
                        );
                        assert_eq!(embeddings.len(), documents.len());

                        for embedding in &embeddings {
                            assert_eq!(embedding.len(), supported_model.dim);
                        }

                        match verify_embeddings(&supported_model.model, &embeddings) {
                            Ok(_) => {}
                            Err(mismatched_indices) => {
                                panic!(
                                    "Mismatched embeddings for model {model}: {sentences:?}",
                                    model = supported_model.model,
                                    sentences = &mismatched_indices
                                        .iter()
                                        .map(|&i| documents[i])
                                        .collect::<Vec<_>>()
                                );
                            }
                        }
                    }
                });
        }

    };
}
create_embeddings_test!(
    name: test_batch_size_default,
    batch_size: None,
);
create_embeddings_test!(
    name: test_batch_size_less_than_document_count,
    batch_size: Some(3),
);

#[test]
fn test_sparse_embeddings() {
    SparseTextEmbedding::list_supported_models()
        .par_iter()
        .for_each(|supported_model| {
            let model: SparseTextEmbedding =
                SparseTextEmbedding::try_new(SparseInitOptions::new(supported_model.model.clone()))
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
    let test_model_info = TextEmbedding::get_model_info(&EmbeddingModel::AllMiniLML6V2).unwrap();

    TextEmbedding::try_new(InitOptions::new(test_model_info.model.clone())).unwrap();

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

    // Find the onnx file - it will be any file ending with .onnx
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
    let user_defined_model =
        UserDefinedEmbeddingModel::new(onnx_file, tokenizer_files).with_pooling(Pooling::Mean);

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

            println!("supported_model: {:?}", supported_model);

        let result = TextRerank::try_new(RerankInitOptions::new(supported_model.model.clone()))
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

        let option_a = "panda is an animal";
        let option_b = "The giant panda, sometimes called a panda bear or simply panda, is a bear species endemic to China.";

        assert!(
            results[0].document.as_ref().unwrap() == option_a ||
            results[0].document.as_ref().unwrap() == option_b
        );
        assert!(
            results[1].document.as_ref().unwrap() == option_a ||
            results[1].document.as_ref().unwrap() == option_b
        );
        assert_ne!(results[0].document, results[1].document, "The top two results should be different");

        // Clear the model cache to avoid running out of space on GitHub Actions.
        clean_cache(supported_model.model_code.clone())
    });
}

#[ignore]
#[test]
fn test_user_defined_reranking_large_model() {
    // Setup model to download from Hugging Face
    let cache = hf_hub::Cache::new(std::path::PathBuf::from(fastembed::DEFAULT_CACHE_DIR));
    let api = hf_hub::api::sync::ApiBuilder::from_cache(cache)
        .with_progress(true)
        .build()
        .expect("Failed to build API from cache");
    let model_repo = api.model("rozgo/bge-reranker-v2-m3".to_string());

    // Download the onnx model file
    let onnx_file = model_repo.download("model.onnx").unwrap();
    // Onnx model exceeds the limit of 2GB for a file, so we need to download the data file separately
    let _onnx_data_file = model_repo.get("model.onnx.data").unwrap();

    // OnnxSource::File is used to load the onnx file using onnx session builder commit_from_file
    let onnx_source = OnnxSource::File(onnx_file);

    // Load the tokenizer files
    let tokenizer_files: TokenizerFiles = TokenizerFiles {
        tokenizer_file: read_file_to_bytes(&model_repo.get("tokenizer.json").unwrap()).unwrap(),
        config_file: read_file_to_bytes(&model_repo.get("config.json").unwrap()).unwrap(),
        special_tokens_map_file: read_file_to_bytes(
            &model_repo.get("special_tokens_map.json").unwrap(),
        )
        .unwrap(),

        tokenizer_config_file: read_file_to_bytes(
            &model_repo.get("tokenizer_config.json").unwrap(),
        )
        .unwrap(),
    };

    let model = UserDefinedRerankingModel::new(onnx_source, tokenizer_files);

    let user_defined_reranker =
        TextRerank::try_new_from_user_defined(model, Default::default()).unwrap();

    let documents = vec![
        "Hello, World!",
        "This is an example passage.",
        "fastembed-rs is licensed under Apache-2.0",
        "Some other short text here blah blah blah",
    ];

    let results = user_defined_reranker
        .rerank("Ciao, Earth!", documents.clone(), false, None)
        .unwrap();

    assert_eq!(results.len(), documents.len());
    assert_eq!(results.first().unwrap().index, 0);
}

#[test]
fn test_user_defined_reranking_model() {
    // Constitute the model in order to ensure it's downloaded and cached
    let test_model_info: fastembed::RerankerModelInfo =
        TextRerank::get_model_info(&RerankerModel::JINARerankerV1TurboEn);

    TextRerank::try_new(RerankInitOptions::new(test_model_info.model)).unwrap();

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

    // Find the onnx file - it will be any file in ./onnx ending with .onnx
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
    let user_defined_model = UserDefinedRerankingModel::new(onnx_file, tokenizer_files);

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
            let model: ImageEmbedding =
                ImageEmbedding::try_new(ImageInitOptions::new(supported_model.model.clone()))
                    .unwrap();

            let images = vec!["tests/assets/image_0.png", "tests/assets/image_1.png"];

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

#[test]
#[ignore]
fn test_nomic_embed_vision_v1_5() {
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot_product = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot_product / (norm_a * norm_b)
    }

    fn cosine_similarity_matrix(
        embeddings_a: &[Vec<f32>],
        embeddings_b: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        embeddings_a
            .iter()
            .map(|a| {
                embeddings_b
                    .iter()
                    .map(|b| cosine_similarity(a, b))
                    .collect()
            })
            .collect()
    }

    // Test the NomicEmbedVisionV15 model specifically because it outputs a 3D tensor with a different
    // output key ('last_hidden_state') compared to other models. This test ensures our tensor extraction
    // logic can handle both standard output keys and this model's specific naming convention.
    let image_model = ImageEmbedding::try_new(ImageInitOptions::new(
        fastembed::ImageEmbeddingModel::NomicEmbedVisionV15,
    ))
    .unwrap();

    // tests/assets/image_0.png is a blue cat
    // tests/assets/image_1.png is a red cat
    let images = vec!["tests/assets/image_0.png", "tests/assets/image_1.png"];
    let image_embeddings = image_model.embed(images.clone(), None).unwrap();
    assert_eq!(image_embeddings.len(), images.len());

    let text_model = TextEmbedding::try_new(InitOptions::new(
        fastembed::EmbeddingModel::NomicEmbedTextV15,
    ))
    .unwrap();
    let texts = vec!["green cat", "blue cat", "red cat", "yellow cat", "dog"];
    let text_embeddings = text_model.embed(texts.clone(), None).unwrap();

    // Generate similarity matrix
    let similarity_matrix = cosine_similarity_matrix(&text_embeddings, &image_embeddings);
    // Print the similarity matrix with text labels
    for (i, row) in similarity_matrix.iter().enumerate() {
        println!("{}: {:?}", texts[i], row);
    }

    assert_eq!(text_embeddings.len(), texts.len());
    assert_eq!(text_embeddings[0].len(), 768);
}

fn clean_cache(model_code: String) {
    let repo = Repo::model(model_code);
    let cache_dir = format!("{}/{}", DEFAULT_CACHE_DIR, repo.folder_name());
    fs::remove_dir_all(cache_dir).ok();
}
// This is item "test-environment-aeghhgwpe-pro02a" of the [Aguana corpus](http://argumentation.bplaced.net/arguana/data)
fn get_sample_text() -> String {
    let t = "animals environment general health health general weight philosophy ethics Being vegetarian helps the environment  Becoming a vegetarian is an environmentally friendly thing to do. Modern farming is one of the main sources of pollution in our rivers. Beef farming is one of the main causes of deforestation, and as long as people continue to buy fast food in their billions, there will be a financial incentive to continue cutting down trees to make room for cattle. Because of our desire to eat fish, our rivers and seas are being emptied of fish and many species are facing extinction. Energy resources are used up much more greedily by meat farming than my farming cereals, pulses etc. Eating meat and fish not only causes cruelty to animals, it causes serious harm to the environment and to biodiversity. For example consider Meat production related pollution and deforestation  At Toronto\u{2019}s 1992 Royal Agricultural Winter Fair, Agriculture Canada displayed two contrasting statistics: \u{201c}it takes four football fields of land (about 1.6 hectares) to feed each Canadian\u{201d} and \u{201c}one apple tree produces enough fruit to make 320 pies.\u{201d} Think about it \u{2014} a couple of apple trees and a few rows of wheat on a mere fraction of a hectare could produce enough food for one person! [1]  The 2006 U.N. Food and Agriculture Organization (FAO) report concluded that worldwide livestock farming generates 18% of the planet's greenhouse gas emissions \u{2014} by comparison, all the world's cars, trains, planes and boats account for a combined 13% of greenhouse gas emissions. [2]  As a result of the above point producing meat damages the environment. The demand for meat drives deforestation. Daniel Cesar Avelino of Brazil's Federal Public Prosecution Office says \u{201c}We know that the single biggest driver of deforestation in the Amazon is cattle.\u{201d} This clearing of tropical rainforests such as the Amazon for agriculture is estimated to produce 17% of the world's greenhouse gas emissions. [3] Not only this but the production of meat takes a lot more energy than it ultimately gives us chicken meat production consumes energy in a 4:1 ratio to protein output; beef cattle production requires an energy input to protein output ratio of 54:1.  The same is true with water use due to the same phenomenon of meat being inefficient to produce in terms of the amount of grain needed to produce the same weight of meat, production requires a lot of water. Water is another scarce resource that we will soon not have enough of in various areas of the globe. Grain-fed beef production takes 100,000 liters of water for every kilogram of food. Raising broiler chickens takes 3,500 liters of water to make a kilogram of meat. In comparison, soybean production uses 2,000 liters for kilogram of food produced; rice, 1,912; wheat, 900; and potatoes, 500 liters. [4] This is while there are areas of the globe that have severe water shortages. With farming using up to 70 times more water than is used for domestic purposes: cooking and washing. A third of the population of the world is already suffering from a shortage of water. [5] Groundwater levels are falling all over the world and rivers are beginning to dry up. Already some of the biggest rivers such as China\u{2019}s Yellow river do not reach the sea. [6]  With a rising population becoming vegetarian is the only responsible way to eat.  [1] Stephen Leckie, \u{2018}How Meat-centred Eating Patterns Affect Food Security and the Environment\u{2019}, International development research center  [2] Bryan Walsh, Meat: Making Global Warming Worse, Time magazine, 10 September 2008 .  [3] David Adam, Supermarket suppliers \u{2018}helping to destroy Amazon rainforest\u{2019}, The Guardian, 21st June 2009.  [4] Roger Segelken, U.S. could feed 800 million people with grain that livestock eat, Cornell Science News, 7th August 1997.  [5] Fiona Harvey, Water scarcity affects one in three, FT.com, 21st August 2003  [6] Rupert Wingfield-Hayes, Yellow river \u{2018}drying up\u{2019}, BBC News, 29th July 2004";
    t.to_string()
}
#[test]
fn test_batch_size_does_not_change_output() {
    let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_max_length(384),
    )
    .expect("Create model succesfully");

    let sentences = vec![
        "Books are no more threatened by Kindle than stairs by elevators.",
        "You are who you are when nobody's watching.",
        "An original idea. That can't be too hard. The library must be full of them.",
        "Gaia visited her daughter Mnemosyne, who was busy being unpronounceable.",
        "You can never be overdressed or overeducated.",
        "I don't want to go to heaven. None of my friends are there.",
        "I never travel without my diary. One should always have something sensational to read in the train.",
        "I can resist anything except temptation.",
        "It is absurd to divide people into good and bad. People are either charming or tedious."
    ];

    let single_batch = model
        .embed(sentences.clone(), None)
        .expect("create successfully");
    let small_batch = model
        .embed(sentences, Some(3))
        .expect("create successfully");

    assert_eq!(single_batch.len(), small_batch.len());
    for (a, b) in single_batch.into_iter().zip(small_batch.into_iter()) {
        assert!(a == b, "Expect each sentence embedding are equal.");
    }
}

#[test]
fn test_bgesmallen1point5_match_python_counterpart() {
    let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::BGESmallENV15).with_max_length(384),
    )
    .expect("Create model succesfully");
    let text = get_sample_text();

    // baseline is generated in python using Xenova/bge-small-en-v1.5.onnx
    // Tokenize with python SentenceTransformer("BAAI/bge-small-en-v1.5") default tokenizer
    // with (text, padding="max_length",max_length=384,truncation=True, return_tensors="np").
    // Normalized and pooled with SentenceTransformer("BAAI/bge-small-en-v1.5") default pooling settings.
    // we only take a 10 items to keep the test file polite
    let baseline: Vec<f32> = vec![
        4.208_193_7e-2,
        -2.748_133_2e-2,
        6.742_810_5e-2,
        2.282_790_5e-2,
        4.257_192e-2,
        -4.163_983_5e-2,
        6.814_807_4e-6,
        -9.643_933e-3,
        -3.475_583e-3,
        6.606_272e-2,
    ];

    let embeddings = model.embed(vec![text], None).expect("create successfully");
    let tolerance: f32 = 1e-3;
    for (expected, actual) in embeddings[0]
        .clone()
        .into_iter()
        .take(baseline.len())
        .zip(baseline.into_iter())
    {
        assert!((expected - actual).abs() < tolerance);
    }
}

#[test]
fn test_allminilml6v2_match_python_counterpart() {
    let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_max_length(384),
    )
    .expect("Create model succesfully");

    let text = get_sample_text();

    // baseline is generated in python using qdrant/all-mini-lm-l6-v2.onnx
    // Tokenizer with python SentenceTransformer("all-mini-lm-l6-v2") default tokenizer
    // with (text, padding="max_length",max_length=384,truncation=True, return_tensors="np").
    // Normalized and pooled with SentenceTransformer("all-mini-lm-l6-v2") default pooling settings.
    // we only take a 10 items to keep the test file polite
    let baseline: Vec<f32> = vec![
        3.510_517_6e-2,
        1.046_043e-2,
        3.767_998_5e-2,
        7.073_633_4e-2,
        9.097_775e-2,
        -2.507_714_7e-2,
        -2.214_382e-2,
        -1.016_435_9e-2,
        4.660_127_3e-2,
        7.431_366e-2,
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
