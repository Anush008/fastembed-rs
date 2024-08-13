use std::path::Path;

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::common::DEFAULT_CACHE_DIR;
use crate::pooling::Pooling;
use crate::sparse_text_embedding::SparseTextEmbedding;
use crate::{
    read_file_to_bytes, EmbeddingModel, InitOptions, InitOptionsUserDefined, RerankInitOptions,
    RerankInitOptionsUserDefined, RerankerModel, SparseInitOptions, TextEmbedding, TextRerank,
    TokenizerFiles, UserDefinedEmbeddingModel, UserDefinedRerankingModel,
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

        assert_eq!(results.len(), documents.len());
        assert_eq!(results[0].document.as_ref().unwrap(), "panda is an animal");
        assert_eq!(results[1].document.as_ref().unwrap(), "The giant panda, sometimes called a panda bear or simply panda, is a bear species endemic to China.");
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
    let user_defined_model = UserDefinedRerankingModel {
        onnx_file,
        tokenizer_files,
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

// This is item "test-environment-aeghhgwpe-pro02a" of the [Aguana corpus](http://argumentation.bplaced.net/arguana/data)
fn get_sample_text() -> String {
    let t = "animals environment general health health general weight philosophy ethics Being vegetarian helps the environment  Becoming a vegetarian is an environmentally friendly thing to do. Modern farming is one of the main sources of pollution in our rivers. Beef farming is one of the main causes of deforestation, and as long as people continue to buy fast food in their billions, there will be a financial incentive to continue cutting down trees to make room for cattle. Because of our desire to eat fish, our rivers and seas are being emptied of fish and many species are facing extinction. Energy resources are used up much more greedily by meat farming than my farming cereals, pulses etc. Eating meat and fish not only causes cruelty to animals, it causes serious harm to the environment and to biodiversity. For example consider Meat production related pollution and deforestation  At Toronto\u{2019}s 1992 Royal Agricultural Winter Fair, Agriculture Canada displayed two contrasting statistics: \u{201c}it takes four football fields of land (about 1.6 hectares) to feed each Canadian\u{201d} and \u{201c}one apple tree produces enough fruit to make 320 pies.\u{201d} Think about it \u{2014} a couple of apple trees and a few rows of wheat on a mere fraction of a hectare could produce enough food for one person! [1]  The 2006 U.N. Food and Agriculture Organization (FAO) report concluded that worldwide livestock farming generates 18% of the planet's greenhouse gas emissions \u{2014} by comparison, all the world's cars, trains, planes and boats account for a combined 13% of greenhouse gas emissions. [2]  As a result of the above point producing meat damages the environment. The demand for meat drives deforestation. Daniel Cesar Avelino of Brazil's Federal Public Prosecution Office says \u{201c}We know that the single biggest driver of deforestation in the Amazon is cattle.\u{201d} This clearing of tropical rainforests such as the Amazon for agriculture is estimated to produce 17% of the world's greenhouse gas emissions. [3] Not only this but the production of meat takes a lot more energy than it ultimately gives us chicken meat production consumes energy in a 4:1 ratio to protein output; beef cattle production requires an energy input to protein output ratio of 54:1.  The same is true with water use due to the same phenomenon of meat being inefficient to produce in terms of the amount of grain needed to produce the same weight of meat, production requires a lot of water. Water is another scarce resource that we will soon not have enough of in various areas of the globe. Grain-fed beef production takes 100,000 liters of water for every kilogram of food. Raising broiler chickens takes 3,500 liters of water to make a kilogram of meat. In comparison, soybean production uses 2,000 liters for kilogram of food produced; rice, 1,912; wheat, 900; and potatoes, 500 liters. [4] This is while there are areas of the globe that have severe water shortages. With farming using up to 70 times more water than is used for domestic purposes: cooking and washing. A third of the population of the world is already suffering from a shortage of water. [5] Groundwater levels are falling all over the world and rivers are beginning to dry up. Already some of the biggest rivers such as China\u{2019}s Yellow river do not reach the sea. [6]  With a rising population becoming vegetarian is the only responsible way to eat.  [1] Stephen Leckie, \u{2018}How Meat-centred Eating Patterns Affect Food Security and the Environment\u{2019}, International development research center  [2] Bryan Walsh, Meat: Making Global Warming Worse, Time magazine, 10 September 2008 .  [3] David Adam, Supermarket suppliers \u{2018}helping to destroy Amazon rainforest\u{2019}, The Guardian, 21st June 2009.  [4] Roger Segelken, U.S. could feed 800 million people with grain that livestock eat, Cornell Science News, 7th August 1997.  [5] Fiona Harvey, Water scarcity affects one in three, FT.com, 21st August 2003  [6] Rupert Wingfield-Hayes, Yellow river \u{2018}drying up\u{2019}, BBC News, 29th July 2004";
    t.to_string()
}
#[test]
fn test_batch_size_does_not_change_output() {
    let model = TextEmbedding::try_new(InitOptions {
        model_name: EmbeddingModel::AllMiniLML6V2,
        max_length: 384,
        show_download_progress: true,
        ..Default::default()
    })
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
    let model = TextEmbedding::try_new(InitOptions {
        model_name: EmbeddingModel::BGESmallENV15,
        max_length: 384,
        show_download_progress: true,
        ..Default::default()
    })
    .expect("Create model succesfully");
    let text = get_sample_text();

    // baseline is generated in python using Xenova/bge-small-en-v1.5.onnx
    // Tokenize with python SentenceTransformer("BAAI/bge-small-en-v1.5") default tokenizer
    // with (text, padding="max_length",max_length=384,truncation=True, return_tensors="np").
    // Normalized and pooled with SentenceTransformer("BAAI/bge-small-en-v1.5") default pooling settings.
    // we only take a 10 items to keep the test file polite
    let baseline: Vec<f32> = vec![
        4.20819372e-02,
        -2.74813324e-02,
        6.74281046e-02,
        2.28279047e-02,
        4.25719209e-02,
        -4.16398346e-02,
        6.81480742e-06,
        -9.64393280e-03,
        -3.47558293e-03,
        6.60627186e-02,
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

#[test]
fn test_allminilml6v2_match_python_counterpart() {
    let model = TextEmbedding::try_new(InitOptions {
        model_name: EmbeddingModel::AllMiniLML6V2,
        max_length: 384,
        show_download_progress: true,
        ..Default::default()
    })
    .expect("Create model succesfully");

    let text = get_sample_text();

    // baseline is generated in python using qdrant/all-mini-lm-l6-v2.onnx
    // Tokenizer with python SentenceTransformer("all-mini-lm-l6-v2") default tokenizer
    // with (text, padding="max_length",max_length=384,truncation=True, return_tensors="np").
    // Normalized and pooled with SentenceTransformer("all-mini-lm-l6-v2") default pooling settings.
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
