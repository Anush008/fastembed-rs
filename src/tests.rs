use std::path::Path;

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::common::DEFAULT_CACHE_DIR;
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


#[test]
fn compare_python_embeddings() {
    let model = TextEmbedding::try_new(InitOptions {
        model_name: EmbeddingModel::AllMiniLML6V2,
        max_length:384,
        show_download_progress: true,
        ..Default::default()
    }).expect("Create model succesfully");

    // This is item "test-environment-aeghhgwpe-pro02a" of the [Aguana corpus](http://argumentation.bplaced.net/arguana/data)
    let text = "animals environment general health health general weight philosophy ethics Being vegetarian helps the environment  Becoming a vegetarian is an environmentally friendly thing to do. Modern farming is one of the main sources of pollution in our rivers. Beef farming is one of the main causes of deforestation, and as long as people continue to buy fast food in their billions, there will be a financial incentive to continue cutting down trees to make room for cattle. Because of our desire to eat fish, our rivers and seas are being emptied of fish and many species are facing extinction. Energy resources are used up much more greedily by meat farming than my farming cereals, pulses etc. Eating meat and fish not only causes cruelty to animals, it causes serious harm to the environment and to biodiversity. For example consider Meat production related pollution and deforestation  At Toronto\u{2019}s 1992 Royal Agricultural Winter Fair, Agriculture Canada displayed two contrasting statistics: \u{201c}it takes four football fields of land (about 1.6 hectares) to feed each Canadian\u{201d} and \u{201c}one apple tree produces enough fruit to make 320 pies.\u{201d} Think about it \u{2014} a couple of apple trees and a few rows of wheat on a mere fraction of a hectare could produce enough food for one person! [1]  The 2006 U.N. Food and Agriculture Organization (FAO) report concluded that worldwide livestock farming generates 18% of the planet's greenhouse gas emissions \u{2014} by comparison, all the world's cars, trains, planes and boats account for a combined 13% of greenhouse gas emissions. [2]  As a result of the above point producing meat damages the environment. The demand for meat drives deforestation. Daniel Cesar Avelino of Brazil's Federal Public Prosecution Office says \u{201c}We know that the single biggest driver of deforestation in the Amazon is cattle.\u{201d} This clearing of tropical rainforests such as the Amazon for agriculture is estimated to produce 17% of the world's greenhouse gas emissions. [3] Not only this but the production of meat takes a lot more energy than it ultimately gives us chicken meat production consumes energy in a 4:1 ratio to protein output; beef cattle production requires an energy input to protein output ratio of 54:1.  The same is true with water use due to the same phenomenon of meat being inefficient to produce in terms of the amount of grain needed to produce the same weight of meat, production requires a lot of water. Water is another scarce resource that we will soon not have enough of in various areas of the globe. Grain-fed beef production takes 100,000 liters of water for every kilogram of food. Raising broiler chickens takes 3,500 liters of water to make a kilogram of meat. In comparison, soybean production uses 2,000 liters for kilogram of food produced; rice, 1,912; wheat, 900; and potatoes, 500 liters. [4] This is while there are areas of the globe that have severe water shortages. With farming using up to 70 times more water than is used for domestic purposes: cooking and washing. A third of the population of the world is already suffering from a shortage of water. [5] Groundwater levels are falling all over the world and rivers are beginning to dry up. Already some of the biggest rivers such as China\u{2019}s Yellow river do not reach the sea. [6]  With a rising population becoming vegetarian is the only responsible way to eat.  [1] Stephen Leckie, \u{2018}How Meat-centred Eating Patterns Affect Food Security and the Environment\u{2019}, International development research center  [2] Bryan Walsh, Meat: Making Global Warming Worse, Time magazine, 10 September 2008 .  [3] David Adam, Supermarket suppliers \u{2018}helping to destroy Amazon rainforest\u{2019}, The Guardian, 21st June 2009.  [4] Roger Segelken, U.S. could feed 800 million people with grain that livestock eat, Cornell Science News, 7th August 1997.  [5] Fiona Harvey, Water scarcity affects one in three, FT.com, 21st August 2003  [6] Rupert Wingfield-Hayes, Yellow river \u{2018}drying up\u{2019}, BBC News, 29th July 2004";

    // baseline is generated in python using qdrant/all-mini-lm-l6-v2.onnx
    let baseline: Vec<f32> = vec![
        3.51051763e-02,  1.04604298e-02,  3.76799852e-02,  7.07363337e-02,
        9.09777507e-02, -2.50771474e-02, -2.21438203e-02, -1.01643587e-02,
        4.66012731e-02,  7.43136629e-02,  3.81156653e-02, -9.31796506e-02,
       -6.91598505e-02,  2.26031197e-03,  8.37039575e-03,  1.12127541e-02,
        4.59930897e-02,  9.98165645e-03, -9.06514674e-02,  6.60405084e-02,
        2.90363748e-02,  3.66936028e-02, -1.32901650e-02,  2.58240085e-02,
       -9.10237357e-02, -9.41415802e-02, -5.11121228e-02, -5.45051582e-02,
       -1.02085494e-01,  2.44955495e-02, -3.88111384e-03,  3.23626734e-02,
        3.60895582e-02,  7.69520339e-05, -2.35975832e-02,  6.61284383e-03,
        1.14524141e-01, -7.93305412e-02, -3.84794292e-03,  4.06287611e-02,
       -4.33373488e-02, -6.96017221e-02,  1.67226363e-02, -3.80154066e-02,
        6.02646498e-03,  2.22117919e-03, -1.51069155e-02, -3.68320011e-03,
        1.05054993e-02, -1.23643398e-01,  1.26858708e-02,  3.31539325e-02,
       -7.92319924e-02, -5.65977357e-02,  4.78964746e-02, -1.21093743e-01,
        4.58319159e-03, -6.07872456e-02, -3.30011584e-02,  3.23408432e-02,
        7.18807206e-02, -4.72615063e-02, -1.26546444e-02,  5.74186957e-03,
        8.38401541e-02, -1.83143374e-02, -7.34603480e-02,  5.57967350e-02,
       -8.79004523e-02, -7.68315862e-04,  1.40642971e-02, -5.51650487e-02,
       -3.65579017e-02, -6.83044940e-02, -2.57119145e-02,  5.65035045e-02,
        2.15697233e-02,  4.12733369e-02,  1.19941108e-01, -3.22536603e-02,
        6.40394315e-02,  9.86634847e-03,  3.52630997e-03, -7.60268643e-02,
        3.68730202e-02, -4.12338562e-02, -3.15781753e-03,  2.62700599e-02,
        6.40113978e-03,  3.41433361e-02, -3.76855135e-02, -6.15284927e-02,
        9.55483839e-02,  7.54997134e-02,  1.25792855e-02,  6.02337383e-02,
        1.34278778e-02, -9.86816287e-02,  4.98550236e-02, -1.93966012e-02,
       -2.33738981e-02, -3.56944874e-02,  5.74744400e-03, -4.96314093e-02,
        8.24571296e-04,  1.68628339e-02, -1.29957780e-01,  4.58944179e-02,
        4.00895551e-02,  8.89468193e-02, -6.99186698e-02,  5.58378175e-02,
       -1.06258005e-01,  9.29498747e-02,  3.99756283e-02,  4.46220301e-02,
        4.88199219e-02, -8.97670835e-02, -1.28866658e-02, -4.10931418e-03,
        7.35270837e-03,  2.33679861e-02, -1.05916904e-02,  3.50892171e-02,
        9.64816958e-02, -4.57886383e-02,  4.26217839e-02,  4.45787535e-33,
       -4.79732603e-02, -7.55402520e-02,  7.38559291e-02, -9.96180028e-02,
        2.18187980e-02,  4.43287892e-03, -8.54854807e-02,  3.93715091e-02,
        8.10520053e-02,  1.80339161e-02, -1.75382495e-02, -3.75687554e-02,
        5.99994846e-02,  2.47306880e-02,  4.51350324e-02, -5.05464673e-02,
       -7.44671607e-03, -2.51467489e-02,  2.55274177e-02,  1.41209867e-02,
       -1.60762798e-02, -3.37061174e-02,  3.08316965e-02,  5.34818061e-02,
        1.61754396e-02, -3.09070870e-02,  2.05388181e-02, -1.16848901e-01,
        6.60440745e-03,  3.52322496e-03,  3.31513658e-02, -8.33946019e-02,
       -4.24109809e-02, -4.75227758e-02, -3.94212604e-02, -4.97223213e-02,
        1.74339656e-02,  3.34502757e-02, -6.77237585e-02,  4.77080680e-02,
       -1.79492719e-02,  5.11257816e-03,  8.47059712e-02, -5.84509782e-02,
        4.51399386e-02,  1.06090698e-02,  4.23551053e-02,  6.25070781e-02,
       -6.81258887e-02,  1.98892150e-02, -3.84888165e-02,  4.61487509e-02,
        7.10240304e-02, -4.88134213e-02,  2.96834148e-02, -3.12856138e-02,
        2.60314960e-02, -3.52520682e-02, -7.00081736e-02, -1.46203404e-02,
       -5.08719422e-02,  6.48945794e-02, -2.28208434e-02,  1.12276636e-02,
       -1.67943481e-02, -2.08180938e-02,  5.47553487e-02, -9.45482776e-03,
       -2.68401541e-02,  8.00292715e-02,  3.14962156e-02, -4.73631546e-02,
       -1.36573762e-02, -6.63933605e-02,  7.25492276e-03, -2.17820201e-02,
       -2.66148765e-02,  9.31434985e-03, -1.10124201e-01,  6.58026487e-02,
       -3.00105549e-02,  5.88533506e-02, -2.49483176e-02, -4.95315567e-02,
       -6.71937317e-02,  3.24900560e-02, -2.04695389e-02, -3.34003232e-02,
        1.23592637e-01, -1.65716931e-02,  2.00482644e-02,  8.26424081e-03,
       -1.23888822e-02, -2.73734480e-02, -5.49323531e-03, -4.47739517e-33,
       -2.95793470e-02, -1.66377833e-03,  5.38532389e-03,  3.95982973e-02,
        2.12997869e-02, -9.62312147e-02,  9.29667521e-03, -3.34646069e-02,
        1.09288469e-02, -8.76594335e-02, -4.90870439e-02,  7.49579668e-02,
       -1.59861464e-02,  3.29942480e-02,  1.93475261e-02,  1.28691904e-02,
       -3.21327001e-02, -1.05767399e-02, -6.04807325e-02, -8.56276602e-02,
       -3.06638498e-02,  3.80774662e-02,  5.69422208e-02,  1.81423221e-02,
       -7.01246709e-02,  5.10974526e-02, -3.55399102e-02,  6.02702126e-02,
        8.00985321e-02, -7.05354959e-02, -1.03998091e-02,  9.41087082e-02,
       -5.92826158e-02, -8.06960240e-02, -3.76178138e-02, -2.79329792e-02,
        2.09042337e-02,  7.70026306e-03, -9.72199347e-03,  3.59404162e-02,
        7.51863420e-02,  1.79292187e-02, -6.34065568e-02,  1.71660935e-03,
       -4.99649867e-02,  1.65790860e-02,  3.34282219e-02, -1.83252506e-02,
        7.98043013e-02,  4.19670418e-02,  1.34012520e-01,  2.62213517e-02,
       -6.87130764e-02,  2.53241360e-02,  5.69513366e-02, -8.02854251e-04,
        7.59661570e-02,  9.35709663e-03, -5.76349683e-02, -7.44777033e-03,
        2.56016199e-02,  7.64308348e-02, -5.37360087e-03,  1.60237588e-02,
        2.63605937e-02, -6.94938237e-03, -2.11461391e-02,  1.58795866e-03,
        3.94684598e-02, -1.19603416e-02, -6.80993125e-02,  2.67107822e-02,
       -3.87540273e-02, -3.65402587e-02, -8.27274099e-02,  1.05797641e-01,
        4.18990292e-02, -6.45376891e-02,  1.89866836e-03,  4.17896025e-02,
        1.34420849e-03, -5.48073165e-02,  8.10993463e-02, -7.92252738e-03,
        2.97114495e-02, -4.45416532e-02, -7.20333830e-02, -3.09838280e-02,
        4.23357785e-02,  1.33295074e-01, -1.33586153e-01, -7.87638724e-02,
        1.10902300e-03,  4.47123572e-02,  2.10078824e-02, -5.87940221e-08,
        3.81310540e-03, -3.10390238e-02,  2.35545468e-02,  1.74355842e-02,
        2.67300825e-03,  3.66347353e-03,  1.23544177e-02,  3.21473926e-02,
        4.92791720e-02,  7.21906126e-02, -3.43316863e-03,  1.05575971e-01,
       -2.29346529e-02,  1.18679814e-01, -3.90798897e-02,  9.11866780e-04,
        4.85025942e-02, -6.60484210e-02, -2.30571553e-02,  1.03314705e-02,
       -6.10778257e-02, -7.41297472e-03, -1.41818345e-01, -6.76135495e-02,
        6.92776144e-02, -7.13223591e-02, -3.19796838e-02,  3.25076953e-02,
        8.41270834e-02, -2.10842788e-02,  3.39241438e-02,  3.98571510e-03,
       -2.29800697e-02, -8.23784433e-03,  1.13272443e-02, -8.72365385e-02,
       -1.34138642e-02, -1.99352074e-02, -1.10229664e-02,  4.80700983e-03,
        1.16155427e-02,  4.66864929e-02,  5.68014290e-03,  1.52975358e-02,
        1.33216362e-02, -5.64585254e-02, -4.54223007e-02,  6.13160841e-02,
        2.72959080e-02,  1.41720509e-03, -4.81577590e-02, -3.52402851e-02,
        5.11605963e-02, -4.39299159e-02, -3.29744024e-03,  6.72190730e-03,
        2.01416835e-02, -4.58171926e-02,  2.07999609e-02,  3.75747420e-02,
        8.51716474e-02, -7.08120465e-02,  2.96163429e-02,  4.38641012e-02
    ];

    let embeddings = model.embed(vec![text], None).expect("create successfully");
    let tolerance: f32 = 1e-6;
    for (expected, actual) in embeddings[0].clone().into_iter().zip(baseline.into_iter()) {
        assert!((expected - actual).abs() < tolerance);
    }
}