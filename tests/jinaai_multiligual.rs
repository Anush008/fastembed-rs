use fastembed::{RerankInitOptions, RerankerModel, TextRerank};

#[test]
fn multiligual_reranker() {
    let model = TextRerank::try_new(RerankInitOptions {
        model_name: RerankerModel::JINARerankerV2BaseMultiligual,
        show_download_progress: true,
        ..Default::default()
    })
    .unwrap();

    let query = "Organic skincare products for sensitive skin";
    let documents = vec![
        "Organic skincare for sensitive skin with aloe vera and chamomile.",
        "New makeup trends focus on bold colors and innovative techniques",
        "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille",
        "Neue Make-up-Trends setzen auf kräftige Farben und innovative Techniken",
        "Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla",
        "Las nuevas tendencias de maquillaje se centran en colores vivos y técnicas innovadoras",
        "针对敏感肌专门设计的天然有机护肤产品",
        "新的化妆趋势注重鲜艳的颜色和创新的技巧",
        "敏感肌のために特別に設計された天然有機スキンケア製品",
        "新しいメイクのトレンドは鮮やかな色と革新的な技術に焦点を当てています",
    ];

    let result = model.rerank(query, documents, true, None);
    assert!(result.is_ok());

    if let Ok(res) = result {
        for doc in res.iter() {
            assert!(doc.document.is_some());
            println!(
                "ID: {}, Score: {:+.3}, Text: {}",
                doc.index,
                doc.score,
                doc.document.as_ref().unwrap()
            );
        }
    }
}
