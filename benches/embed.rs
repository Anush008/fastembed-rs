use codspeed_criterion_compat::{criterion_group, criterion_main, Criterion};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::time::Duration;

fn criterion_benchmark(c: &mut Criterion) {
    let model: TextEmbedding = TextEmbedding::try_new(InitOptions {
        model_name: EmbeddingModel::BGESmallENV15,
        show_download_progress: false,
        ..Default::default()
    })
    .unwrap();

    let short_texts = [
        "Hello, World!",
        "This is an example passage.",
        "fastembed-rs is licensed under Apache-2.0",
        "Some other short text here blah blah blah",
    ]
    .iter()
    .cycle()
    .take(100)
    .map(|x| x.to_string())
    .collect::<Vec<_>>();

    let long_texts = [
        "Contribution shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, submitted
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as Not a Contribution.
",
        "Contributor shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.
",
        "Derivative Works shall mean any work, whether in Source or Object form,
        that is based on (or derived from) the Work and for which the editorial
        revisions, annotations, elaborations, or other modifications represent,
        as a whole, an original work of authorship. For the purposes of this
        License, Derivative Works shall not include works that remain
        separable from, or merely link (or bind by name) to the interfaces of,
        the Work and Derivative Works thereof.",
    ]
    .iter()
    .cycle()
    .take(20)
    .map(|x| x.to_string())
    .collect::<Vec<_>>();

    c.bench_function("embed BGESmallENV15 short", |b| {
        b.iter(|| model.embed(short_texts.clone(), None).unwrap())
    });
    c.bench_function("embed BGESmallENV15 long", |b| {
        b.iter(|| model.embed(long_texts.clone(), None).unwrap())
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(60));
    targets = criterion_benchmark,
);
criterion_main!(benches);
