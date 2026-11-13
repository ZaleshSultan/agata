# Book curation for ABC v1

The private source is Lyubov Polivalina's *Encyclopedia of the Young Family*, Nauchnaya Kniga, 2013: a 98-page Litres preview, not a complete book. No source extracts or derived question/answer text are stored in this tracked note.

New private files live under `books/processed/abc_v1/`. Earlier datasets and results are preserved. `reviewed_claims.jsonl` records sixteen reviewed source propositions with complete verbatim evidence, PDF page, chapter, section, chunk identifier, source hash, and review notes. Thirty-two training examples use two different tasks per proposition: content reproduction and paraphrase. Four additional training examples teach the verified limits of the available preview. Those four are not additional content facts.

The curator read the complete supporting page passages and checked sentence boundaries, negation, source fidelity, and answer scope. No automatic candidate was promoted without review. Medical, legal, coercive, and unsupported modern factual advice was excluded. Two retained evidence sentences contain unrelated stereotyping or a causal guarantee; the corresponding targets teach only the specifically requested neutral examples and omit those claims. This distinction is recorded in their private review notes. Curation is AI-assisted, not an independent human review.

Training and validation use chapter I only. The entire Introduction is excluded. Validation contains four new questions about known training propositions and is for loss monitoring only; it cannot serve as evidence of unseen-fact learning or guide configuration selection.

The book benchmark has ten new questions about taught facts, four compositions of two taught facts, four questions from the excluded Introduction, and four requests whose answers cannot be verified from the available preview. Eight exact training questions form a separate memorization diagnostic. Exact normalized training, validation, and independent benchmark questions do not overlap. Known-fact probes deliberately share facts with training; that is the object of this test, not independent knowledge discovery. Compositions and excluded-chapter results are reported separately from the primary score. All semantic rubrics were fixed before model outputs were inspected.

Each benchmark rubric requires the specified meaning and rejects contradictory claims or fabricated details. Keywords remain diagnostics. A refusal on a known-fact question is not successful recall; a clean refusal about excluded material is distinguished from a hallucination. No inference is made that unavailable material is absent from the full edition.

This corpus has sixteen distinct propositions but fewer independent themes. Many are ordinary relationship ideas or statements about the author's views. Successful paraphrases would establish narrow transfer across wording; exact recall, source-example memorization, or refusal alone would not establish broad new knowledge. The small, correlated sample cannot justify a population-level superiority claim.

Private reproducibility files: `build_book_private.py`, `finalize_metadata_private.py`, `curation_summary.json`, the five JSONL files above. The summary records counts, evidence checks, limitations, and SHA-256 hashes. Raw or derived book data must remain ignored and must not be committed or uploaded.
