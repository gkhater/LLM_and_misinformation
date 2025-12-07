# Test Log: build_wiki_corpus & compileall

- Command: `python scripts/build_wiki_corpus.py --output data/wiki_passages.tsv --pages 200 --lang en --user-agent "llm-misinfo-benchmark/0.1"`
- Result: **pass**; wrote 2,328 passages to `data/wiki_passages.tsv`.
- Sample rows show `id<TAB>title [SEP] chunk`.

- Command: `python -m compileall src`
- Result: **pass** after adding corpora/retrieval tweaks.
