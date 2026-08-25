# Book Recommender

Sistema de recomendação de livros baseado em similaridade de conteúdo, usando embeddings de linguagem natural (Sentence Transformers) sobre gêneros literários extraídos do dataset Goodbooks-10k.

## Problema

Dado um livro que o usuário gostou, recomendar outros livros com perfil de gênero semelhante — sem depender apenas de tags exatas, mas de similaridade semântica entre elas.

## Dataset

[Goodbooks-10k](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k) (Kaggle) — ~10.000 livros do Goodreads, com metadados, tags de usuários e ~6M avaliações.

Arquivos utilizados:
- `books.csv` — metadados (título, autor, avaliação média)
- `tags.csv` — dicionário de tags
- `book_tags.csv` — relação livro ↔ tag ↔ contagem de votos
- `ratings.csv` — avaliações individuais (usado para validação)

## Abordagem

**Content-based filtering** usando embeddings semânticos:

1. **Limpeza de tags**: o Goodreads permite tags livres de usuários, então a maioria (`to-read`, `owned`, `kindle`...) não representa gênero literário. Foi construída uma whitelist de ~29 gêneros válidos para filtrar o ruído.
2. **Unificação de sinônimos**: tags equivalentes com grafias diferentes (`sci-fi`/`science-fiction`, `nonfiction`/`non-fiction`) foram unificadas.
3. **Agregação por livro**: as 5 tags mais votadas (dentro da whitelist) foram combinadas em uma string de texto por livro.
4. **Embeddings**: cada string de gênero foi transformada em um vetor de 384 dimensões usando o modelo `all-MiniLM-L6-v2` (Sentence Transformers).
5. **Similaridade**: comparação por similaridade de cosseno entre todos os pares de livros.
6. **Validação cruzada**: a média de rating calculada a partir de `ratings.csv` foi comparada com `average_rating` (já presente no dataset) para checar consistência dos dados.

## Resultado

Dado um livro de entrada, a função retorna os N livros mais similares por gênero, ordenados por score de similaridade. Exemplo: recomendações para *The Hobbit* trouxeram *Peter Pan*, *Alice no País das Maravilhas* e *101 Dálmatas* — todos clássicos infantojuvenis de fantasia/aventura, validando que o modelo captura similaridade semântica coerente, não apenas correspondência exata de palavras.

## Desafios encontrados

- **Chaves de merge inconsistentes**: o dataset usa três IDs diferentes para livro (`id`, `book_id`, `goodreads_book_id`) em contextos distintos, exigindo atenção redobrada nos merges entre tabelas.
- **Duplicação por sinônimos**: a unificação de tags equivalentes gerava duplicatas no texto agregado até a correção (soma de contagens antes da agregação, em vez de depois).
- **Ruído nas tags de usuário**: resolvido com whitelist manual de gêneros válidos.

## Limitações

- Recomendação baseada apenas em **gênero/tags**, não em sinopse ou enredo — dois livros de "fantasy young-adult" podem ter histórias completamente diferentes.
- Livros com exatamente as mesmas top-5 tags recebem similaridade máxima (1.0), mesmo sem terem sido comparados por conteúdo real.

## Próximos passos

- Incorporar **sinopses reais** (via API de livros) para embeddings mais ricos semanticamente.
- Implementar **filtragem colaborativa** com `ratings.csv` (recomendação baseada em usuários com gostos parecidos, não só conteúdo do livro).
- Combinar as duas abordagens em um sistema **híbrido**.
- Evoluir para um **agente com LLM**: em vez de buscar por título exato, o usuário descreve o que quer em linguagem natural (ex: "gosto de ficção científica com crítica social") e o agente busca via RAG na base de embeddings, retornando recomendações com justificativa textual.

## Como rodar

### Pré-requisitos
```bash
pip install pandas numpy scikit-learn sentence-transformers
```

### Estrutura de pastas

book-recommender/
├── data/
│   ├── kaggle/
│   │   ├── books.csv
│   │   ├── tags.csv
│   │   ├── book_tags.csv
│   │   └── ratings.csv
│   ├── books_final.csv       (gerado)
│   └── embeddings.npy        (gerado)
├── 01_preparacao_dados.ipynb
├── 02_embeddings_recomendacao.ipynb
└── README.md



### Passos
1. Baixe o dataset [Goodbooks-10k](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k) e coloque os CSVs em `data/`.
2. Rode `01_preparacao_dados.ipynb` de ponta a ponta — gera `data/books_final.csv`.
3. Rode `02_embeddings_recomendacao.ipynb` de ponta a ponta — gera `data/embeddings.npy` e disponibiliza a função `recomendar(titulo, n=5)`.
4. Use a função:
```python
recomendar('The Hobbit', n=5)
```

## Stack

Python · pandas · scikit-learn · Sentence Transformers (`all-MiniLM-L6-v2`)