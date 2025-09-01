# NLPtesting1
Análise de Sentimento Multilíngue de Reviews de Apps

Este projeto realiza classificação de reviews de aplicativos móveis em múltiplos idiomas (inglês, português e espanhol) em **positivo** ou **negativo**, usando modelos de Machine Learning.

## Tecnologias utilizadas
- Python 3.11
- Pandas, NLTK, Langdetect
- Scikit-learn (LogisticRegression, LinearSVC, GridSearchCV)
- TF-IDF para vetorização de texto

## Pré-processamento
1. Remoção de reviews vazias ou muito curtas (<3 palavras).
2. Detecção de linguagem (`langdetect`) e filtragem para `en`, `pt`, `es`.
3. Limpeza do texto:
   - Lowercase
   - Remoção de caracteres especiais
   - Remoção de stopwords por idioma
4. Criação da coluna `pos/neg`:
   - Rating >= 4 → `pos`
   - Rating <= 3 → `neg`
5. Balanceamento das classes por oversampling de reviews positivas.

## Vetorização
- TF-IDF com `max_features=10000` e `ngram_range=(1,3)`.

## Modelos
- **Logistic Regression**
  - GridSearchCV com parâmetros `C` e `class_weight`
- **LinearSVC**
  - GridSearchCV com parâmetros `C` e `class_weight`
- Avaliação via `classification_report` (f1-score, precisão, recall).

## Observações
- Oversampling pode causar overfitting.
- Stopwords são carregadas a cada linha, otimização possível.
- GridSearch define automaticamente os melhores hiperparâmetros.
- Pode-se comparar desempenho com modelos default e best estimator.

## Resultados
- O script imprime:
  - Classification report para cada modelo
  - Melhores parâmetros encontrados via GridSearchCV

## Uso
1. Baixar dataset `multilingual_mobile_app_reviews_2025.csv`.
2. Rodar o script Python.
3. Conferir métricas no console.
