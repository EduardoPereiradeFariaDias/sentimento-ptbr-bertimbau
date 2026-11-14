Untitled0.ipynb
# Análise de sentimento em avaliações de e-commerce com BERTimbau

Classificador de sentimento (negativo, neutro, positivo) para avaliações de produtos em português, feito com fine-tuning do [BERTimbau Base](https://huggingface.co/neuralmind/bert-base-portuguese-cased) e comparado com um baseline clássico (TF-IDF + regressão logística).

**Modelo publicado:** [Eduardo211000/bertimbau-sentimento-b2w](https://huggingface.co/Eduardo211000/bertimbau-sentimento-b2w)

## Resultados

Avaliação em 3.000 exemplos de teste, separados antes do treino.

| Modelo | Accuracy | F1 macro |
|---|---|---|
| TF-IDF + regressão logística (baseline) | 0,830 | 0,658 |
| **BERTimbau (fine-tuning)** | **0,855** | **0,744** |

| Classe | Precisão | Recall | F1 |
|---|---|---|---|
| negativo | 0,85 | 0,92 | 0,88 |
| neutro | 0,52 | 0,37 | 0,43 |
| positivo | 0,90 | 0,93 | 0,91 |

O principal ganho sobre o baseline está na classe **neutra**: o recall subiu de 0,15 para 0,37 e o F1 de 0,23 para 0,43. Como a base é desbalanceada (cerca de 61% positivo, 26% negativo e 12% neutro), o F1 macro é a métrica principal; a accuracy sozinha engana.

Matriz de confusão do BERTimbau (linhas = real, colunas = previsto):

| | negativo | neutro | positivo |
|---|---|---|---|
| **negativo** | 694 | 39 | 23 |
| **neutro** | 74 | 139 | 163 |
| **positivo** | 46 | 90 | 1732 |

## Dados

- **Dataset:** [B2W-Reviews01](https://huggingface.co/datasets/ruanchaves/b2w-reviews01) (avaliações da Americanas.com).
- **Rótulos:** criados a partir da nota (1 a 2 = negativo, 3 = neutro, 4 a 5 = positivo).
- **Amostra:** 30.000 avaliações escolhidas aleatoriamente (seed 42), divididas em 24.000 treino, 3.000 validação e 3.000 teste.

## Método

1. **Baseline:** TF-IDF (unigramas e bigramas, 50 mil atributos) + regressão logística.
2. **Fine-tuning:** BERTimbau Base com learning rate 2e-5, batch size 32, 2 épocas, weight decay 0,01, `max_length` 128 e fp16, em GPU T4 do Google Colab. O melhor checkpoint foi escolhido pelo F1 macro na validação.
3. **Avaliação:** relatório por classe, matriz de confusão e leitura manual de erros.

## Análise de erros

Li uma amostra de 30 dos 435 erros do conjunto de teste. Os padrões mais frequentes:

1. **Rótulo incoerente com o texto.** Como a classe vem da nota, há reviews com nota 3 e texto claramente positivo ("Atende minhas necessidades. Gostei do produto."). Nesses casos o modelo parece mais coerente com o texto do que o rótulo.
2. **Crítica à loja em avaliação de nota alta**, como demora na aprovação do pagamento ou atendimento, em reviews que o modelo classifica como negativas.
3. **Reviews sem opinião sobre o produto**, como perguntas ("como faço para receber o carregador?") e frases do tipo "ainda não usei".
4. **Sentimento misto** com "mas" e "porém", em que o rótulo depende do peso que o cliente deu a cada parte.
5. **Texto com ruído**, como repetições e linhas preenchidas com pontos.

A amostra é pequena, então esses padrões são indícios, não conclusões estatísticas. Parte dos erros reflete limitação do rótulo, não do modelo.

## Limitações

- Rótulos derivados da nota, sem anotação manual.
- Neutro é a classe mais difícil: poucos exemplos e natureza ambígua.
- Treinado em avaliações da Americanas.com de 2018; pode ir pior em outros domínios.
- Textos truncados em 128 tokens.
- Não foi avaliado quanto a vieses.

## Como reproduzir

1. Abra o notebook [`Untitled0.ipynb`](Untitled0.ipynb) no Google Colab com GPU (T4).
2. Rode as células em ordem: instalação, carga dos dados, baseline, fine-tuning e avaliação.
3. Para publicar o seu próprio modelo, faça login no Hugging Face (`notebook_login()`) com um token de escrita. Nunca escreva o token no código.
**Observação:** versões novas da biblioteca `datasets` não aceitam o script original do B2W-Reviews01. O notebook carrega a versão em Parquet do dataset (`revision="refs/convert/parquet"`).

## Usar o modelo

```python
from transformers import pipeline

clf = pipeline("text-classification", model="Eduardo211000/bertimbau-sentimento-b2w")
clf("Produto excelente, chegou rápido!")
```

## Próximos passos

- Anotar manualmente uma amostra do teste para medir o ruído dos rótulos.
- Tratar o desbalanceamento (pesos de classe ou reamostragem) para melhorar o recall do neutro.
  

## Licenças e créditos

- Código deste repositório: MIT.
- Modelo base: BERTimbau (Souza et al., 2020), licença MIT.
- Dataset: B2W-Reviews01 (Real et al., 2019). As fontes divergem entre CC BY 4.0 e CC BY-NC-SA 4.0; por precaução, o modelo publicado usa CC BY-NC-SA 4.0.
