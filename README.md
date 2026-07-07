# Emotion Classification: LSTM vs BiLSTM vs BiRNN vs Transformer



**Author:** Jana Ashour

## Task

Classify short English sentences into one of six emotions using a labeled corpus:

| Label | Emotion  |
|-------|----------|
| 0     | sadness  |
| 1     | joy      |
| 2     | love     |
| 3     | anger    |
| 4     | fear     |
| 5     | surprise |

The dataset is pre-split into train / dev / test sets. Train is used only for
fitting parameters, dev only for hyperparameter tuning and model selection,
and test strictly for final reported results.

## Contents

- `emotion_classification_lstm_bilstm_birnn_transformer.ipynb` — full notebook with data
  loading, model implementations, training, and evaluation
- `emotion_classification_lstm_bilstm_birnn_transformer_report.pdf` — written report with
  results tables, confusion matrices, and analysis

## Parts

1. **LSTM vs BiLSTM** — hyperparameter search (4 trials each), best model comparison
2. **BiLSTM vs BiRNN** — reusing the best BiLSTM config, evaluating a plain bidirectional RNN
3. **Effect of Additive Attention on BiLSTM** — attention-augmented BiLSTM, with attention
   heatmap visualizations for correct/incorrect predictions
4. **Self-Attention Transformer Encoder** with learned positional encodings

## Key Results (Test set, best configs)

| Model                | Accuracy | Macro F1 |
|----------------------|----------|----------|
| LSTM (unidirectional) | 0.8900  | 0.8262   |
| BiLSTM                | 0.9160  | 0.8664   |
| BiRNN                 | 0.7450  | 0.4803   |
| BiLSTM + Attention    | 0.9295  | 0.8885   |
| Transformer Encoder   | 0.9005  | 0.8495   |

## Findings

- BiLSTM substantially outperforms both unidirectional LSTM and BiRNN, especially
  on minority classes (love, surprise), where BiRNN collapses entirely (F1 = 0).
- Adding additive attention to BiLSTM gives the best overall test performance.
- The Transformer encoder is competitive but doesn't clearly beat BiLSTM + Attention
  on this dataset size, while costing roughly 2x the training time per epoch.
- Self-attention helps most on emotions with cues spread across the sentence
  (joy, anger, sadness); BiLSTM still generalizes better on short/subtle
  emotions (surprise, love) given the limited data.
