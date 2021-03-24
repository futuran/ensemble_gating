# Ensemble Gating

２つの翻訳モデルの出力のうち「良い訳文」を選択するとっても簡単なNN.

A語→B語への２つの翻訳モデルの入力文Aと出力文B1とB2について
- gating.py:B1とB2のうちSentence-BLEUの高い方を分類問題として予測する（B2：ラベル１, B1：ラベル０）
- gating.rec.py:(B2のSentence-BLEU) - (B1のSentence-BLEU)の値を回帰問題として予測する
