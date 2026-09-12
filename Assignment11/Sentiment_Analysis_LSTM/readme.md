# Customer Review Sentiment Analysis using NLP and LSTM

## 1. Project Overview and Objective
Organizations receive large volumes of customer reviews online, and manually
analyzing them is time-consuming. This project builds a Natural Language
Processing (NLP) based deep learning model — using an LSTM (Long Short-Term
Memory) network — to automatically classify customer/movie reviews as
**Positive** or **Negative**.

The pipeline covers:
- Text preprocessing and sequence preparation
- Traditional NLP feature representations (Bag of Words, TF-IDF)
- Deep learning word embeddings
- An LSTM-based binary sentiment classifier
- Full model evaluation with metrics and visualizations

## 2. Dataset Details
- **Source:** Keras built-in IMDB dataset (`tensorflow.keras.datasets.imdb`)
- **Total reviews:** 50,000
- **Training samples:** 25,000
- **Testing samples:** 25,000
- **Vocabulary size used:** Top 10,000 most frequent words
- **Target variable:**
  - `0` → Negative sentiment
  - `1` → Positive sentiment

The dataset is provided pre-tokenized as integer sequences (word indices), and
comes with an official 50/50 train-test split, so no separate splitting step
was required.

## 3. Preprocessing Steps
1. **Loading:** Loaded the IMDB dataset with `num_words=10000` to cap the vocabulary.
2. **Decoding demo:** Reconstructed a sample review from its word-index sequence
   using the IMDB word index, to illustrate the raw text.
3. **Text cleaning:** Lower-cased text and removed punctuation/numbers.
4. **Tokenization:** Split cleaned text into word tokens.
5. **Stopword removal:** Removed common English stopwords (NLTK stopword list).
6. **Sequence generation:** Used the built-in Keras integer-encoded sequences
   as model input (word indexing already provided by the dataset).
7. **Padding:** Padded/truncated all sequences to a fixed length (`MAX_LEN = 200`)
   using post-padding and post-truncation.
8. **Train-test split:** Used the dataset's existing 25,000/25,000 split; an
   additional 20% of the training set was reserved for validation during training.

## 4. Text Representation Techniques
| Technique | Description |
|---|---|
| **Bag of Words** | Counts word occurrences per review, ignoring order and context (`CountVectorizer`) |
| **TF-IDF** | Weighs words by frequency in a review relative to their frequency across all reviews (`TfidfVectorizer`) |
| **Word Embeddings** | Dense, trainable vector representations learned inside the LSTM model's `Embedding` layer, capturing semantic similarity and used with sequence order |

## 5. Model Architecture
```
Input Text (padded sequence, length=200)
        |
Embedding Layer (vocab=10000, dim=64)
        |
LSTM Layer (64 units)
        |
Dropout (0.3)
        |
Dense Layer (16 units, ReLU)
        |
Dense Output Layer (1 unit, Sigmoid)
```
- **Loss function:** Binary Crossentropy
- **Optimizer:** Adam
- **Metric:** Accuracy
- **Epochs:** 5
- **Batch size:** 128

## 6. Training Results
The model was trained for 5 epochs with an 80/20 train/validation split on the
training data. Training and validation accuracy/loss curves are plotted in the
notebook (`Sentiment_Analysis_LSTM.ipynb`) to visualize learning progress and
check for overfitting.

> Exact accuracy/loss values will vary slightly by run/environment — see the
> notebook outputs for the specific numbers obtained.

## 7. Evaluation Metrics
The trained model was evaluated on the 25,000-sample test set using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

Additional visualizations included in the notebook:
- Training vs Validation Accuracy curve
- Training vs Validation Loss curve
- Confusion Matrix heatmap
- Actual vs Predicted sentiment comparison table (sample of test reviews)
- Sample predictions on custom example reviews

## 8. Performance Analysis and Conclusion
- **Sequence length:** Longer `MAX_LEN` retains more context but increases
  padding/computation; too short truncates useful information.
- **Embedding layer:** Converts sparse word indices into dense vectors that
  capture semantic relationships, improving generalization over BoW/TF-IDF.
- **LSTM memory cells:** Allow the model to retain context across the sequence,
  helping with long-range dependencies and negation handling.
- **Overfitting:** Training accuracy typically exceeds validation accuracy after
  a few epochs. Mitigations: dropout, early stopping, simpler architecture,
  regularization, more data.
- **Limitations:** Fixed 10,000-word vocabulary drops rare/domain words, fixed
  sequence length may truncate long reviews, and the model may not generalize
  well to sarcasm, slang, or entirely different review domains (e.g., products
  vs. movies).

**Conclusion:** The LSTM-based model successfully classifies review sentiment
and outperforms traditional bag-of-words style approaches by capturing word
order and context, at the cost of higher computational complexity.

## 9. Repository Contents
```
├── Sentiment_Analysis_LSTM.ipynb   # Complete notebook (all 5 tasks, code + outputs)
├── README.md                       # This file
```

## 10. How to Run
1. Install dependencies:
   ```bash
   pip install tensorflow scikit-learn nltk pandas matplotlib seaborn
   ```
2. Open `Sentiment_Analysis_LSTM.ipynb` in Jupyter Notebook / JupyterLab / Google Colab.
3. Run all cells sequentially (Task 1 → Task 5).

## 11. Submission Link
_Add your GitHub repository / Google Drive link here before submission._
