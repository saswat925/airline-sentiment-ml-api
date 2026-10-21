# Airline Sentiment Analysis

An end-to-end machine-learning project that classifies airline-related tweets as **negative**, **neutral** or **positive**. It covers data understanding, exploratory analysis, NLP preprocessing, model selection with cross-validation, error analysis, topic modelling, and delivery of the trained model through a **FastAPI** service and a **Streamlit** dashboard.

## At a glance

| | |
| :--- | :--- |
| **Task** | 3-class sentiment classification of tweets directed at airlines |
| **Data** | 14,640 tweets, six US airlines (Twitter US Airline Sentiment dataset) |
| **Selected model** | TF-IDF (1-2 grams) + Linear SVC |
| **Test accuracy** | **78.4 %** (baseline: 62.7 %) |
| **Test Macro-F1** | **0.714** (baseline: 0.257) |
| **Primary metric** | Macro-F1, because 63 % of the data is a single class |
| **Deliverables** | Jupyter notebook, serialized pipeline, REST API with tests, analytics dashboard |

## Contents

1. [Business problem](#1-business-problem)
2. [Dataset](#2-dataset)
3. [Exploratory data analysis](#3-exploratory-data-analysis)
4. [Methodology](#4-methodology)
5. [Results](#5-results)
6. [Error analysis and interpretability](#6-error-analysis-and-interpretability)
7. [Topic modelling of negative tweets](#7-topic-modelling-of-negative-tweets)
8. [Limitations and recommended improvements](#8-limitations-and-recommended-improvements)
9. [Project structure](#9-project-structure)
10. [Getting started](#10-getting-started)
11. [API reference](#11-api-reference)
12. [Tech stack](#12-tech-stack)

---

## 1. Business problem

Passengers increasingly report delays, cancellations and baggage problems on social media first. Reading those posts manually is slow, subjective and reactive, so a crisis can grow before anyone responds. An automated classifier can triage the stream in real time.

| Stakeholder | How they use the system |
| :--- | :--- |
| **Customer-experience and operations managers** | Spot systemic issues (for example recurring delay or hold-time complaints) early |
| **Brand and PR teams** | Detect sudden spikes in negative sentiment |
| **Support agents** | Prioritise the most distressed passengers first |

**Success measures.** Business KPIs are response-time reduction, net-sentiment improvement and fewer escalated tickets. The ML metrics that support them are **Macro-F1** (balanced quality across all three classes), **recall on the negative class** (do not miss complaints) and **precision on the negative class** (do not waste agent time).

## 2. Dataset

* **Size:** 14,640 tweets x 15 columns; six airlines (American, Delta, Southwest, US Airways, United, Virgin America).
* **Target:** `airline_sentiment` (negative / neutral / positive).
* **Model input:** the `text` column only.

| Class | Tweets | Share |
| :--- | ---: | ---: |
| Negative | 9,178 | 62.7 % |
| Neutral | 3,099 | 21.2 % |
| Positive | 2,363 | 16.1 % |

**Data quality**

| Column | Missing | Comment |
| :--- | ---: | :--- |
| `negativereason_gold` | 99.8 % | Almost empty, unused |
| `airline_sentiment_gold` | 99.7 % | Almost empty, unused |
| `tweet_coord` | 93.0 % | Unused |
| `negativereason` | 37.3 % | **Target leakage** (see below) |
| `user_timezone` / `tweet_location` | 32.9 % / 32.3 % | Unused |
| `negativereason_confidence` | 28.1 % | **Target leakage** (see below) |

* **36 fully duplicated rows** exist (0.25 %). They were not removed.
* **Leakage check.** `negativereason` is empty for 5,462 tweets, which is exactly the number of non-negative tweets (3,099 + 2,363), and populated for 9,178, exactly the number of negative tweets. It therefore effectively mirrors the label. It and `negativereason_confidence` are only known after a tweet has been judged, so they cannot exist for a new, unlabelled tweet and were excluded. Training on the text alone keeps the evaluation honest.

## 3. Exploratory data analysis

![EDA overview](assets/eda_overview.png)

**Sentiment by airline** (share of each airline's tweets)

| Airline | Negative | Note |
| :--- | ---: | :--- |
| US Airways | 77.7 % | Highest negative share |
| American | 71.0 % | |
| United | ~69 % | Read from chart |
| Southwest | ~49 % | Read from chart |
| Delta | ~43 % | Read from chart |
| Virgin America | 35.9 % | Lowest negative share and highest positive share (30.2 %) |

**Why customers complain** (`negativereason`, as a share of the 9,178 negative tweets)

| Reason | Tweets | Share |
| :--- | ---: | ---: |
| Customer Service Issue | 2,910 | 31.7 % |
| Late Flight | 1,665 | 18.1 % |
| Can't Tell | 1,190 | 13.0 % |
| Cancelled Flight | 847 | 9.2 % |
| Lost Luggage | 724 | 7.9 % |
| Bad Flight | 580 | 6.3 % |
| Flight Booking Problems | 529 | 5.8 % |
| Flight Attendant Complaints | 481 | 5.2 % |
| Long lines | 178 | 1.9 % |
| Damaged Luggage | 74 | 0.8 % |

**Tweet length by sentiment**

| Sentiment | Avg. characters | Avg. words |
| :--- | ---: | ---: |
| Negative | 113.9 | 19.7 |
| Neutral | 87.4 | 14.4 |
| Positive | 86.1 | 14.0 |

**Key insights**

1. **People tweet airlines mostly to complain.** Negative tweets outnumber positive ones roughly 4 to 1, which makes accuracy a misleading metric on its own.
2. **Service, not just schedule, drives dissatisfaction.** Customer Service Issue is the largest single reason (31.7 %). Together with late and cancelled flights it covers **59 %** of all negative tweets. Delays are often outside an airline's control; how it communicates and rebooks is not.
3. **Brands differ sharply.** Virgin America's 35.9 % negative share versus US Airways' 77.7 % suggests there are practices worth benchmarking.
4. **Complaints are wordier.** Negative tweets are about 28 characters longer than positive ones on average, consistent with passengers explaining a problem rather than saying thanks.

## 4. Methodology

```
Tweets.csv
   |
   v
Text cleaning --> Stratified split 70/15/15 --> TF-IDF (1-2 grams) --> Classifier --> Evaluation --> Joblib pipeline
                                                                          ^
                                                          5-fold GridSearchCV (Macro-F1)
```

### 4.1 Text preprocessing

| Step | Reason |
| :--- | :--- |
| Lowercase | Uniform vocabulary |
| Remove URLs, `@mentions`, HTML tags | Carry no sentiment and would cause overfitting to handles |
| Remove digits and special characters (apostrophes kept) | Reduces noise |
| Remove stop-words **except a negation whitelist** (`not`, `no`, `never`, `nor`, `neither`, `none`, `cannot`, `hardly`, `scarcely`, `barely`) | "not good" must not collapse into "good" |
| WordNet lemmatization | Merges inflected forms and keeps tokens human-readable for interpretation |

### 4.2 Data splitting

A **stratified 70 / 15 / 15** split (random state 42) gives 10,248 training, 2,196 validation and 2,196 test tweets. Class proportions are identical in all three (62.7 / 21.2 / 16.1 %). The validation set is used for model selection and the **test set is used once**, for the final score.

### 4.3 Features and models

* **Features:** `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9)`. Bigrams capture phrases such as "customer service" and "no problem". The vectorizer lives inside the scikit-learn `Pipeline`, so it is fitted on training folds only and cannot leak.
* **Candidates and search grids** (5-fold `GridSearchCV`, scoring `f1_macro`):

| Model | Grid | Best setting |
| :--- | :--- | :---: |
| Multinomial Naive Bayes | `alpha` in {0.1, 0.5, 1.0} | 0.1 |
| Logistic Regression | `C` in {0.1, 1, 10} | 10 |
| Linear SVC | `C` in {0.1, 1, 10} | 1.0 |

* **Baseline:** a majority-class `DummyClassifier`, the minimum every model must beat.

## 5. Results

### 5.1 Model comparison (validation set, 2,196 tweets)

| Model | CV Macro-F1 | Val. accuracy | Val. Macro-F1 |
| :--- | :---: | :---: | :---: |
| Dummy (most frequent) | n/a | 0.6270 | 0.2569 |
| Naive Bayes | 0.6498 | 0.7696 | 0.6778 |
| Logistic Regression | 0.6938 | 0.7737 | 0.7021 |
| **Linear SVC (selected)** | **0.6994** | **0.7760** | **0.7053** |

All three models beat the baseline by a wide margin (Macro-F1 at least 2.6x higher; 2.7x for Linear SVC). Linear SVC and Logistic Regression are practically tied (a 0.003 gap in validation Macro-F1 is unlikely to be meaningful on a validation set of this size). Linear SVC is the nominal winner; Logistic Regression is the natural alternative if calibrated probabilities are needed. Naive Bayes shows the typical majority-class bias: the best negative recall (0.94) but the worst neutral recall (0.40).

### 5.2 Final evaluation of Linear SVC (unseen test set, 2,196 tweets)

**Accuracy 0.7837 | Macro-F1 0.7137 | Weighted-F1 0.78**

| Class | Precision | Recall | F1 | Support |
| :--- | :---: | :---: | :---: | ---: |
| Negative | 0.84 | 0.90 | 0.87 | 1,377 |
| Neutral | 0.60 | 0.58 | 0.59 | 465 |
| Positive | 0.76 | 0.62 | 0.68 | 354 |
| **Macro average** | **0.74** | **0.70** | **0.71** | 2,196 |

* **Generalisation is healthy.** Cross-validation (0.699), validation (0.705) and test (0.714) Macro-F1 agree closely, so the model is not overfitting to the validation set. With 2,196 test tweets, the 95 % confidence interval on accuracy is roughly +/- 1.7 percentage points (normal approximation).
* **Negative class works well for triage.** Recall of 0.90 means about nine in ten complaints are caught; precision of 0.84 means about one in six flagged tweets is a false alarm.
* **Neutral is the weak spot** (F1 0.59), followed by positive recall (0.62).

### 5.3 Confusion matrices (validation set)

![Confusion matrices](assets/confusion_matrices.png)

Linear SVC on validation (rows = true class, columns = predicted):

| True \ Predicted | Negative | Neutral | Positive |
| :--- | ---: | ---: | ---: |
| **Negative** | 1,226 | 115 | 36 |
| **Neutral** | 172 | 251 | 42 |
| **Positive** | 69 | 58 | 227 |

## 6. Error analysis and interpretability

### 6.1 Where the model fails

Of the 492 validation errors, **387 (79 %) involve the neutral class**. The largest single confusion is *neutral predicted as negative* (172 tweets, 35 % of all errors), followed by *negative predicted as neutral* (115). Judging by the neutral class's top features (*whether, maybe get, hi, care team, appreciate patience*), neutral tweets lean towards questions and polite or informational phrasing, and they share a lot of flight-logistics vocabulary with complaints, so the boundary with negative is genuinely blurry. Some of the confusion is plausibly label noise; the dataset's `airline_sentiment_confidence` column, which was not analysed here, could be used to check.

Sample misclassifications (text shown after preprocessing):

| Preprocessed tweet | True | Predicted | Likely cause (interpretation) |
| :--- | :---: | :---: | :--- |
| thnx response bot seems 'improvement' could easily attained st class service really worth | neutral | negative | Sarcasm and quoted words |
| not possible carryon put closetoverhead success airline thanks | neutral | negative | Mixed polarity ("not possible" plus "thanks") |
| really going let get best | negative | positive | Sarcasm; positive words, context lost |
| flight next two week let's make good memory | positive | negative | Short text with weak sentiment words |
| pwm direct orlando saturday happening march thought going permanent thing | negative | neutral | Complaint expressed without sentiment vocabulary |

A further sanity check on the saved pipeline shows the same weakness: the clearly neutral sentence *"My flight is scheduled for tomorrow at 5 PM."* was predicted **negative**. This is likely because flight-logistics vocabulary is mostly associated with complaints in the training data (62.7 % negative).

### 6.2 What the model learned

Top Linear SVC features per class (highest positive coefficients):

| Class | Strongest indicators |
| :--- | :--- |
| **Negative** | hour, worst, nothing, hold, bag, hr, disappointed, ridiculous, ruining, **no**, stuck, fail, luggage, mess, never fly |
| **Positive** | great, excellent, awesome, amazing, love, can wait, wonderful, thanks, kudos, best, impressed, thx, yes please, no problem, appreciate |
| **Neutral** | arrived yet, hi, care team, appreciate patience, delay cancelled, whether, maybe get, still fly, still use, promo |

* **The signals are sensible.** Time-wasting words (*hour, hold, stuck*), baggage words and strong negative adjectives point to negative; gratitude and praise point to positive.
* **Bigrams and negations pay off.** `no` is itself a top negative feature, while the bigrams `no problem` and `no worry` act as *positive* (or anti-negative) signals. A unigram-only model could not separate these, which supports the choice of `ngram_range=(1, 2)` and of keeping negation words.
* **Neutral features are diffuse.** They are largely question-like or polite phrases (*whether, maybe get, hi*) rather than distinctive content words, which helps explain why neutral is hardest to separate.
* A few tokens (for example `dal`, `tab`) look dataset-specific rather than linguistic, a reminder to check for brand or topic artefacts before trusting the model on a different airline set.

## 7. Topic modelling of negative tweets

Latent Dirichlet Allocation (5 topics) was fitted on the 9,178 negative tweets to look for complaint themes:

| Topic | Top keywords | Interpretation |
| :--- | :--- | :--- |
| 1 | flight, cancelled, problem, flightled, help, time, customer, plane, dm, change | Cancellations and requests for rebooking help |
| 2 | hold, hour, flight, phone, service, customer, help, minute, wait, waiting | Long hold times and phone support |
| 3 | flight, cancelled, time, airline, flightled, worst, like, customer, service, fly | Cancellations combined with general anger at service |
| 4 | flight, gate, bag, ua, plane, delayed, fleet, checked, fleek, delay | Gate, delay and baggage-check issues |
| 5 | flight, hour, plane, delayed, late, bag, delay, waiting, gate, luggage | Delays, waiting and luggage |

**Reading the result honestly:** the topics are only partly distinct. "flight" appears in all five and several topics overlap (1 and 3; 4 and 5). Tokens such as `fleet` and `fleek` in topic 4 look like a recurring hashtag rather than a theme. Treat LDA here as exploratory colour. The *quantified* breakdown of complaint drivers is the labelled `negativereason` table in section 3. Hold times (topic 2) and cancellations (topics 1 and 3) are the clearest additional themes.

*Technical note:* LDA was fitted on TF-IDF weights with a fixed number of topics. Raw counts (`CountVectorizer`) and a coherence search over the number of topics would be the more standard setup.

## 8. Limitations and recommended improvements

### Known limitations

| Area | Observation |
| :--- | :--- |
| **Neutral class** | F1 of 0.59; most errors involve neutral (section 6.1) |
| **Negation handling is partial** | The whitelist keeps `not`, `no` and `nor`, but NLTK's stop-word list also contains contracted negations (`didn't`, `wasn't`, `don't`, `won't`, `couldn't`, ...) that are **still removed**. "wasn't helpful" therefore loses its negation. |
| **Information discarded** | The cleaning regex removes `!`, `?`, digits and emojis, which can carry sentiment |
| **Bag-of-words ceiling** | No word order beyond bigrams, so sarcasm and long-range context are missed |
| **Class imbalance** | Not addressed in training (no class weights); recall for neutral and positive trails precision |
| **Evaluation** | One split, no repeated CV or confidence intervals on Macro-F1; 36 duplicate rows not removed before splitting |
| **Tuning** | Naive Bayes (`alpha=0.1`) and Logistic Regression (`C=10`) landed on the edge of their grids, so their optimum may lie outside the search range |
| **Training / serving skew risk** | The preprocessing function is copy-pasted into the notebook, `app.py` and `dashboard.py`; a change in one place can silently diverge from the others |
| **API** | `/health` returns HTTP 200 even when the model failed to load (the body says `unhealthy`), so a container probe would not detect it. Whitespace-only or symbol-only text passes validation, is cleaned to an empty string and still receives a label. |
| **Dashboard** | The "Model Architecture & Metrics" tab shows hard-coded numbers that must be updated by hand after retraining |
| **Scope** | English text only; Twitter users are not representative of all passengers; a single snapshot of data means vocabulary drift is not measured |

### Recommended next steps (roughly by effort / impact)

1. **Fix contracted negations** by adding them to `NEGATION_WORDS` and keep `!` / `?` tokens or emoji-to-text mapping; retrain and compare Macro-F1.
2. **Try `class_weight="balanced"`** and a wider grid (including `C=100` for Logistic Regression and smaller `alpha` for Naive Bayes) to lift minority-class recall.
3. **Centralise preprocessing** in one module (or wrap it in a `FunctionTransformer` inside the pipeline) so training and serving share a single implementation.
4. **Expose confidence** (Logistic Regression probabilities or `CalibratedClassifierCV`), and route low-confidence tweets to human review. This also provides labels for retraining and drift monitoring.
5. **Benchmark a transformer** (DistilBERT / RoBERTa) to quantify the gain on neutral tweets and sarcasm against the added latency and cost.
6. **Harden the API:** return 503 from `/health` when the model is missing, reject empty-after-cleaning text, add a batch endpoint, pin dependency versions (a joblib model should be loaded with the scikit-learn version that created it) and containerise.
7. **Deduplicate before splitting** and report cross-validated Macro-F1 with confidence intervals.

## 9. Project structure

```
airline-sentiment-analysis/
├── README.md                          # Project documentation (this file)
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Ignores caches, virtual envs, .env, notebook checkpoints
│
├── sentimental_ML.ipynb               # Full ML workflow: EDA -> preprocessing -> training
│                                      #   -> evaluation -> LDA -> model serialization
├── app.py                             # FastAPI inference service (/health, /predict)
├── test_app.py                        # Pytest suite for the API
├── dashboard.py                       # Streamlit analytics dashboard
│
├── Tweets.csv                         # Dataset  (you provide; see Getting started)
├── best_sentiment_pipeline.joblib     # Trained pipeline  (generated by the notebook)
│
└── assets/
    ├── eda_overview.png               # Figures used in this README
    └── confusion_matrices.png
```

`Tweets.csv` and `best_sentiment_pipeline.joblib` are not shipped in the source tree. All scripts use relative paths, so run every command **from the project root**.

## 10. Getting started

**Prerequisites:** Python 3.9 or newer.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# 2. Install dependencies and the NLTK data used by the preprocessing
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4
```

**3. Add the dataset.** Download the Twitter US Airline Sentiment `Tweets.csv` and place it in the project root.

**4. Train the model.** Open `sentimental_ML.ipynb` and run all cells (Jupyter is only needed for this step: `pip install notebook`, or use Google Colab / VS Code). The notebook writes `best_sentiment_pipeline.joblib` to the project root.

**5. Serve the API**

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Interactive documentation: <http://127.0.0.1:8000/docs>

**6. Run the tests** (they load the trained model and assert on its predictions, so complete step 4 first)

```bash
pytest test_app.py
```

**7. Launch the dashboard**

```bash
streamlit run dashboard.py
```

Opens at <http://localhost:8501> with three tabs: real-time predictor, historical patterns (needs `Tweets.csv`) and model metrics.

## 11. API reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Reports whether the model pipeline is loaded |
| `POST` | `/predict` | Returns the sentiment of one text |

**Request**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I had a terrible experience, my baggage was completely lost!"}'
```

**Response**

```json
{
  "text": "I had a terrible experience, my baggage was completely lost!",
  "cleaned_text": "terrible experience baggage completely lost",
  "prediction": "negative"
}
```

| Status | Meaning |
| :--- | :--- |
| `200` | Prediction returned |
| `422` | Body is missing `text`, or `text` is an empty string |
| `503` | Model pipeline is not loaded |

The test suite covers the health endpoint, one positive and one negative prediction, and the validation error.

## 12. Tech stack

| Area | Tools |
| :--- | :--- |
| Data and ML | pandas, NumPy, scikit-learn (TF-IDF, Naive Bayes, Logistic Regression, Linear SVC, LDA, GridSearchCV), NLTK |
| Visualisation | Matplotlib, Seaborn |
| Serving | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit |
| Testing | Pytest, HTTPX (`TestClient`) |
| Persistence | Joblib |
