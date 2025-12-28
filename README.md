

# Assignment 2: IMDB & FashionMNIST - Ολοκληρωμένο ML Pipeline

## Περιγραφή Project
Αυτό το repository υλοποιεί ένα πλήρως αυτοματοποιημένο και αναπαραγώγιμο ML pipeline για ταξινόμηση κειμένου (IMDB) και εικόνας (FashionMNIST), με έμφαση στη modular αρχιτεκτονική, την επαληθευσιμότητα και την αυτοματοποίηση ελέγχων/αποτελεσμάτων.

### Κύρια μέρη:
- **Μέρος Α: IMDB Classic**
	- Προεπεξεργασία, IG-based λεξιλόγιο, custom vectorizer
	- Εκπαίδευση Logistic Regression & BernoulliNB
	- Learning curves, αξιολόγηση, αυτόματη παραγωγή πινάκων/plots
- **Μέρος Β: IMDB RNN**
	- RNN (LSTM/GRU) με pre-trained GloVe embeddings
	- Custom DataLoader, training loop, loss curves, αξιολόγηση
- **Μέρος Γ: FashionMNIST CNN**
	- Εκπαίδευση CNN σε FashionMNIST
	- Training/validation/test split, loss curves, αξιολόγηση

Όλα τα pipelines υποστηρίζουν quick mode για ταχύτατο smoke test & πλήρη mode για κανονικά πειράματα.

---

## Δομή Φακέλων & Κώδικα

- **src/**: Όλος ο πηγαίος κώδικας
	- **imdb/**: Κλασικά μοντέλα, vectorizer, vocab, training, evaluation, learning curves
	- **rnn_imdb/**: RNN, embeddings, custom DataLoader, training/evaluation
	- **fashion/**: CNN, data loaders, training/evaluation
	- **utils/**: Κοινές βοηθητικές συναρτήσεις (metrics, plots, io, logger, seed, device)
	- **make_report.py**: Αυτόματη παραγωγή markdown αναφοράς με όλα τα αποτελέσματα/plots
- **configs/**: Όλα τα config files (YAML/JSON) για reproducibility & εύκολη παραμετροποίηση
- **outputs/**: Παράγονται αυτόματα (plots, tables, checkpoints)
- **report/**: Η τελική αναφορά (assignment2_report.md)
- **scripts/**: Εκτελέσιμα scripts για πλήρη/quick pipeline
- **tests/**: Πλήρες suite ελέγχων (pytest)

---

## Πλήρης Ροή Pipeline & Scripts

Όλη η ροή εκτελείται με:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_all.sh --quick   # ή χωρίς --quick για πλήρη run
pytest -q                        # για αυτόματο έλεγχο όλων των pipelines
```

Το `scripts/run_all.sh` εκτελεί διαδοχικά:
1. **IMDB Classic**: Δημιουργία λεξιλογίου (IG), εκπαίδευση logreg/nb, learning curves, αξιολόγηση
2. **IMDB RNN**: Εκπαίδευση & αξιολόγηση RNN με GloVe embeddings
3. **FashionMNIST CNN**: Εκπαίδευση & αξιολόγηση CNN
4. **Αυτόματη παραγωγή αναφοράς**: Δημιουργεί το `report/assignment2_report.md` με όλα τα αποτελέσματα/plots

Όλα τα scripts CLI τρέχουν ως modules (π.χ. `python -m src.imdb.train_classical ...`).

---

## Αναλυτική Περιγραφή Pipelines

### Μέρος Α: IMDB Classic (src/imdb)
- **Vocab/IG**: Δημιουργία λεξιλογίου με IG, αφαίρεση συχνών/σπάνιων, bulletproof fallback για quick mode
- **Vectorizer**: Custom, συμβατός με pre-tokenized input, πλήρως reproducible
- **Training**: Εκπαίδευση logreg/nb με GridSearchCV, αποθήκευση μοντέλων, παραγωγή learning curves
- **Evaluation**: Αυτόματη παραγωγή πινάκων metrics (csv/markdown), plots
- **Quick mode**: Εγγυημένα X_train.shape[1] ≥ 1, robust fallback, πλήρης έλεγχος με tests

### Μέρος Β: IMDB RNN (src/rnn_imdb)
- **DataLoader**: Custom, με tokenization, padding, split
- **Embeddings**: Φόρτωση GloVe, fallback για quick mode
- **Model**: RNN (LSTM/GRU), bidirectional, dropout, training loop
- **Evaluation**: Αποθήκευση loss curves, test metrics, plots

### Μέρος Γ: FashionMNIST CNN (src/fashion)
- **DataLoader**: Custom split (train/dev/test), reproducible seed
- **Model**: CNN, training loop, early stopping
- **Evaluation**: Αποθήκευση loss curves, test metrics, plots

---

## Testing & Reproducibility

- Όλα τα pipelines έχουν πλήρη κάλυψη με tests (tests/)
- Υπάρχει end-to-end test (test_end_to_end_quick.py) που τρέχει όλο το pipeline σε quick mode και ελέγχει artifacts, features, outputs
- Όλα τα scripts ελέγχονται ως modules (python -m ...)
- Τα tests καλύπτουν: training, evaluation, metrics, data splits, CLI, artifacts, reproducibility
- Το pytest.ini διασφαλίζει σωστό PYTHONPATH (δεν χρειάζεται export)

---

## Παραγόμενα Αρχεία & Αναφορά

Μετά την εκτέλεση:
- **outputs/plots/**: Όλα τα γραφήματα (learning/loss curves, αρχιτεκτονικές)
- **outputs/tables/**: Όλα τα αποτελέσματα (csv)
- **outputs/checkpoints/**: Checkpoints μοντέλων
- **report/assignment2_report.md**: Αυτόματη markdown αναφορά με όλα τα αποτελέσματα/plots

Η αναφορά παράγεται αυτόματα από το script `src/make_report.py` και περιλαμβάνει όλα τα αποτελέσματα, plots, πίνακες, hyperparameters, datasets, reproducibility info.

---

## Dependencies

Όλες οι βασικές βιβλιοθήκες περιγράφονται στο requirements.txt:
- numpy, pandas, scikit-learn, matplotlib
- torch, torchvision
- pytest, tabulate

---

## Χρήσιμα Tips & Troubleshooting

- Όλα τα scripts τρέχουν ως modules (python -m ...)
- Δεν απαιτείται χειροκίνητο PYTHONPATH (βλ. pytest.ini)
- Τα configs είναι yaml/json για εύκολη παραμετροποίηση
- Αν κάποιο artifact λείπει, τρέξε `bash scripts/run_all.sh --quick`
- Για debugging, δες τα logs/outputs σε κάθε script

---

## Επικοινωνία
Για απορίες/σχόλια, επικοινώνησε με τον συγγραφέα της εργασίας.

## Δομή Φακέλων
- **src/**: Όλος ο πηγαίος κώδικας
	- **imdb/**: Κλασικά μοντέλα για IMDB (vectorizer, train, eval, vocab)
	- **rnn_imdb/**: RNN για IMDB (μοντέλο, embeddings, train, evaluate)
	- **fashion/**: CNN για FashionMNIST (μοντέλο, train, eval)
	- **utils/**: Βοηθητικές συναρτήσεις (metrics, plots, io, logger, seed, device)
- **configs/**: Αρχεία ρυθμίσεων (YAML/JSON) για τα πειράματα
- **outputs/**: Όλα τα παραγόμενα αρχεία (πίνακες, γραφήματα, checkpoints)
- **report/**: Η τελική αναφορά (assignment2_report.md)
- **scripts/**: Εκτελέσιμα scripts για αυτοματοποίηση pipelines
- **tests/**: Μονάδες ελέγχου (pytest)

## Εγκατάσταση & Εκτέλεση
1. **Εγκατάσταση dependencies**
	 ```sh
	 python3 -m venv .venv
	 source .venv/bin/activate
	 pip install -r requirements.txt
	 ```
2. **Γρήγορη εκτέλεση (smoke test):**
	 ```sh
	 bash scripts/run_all.sh --quick
	 pytest -q
	 ```
3. **Πλήρης εκτέλεση:**
	 ```sh
	 bash scripts/run_all.sh
	 ```

## Pipelines
Το script `scripts/run_all.sh` εκτελεί διαδοχικά:
- **IMDB Classic**: Δημιουργία λεξιλογίου, εκπαίδευση Logistic Regression & BernoulliNB, learning curves, αξιολόγηση
- **IMDB RNN**: Εκπαίδευση & αξιολόγηση RNN με embeddings
- **FashionMNIST CNN**: Εκπαίδευση & αξιολόγηση CNN
- **Αυτόματη παραγωγή αναφοράς**: Δημιουργεί το `report/assignment2_report.md` με όλα τα αποτελέσματα και γραφήματα

Όλα τα pipelines υποστηρίζουν το flag `--quick` για ταχύ έλεγχο (μικρότερο dataset, γρήγορη εκτέλεση).

## Παραγόμενα Αρχεία
Μετά την εκτέλεση, θα βρείτε:
- **outputs/plots/**: Γραφήματα learning/loss curves, αρχιτεκτονικές
- **outputs/tables/**: Αποτελέσματα πειραμάτων σε csv
- **outputs/checkpoints/**: Checkpoints μοντέλων
- **report/assignment2_report.md**: Η τελική markdown αναφορά

## Έλεγχος & Αναπαραγωγή
- Όλα τα tests εκτελούνται με:
	```sh
	pytest -q
	```
- Η αναφορά και τα αποτελέσματα παράγονται αυτόματα με:
	```sh
	bash scripts/run_all.sh --quick
	```

## Dependencies
Οι βασικές βιβλιοθήκες είναι:
- numpy, pandas, scikit-learn, matplotlib
- torch, torchvision (Apple Silicon ready)
- pytest (testing)
- tabulate (για markdown πίνακες στην αναφορά)
Δείτε το `requirements.txt` για πλήρη λίστα.

## Χρήσιμα Σημεία
- Όλος ο κώδικας τρέχει ως module (π.χ. `python -m src.imdb.train_classical ...`)
- Δεν απαιτείται χειροκίνητο PYTHONPATH (βλ. pytest.ini)
- Τα configs είναι σε yaml/json για εύκολη παραμετροποίηση
- Τα tests καλύπτουν pipelines, metrics, data splits, training, utils

## Επικοινωνία
Για απορίες/σχόλια, επικοινωνήστε με τον συγγραφέα της εργασίας.
