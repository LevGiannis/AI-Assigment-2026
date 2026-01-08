
# Assignment 2: IMDB & FashionMNIST

## Περιγραφή
Αυτό το project υλοποιεί pipelines για επεξεργασία και ταξινόμηση κειμένου (IMDB) και εικόνας (FashionMNIST) με κλασικές και νευρωνικές μεθόδους. Περιλαμβάνει:
- Κλασική επεξεργασία κειμένου IMDB (Logistic Regression, BernoulliNB)
- RNN για IMDB
- CNN για FashionMNIST
- Αυτόματη παραγωγή αναφοράς και γραφημάτων

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

## Linux σημειώσεις (προτεινόμενος τρόπος εκτέλεσης)
- **Tests (σταθερά σε Linux):**
	```sh
	.venv/bin/python -m pytest tests/
	```

- **Quick end-to-end (χωρίς εξωτερικό GloVe):**
	```sh
	PYTHON=.venv/bin/python bash scripts/run_all.sh --quick
	```

- **Full end-to-end (με πραγματικό GloVe):**
	```sh
	export GLOVE_PATH=/path/to/glove.6B.100d.txt
	PYTHON=.venv/bin/python bash scripts/run_all.sh
	```

Σημείωση: Το `scripts/run_all.sh` υποστηρίζει τη μεταβλητή `PYTHON` (ώστε να χρησιμοποιεί το venv) και το `GLOVE_PATH` για το IMDB RNN σε full mode.

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
