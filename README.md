# HeartCare AI

HeartCare AI is a conversational web application and research toolkit for heart disease risk assessment using federated learning and deep learning. It features an interactive Streamlit chatbot for user-friendly risk assessment, as well as scripts for federated and centralized model training and evaluation.

---

## Features

- **Conversational AI Assistant:** Empathetic, step-by-step chat interface for collecting user health data and providing risk assessment (`app.py`).
- **Federated Learning Simulation:** Advanced federated learning pipeline for heart disease prediction, with model/result saving and visualizations (`federated_learning.py`).
- **Centralized Training:** Baseline centralized model training for comparison.
- **Data Extraction with LLM:** Uses LLMs (via Groq API) to extract structured medical data from user input.
- **Progress Tracking:** Sidebar in the app shows assessment progress and collected data.
- **Emergency Detection:** Detects emergency symptoms and recommends immediate medical attention.
- **Visualization:** Automatically generates and saves plots of data distribution, training history, confusion matrices, and performance comparisons.
- **Reproducible Experiments:** Stores experiment configs and results in organized folders.

---

## Folder Structure

```
.
├── app.py                       # Streamlit conversational assistant
├── federated_learning.py        # Federated and centralized model training/evaluation
├── models/
│   ├── best_global_fl_model.h5  # Best federated model (used by app.py)
│   ├── centralized_model.h5     # Centralized model
│   ├── global_fl_model.h5       # Final federated model
│   └── scaler.pkl               # Scaler for feature normalization
├── Data/
│   └── heart_disease_uci.csv    # Heart disease dataset
├── federated_learning_results/  # Experiment configs and results
├── visualizations/              # Generated plots (created at runtime)
├── results/                     # Model evaluation results (created at runtime)
└── .env                         # Environment variables (Groq API key, etc.)
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- TensorFlow
- NumPy, pandas, scikit-learn, matplotlib, seaborn
- [Groq Python SDK](https://github.com/groq/groq-python)
- python-dotenv

Install dependencies:
```sh
pip install -r requirements.txt
```
*(Create `requirements.txt` if missing, based on imports in your code.)*

### Setup

1. **Add your Groq API key**  
   Create a `.env` file in the root directory:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

2. **Prepare models and scaler**  
   Make sure `models/best_global_fl_model.h5` and `models/scaler.pkl` exist.  
   You can generate them by running federated learning:
   ```sh
   python federated_learning.py
   ```

3. **Dataset**  
   Ensure `Data/heart_disease_uci.csv` is present.

---

## Running the Application

Start the Streamlit assistant:
```sh
streamlit run app.py
```
The app will open in your browser at [http://localhost:8501](http://localhost:8501).

---

## Training & Experiments

To run federated and centralized training, and generate results/plots:
```sh
python federated_learning.py
```
- Models are saved in `models/`
- Visualizations in `visualizations/`
- Results in `results/`
- Experiment configs/results in `federated_learning_results/`

---

## Usage

- **Conversational Assistant:**  
  Answer the chatbot's questions about your health. The assistant will extract relevant data, track progress, and provide a risk assessment at the end.
- **Emergency Detection:**  
  If you mention emergency symptoms (e.g., "severe chest pain"), the assistant will recommend immediate medical attention.
- **Experimentation:**  
  Modify `federated_learning.py` or experiment configs to try different federated learning settings.

---

## Disclaimer

**HeartCare AI is a research and educational tool. It does not provide medical diagnosis. Always consult a healthcare professional for medical advice.**

---

## License

MIT License

---

## Acknowledgments

- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- [Streamlit](https://streamlit.io/)
- [Groq LLMs](https://groq.com/)
