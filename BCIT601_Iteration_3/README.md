# BCIT601 -- Assessment 2 -- Iteration 3

**Student:** Khoyani Vrushang Vipulbhai
**Student ID:** 99342118
**Project:** AI-Powered Local City Council Service Request
Classification System

## Project overview

The aim of this project is to design a machine learning model that will classify requests for the city council services from the community into four classes – Road, Waste, Water, and Safety. The current iteration, which is iteration number three, extends from the previous efforts in data processing and development of baseline models.

## Project structure

-   `data/raw/` -- original supplied CSV dataset
-   `data/processed/` -- processed CSV containing cleaned request text
-   `notebooks_or_scripts/preprocess_data.py` -- checks and cleans the
    dataset
-   `notebooks_or_scripts/baseline_model.py` -- trains and evaluates the
    baseline model
-   `notebooks_or_scripts/model_improvement.py` -- earlier model
    comparison
-   `notebooks_or_scripts/iteration3_model_comparison.py` -- compares
    seven model configurations using a grouped holdout split
-   `model/` -- saved model and TF-IDF vectoriser
-   `outputs/` -- data checks, model metrics, reports, predictions and
    test evidence
-   `api/app.py` -- FastAPI application
-   `tests/test_api.py` -- automated API tests

## Dataset and preprocessing

The dataset provided has 80 data points in four different groups,
which include the road, waste, water, and safety, each with 20 data
points. The preprocessing program analyses the data points, cleans the
requests text, normalizes the spaces, and stores the new dataset. The
original raw dataset remains unchanged.

There are instances where requests texts are similar among different
data points. In the case of Iteration 3, in the model comparison,
the split will be done by groups, such that the similar cleaned
requests will be in the same partition, thus minimizing duplicate-text
leakage from train to test datasets.

This data set is small and includes created samples. Consequently, these
outcomes should be interpreted as a pilot study in the classroom, not as
evidence for the model's performance on requests to actual council
departments.

## Model development and results

The earlier baseline used TF-IDF features with Logistic Regression. Its
row-based 80/20 stratified split achieved:

-   Accuracy: **87.50%**
-   Macro F1: **0.8786**

This earlier result may be optimistic because repeated request texts
could appear in both the training and test partitions.

For Iteration 3, seven model configurations were compared using the same
grouped holdout split. The selected model is Logistic Regression with
character TF-IDF features:

-   `analyzer="char_wb"`
-   `ngram_range=(3, 5)`
-   `sublinear_tf=True`

On the grouped holdout test set, the selected model achieved:

-   Accuracy: **86.67%**
-   Macro F1: **0.8452**

The evaluation prioritised macro F1 so that all four categories received equal consideration. The test set was also small with 15 records, making it difficult to gauge real-world performance with any level of confidence.

All records (80) were used for the API demonstration, for which the model was re-fitted. The final performance scores for the model still pertain to the grouped test set.
## API endpoints

The FastAPI application provides these endpoints:

-   `GET /health` -- checks the service and confirms whether the model
    is loaded
-   `GET /categories` -- returns the available request categories
-   `POST /predict` -- accepts request text and returns a predicted
    category, confidence value and review information

Example request:

``` json
{
  "request_text": "There is a pothole near the school"
}
```

The model and vectorizer are loaded at startup. This API does not train
the model per request. Inputs that do not contain any valid text,
have text that is empty or has an invalid datatype will be rejected
via validation error.

The confidence score is the maximum of the class probabilities from
the model. It is not an indicator that the prediction is always
correct. The review threshold is tentative and requires more
validation.
## Setup and run on Windows

Open PowerShell in the extracted project folder and run:

``` powershell
py -3.13 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
py -3.13 -m pip install -r requirements.txt
py -3.13 notebooks_or_scripts\preprocess_data.py
py -3.13 notebooks_or_scripts\iteration3_model_comparison.py
py -3.13 -m pytest -v
py -3.13 -m uvicorn api.app:app --reload
```

If the virtual environment is already created and activated, there is no
need to create it again.

When Uvicorn is running, open:

-   `http://127.0.0.1:8000/health`
-   `http://127.0.0.1:8000/categories`
-   `http://127.0.0.1:8000/docs`

Use the `/docs` page to send a test request to `POST /predict`.

## Testing

I ran the automated tests on my computer. All 10 tests passed, with 0 failures. The test checks the behavior and input validation of API endpoints.

In addition to automated testing, I performed a manual test with both
valid and invalid data input. The examples of valid input included road,
waste, water and safety requests. The examples of invalid data included
an empty text input, a missing request field and an input in numbers.
In a live Swagger UI test on 18 November 2026, the request `Broken glass is creating a safety risk` returned HTTP 200 with `predicted_category: road`, `confidence: 0.3206`, and `review_recommended: true`. This was an incorrect category for the described safety issue. The low-confidence review flag demonstrates why staff must review uncertain and safety-related requests. This is one observed example, not a general performance measure.

The test doesn’t guarantee that all predictions are correct.
## Ethics and responsible AI

This is a supervised classroom prototype system and not a production
council decision making system. This system must not automatically deny
requests, determine eligibility, establish priority and/or dispatch the
council crews. The council staff must continue being responsible for
the decisions they make, and must be able to correct model
predictions. If a request is urgent or safety related, it should always
be assessed by a person regardless of model confidence.

The dataset contains only 80 fabricated data points. The dataset does
not represent the range of communities, languages, locations, and
writing styles that occur in real service requests. Any production
system will need a better dataset, error checking, fairness testing,
monitoring of the changes in the data, and human review procedure.

Real service requests may contain personal information. Production system
should collect only information needed, inform about how the
information is used, limit access, protect stored and communicated
information, and specify information retention/deletion procedures. Personal information should not be used in the model training and logging.

For Aotearoa New Zealand, a deployment would require engagement with
mana whenua and with the impacted Māori community and taking into
account the considerations arising from te Tiriti o Waitangi. Māori
Data Sovereignty requires that the interests of Māori be taken into
account in decision-making about the use, reuse, sharing, and
interpretation of data. Relevant principles include: Collective
Benefit, Authority to Control, Responsibility and Ethics. Stating these
principles is not engagement or governance. This work has not been done
and no endorsement by Māori is claimed.

## Iteration 3 evidence files

The project includes the following evidence, where available:

-   `Iteration_3_Progress_and_Ethics.docx` -- progress summary and
    ethics evaluation
-   `Ethics_and_Responsible_AI_Addendum.docx` -- additional
    responsible-AI discussion
-   `outputs/iteration3_model_comparison.csv` -- comparison of the seven
    model configurations
-   `outputs/iteration3_selected_model_metrics.json` -- selected model
    metrics
-   `outputs/iteration3_selected_classification_report.txt` --
    classification report
-   `outputs/model_selection_notes.json` -- model selection notes
-   `outputs/pytest_results_iteration3.txt` -- automated test output
-   `outputs/api_test_results_iteration3.txt` -- API test evidence
-   `outputs/broken_glass_prediction_evidence.png` -- live Swagger UI response for the broken-glass example

