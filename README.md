# DSE_FINAL
DSE final project

Detailed information in poster.

Current Stage:
- fine-tune `meta-llama/Meta-Llama-3-8B-Instruct` model for each breast cancer staging factors on 3,669 real-world pathology reports.
- Did not include M_Classifier as I need to consult the clinician for detailed information on how to determine `M` from the reports, will add later.

Future work:
- Apply on more pathology reports to let the model learn from those class with too few samples.
- Incorporates and let the model output all the fields together, along with a final staging (possible staging if missing value exist) results for clinician's review.
