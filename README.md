# Citation Analysis Script

The provided script `citation_analysis.py` is a pipeline developed in the DFG-funded project Quoting as a narrative strategy: A digital-hermeneutic analysis of intertextual phenomena using the example of the letters of Church Father Jerome".

It compares two Latin texts, identifies potential citations and intertextual references, and returns an Excel file.

## Overview

The algorithm operates at the word level and searches for **at least two exact matching words within the same clause**. Stopwords are filtered out. Multiple filter layers are then applied to reduce irrelevant matches, including:

* Punctuation handling
* Part-of-speech (POS) filtering
* Semantic similarity checks

A second, broader algorithm looks for **groups of at least four matching words** (including stopwords) to catch potentially missed references.

For a detailed description of the pipeline, see the following publications:

* [Description of the Pipeline](https://www.digitalhumanities.org/dhq/vol/18/3/000716/000716.html)
* [HTRG Filter Details](https://doi.org/10.1093/llc/fqae078)

## Input Requirements

You will need:

* **Two Latin text files** in **tab-separated format**:

  * Left column: meta-information (e.g., verse, chapter)
  * Right column: actual text
  * *See example files (Virgils Aeneid and Jeromes Epistles are provided) for reference.*

* **Stopword list**:

  * A default stopword list (based on the works Virgil and Jerome) is provided.
  * You can use your own by providing `--stoplist filename.txt`.

* **Python**:

  * Tested with Python `3.10.10`.
  * Other versions may work but could require additional package management.

## Getting Started

```bash
# Clone the repository and move into it
git clone https://github.com/MWittweiler/citation_analysis.git
cd citation_analysis

# Create and activate a virtual environment
python -m venv ca_env
source ca_env/bin/activate  # or `ca_env\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Run the script with the necessary arguments
python citation_analysis.py --input_1 data/aeneid.txt --input_2 data/hieronymus_epistles.txt --genre_1 poetry --genre_2 prose
```
## Note: The runtime increases exponentially the larger the input files are. For the sample files it took us about half an hour to run the script.

### Mandatory Arguments

* `--input_1`, `--input_2`: Tab-separated text files
* `--genre_1`, `--genre_2`: Either `poetry` or `prose`

### Optional Arguments

* `--stoplist filename.txt`: Use a custom stopword list
* `--htrg False`: Disable the HTRG filter, e.g. if you can't install Cracovia system. This improves speed by a lot but expect about 25-50% more irrelevant findings.
* `--similarity False`: Disable the semantic similarity filter, e.g. if you cant install LatinCy. Expect about 20-30% more irrelevant findings.

## Output

The script will produce:

* `*_results.xlsx`: Results using the bigram matching method


## Adjusting Parameters

At the top of the script, four global variables can be changed. The values in the code reflect what worked best in our experiments with Jerome and classical authors. For your setting other values might work better. Some parts of the code might not work as expected when changing the values!


* **min_number_of_shared_words** (default: `2`): How many words constitute a match excluding stop words. Expect fewer findings when increasing the value.
* **min_number_complura** (default: `4`): How many words constitute a match with stop words. Expect fewer findings when increasing the value.
* **maximum_distance_between_shared_words** (default: `2`): Maximum distance allowed between the shared words, so citations are found even with reorderings and words in between. Expect more findings when increasing the value.
* **similarity_threshold** (default: `0.3`): As the last filtering step the cosine similarity of the word embeddings of the two shared words is calculated. If the resulting value is above the threshold the findings is considered irrelevant, because it is likely an everyday collocation. Expect more findings when increasing the value. When there are more than two shared words, this filter is skipped.


## Contact

Author: Michael Wittweiler based on work by Franziska Schropp and Marie Revellio

Do not hesitate to contact the author, if you spot mistakes or have questions! (michael.wittweiler@sglp.uzh.ch)

This work was carried out as part of the project "Zitieren als narrative Strategie. Eine digital-hermeneutische Untersuchung von Intertextualitätsphänomenen am Beispiel des Briefcorpus des Kirchenlehrers Hieronymus." under the supervision of Prof. Dr. Barbara Feichtinger and Dr. Marie Revellio, and was supported by the German Research Foundation (DFG, Forschungsgemeinschaft) [382880410].
