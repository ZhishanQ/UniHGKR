# CompMix-IR Benchmark

This repository contains various components of the CompMix-IR benchmark, which is designed for information retrieval on a heterogeneous knowledge corpus. Below is an overview of the folder structure and the purpose of each file:


## Data Access

- **CompMix-IR Corpus**:
  - [Google Drive](https://drive.google.com/file/d/1sDmPieBkAnO9Rb7oDDXAgRDd5SRo_rPP/view?usp=sharing)
  - [HuggingFace Dataset](https://huggingface.co/datasets/ZhishanQ/CompMix-IR)

- **Data-Text Pairs**:
  - [Google Drive](https://drive.google.com/file/d/1AOyY0T_FQo7Br6o7KfkNSnoW9L9dZYXb/view?usp=sharing)

 
## Folder Structure

- **CompMix**: 
  - Contains QA pairs from the CompMix dataset.

- **ConvMix_annotated**: 
  - Contains QA pairs from the ConvMix dataset.

- **eval_part**: 
  - Used to evaluate whether the retrieved evidence is a positive match for a query. This is judged by checking if the evidence contains the answer text or entity for the query.

- **data_2_text_subset.json**: 
  - A subset of Data-Text Pairs used in training stages 1 and 2. Each item is a dictionary with the following keys:
    - `evi_id`: Index of the evidence in the corpus.
    - `source`: Origin of the evidence (e.g., `kb` for knowledge graph triples, `table` for tables, `info` for infoboxes, `text` for natural language text).
    - `raw_data`: Linearized form of the data as it appears in the corpus.
    - `nl_text`: Natural language text generated by GPT-4o-mini with the same semantic information as `raw_data`.

  **Sample:**

  ```json
  {
      "index": 724218,
      "evi_id": 5088563,
      "source": "table",
      "raw_data": "2010 Man Booker Prize, Author is Howard Jacobson, Title is The Finkler Question, Genre(s) is Novel, Country is UK, Publisher is Bloomsbury",
      "nl_text": "The 2010 Man Booker Prize was awarded to Howard Jacobson for his novel \"The Finkler Question,\" which was published by Bloomsbury in the UK."
  }
  ```

- **subset_kb_wikipedia_mixed_rd.json**: 
  - A subset of the CompMix-IR corpus. Each item is a dictionary with the following keys:
    - `source`: Origin of the evidence.
    - `evidence_text`: Content of the evidence, linearized if it's structured data.
    - `wikidata_entities`: Corresponding entities in the Wikidata knowledge graph.
    - `disambiguations`: List of disambiguated entities.
    - `retrieved_for_entity`: Entity for which the evidence was retrieved.

  **Sample:**

  ```json
  {
      "evidence_text": "The Twilight Zone (1959 TV series), Season is 4 (1963), Time slot is Thursday at 9:00-10:00 pm E.T.",
      "source": "table",
      "wikidata_entities": [
          {"id": "Q4571012", "label": "1962\\u201363 United States network television schedule"},
          {"id": "1959-01-01T00:00:00Z", "label": "1959"},
          {"id": "Q559270", "label": "The Twilight Zone"},
          {"id": "Q742103", "label": "The Twilight Zone"},
          {"id": "1963-01-01T00:00:00Z", "label": "1963"},
          {"id": "Q2228924", "label": "The Twilight Zone Radio Dramas"}
      ],
      "disambiguations": [
          ["1959", "1959-01-01T00:00:00Z"],
          ["1963", "1963-01-01T00:00:00Z"],
          ["The Twilight Zone", "Q559270"],
          ["4 (1963)", "Q4571012"],
          ["series", "Q2228924"],
          ["The Twilight Zone (1959 TV series)", "Q742103"]
      ],
      "retrieved_for_entity": {"id": "Q742103", "label": "The Twilight Zone"}
  }
  ```