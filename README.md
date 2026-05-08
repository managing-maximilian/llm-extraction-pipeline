# LLM Extraction Pipeline — Template

A minimal, reusable skeleton of the local-LLM information extraction pipeline
developed for the **ManMax** project. The framework itself does not assume any
particular subject matter — to adapt it to your own use case you provide three
things:

1. a **Pydantic output model** describing the JSON structure the LLM should return,
2. a **prompt** (system + user message) telling the LLM what to extract,
3. (optionally) a **step** that wires the two together, if you need anything
   beyond the default pattern.

Copy this folder, replace those pieces, and you have a working extraction
pipeline.

## Background

The original pipeline extracts **prosopographical statements** from historical source texts (digests, charters, letters, ...) and turns them into ManMax-compatible factoids for the APIS
database. It runs locally on a single GPU via
[`llama-server`](https://github.com/ggml-org/llama.cpp) with a Qwen GGUF model,
chained over ~12 steps (translation (optional) → statement extraction → typing → nesting →
semantic-field extraction → APIS JSON).

This template strips all of that domain content out and keeps only the
reusable framework: the step runner, the LLM provider, the JSON
parser + auto-fixer, and a single placeholder `step1` that you replace.

## What you get

| File                                                   | Purpose                                                                                                            |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| [pipeline/run.py](pipeline/run.py)                     | Step runner — `step_by_step` and `file_by_file` execution modes, log tee-ing, error handling, input-dir chaining   |
| [pipeline/settings.py](pipeline/settings.py)           | Pydantic settings + `MODEL_PRESETS` + YAML loader                                                                  |
| [pipeline/provider.py](pipeline/provider.py)           | Auto-spawns `llama-server`; also supports OpenAI / Anthropic                                                       |
| [pipeline/parser.py](pipeline/parser.py)               | Extracts JSON from LLM output, validates against Pydantic, auto-fixes malformed responses via a follow-up LLM call |
| [pipeline/helpers.py](pipeline/helpers.py)             | `load_json`, `save_json`, `preserve_metadata`, `save_prompt`, `save_raw`, `get_text`                               |
| [pipeline/steps/step1.py](pipeline/steps/step1.py)     | Placeholder step — replace this                                                                                    |
| [pipeline/prompts/step1.py](pipeline/prompts/step1.py) | Placeholder prompt — replace this                                                                                  |
| [pipeline/models/step1.py](pipeline/models/step1.py)   | Placeholder Pydantic output model — replace this                                                                   |
| [config.yaml](config.yaml)                             | All runtime configuration                                                                                          |
| [data/input/example.json](data/input/example.json)     | Example input file                                                                                                 |

The placeholder `step1` performs a small example extraction — it pulls
`persons` (with optional `role`) and `events` (with the persons involved)
from the input text. Use it as a working reference; replace the model and
prompt with your own schema.

## Quick start

1. **Install dependencies.**

   ```bash
   pip install -r requirements.txt
   ```

2. **Point at your `llama-server` binary** in [config.yaml](config.yaml):

   ```yaml
   llm:
     llama_server_bin: /path/to/llama.cpp/build/bin/llama-server
     model_preset: qwen3.5-35b # or any other preset in settings.py
   ```

   The model GGUF is pulled from HuggingFace on first run.

3. **Drop input files** into `data/input/` (one JSON per document — see
   [example.json](data/input/example.json) for the shape).

4. **Run the pipeline:**

   ```bash
   python -m pipeline
   ```

   Output goes to `data/output/step1/`, with prompts and raw LLM responses
   saved alongside for debugging.

## Customising for your use case

### 1. Configure

Edit [config.yaml](config.yaml):

- `llm.model_preset` — pick from `MODEL_PRESETS` in
  [settings.py](pipeline/settings.py), or set `model_path` to an already-downloaded `.gguf`
- `output_language` / `prompt_language` — what the LLM writes / the prompt language
- `input_dir`, `output_dir`, `steps_to_run`, `execution_mode`
- `fixer.enabled` — auto-repair malformed LLM JSON via a second LLM call

### 2. Define your output schema

Replace [models/step1.py](pipeline/models/step1.py) with the structure
you want the LLM to return. The Pydantic model is used both for validation
_and_ shown to the LLM as an example via `get_example_output()`.

### 3. Write your prompt

Replace [prompts/step1.py](pipeline/prompts/step1.py). Each prompt module
exposes:

- `build_system_prompt(output_language, lang) -> str`
- `build_user_message(text, lang) -> str`

The system prompt should include the JSON schema example (the placeholder
already does this with `OUTPUT_FORMAT_EXAMPLE`).

### 4. Adjust the step (usually unchanged)

[steps/step1.py](pipeline/steps/step1.py) follows a fixed pattern:
get text → build prompts → call LLM → parse + validate → return result.
You only need to edit this if your step needs cross-step inputs or extra
post-processing.

### 5. Add more steps

To add `step2`:

1. Create `pipeline/steps/step2.py`, `pipeline/prompts/step2.py`,
   `pipeline/models/step2.py` (copy from `step1.*`).
2. Register the step in [run.py](pipeline/run.py):

   ```python
   from .steps import step1, step2

   STEP_REGISTRY = {
       "step1": step1.run,
       "step2": step2.run,
   }

   _STEP_INPUT_MAP = {
       "step2": "step1",   # step2 reads step1's output
   }
   ```

3. Add a `step2: StepConfig` field on `Settings` in [settings.py](pipeline/settings.py).
4. Add `step2` to `steps_to_run` in [config.yaml](config.yaml).

## CLI overrides

```bash
python -m pipeline                              # use config.yaml defaults
python -m pipeline --config my_config.yaml      # different config file
python -m pipeline --steps step1 step2          # run subset of steps
python -m pipeline --input-dir data/input/v2    # override input dir
python -m pipeline --output-dir data/output/v2  # override output dir
python -m pipeline --mode file_by_file          # run all steps per file
```

## Execution modes

- **`step_by_step`** (default) — process all input files for step N, then for
  step N+1, etc. Best when steps have very different runtimes or when you want
  to inspect a step's output before the next one starts.
- **`file_by_file`** — run the full pipeline on file 1, then file 2, etc.
  Best for early debugging of a single document.

## Cloud providers

Set `provider: openai` or `provider: anthropic` in [config.yaml](config.yaml)
and provide `model_name` + `api_key` (or set `OPENAI_API_KEY` /
`ANTHROPIC_API_KEY` in the environment). Install the corresponding SDK from
the commented section in [requirements.txt](requirements.txt).

## Citation

If you use this software, please cite it using the metadata in
[CITATION.cff](CITATION.cff):

> Sagadin, S., Hadden, R., Tambuscio, M., & Vogeler, G. (2025).
> _LLM Extraction Pipeline_ [Software]. Managing Maximilian (ManMax).
> Institute for Digital Humanities, University of Graz.
> https://github.com/managing-maximilian/llm-extraction-pipeline

## Acknowledgements

This software was developed as part of the [Managing Maximilian (ManMax)](https://www.oeaw.ac.at/imafo/forschung/editionsunternehmen-quellenforschungmir/managing-maximilian-sfb/managing-maximilian-1493-1519) project at the Institute for Digital Humanities, University of Graz, supported by the Austrian Science Fund (FWF) within the Special Research Programme SFB F92 Managing Maximilian (DOI: [10.55776/F92](https://doi.org/10.55776/F92)).

## License

This project is licensed under the [MIT License](LICENSE).
