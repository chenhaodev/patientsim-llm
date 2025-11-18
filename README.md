# PatientSim v1.0.0

**A Persona-Driven Simulator for Realistic Doctor-Patient Interactions**

This repository contains the PatientSim dataset from PhysioNet and an updated simulation system for generating doctor-patient dialogues using modern Large Language Models.

## Overview

PatientSim evaluates how well LLMs can simulate realistic patient personas during medical consultations, addressing three research questions:

1. **RQ1**: Do LLMs naturally reflect diverse persona traits in their responses?
2. **RQ2**: Do LLMs accurately derive responses based on the given profile?
3. **RQ3**: Can LLMs reasonably fill in the blanks?

## Dataset

### Patient Profiles
- **170 patient profiles** derived from real medical records (MIMIC-ED dataset)
- **5 main diagnoses**: Intestinal obstruction (39), Pneumonia (34), UTI (34), MI (34), Cerebral infarction (29)
- **Dataset splits**:
  - Persona test: 108 patients
  - Info test: 52 patients
  - Validation: 10 patients

### Persona Attributes
Each profile includes:
- **CEFR Level** (A/B/C): Language proficiency
- **Personality** (plain/distrust): Cooperation level
- **Recall Level** (low/medium/high): Memory ability
- **Dazed Level** (normal/confused): Mental clarity

### Original Models Evaluated
The dataset includes simulation results from 8 models:
- gemini-2.5-flash-preview-04-17
- gpt-4o-mini
- deepseek-llama-70b
- llama3.1-70b-instruct, llama3.1-8b-instruct, llama3.3-70b-instruct
- qwen2.5-72b-instruct, qwen2.5-7b-instruct

## Updated Simulation System

We've implemented a new simulation system using modern APIs:

### New Models

| Old Models | New Model | Purpose |
|------------|-----------|---------|
| deepseek-llama-70b, llama3.x, qwen2.5-72b | **deepseek-api** | Primary patient simulator |
| gpt-4o-mini, gemini-2.5-flash | **gpt-4.1-api** | Doctor agent |
| qwen2.5-7b | **ollama:qwen3** | Lightweight local patient simulator |

### Quick Start

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your API keys:
# DEEPSEEK_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here
```

#### 3. Start Ollama (Optional)

If using `ollama:qwen3`:

```bash
ollama serve
ollama pull qwen3
```

#### 4. Test Connections

```bash
python generate_dialogues.py --test-connection
```

Expected output:
```
✓ gpt-4.1-api
✓ ollama:qwen3
✓ deepseek-api (if you have credits)
```

#### 5. Generate Test Dialogues

Run on 2 patients to test:

```bash
python generate_dialogues.py \
  --doctor-model gpt-4.1-api \
  --patient-model ollama:qwen3 \
  --splits persona \
  --limit 2
```

#### 6. Full Simulation

Generate all dialogues:

```bash
python generate_dialogues.py \
  --doctor-model gpt-4.1-api \
  --patient-model ollama:qwen3 \
  --splits persona,info
```

## Directory Structure

```
.
├── README.md                   # This file
├── README_SIMULATION.md        # Detailed simulation guide
├── LICENSE.txt                 # PhysioNet Credentialed Health Data License
├── config.yaml                 # Model configuration
├── requirements.txt            # Python dependencies
├── .env.example               # API key template
│
# Simulation Code
├── llm_client.py              # Unified LLM client
├── patient_agent.py           # Patient simulator with persona
├── doctor_agent.py            # Doctor interviewer
├── generate_dialogues.py      # Main simulation script
│
# Original Dataset
├── patient_profile.json       # 170 patient profiles
├── analysis.ipynb             # Original analysis notebook
├── persona_test/              # Persona fidelity evaluation
│   ├── dialogue.jsonl
│   ├── expert_dialogue.jsonl
│   └── llm_simulation/
├── info_test/                 # Information accuracy evaluation
│   ├── dialogue.jsonl
│   ├── expert_plausibility_label.jsonl
│   ├── llm_plausibility_label.jsonl
│   └── llm_simulation/
└── sentence_cls_valid/        # Validation set
│
# Generated Output (created on run)
└── simulation_output/
    ├── persona_test/llm_simulation/{model}/llm_dialogue.jsonl
    └── info_test/llm_simulation/{model}/llm_dialogue.jsonl
```

## Output Format

Each dialogue is saved in JSONL format:

```json
{
  "hadm_id": "28162080",
  "doctor_engine_name": "gpt-4.1-api",
  "patient_engine_name": "ollama:qwen3",
  "cefr_type": "B",
  "personality_type": "plain",
  "recall_level_type": "high",
  "dazed_level_type": "normal",
  "dialog_history": [
    {"role": "Doctor", "content": "Hello, I'm Dr. Smith..."},
    {"role": "Patient", "content": "Hi doctor, I've been having..."}
  ],
  "diagnosis": "Cerebral infarction"
}
```

## Key Features

### Patient Agent
- ✅ Persona-driven responses (CEFR, personality, recall, dazed)
- ✅ CEFR-appropriate vocabulary selection
- ✅ Natural patient behaviors (reluctance, confusion, emotion)
- ✅ Profile consistency (symptoms, history, medications)

### Doctor Agent
- ✅ Structured medical interview (HPI, ROS, PMH, social/family history)
- ✅ Adaptive questioning based on responses
- ✅ Empathetic and professional communication
- ✅ Differential diagnosis thinking

### LLM Client
- ✅ Unified interface for multiple providers
- ✅ OpenAI API (GPT-4.1)
- ✅ DeepSeek API
- ✅ Ollama (local models)

## Evaluation Metrics

The original dataset includes comprehensive evaluation:

### Persona Fidelity
- Personality alignment
- CEFR level accuracy
- Recall ability consistency
- Confusion level realism
- Overall realism and tool usefulness

### Information Accuracy
- Profile consistency (dialogue-level)
- Sentence-level support/entailment
- Contradiction rates
- Clinical plausibility scores

### Expert Annotations
- 4 expert labelers
- 821 utterances rated
- High inter-rater agreement (Gwet's AC1: 0.85-0.97)

## Original Results Summary

**Persona Fidelity** (5-point scale):
- Human baseline: 3.84-4.00
- Best LLM (Llama3.3-70b): 3.68
- Gemini 2.5 Flash: 3.57

**Information Accuracy**:
- Current visit info: 88-91% valid
- Medical history: 75-78% valid
- Social history: 44-61% valid
- Contradiction rate: 2-6%
- Expert plausibility: 3.91/5

## Advanced Usage

### Multi-Model Comparison

```bash
python generate_dialogues.py \
  --doctor-model gpt-4.1-api \
  --patient-model "deepseek-api,ollama:qwen3" \
  --splits persona,info
```

### Custom Configuration

```bash
python generate_dialogues.py \
  --config my_config.yaml \
  --doctor-model gpt-4.1-api \
  --patient-model ollama:qwen3
```

### Process Specific Splits

```bash
# Only persona test (108 patients)
python generate_dialogues.py --splits persona

# Only info test (52 patients)
python generate_dialogues.py --splits info
```

## Configuration

Edit `config.yaml` to customize:

```yaml
models:
  deepseek-api:
    temperature: 0.7      # Creativity (0.0-2.0)
    max_tokens: 2048      # Response length

  gpt-4.1-api:
    temperature: 0.7
    max_tokens: 2048

  ollama:qwen3:
    temperature: 0.7
    max_tokens: 2048

simulation:
  max_turns: 20          # Dialogue length
  output_dir: ./simulation_output
```

## Troubleshooting

### API Key Issues
```
Warning: DEEPSEEK_API_KEY not found
```
**Solution**: Add keys to `.env` and reload:
```bash
export $(cat .env | xargs)
```

### Ollama Connection Failed
```
Connection refused on localhost:11434
```
**Solution**: Start Ollama server:
```bash
ollama serve
```

### Model Not Found
```
Model not available
```
**Solution**: Check model name in `config.yaml` matches your API access

### Rate Limits
Add delays in `llm_client.py` if hitting rate limits.

## Performance Notes

- **deepseek-api**: ~2-3 sec/response, cost-effective
- **gpt-4.1-api**: ~1-2 sec/response, higher quality
- **ollama:qwen3**: <1 sec/response, free (local, requires GPU)

Estimated time for full simulation (160 patients):
- **~2-3 hours** depending on model speed

## Documentation

- **`README.md`**: This file - overview and quick start
- **`README_SIMULATION.md`**: Detailed simulation system guide
- **`analysis.ipynb`**: Original evaluation notebook

---

**For detailed simulation instructions, see [`README_SIMULATION.md`](README_SIMULATION.md)**
