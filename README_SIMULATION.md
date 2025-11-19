# PatientSim - Dialogue Generation System

This directory contains the updated simulation system for generating doctor-patient dialogues using modern LLM APIs.

## Updated Models

The new system uses these models instead of the legacy ones:

| Old Model | New Model | Purpose |
|-----------|-----------|---------|
| deepseek-llama-70b, llama3.x, qwen2.5-72b | **deepseek-api** | Primary patient simulator |
| gpt-4o-mini, gemini-2.5-flash | **gpt-5-mini** | Doctor agent (faster & cheaper) |
| qwen2.5-7b | **ollama:qwen3** | Lightweight patient simulator |

### Pricing (per 1M tokens)

**gpt-5-mini**: Input $0.25, Cached $0.025, Output $2.00
**deepseek-api**: Input $2.00, Cached $0.125, Output ~$6-8
**ollama:qwen3**: Free (local deployment)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your keys:
```bash
DEEPSEEK_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

### 3. Start Ollama (Optional)

If using `ollama:qwen3`, ensure Ollama is running:

```bash
ollama serve
ollama pull qwen3
```

### 4. Test Connections

Verify all APIs are accessible:

```bash
python generate_dialogues.py --test-connection
```

You should see:
```
✓ deepseek-api
✓ gpt-5-mini
✓ ollama:qwen3
```

## Configuration

Edit `config.yaml` to customize:

```yaml
models:
  deepseek-api:
    temperature: 0.7      # Adjust creativity
    max_tokens: 2048      # Response length

simulation:
  max_turns: 20          # Dialogue length
  output_dir: ./simulation_output
```

## Usage

### Quick Start - Generate Test Dialogues

Run on just 2 patients to test:

```bash
python generate_dialogues.py \
  --doctor-model gpt-5-mini \
  --patient-model deepseek-api \
  --splits persona \
  --limit 2
```

### Full Simulation - Single Patient Model

Generate all dialogues with one patient model:

```bash
python generate_dialogues.py \
  --doctor-model gpt-5-mini \
  --patient-model deepseek-api \
  --splits persona,info
```

This will:
- Process 108 patients from `persona` split
- Process 52 patients from `info` split
- Save to `simulation_output/{split}_test/llm_simulation/deepseek-api/`

### Multi-Model Comparison

Run with multiple patient models:

```bash
python generate_dialogues.py \
  --doctor-model gpt-5-mini \
  --patient-model "deepseek-api,ollama:qwen3" \
  --splits persona,info
```

This generates dialogues for each patient model separately.

## Output Structure

Dialogues are saved matching the original dataset structure:

```
simulation_output/
├── persona_test/
│   └── llm_simulation/
│       ├── deepseek-api/
│       │   └── llm_dialogue.jsonl
│       ├── gpt-5-mini/
│       │   └── llm_dialogue.jsonl
│       └── ollama:qwen3/
│           └── llm_dialogue.jsonl
└── info_test/
    └── llm_simulation/
        └── ...
```

Each `llm_dialogue.jsonl` contains:

```json
{
  "hadm_id": "28162080",
  "doctor_engine_name": "gpt-5-mini",
  "patient_engine_name": "deepseek-api",
  "cefr_type": "B",
  "personality_type": "plain",
  "recall_level_type": "high",
  "dazed_level_type": "normal",
  "dialog_history": [
    {"role": "Doctor", "content": "Hello, I'm Dr. Smith..."},
    {"role": "Patient", "content": "Hi doctor..."}
  ],
  "diagnosis": "Cerebral infarction"
}
```

## Advanced Usage

### Custom Configuration

Create a custom config file:

```bash
python generate_dialogues.py \
  --config my_config.yaml \
  --doctor-model gpt-5-mini \
  --patient-model deepseek-api
```

### Process Specific Splits

```bash
# Only persona test
python generate_dialogues.py --splits persona

# Only info test
python generate_dialogues.py --splits info

# Both (default)
python generate_dialogues.py --splits persona,info
```

### Limit Processing (for Testing)

```bash
# Process only 5 patients per split
python generate_dialogues.py --limit 5
```

## Persona Simulation

The system simulates diverse patient personas based on profile attributes:

### Language Proficiency (CEFR)
- **A**: Basic English, simple sentences, limited vocabulary
- **B**: Intermediate English, everyday vocabulary
- **C**: Advanced English, sophisticated vocabulary

### Personality
- **plain**: Cooperative, trusting, straightforward
- **distrust**: Guarded, skeptical, reluctant to share

### Memory/Recall
- **low**: Forgets details, vague responses
- **medium**: Remembers most information
- **high**: Clear memory, specific details

### Mental Clarity
- **normal**: Clear-headed
- **confused**: Disoriented, loses track of conversation

## Troubleshooting

### API Key Errors

```
Warning: DEEPSEEK_API_KEY not found for deepseek-api
```

**Solution**: Add the key to `.env` file and reload:
```bash
export $(cat .env | xargs)
python generate_dialogues.py --test-connection
```

### Ollama Connection Failed

```
Error generating from ollama:qwen3: Connection refused
```

**Solution**: Start Ollama server:
```bash
ollama serve
```

### Model Not Available

```
Model gpt-4-turbo-2024-04-09 not found
```

**Solution**: Update `config.yaml` with correct model name for your API access level.

### Rate Limits

If hitting API rate limits, add delays in `llm_client.py`:

```python
import time
time.sleep(1)  # Add after each API call
```

## Evaluation

After generating dialogues, use the original `analysis.ipynb` to evaluate:

1. Update analysis notebook to point to your output directory
2. Run evaluation metrics (persona fidelity, information accuracy)
3. Compare with baseline models

## File Structure

```
.
├── config.yaml              # Model and simulation configuration
├── requirements.txt         # Python dependencies
├── .env.example            # API key template
├── llm_client.py           # Unified LLM client wrapper
├── patient_agent.py        # Patient simulator with persona
├── doctor_agent.py         # Doctor interviewer
├── generate_dialogues.py   # Main simulation script
├── patient_profile.json    # 170 patient profiles (original)
└── simulation_output/      # Generated dialogues (created on run)
```

## Performance Notes

- **deepseek-api**: ~2-3 sec/response, cost-effective
- **gpt-5-mini**: ~1-2 sec/response, fast & cheap ($0.25/1M input)
- **ollama:qwen3**: <1 sec/response (local), free but requires GPU

Estimated time for full simulation:
- 160 patients × 20 turns × 2.5 sec/turn ≈ **2.2 hours**

## Citation

If you use this updated simulation system, please cite both the original PatientSim dataset and note the model updates:

```
PatientSim v1.0.0 (PhysioNet)
Updated with: DeepSeek API, GPT-4.1 API, Ollama Qwen3
```

## Support

For issues with:
- Original dataset: See PhysioNet documentation
- This simulation code: Check logs, verify API keys, test connections
- Model APIs: Consult provider documentation (OpenAI, DeepSeek, Ollama)
