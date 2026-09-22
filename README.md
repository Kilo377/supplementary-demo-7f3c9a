# GEMS

Official anonymous implementation of **Generative Embodied Multiple Behavior Control Systems for Human-Like Agents**.

<div align="center">
  <video src="https://anonymous.4open.science/api/repo/review-video-82f4/file/demo.mp4" width="100%" autoplay muted loop controls playsinline></video>
</div>

## Setup

Python 3.10 or later is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY="YOUR_API_KEY"
ollama pull bge-m3
ollama serve
```

## Easy run

This lightweight configuration uses a three-step greedy world-model rollout and a three-turn execution limit, so the behavior trace can be inspected quickly.

```bash
python3 run_gems.py \
  --agent-avatar michael_anderson \
  --scene unity_home \
  --intent "Michael plans to prepare dinner." \
  --computation-architecture world_model \
  --planning-strategy greedy \
  --iteration-number 3 \
  --safety-turns 3
```

The generated behavior log and HTML trace are written to the current directory.

## Paper configuration

For the full paper setting, use the world-model controller with a 30-step planning horizon:

```bash
python3 run_gems.py \
  --agent-avatar michael_anderson \
  --scene unity_home \
  --intent "Michael plans to prepare dinner." \
  --computation-architecture world_model \
  --world-model-mode complex \
  --planning-strategy greedy \
  --iteration-number 30 \
  --safety-turns 30
```

Use `python3 run_gems.py --help` for the remaining controller, planning, arbitration, and provider options.
