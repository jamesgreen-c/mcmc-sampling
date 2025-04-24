# MCMC Decryption with Stationary and Transition Probabilities

This project uses **Markov Chain Monte Carlo (MCMC)** to decrypt an encrypted message, assuming that encryption is a character-level substitution. We leverage:

- **Stationary distributions** of characters in English (from *War and Peace*)
- **Transition probabilities** between characters
- A **Metropolis-Hastings** sampler to discover the most probable decryption

---

## Overview

The decryption process assumes:

- A one-to-one mapping from encrypted characters to actual English characters
- The original message is written in a way that respects the character-level transition structure of English text

---

## Key Components

### 📚 Text Preprocessing
We use `TextPreprocessor` to clean and lower the input texts, remove newlines, and restrict to valid symbols.

## 🔧 Setup

First, load the required files:

```python
# Load required data files
with open("../data/war_and_peace.txt", "r", encoding="utf-8") as file:
    text = file.read()
with open("../data/symbols.txt", "r", encoding="utf-8") as file:
    valid_symbols = file.read()
with open("../data/message.txt", "r", encoding="utf-8") as file:
    message = file.read()
```

### 📊 Stationary Distribution

Computed using the frequency of each character in *War and Peace*:

```python
from core.models.distribution import StationaryDistribution
phi = StationaryDistribution(text, valid_symbols).distribution
```

### 🔁 Transition Matrix

Represents first-order character transitions (e.g., 't' → 'h'):

```python
from core.models.distribution import Transitions
psi = Transitions(text, valid_symbols).transitions
```

### 🔐 MCMC Decryption

We use a Metropolis-Hastings sampler to:
- Propose a new sigma (character mapping)
- Accept or reject based on joint log-likelihood of the decrypted message

```python
from core.models.mcmc import MCMCDecrypter
decrypter = MCMCDecrypter(text, valid_symbols, message)
```

---

## Directory Structure

```
core/
├── models/
│   ├── distribution.py     # StationaryDistribution, Transitions
│   └── mcmc.py             # MCMCDecrypter and utilities
data/
├── war_and_peace.txt       # Source text for building probabilities
├── symbols.txt             # Valid symbols (e.g., lowercase letters)
├── message.txt             # Encrypted message to decrypt
scripts/
└── run_decryption.py       # Main script
```

---

## How It Works

1. **Compute `phi`** (stationary probabilities) from *War and Peace*
2. **Compute `psi`** (transition probabilities) from the same text
3. Initialize `sigma` with frequency-matching of message to text
4. Iterate:
   - Propose `sigma'` by swapping two characters
   - Accept `sigma'` with probability:

<pre> ``` min(1, [P_phi(s₁) × ∏ P_psi(s_t | s_{t−1})] / [P_phi(s₁′) × ∏ P_psi(s_t′ | s_{t−1}′)]) ``` </pre>

5. Track the best `sigma` and log-likelihoods

---

## Usage

```bash
python scripts/run_decryption.py
```

This will:
- Load data from the `data/` directory
- Run the MCMC sampler
- Print the partial decrypted text every 100 iterations
- Plot the log-likelihood trace

---

## Example Output

```
on lw whunges tnm lhse furnestpre wetsi lw ytades gtfe le ih
as ly yousder is. lore cumserifme yeirn ly kither dice le no
an my younger ind more culnerible yeirs my kither gice me so
in my younger and more culnerable years my kather gace me so
in my younger and more vulnerable years my father gave me so
```

(decryption improves with iterations)

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Notes

- Decryption assumes no missing or extra characters — just scrambled
- MCMC sampler swaps character mappings and accepts based on log joint probabilities
- Best viewed after a few thousand iterations for convergence

---

## License

MIT License

---

## Acknowledgements

Inspired by MCMC applications in cryptography and language modeling.
