# Transformer from Scratch (Harvard Annotated Version)

This repository contains a PyTorch-based, from-scratch implementation of the original Transformer model, based on the paper *Attention Is All You Need* (Vaswani et al., 2017) and the [Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) tutorial by Harvard NLP.

**Author:** Jeonghwan Lee (이정환)  
**Goal:** To gain a deep understanding of the Transformer architecture by implementing each component step-by-step without relying on high-level libraries.

---

## Features

- Implemented entirely from scratch using PyTorch
- Closely follows the structure and logic of Harvard's Annotated Transformer
- Modular design for clarity and extensibility
- Suitable for learning, experimentation, and research adaptation

---

## Project Structure

```text
transformer-from-scratch/
├── model/
│   ├── embedding.py             # Token & Positional Embedding
│   ├── attention.py             # Scaled Dot-Product & Multi-Head Attention
│   ├── encoder.py               # Encoder Layer
│   ├── decoder.py               # Decoder Layer
│   ├── transformer.py           # Full Transformer Model
│   └── utils.py                 # Masking and helper functions
├── train.py                     # Training loop (to be implemented)
├── inference.py                 # Inference example (to be implemented)
├── data/                        # Optional dataset directory
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## References

- Vaswani et al., 2017. [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- Harvard NLP. [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

## Author

**Jeonghwan Lee (이정환)**  
Catholic University of Korea — AI Undergraduate & Research Intern  
GitHub: [https://github.com/go2gym365](https://github.com/go2gym365)