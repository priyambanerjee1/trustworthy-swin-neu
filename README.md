# Trustworthy Vision Transformer for Industrial Surface Defect Detection

Implementation of the adaptive trust-aware decision framework for industrial surface defect classification using Swin Transformer, post-hoc temperature scaling, and Monte Carlo (MC) Dropout.

## Overview
Rather than relying on uncalibrated softmax confidence, this framework derives an operational **Trust Score** ($T$):
$$T = C_{\text{cal}} \times (1 - U_{\text{norm}})$$
governing a 3-tier action policy:
* **Accept** ($T \ge 0.90$)
* **Human Review** ($0.60 \le T < 0.90$)
* **Reject** ($T < 0.60$)

Crucially, predictive uncertainty $U$ is normalized against **fixed in-domain reference bounds** ($U_{\min}^{\text{ref}}, U_{\max}^{\text{ref}}$) to prevent scale resetting under distribution shifts.

## Repository Setup
```bash
git clone [https://github.com/](https://github.com/)<your-username>/trustworthy-swin-neu.git
cd trustworthy-swin-neu
pip install -r requirements.txt
