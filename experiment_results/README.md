# Experiment Results Summary

## WMVeID863 Tri-modal Vehicle ReID under Flare

**Base**: CLIP ViT-B/16 + MFMP + MC Loss
**Original v2**: mAP 68.4%

## Experiment Grid

| Exp | Architecture | Best mAP | vs A (67.8%) | Status |
|-----|-------------|----------|---------------|--------|
| A | v2 baseline (seed fixed) | 67.8% | baseline | ✅ |
| B | v2 + FlareSampler | 58.5% (E24) | -9.3% | ❌ stopped E25 |
| C | v2 + ModalityDropout p=0.1 | 63.6% | -4.2% | ❌ |
| F | v2 + CoEN-lite | 68.0% | +0.2% | ✅ |
| G | v2 + CoEN-lite + FlareSampler | — | — | 🔄 running |

## Key Findings

1. **Flare sampler alone hurts** (B: -9.9% vs v2)
2. **Modality dropout alone hurts** (C: -4.8% vs v2)  
3. **CoEN-lite breaks even** (F: 68.0% vs v2 68.4%)
4. **CoEN + sampler** (G) — testing three-mechanism synergy

## Architecture

Three-layer defense: FlareBalancedSampler → CoEN Repair (feature) → CoEN Gate (loss)
