# Baselines

Upstream implementations for the four baselines with public repositories, plus
an included SBR implementation.

| Folder | Upstream repository | Commit | License |
|---|---|---|---|
| `Vaccine/` | https://github.com/git-disl/Vaccine | `d4bd30098c2afda40d12a89197bb0ca898899e1f` | Apache-2.0 |
| `LiSA/` | https://github.com/git-disl/Lisa | `68e36595d3643ae5b18e955ee385bad71851d021` | Apache-2.0 |
| `SafeMERGE/` | https://github.com/aladinD/SafeMERGE | `672444428da73e4a8146239a23857f2cd4e056e6` | See upstream |
| `BAR/` | https://github.com/SeokilHam/Buffer-and-Reinforce | `8cf9ca04aad5d6bb8eca2f6913eb844563afe407` | Apache-2.0 |
| `SBR/` | — | — | Included implementation |

Each vendored folder keeps its upstream README and scripts. Run each baseline
according to its official instructions and default configuration.

- Vaccine: `baselines/Vaccine/script/alignment/Vaccine.sh`
- LiSA: `baselines/LiSA/script/finetune/lisa_poison_ratio.sh`
- SafeMERGE: `baselines/SafeMERGE/get_safemerge_model.py`
- BAR: `baselines/BAR/script/finetune/bufferlora_llama.sh`,
  `reinforcelora_llama.sh`, and `bufferlora_ft_llama.sh`
- SBR: `baselines/SBR/paper_protocol.py`
