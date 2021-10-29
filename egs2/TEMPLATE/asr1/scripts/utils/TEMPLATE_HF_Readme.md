---
tags:
- espnet
- audio
- ${hf_task}
language: ${lang}
datasets:
- ${_corpus}
license: cc-by-4.0
---

## ESPnet2 ${espnet_task} model 

### \`${hf_repo}\`

This model was trained by ${_creator_name} using ${_task} recipe in [espnet](https://github.com/espnet/espnet/).

### Demo: How to use in ESPnet2

\`\`\`bash
cd espnet
${_checkout}
pip install -e .
cd $(pwd | rev | cut -d/ -f1-3 | rev)
./run.sh --skip_data_prep false --skip_train true --download_model ${hf_repo}
\`\`\`

$(cat "${task_exp}"/RESULTS.md)

## ${espnet_task} config

<details><summary>expand</summary>

\`\`\`
$(cat "${task_exp}"/config.yaml)
\`\`\`

</details>

$(if [ -z ${var+use_lm} ]; then 
    ${use_lm} && echo "## LM config
    
<details><summary>expand</summary>

\`\`\`
  $(cat "${lm_exp}"/config.yaml)
\`\`\`

</details>
    ";
fi)
