# DUAL-REFLECT: Enhanced Translation with Dual-Reflective Learning

## Background

Large language models (LLMs) have shown remarkable abilities in various tasks, including machine translation. Recent advancements have demonstrated that LLMs can improve translation quality by employing self-reflective methods to refine initial drafts through feedback loops. However, the effectiveness of this self-reflection is often constrained by limited feedback, impacting the continuous improvement of translations.

![intro-main-v2_00](https://github.com/user-attachments/assets/ff9d89d6-b000-4306-a7bc-fd2c3f794bda)

To tackle this issue, we introduce **DUAL-REFLECT**, a framework that leverages the duality property of translation tasks to provide effective feedback to LLMs, thereby enhancing their reflective capabilities and improving translation performance. DUAL-REFLECT stands for **DUAL** learning enhanced auto-**REFLEC**tive **T**ranslation and consists of five stages:



1. **Draft Translation**: LLMs generate an initial translation.
2. **Back Translation**: The draft translation is translated back to the source language.
3. **Process Assessment**: An LLM-based agent evaluates whether dual reflection is needed.
4. **Dual Reflection**: LLMs analyze discrepancies between back-translation and the original source to identify biases and propose improvements.
5. **Auto Revision**: LLMs revise the initial translation based on the analysis and suggestions.

Our experiments show that DUAL-REFLECT significantly enhances translation performance across various languages and benchmarks. It outperforms strong baseline methods and achieves superior results, especially in low-resource translation tasks.

## Installation

To use DUAL-REFLECT, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/dual-reflect.git
   ```

2. Navigate to the project directory:
   ```bash
   cd dual-reflect
   ```

3. Use the Dual-Reflect Method:
   ```evaluate.pybash
   python agent_with_LLM_as_judge.py

   python agent_with_QE_as_judge.py
   ```

4. If you want to debug the code:
   ```evaluate.py
   python evaluate.py
   ```

## Usage

To run DUAL-REFLECT, execute the following command:

```bash
python main.py --config config.yaml
```

Adjust the `config.yaml` file as needed for your specific translation tasks and data.

## Contributions

We welcome contributions to DUAL-REFLECT! 

```
@inproceedings{chen-etal-2024-dual,
    title = "{DUAL}-{REFLECT}: Enhancing Large Language Models for Reflective Translation through Dual Learning Feedback Mechanisms",
    author = "Chen, Andong  and
      Lou, Lianzhang  and
      Chen, Kehai  and
      Bai, Xuefeng  and
      Xiang, Yang  and
      Yang, Muyun  and
      Zhao, Tiejun  and
      Zhang, Min",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-short.64",
    pages = "693--704",
    abstract = "Recently, large language models (LLMs) enhanced by self-reflection have achieved promising performance on machine transla004 tion. The key idea is guiding LLMs to generate translation with human-like feedback. However, existing self-reflection methods lack effective feedback information, limiting the translation performance. To address this, we introduce a DUAL-REFLECT framework, leveraging the dual learning of translation tasks to provide effective feedback, thereby enhancing the models{'} self-reflective abilities and improving translation performance. The application of this method across various translation tasks has proven its effectiveness in improving translation accuracy and eliminating ambiguities, especially in translation tasks with low-resource language pairs.",
}
```
