# Comfyui-InfiniNode-Model-Suite 🚀

[cite_start]Welcome to the **InfiniNode Model Suite**, a custom node pack for ComfyUI that transforms the process of manipulating generative AI models[cite: 1]. [cite_start]Our suite is a direct implementation of the "GUI-Based Key Converter Development Plan," designed to remove technical barriers for advanced AI practitioners and integrate seamlessly with existing image generation pipelines[cite: 2, 3, 4].

[cite_start]The InfiniNode Model Suite is engineered as a definitive toolkit for model merging, customization, and optimization, empowering users to automate complex workflows and experiment directly within the ComfyUI environment[cite: 4, 5].

---

## 🔑 Features at a Glance

[cite_start]The suite is divided into three core categories: Core I/O, StateDict Manipulation, and Advanced Merging[cite: 7]. [cite_start]These nodes utilize PyTorch and the `safetensors` library for secure and high-performance operations[cite: 8, 9].

### 1. Core I/O & Inspection Nodes

[cite_start]These nodes are the foundation for any model manipulation workflow, handling the loading, inspection, and saving of model data[cite: 11].

* [cite_start]**Load Model StateDict**: Loads a generative AI model file from disk into a `state_dict` object[cite: 12, 13].
    * [cite_start]Supports `.safetensors` and legacy `.ckpt` formats[cite: 15, 16].
    * [cite_start]Prioritizes `.safetensors` for superior security and performance[cite: 17, 18].
    * [cite_start]Includes a `weights_only=True` toggle for secure loading of `.ckpt` files, mitigating risks from the `pickle` module[cite: 19, 20].
    * [cite_start]Automatically handles nested `state_dict` keys[cite: 21, 22].

* [cite_start]**Save Model StateDict**: Saves a modified `state_dict` object back to a file[cite: 23, 24].
    * [cite_start]Outputs primarily to the recommended `.safetensors` format for security and efficiency[cite: 26, 27].
    * [cite_start]Option to save to `.ckpt` for backward compatibility[cite: 28, 29].
    * [cite_start]Allows users to select output precision between `fp32` (full) and `fp16` (half) to optimize for inference and VRAM usage[cite: 30, 31].

* **Compare StateDicts & Gen-Keymap (InfiniNode)**: A utility node that serves as the starting point for any complex merge. It's designed to be the starting point for any complex merge. This node takes `state_dict_a` (source model) and `state_dict_b` (target model) as inputs. [cite_start]It outputs a `comparison_report` and a `keymap_template`[cite: 32, 33, 34, 35]. [cite_start]The `keymap_template` is a ready-to-use YAML-formatted string that can be copied, edited, and fed directly into the `Apply Keymap File` node[cite: 35].

---

### 2. StateDict Manipulation Nodes

[cite_start]These nodes act as surgical tools for renaming, pruning, and aligning model parameters[cite: 37, 38].

* [cite_start]**Rename Keys**: Offers flexible methods for renaming keys within a `state_dict`[cite: 39, 40].
    * [cite_start]**Simple Find/Replace**: Basic text fields for direct key name changes[cite: 41, 42].
    * [cite_start]**Regex Support**: A toggle for advanced, pattern-based renaming[cite: 43, 44].
    * [cite_start]**Preview Mode**: An option to output a text log of proposed changes without altering the `state_dict`[cite: 44, 45].

* [cite_start]**Prune Keys**: Intelligently removes unwanted parameters to optimize model size and performance[cite: 45, 46].
    * [cite_start]**Pattern-Based Pruning**: Uses text patterns (with regex support) to remove keys matching specific criteria, such as all optimizer states[cite: 47, 48].
    * [cite_start]**Component Selection**: Checkboxes to easily remove entire architectural components like the VAE or Text Encoder[cite: 48, 49].

* [cite_start]**Apply Keymap File**: Applies a complex set of renaming rules from an external `.json` or `.yaml` configuration file[cite: 51, 52, 53]. [cite_start]This is essential for standardizing models or tackling obfuscated keys[cite: 54, 55].

---

### 3. Advanced Merging Nodes

[cite_start]This is the core of the suite, providing access to sophisticated model merging algorithms[cite: 56, 57].

* [cite_start]**Merge (Weighted Sum)**: Combines two or more models using a weighted average, which is ideal for blending different styles or aesthetics[cite: 58, 59, 61].

* [cite_start]**Merge (Add Difference)**: Implements "task arithmetic" to transfer a learned skill (or "task vector") from one model to another[cite: 62, 63, 65].

* [cite_start]**Merge (SLERP)**: Performs Spherical Linear Interpolation for a smoother, more geometrically sound merge, avoiding performance degradation often seen with linear interpolation[cite: 66, 67, 69].

* [cite_start]**Component Splitter / Combiner**: A set of utility nodes for deconstructing a model into its primary components and then reconstructing a new model from selected parts[cite: 70, 71]. [cite_start]This is essential for creating **hybrid models**[cite: 74, 75].

---

## 💡 Example: Hybrid Qwen-Image + SDXL Model

[cite_start]The following workflow demonstrates how the InfiniNode suite can combine the text rendering capabilities of Qwen-Image with the versatile image generation of SDXL[cite: 76, 77].

1.  [cite_start]**Load Models**: Use two `Load Model StateDict` nodes to load `qwen-image.safetensors` and `sdxl-base.safetensors`[cite: 78].
2.  **Inspect & Align**: Connect both outputs to `Compare StateDicts & Gen-Keymap` nodes to observe key differences and generate a `keymap_template`. [cite_start]Use the **`Apply Keymap File`** node with a community-provided `qwen_to_sdxl.yaml` map to translate Qwen-Image's keys to an SDXL-compatible format[cite: 80, 81, 82, 83]. [cite_start]This addresses the core challenge of interoperability[cite: 84].
3.  [cite_start]**Split Components**: Use the **`Component Splitter`** node on both the aligned Qwen-Image state_dict and the SDXL state_dict[cite: 85, 86].
4.  [cite_start]**Create Hybrid**: Take the Text Encoder output from the Qwen-Image splitter and the UNet and VAE outputs from the SDXL splitter[cite: 88, 89]. [cite_start]Feed these into a **`Component Combiner`** node[cite: 90].
5.  [cite_start]**Save Result**: Connect the output of the Component Combiner to a `Save Model StateDict` node to create a new hybrid model, such as `hybrid_qwen-sdxl_model.safetensors`, with potentially novel capabilities[cite: 91].

---

## 🌐 Find Us on GitHub

[cite_start]By launching the **InfiniNode Model Suite**, we are turning a strategic blueprint into a powerful toolkit that will significantly enhance workflow efficiency, foster innovation, and unlock new frontiers in generative AI research and art[cite: 95, 96].

For more information, contributions, or to report issues, please visit our GitHub repository:
[cite_start][https://github.com/InfiniNode](https://github.com/InfiniNode) [cite: 97]
