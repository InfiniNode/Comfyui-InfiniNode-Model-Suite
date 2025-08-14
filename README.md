# Comfyui-InfiniNode-Model-Suite 🚀

Welcome to the **InfiniNode Model Suite**, a custom node pack for ComfyUI that transforms the process of manipulating generative AI models. Our suite is a direct implementation of the "GUI-Based Key Converter Development Plan," designed to remove technical barriers for advanced AI practitioners and integrate seamlessly with existing image generation pipelines.

The InfiniNode Model Suite is engineered as a definitive toolkit for model merging, customization, and optimization, empowering users to automate complex workflows and experiment directly within the ComfyUI environment.

---

## 🔑 Features at a Glance

The suite is divided into three core categories: Core I/O, StateDict Manipulation, and Advanced Merging. These nodes utilize PyTorch and the `safetensors` library for secure and high-performance operations.

### 1. Core I/O & Inspection Nodes

These nodes are the foundation for any model manipulation workflow, handling the loading, inspection, and saving of model data.

* **Load Model StateDict**: Loads a generative AI model file from disk into a `state_dict` object.
    * Supports `.safetensors` and legacy `.ckpt` formats.
    * Prioritizes `.safetensors` for superior security and performance.
    * Includes a `weights_only=True` toggle for secure loading of `.ckpt` files, mitigating risks from the `pickle` module.
    * Automatically handles nested `state_dict` keys.

* **Save Model StateDict**: Saves a modified `state_dict` object back to a file.
    * Outputs primarily to the recommended `.safetensors` format for security and efficiency.
    * Option to save to `.ckpt` for backward compatibility.
    * Allows users to select output precision between `fp32` (full) and `fp16` (half) to optimize for inference and VRAM usage.

* **Compare StateDicts & Gen-Keymap (InfiniNode)**: A utility node that serves as the starting point for any complex merge. It's designed to be the starting point for any complex merge. This node takes `state_dict_a` (source model) and `state_dict_b` (target model) as inputs. It outputs a `comparison_report` and a `keymap_template`. The `keymap_template` is a ready-to-use YAML-formatted string that can be copied, edited, and fed directly into the `Apply Keymap File` node.

![Screenshot_1771](https://github.com/user-attachments/assets/16edccab-cb2f-4176-ac2a-40239935af61)


---

### 2. StateDict Manipulation Nodes

These nodes act as surgical tools for renaming, pruning, and aligning model parameters.

* **Rename Keys**: Offers flexible methods for renaming keys within a `state_dict`.
    * **Simple Find/Replace**: Basic text fields for direct key name changes.
    * **Regex Support**: A toggle for advanced, pattern-based renaming.
    * **Preview Mode**: An option to output a text log of proposed changes without altering the `state_dict`.

* **Prune Keys**: Intelligently removes unwanted parameters to optimize model size and performance.
    * **Pattern-Based Pruning**: Uses text patterns (with regex support) to remove keys matching specific criteria, such as all optimizer states.
    * **Component Selection**: Checkboxes to easily remove entire architectural components like the VAE or Text Encoder.

* **Apply Keymap File**: Applies a complex set of renaming rules from an external `.json` or `.yaml` configuration file. This is essential for standardizing models or tackling obfuscated keys.

---

### 3. Advanced Merging Nodes

This is the core of the suite, providing access to sophisticated model merging algorithms.

* **Merge (Weighted Sum)**: Combines two or more models using a weighted average, which is ideal for blending different styles or aesthetics.

* **Merge (Add Difference)**: Implements "task arithmetic" to transfer a learned skill (or "task vector") from one model to another.

* **Merge (SLERP)**: Performs Spherical Linear Interpolation for a smoother, more geometrically sound merge, avoiding performance degradation often seen with linear interpolation.

* **Component Splitter / Combiner**: A set of utility nodes for deconstructing a model into its primary components and then reconstructing a new model from selected parts. This is essential for creating **hybrid models**.

---

## 💡 Example: Hybrid Qwen-Image + SDXL Model

The following workflow demonstrates how the InfiniNode suite can combine the text rendering capabilities of Qwen-Image with the versatile image generation of SDXL.

1.  **Load Models**: Use two `Load Model StateDict` nodes to load `qwen-image.safetensors` and `sdxl-base.safetensors`.
2.  **Inspect & Align**: Connect both outputs to `Compare StateDicts & Gen-Keymap` nodes to observe key differences and generate a `keymap_template`. Use the **`Apply Keymap File`** node with a community-provided `qwen_to_sdxl.yaml` map to translate Qwen-Image's keys to an SDXL-compatible format. This addresses the core challenge of interoperability.
3.  **Split Components**: Use the **`Component Splitter`** node on both the aligned Qwen-Image state_dict and the SDXL state_dict.
4.  **Create Hybrid**: Take the Text Encoder output from the Qwen-Image splitter and the UNet and VAE outputs from the SDXL splitter. Feed these into a **`Component Combiner`** node.
5.  **Save Result**: Connect the output of the Component Combiner to a `Save Model StateDict` node to create a new hybrid model, such as `hybrid_qwen-sdxl_model.safetensors`, with potentially novel capabilities.

---

## 🌐 Find Us on GitHub

By launching the **InfiniNode Model Suite**, we are turning a strategic blueprint into a powerful toolkit that will significantly enhance workflow efficiency, foster innovation, and unlock new frontiers in generative AI research and art.

For more information, contributions, or to report issues, please visit our GitHub repository:
[https://github.com/InfiniNode](https://github.com/InfiniNode) 
