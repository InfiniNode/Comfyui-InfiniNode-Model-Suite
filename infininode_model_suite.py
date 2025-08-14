import torch
import os
import re
import json
from safetensors.torch import load_file, save_file

# --- Dependency Check for YAML ---
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("\033[93m[InfiniNode] Warning: PyYAML is not installed. .yaml keymap files will not be supported. To enable, run: pip install pyyaml\033[0m")

# --- Helper Functions ---
def get_state_dict_from_checkpoint(pl_sd):
    """Extracts the state_dict from a loaded checkpoint file."""
    if "state_dict" in pl_sd:
        return pl_sd["state_dict"]
    elif "state_dict" in pl_sd:
        return pl_sd["state_dict"]
    return pl_sd

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """
    Spherical Linear Interpolation for PyTorch tensors.
    """
    v0_norm = torch.linalg.norm(v0)
    v1_norm = torch.linalg.norm(v1)
    v0_normalized = v0 / v0_norm
    v1_normalized = v1 / v1_norm
    dot = torch.dot(v0_normalized.flatten(), v1_normalized.flatten())
    if torch.abs(dot) > DOT_THRESHOLD:
        res = (1 - t) * v0 + t * v1
    else:
        theta_0 = torch.acos(dot)
        sin_theta_0 = torch.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = torch.sin(theta_t)
        sin_theta_0_minus_t = torch.sin(theta_0 - theta_t)
        res = (sin_theta_0_minus_t / sin_theta_0) * v0 + (sin_theta_t / sin_theta_0) * v1
    return res

# --- Node Definitions ---

class InfiniNode_LoadStateDict:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "file_path": ("STRING", {"default": "models/checkpoints/model.safetensors"}) } }
    RETURN_TYPES = ("STATE_DICT",)
    FUNCTION = "load_state_dict"
    CATEGORY = "InfiniNode/IO"
    def load_state_dict(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[InfiniNode] File not found: {file_path}")
        print(f"[InfiniNode] Loading model from: {file_path}")
        if file_path.endswith(".safetensors"):
            state_dict = load_file(file_path, device="cpu")
        elif file_path.endswith(".ckpt"):
            print("\033[93m[InfiniNode] Warning: Loading a .ckpt file. For security, it's loaded with 'weights_only=True'.\033[0m")
            full_checkpoint = torch.load(file_path, map_location="cpu", weights_only=True)
            state_dict = get_state_dict_from_checkpoint(full_checkpoint)
        else:
            raise ValueError("[InfiniNode] Unsupported file format. Please use .safetensors or .ckpt.")
        return (state_dict,)

class InfiniNode_SaveStateDict:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "state_dict": ("STATE_DICT",), "file_path": ("STRING", {"default": "models/checkpoints/merged_model.safetensors"}), "precision": (["fp32", "fp16"],) } }
    RETURN_TYPES = ()
    FUNCTION = "save_state_dict"
    OUTPUT_NODE = True
    CATEGORY = "InfiniNode/IO"
    def save_state_dict(self, state_dict, file_path, precision):
        if not file_path.endswith(".safetensors"):
            print(f"\033[93m[InfiniNode] Warning: Output file does not end with .safetensors. It will be saved in this format for security.\033[0m")
            file_path = os.path.splitext(file_path)[0] + ".safetensors"
        print(f"[InfiniNode] Preparing to save to {file_path} with {precision} precision.")
        save_dict = {}
        if precision == "fp16":
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                    save_dict[k] = v.half()
                else:
                    save_dict[k] = v
        else:
            save_dict = state_dict
        save_file(save_dict, file_path)
        print(f"\033[92m[InfiniNode] Successfully saved model to: {file_path}\033[0m")
        return ()

class InfiniNode_CompareStateDicts:
    """Compares two state_dicts and generates a keymap template."""
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "state_dict_a": ("STATE_DICT",), "state_dict_b": ("STATE_DICT",) } }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("comparison_report", "keymap_template")
    FUNCTION = "compare_and_gen"
    CATEGORY = "InfiniNode/Utils"

    def compare_and_gen(self, state_dict_a, state_dict_b):
        keys_a = set(state_dict_a.keys())
        keys_b = set(state_dict_b.keys())
        
        common_keys = keys_a.intersection(keys_b)
        unique_to_a = keys_a.difference(keys_b)
        unique_to_b = keys_b.difference(common_keys)

        # --- Generate Comparison Report ---
        report = f"--- InfiniNode Comparison Report ---\n\n"
        report += f"Total Keys in A: {len(keys_a)}\n"
        report += f"Total Keys in B: {len(keys_b)}\n"
        report += f"Common Keys: {len(common_keys)}\n\n"
        
        report += f"--- Keys Unique to Model A ({len(unique_to_a)}) ---\n"
        for key in list(unique_to_a)[:20]: report += f"{key}\n"
        if len(unique_to_a) > 20: report += "...\n"
            
        report += f"\n--- Keys Unique to Model B ({len(unique_to_b)}) ---\n"
        for key in list(unique_to_b)[:20]: report += f"{key}\n"
        if len(unique_to_b) > 20: report += "...\n"

        # --- Generate Keymap Template (YAML for readability) ---
        template = "# InfiniNode Auto-Generated Keymap Template\n"
        template += "# Format: 'original_key_from_model_A': 'new_key_for_model_B'\n\n"
        
        template += "# --- Keys Unique to Model A (Need Mapping) ---\n"
        for key in unique_to_a:
            template += f"'{key}': 'PASTE_TARGET_KEY_HERE'\n"
            
        template += "\n# --- Common Keys (Mapped 1-to-1 by default) ---\n"
        for key in common_keys:
            template += f"'{key}': '{key}'\n"
            
        return (report, template)

class InfiniNode_ApplyKeymapFile:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "state_dict": ("STATE_DICT",), "keymap_file_path": ("STRING", {"default": "keymaps/my_map.yaml"}), } }
    RETURN_TYPES = ("STATE_DICT",)
    FUNCTION = "apply_keymap"
    CATEGORY = "InfiniNode/Manipulation"
    def apply_keymap(self, state_dict, keymap_file_path):
        if not os.path.exists(keymap_file_path):
            raise FileNotFoundError(f"[InfiniNode] Keymap file not found: {keymap_file_path}")
        if keymap_file_path.endswith(".json"):
            with open(keymap_file_path, 'r') as f: keymap = json.load(f)
        elif keymap_file_path.endswith((".yaml", ".yml")):
            if not YAML_AVAILABLE: raise ImportError("[InfiniNode] PyYAML is required for .yaml keymaps.")
            with open(keymap_file_path, 'r') as f: keymap = yaml.safe_load(f)
        else:
            raise ValueError("[InfiniNode] Unsupported keymap file type. Use .json or .yaml.")
        
        # This logic ensures unmapped keys from the source are discarded, and only mapped keys form the new dict.
        new_dict = {}
        mapped_count = 0
        for old_key, new_key in keymap.items():
            if old_key in state_dict:
                new_dict[new_key] = state_dict[old_key]
                mapped_count += 1
            
        print(f"[InfiniNode] Applied keymap from {os.path.basename(keymap_file_path)}. Mapped {mapped_count} keys to create new dict of size {len(new_dict)}.")
        return (new_dict,)

# --- Other nodes remain the same... ---
# (The code for Rename, Prune, Merging, and Components is unchanged)
class InfiniNode_RenameKeys:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "state_dict": ("STATE_DICT",), "find_pattern": ("STRING", {"default": "old_prefix."}), "replace_with": ("STRING", {"default": "new_prefix."}), "use_regex": ("BOOLEAN", {"default": False}), } }
    RETURN_TYPES = ("STATE_DICT",)
    FUNCTION = "rename_keys"
    CATEGORY = "InfiniNode/Manipulation"
    def rename_keys(self, state_dict, find_pattern, replace_with, use_regex):
        new_dict = {}
        for key, value in state_dict.items():
            if use_regex: new_key = re.sub(find_pattern, replace_with, key)
            else: new_key = key.replace(find_pattern, replace_with)
            new_dict[new_key] = value
        print(f"[InfiniNode] Key renaming complete. Processed {len(new_dict)} keys.")
        return (new_dict,)

class InfiniNode_PruneKeys:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "state_dict": ("STATE_DICT",), "prune_pattern": ("STRING", {"multiline": True, "default": "^ema_.*"}), } }
    RETURN_TYPES = ("STATE_DICT",)
    FUNCTION = "prune_keys"
    CATEGORY = "InfiniNode/Manipulation"
    def prune_keys(self, state_dict, prune_pattern):
        new_dict = {}
        keys_to_remove = set(key for key in state_dict.keys() if re.search(prune_pattern, key))
        for key, value in state_dict.items():
            if key not in keys_to_remove: new_dict[key] = value
        print(f"[InfiniNode] Pruning complete. Removed {len(keys_to_remove)} keys.")
        return (new_dict,)

class InfiniNode_MergeWeightedSum:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "state_dict_a": ("STATE_DICT",), "factor_a": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}), "state_dict_b": ("STATE_DICT",), "factor_b": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}), } }
    RETURN_TYPES = ("STATE_DICT",)
    FUNCTION = "merge"
    CATEGORY = "InfiniNode/Merging"
    def merge(self, state_dict_a, factor_a, state_dict_b, factor_b):
        merged_dict = {}
        keys_a, keys_b = set(state_dict_a.keys()), set(state_dict_b.keys())
        for key in keys_a.intersection(keys_b):
            merged_dict[key] = (state_dict_a[key] * factor_a) + (state_dict_b[key] * factor_b)
        for key in keys_a.difference(keys_b): merged_dict[key] = state_dict_a[key]
        print(f"[InfiniNode] Weighted Sum merge complete.")
        return (merged_dict,)

class InfiniNode_MergeAddDifference:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "base_model_a": ("STATE_DICT",), "finetuned_model_a": ("STATE_DICT",), "base_model_c": ("STATE_DICT",), } }
    RETURN_TYPES = ("STATE_DICT",)
    FUNCTION = "merge"
    CATEGORY = "InfiniNode/Merging"
    def merge(self, base_model_a, finetuned_model_a, base_model_c):
        merged_dict = {}
        for key in base_model_c.keys():
            if key in base_model_a and key in finetuned_model_a:
                merged_dict[key] = base_model_c[key] + (finetuned_model_a[key] - base_model_a[key])
            else: merged_dict[key] = base_model_c[key]
        print(f"[InfiniNode] Add Difference merge complete.")
        return (merged_dict,)

class InfiniNode_MergeSLERP:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "state_dict_a": ("STATE_DICT",), "state_dict_b": ("STATE_DICT",), "t_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}), } }
    RETURN_TYPES = ("STATE_DICT",)
    FUNCTION = "merge"
    CATEGORY = "InfiniNode/Merging"
    def merge(self, state_dict_a, state_dict_b, t_factor):
        merged_dict = {}
        keys_a, keys_b = set(state_dict_a.keys()), set(state_dict_b.keys())
        for key in keys_a.intersection(keys_b):
            merged_dict[key] = slerp(t_factor, state_dict_a[key].float(), state_dict_b[key].float())
        print(f"[InfiniNode] SLERP merge complete.")
        return (merged_dict,)

class InfiniNode_ComponentSplitter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"state_dict": ("STATE_DICT",)}}
    RETURN_TYPES = ("STATE_DICT", "STATE_DICT", "STATE_DICT")
    RETURN_NAMES = ("unet_dict", "vae_dict", "text_encoder_dict")
    FUNCTION = "split"
    CATEGORY = "InfiniNode/Components"
    def split(self, state_dict):
        unet_dict, vae_dict, te_dict = {}, {}, {}
        for key, value in state_dict.items():
            if key.startswith("model.diffusion_model."): unet_dict[key] = value
            elif key.startswith("first_stage_model."): vae_dict[key] = value
            elif key.startswith("cond_stage_model."): te_dict[key] = value
        print(f"[InfiniNode] Split complete: UNet ({len(unet_dict)}), VAE ({len(vae_dict)}), TE ({len(te_dict)}) keys.")
        return (unet_dict, vae_dict, te_dict)

class InfiniNode_ComponentCombiner:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "unet_dict": ("STATE_DICT",), "vae_dict": ("STATE_DICT",), "text_encoder_dict": ("STATE_DICT",), } }
    RETURN_TYPES = ("STATE_DICT",)
    FUNCTION = "combine"
    CATEGORY = "InfiniNode/Components"
    def combine(self, unet_dict, vae_dict, text_encoder_dict):
        combined_dict = {}
        combined_dict.update(unet_dict)
        combined_dict.update(vae_dict)
        combined_dict.update(text_encoder_dict)
        print(f"[InfiniNode] Combine complete. Total keys: {len(combined_dict)}")
        return (combined_dict,)

# --- Node Mappings for ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "InfiniNode_LoadStateDict": InfiniNode_LoadStateDict,
    "InfiniNode_SaveStateDict": InfiniNode_SaveStateDict,
    "InfiniNode_CompareStateDicts": InfiniNode_CompareStateDicts, # <-- NEW NODE
    "InfiniNode_ApplyKeymapFile": InfiniNode_ApplyKeymapFile,
    "InfiniNode_RenameKeys": InfiniNode_RenameKeys,
    "InfiniNode_PruneKeys": InfiniNode_PruneKeys,
    "InfiniNode_MergeWeightedSum": InfiniNode_MergeWeightedSum,
    "InfiniNode_MergeAddDifference": InfiniNode_MergeAddDifference,
    "InfiniNode_MergeSLERP": InfiniNode_MergeSLERP,
    "InfiniNode_ComponentSplitter": InfiniNode_ComponentSplitter,
    "InfiniNode_ComponentCombiner": InfiniNode_ComponentCombiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InfiniNode_LoadStateDict": "Load Model StateDict (InfiniNode)",
    "InfiniNode_SaveStateDict": "Save Model StateDict (InfiniNode)",
    "InfiniNode_CompareStateDicts": "Compare & Gen-Keymap (InfiniNode)", # <-- NEW NODE
    "InfiniNode_ApplyKeymapFile": "Apply Keymap File (InfiniNode)",
    "InfiniNode_RenameKeys": "Rename Keys (InfiniNode)",
    "InfiniNode_PruneKeys": "Prune Keys (InfiniNode)",
    "InfiniNode_MergeWeightedSum": "Merge (Weighted Sum) (InfiniNode)",
    "InfiniNode_MergeAddDifference": "Merge (Add Difference) (InfiniNode)",
    "InfiniNode_MergeSLERP": "Merge (SLERP) (InfiniNode)",
    "InfiniNode_ComponentSplitter": "Component Splitter (InfiniNode)",
    "InfiniNode_ComponentCombiner": "Component Combiner (InfiniNode)",
}