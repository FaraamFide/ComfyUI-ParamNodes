import os
import sys
import torch
import numpy as np
import folder_paths
import node_helpers
from PIL import Image, ImageOps, ImageSequence


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")


class ParamString:
    """
    An input node for a string (text) parameter from the API.
    e.g., for prompts.
    """
    CATEGORY = "Params/Input"
    FUNCTION = "get_value"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("value",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {"default": "", "multiline": True}),
            }
        }

    def get_value(self, value):
        return (value,)

class ParamInt:
    """
    An input node for an integer parameter from the API.
    e.g., for seed, steps, dimensions.
    """
    CATEGORY = "Params/Input"
    FUNCTION = "get_value"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    def get_value(self, value):
        return (value,)

class ParamFloat:
    """
    An input node for a float parameter from the API.
    e.g., for LoRA strength, CFG scale.
    """
    CATEGORY = "Params/Input"
    FUNCTION = "get_value"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "display": "number"
                }),
            }
        }

    def get_value(self, value):
        return (value,)

class ParamBoolean:
    """
    An input node for a boolean (True/False) parameter from the API.
    Used for controlling logic, like enabling/disabling features.
    """
    CATEGORY = "Params/Input"
    FUNCTION = "get_value"
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("value",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("BOOLEAN", {"default": True}),
            }
        }

    def get_value(self, value):
        return (value,)

class ParamUniversal:
    """
    A universal input node that can be connected to any COMBO widget.
    It takes a string from the API (e.g., a model filename) and outputs
    a universal type that the frontend allows connecting anywhere.
    """
    CATEGORY = "Params/Input"
    FUNCTION = "get_value"
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("value",)
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("STRING", {"default": "None", "multiline": False}),
            }
        }

    def get_value(self, value):
        return (value,)

class ParamImage:
    """
    An input node that loads an image from a specified file path.
    The path is provided by the API.
    NOTE: The file must be accessible by the ComfyUI server.
    """
    CATEGORY = "Params/Input"
    FUNCTION = "load_image"
    RETURN_TYPES = ("IMAGE", "MASK")

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image_path": ("STRING", {"default": "input/example.png"})}}

    def load_image(self, image_path):
        if not os.path.isabs(image_path):
            # If the path is relative, assume it's in the ComfyUI base directory
            # This allows for paths like "input/image.png" or "output/image.png"
            image_path = os.path.join(folder_paths.base_path, image_path)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        img = Image.open(image_path)
        output_images = []
        output_masks = []
        
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)



class HelperModelSwitch:
    """
    A simple and reliable switch for model streams.
    Based on a boolean input, it passes through one of two connected models.
    Useful for conditionally applying processes like LoRAs.
    """
    CATEGORY = "Helpers/Logic"
    FUNCTION = "switch"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_a": ("MODEL",),
                "model_b": ("MODEL",),
                "select_b": ("BOOLEAN", {"default": True}),
            }
        }

    def switch(self, model_a, model_b, select_b):
        if select_b:
            return (model_b,)
        else:
            return (model_a,)



NODE_CLASS_MAPPINGS = {
    "ParamString": ParamString,
    "ParamInt": ParamInt,
    "ParamFloat": ParamFloat,
    "ParamBoolean": ParamBoolean,
    "ParamUniversal": ParamUniversal,
    "ParamImage": ParamImage,
    "HelperModelSwitch": HelperModelSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ParamString": "String Param",
    "ParamInt": "Integer Param",
    "ParamFloat": "Float Param",
    "ParamBoolean": "Boolean Param",
    "ParamUniversal": "Universal Param (for Combos)",
    "ParamImage": "Image Path Param",
    "HelperModelSwitch": "Model Switch",
}