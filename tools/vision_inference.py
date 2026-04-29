import os
import torch
from PIL import Image
import base64
from io import BytesIO
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class LocalVisionInference:
    """
    Handles local inference for the Qwen3-VL vision model.
    Loads the model from the local safetensors/transformers directory.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Lazy loading of the vision model to save memory."""
        if self.model is not None:
            return
            
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            logger.info(f"Loading local vision model from {self.model_path} on {self.device}...")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
            # Load model
            # Note: We use flash_attention_2 if available and on GPU
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            if self.device == "cpu":
                self.model = self.model.to("cpu")
                
            logger.info("Local vision model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load local vision model: {e}")
            raise e

    def process_image(self, prompt: str, image_base64: str) -> str:
        """
        Analyze an image using the local model.
        """
        if self.model is None:
            self.load_model()
            
        try:
            # Decode image
            image = Image.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")
            
            # Prepare multimodal prompt
            # Qwen2-VL specific format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # Process input
            from transformers import Qwen2VLForConditionalGeneration
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Note: Qwen2-VL processor handle images internally when using apply_chat_template/process_vision_info
            # But for simplicity in this integration:
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            # Generate
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return output_text
        except Exception as e:
            logger.error(f"Error during vision inference: {e}")
            return f"Error analyzing image: {str(e)}"
